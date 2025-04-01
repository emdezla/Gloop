import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from collections import deque
from torch.utils.data import Dataset, DataLoader

# --------------------------
# Global Settings & Constants
# --------------------------
# Define device early
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Final learning rates for each optimizer (used during warmup)
FINAL_ACTOR_LR = 1e-5
FINAL_CRITIC_LR = 1e-4
FINAL_ALPHA_LR = 1e-5

# --------------------------
# Data Handling
# --------------------------
class DiabetesDataset(Dataset):
    """Processed diabetes management dataset."""
    
    def __init__(self, csv_file):
        # Load data and fill missing values
        df = pd.read_csv(csv_file)
        df = df.ffill().bfill()
        
        # Ensure key features have no missing values
        if df[["glu", "glu_d", "glu_t", "hr", "hr_d", "hr_t", "iob", "hour"]].isna().any().any():
            raise ValueError("Dataset contains NaN values after preprocessing")
        
        # Verify that action values are within [-1, 1]
        assert df["action"].between(-1, 1).all(), "Actions must be between -1 and 1"
        
        # Prepare state features and action values
        self.states = df[["glu", "glu_d", "glu_t", "hr", "hr_d", "hr_t", "iob", "hour"]].values.astype(np.float32)
        self.actions = df["action"].values.astype(np.float32).reshape(-1, 1)
        
        # Compute rewards from the glu_raw values
        self.rewards = self._compute_rewards(df["glu_raw"].values)
        
        # Create transitions: next_states via roll and done flags
        self.next_states = np.roll(self.states, -1, axis=0)
        self.dones = df["done"].values.astype(np.float32)
        
        # Remove last transition (invalid next state)
        self._sanitize_transitions()
    
    def _compute_rewards(self, glucose_next):
        """
        Compute rewards using the Risk Index (RI)-based function.
        Based on Kovatchev et al. (2005) and extended with a penalty for severe hypoglycemia.
        """
        glucose = np.clip(glucose_next.astype(np.float32), 10, 400)  # Clamp extreme values

        # Step 1: Risk transformation function
        log_glucose = np.log(glucose)
        f = 1.509 * (np.power(log_glucose, 1.084) - 5.381)
        r = 10 * np.square(f)

        # Step 2: LBGI and HBGI
        lbgi = np.where(f < 0, r, 0)
        hbgi = np.where(f > 0, r, 0)

        # Step 3: Total Risk Index (RI)
        ri = (lbgi + hbgi)

        # Step 4: Normalize RI to [0, -1] range and convert to reward
        normalized_ri = -ri / 100.0
        rewards = np.clip(normalized_ri, -1.0, 0.0)

        # Step 5: Severe hypoglycemia penalty
        severe_hypo_penalty = np.where(glucose <= 39, -15.0, 0.0)
        rewards += severe_hypo_penalty

        # Step 6: Optional time penalty (to encourage faster control)
        rewards -= 0.01

        return np.clip(rewards, -15.0, 0.0).astype(np.float32)

    
    def _sanitize_transitions(self):
        """Remove the last transition which lacks a valid next state."""
        valid_mask = np.ones(len(self.states), dtype=bool)
        valid_mask[-1] = False
        self.states = self.states[valid_mask]
        self.actions = self.actions[valid_mask]
        self.rewards = self.rewards[valid_mask]
        self.next_states = self.next_states[valid_mask]
        self.dones = self.dones[valid_mask]
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return {
            'state': torch.FloatTensor(self.states[idx]),
            'action': torch.FloatTensor(self.actions[idx]),
            'reward': torch.FloatTensor([self.rewards[idx]]),
            'next_state': torch.FloatTensor(self.next_states[idx]),
            'done': torch.FloatTensor([self.dones[idx]])
        }

# --------------------------
# Neural Network Components: SAC Agent
# --------------------------
class SACAgent(nn.Module):
    """Simplified Soft Actor-Critic (SAC) agent."""
    
    def __init__(self, state_dim=8, action_dim=1,
                 actor_lr=FINAL_ACTOR_LR, critic_lr=FINAL_CRITIC_LR, alpha_lr=FINAL_ALPHA_LR):
        super().__init__()
        
        # Actor network architecture
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # Initialize actor weights
        for layer in self.actor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
        
        # Twin Q-networks and corresponding target networks
        self.q1 = self._create_q_network(state_dim, action_dim)
        self.q2 = self._create_q_network(state_dim, action_dim)
        self.q1_target = self._create_q_network(state_dim, action_dim)
        self.q2_target = self._create_q_network(state_dim, action_dim)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Entropy regularization parameters
        self.target_entropy = -action_dim * 1.5  # Adjusted target entropy
        self.log_alpha = torch.tensor([1.0], requires_grad=True)
        
        # Define optimizers for actor, critic, and temperature parameter
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=alpha_lr)
    
    def _create_q_network(self, state_dim, action_dim):
        """Creates a robust Q-network."""
        net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.LayerNorm(256),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 1)
        )
        for layer in net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('leaky_relu', 0.01))
                nn.init.constant_(layer.bias, 0)
        # Initialize the final layer with small weights
        nn.init.uniform_(net[-1].weight, -3e-3, 3e-3)
        return net
    
    def act(self, state):
        """Returns an action by applying tanh to the actor's output."""
        return torch.tanh(self.actor(state))
    
    def update_targets(self, tau=0.005):
        """Soft-update the target networks."""
        with torch.no_grad():
            for target, source in zip(self.q1_target.parameters(), self.q1.parameters()):
                target.data.copy_(tau * source.data + (1 - tau) * target.data)
            for target, source in zip(self.q2_target.parameters(), self.q2.parameters()):
                target.data.copy_(tau * source.data + (1 - tau) * target.data)

# --------------------------
# Replay Buffer (using deque)
# --------------------------
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, transition):
        self.buffer.append(transition)
            
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

# --------------------------
# Simplified Training Loop
# --------------------------
def train_sac(dataset_path, epochs=100, batch_size=512):
    # Simple dataset setup
    dataset = DiabetesDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize agent
    agent = SACAgent().to(device)
    
    # Main training loop
    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        
        for batch_idx, batch in enumerate(dataloader):
            # Print sample data for debugging
            #state_sample = batch['state'][0].cpu().numpy()
            #action_sample = batch['action'][0].item()
            #reward_sample = batch['reward'][0].item()
            #next_state_sample = batch['next_state'][0].cpu().numpy()
            
            #print(f"Batch {batch_idx+1}:")
            #print(f"State: {state_sample.round(2)}")
            #print(f"Action: {action_sample:.3f}")
            #print(f"Reward: {reward_sample:.3f}")
            #print(f"Next State: {next_state_sample.round(2)}\n")

            # Move data to device
            states = batch['state'].to(device)
            actions = batch['action'].to(device)
            rewards = batch['reward'].to(device)
            next_states = batch['next_state'].to(device)
            dones = batch['done'].to(device)

            # Critic Update
            with torch.no_grad():
                next_actions = agent.act(next_states)
                q_next = torch.min(
                    agent.q1_target(torch.cat([next_states, next_actions], 1)),
                    agent.q2_target(torch.cat([next_states, next_actions], 1))
                )
                target_q = rewards + 0.99 * (1 - dones) * q_next

            current_q1 = agent.q1(torch.cat([states, actions], 1))
            current_q2 = agent.q2(torch.cat([states, actions], 1))
            
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            
            agent.critic_optim.zero_grad()
            critic_loss.backward()
            agent.critic_optim.step()

            # Actor Update
            pred_actions = agent.act(states)
            actor_loss = -torch.min(
                agent.q1(torch.cat([states, pred_actions], 1)),
                agent.q2(torch.cat([states, pred_actions], 1))
            ).mean()
            
            agent.actor_optim.zero_grad()
            actor_loss.backward()
            agent.actor_optim.step()

            # Temperature Update
            alpha_loss = -(agent.log_alpha * (agent.target_entropy + 0.5)).mean()
            agent.alpha_optim.zero_grad()
            alpha_loss.backward()
            agent.alpha_optim.step()

            # Update targets
            agent.update_targets()

        print(f"Critic Loss: {critic_loss.item():.4f}")
        print(f"Actor Loss: {actor_loss.item():.4f}")
        print(f"Alpha Loss: {alpha_loss.item():.4f}")

    return agent


if __name__ == "__main__":
    # Simple execution
    print(f"Training on device: {device}")
    agent = train_sac(
        dataset_path="datasets/processed/559-train.csv",
        epochs=10,
        batch_size=512
    )
