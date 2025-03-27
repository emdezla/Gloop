"""
Simplified SAC-CQL Training for Diabetes Management
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm

# --------------------------
# Data Handling
# --------------------------

class DiabetesDataset(Dataset):
    """Processed diabetes management dataset"""
    
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        
        # State features (8 dimensions)
        self.states = df[["glu", "glu_d", "glu_t", "hr", "hr_d", "hr_t", "iob", "hour"]].values.astype(np.float32)
        
        # Actions (2 dimensions)
        self.actions = df[["basal", "bolus"]].values.astype(np.float32)
        
        # Rewards computed from next glucose values
        self.rewards = self._compute_rewards(df["glu_raw"].values)
        
        # Transition handling
        self.next_states = np.roll(self.states, -1, axis=0)
        self.dones = df["done"].values.astype(np.float32)
        
        # Remove last invalid transition
        self._sanitize_transitions()

    def _compute_rewards(self, glucose_next):
        """Risk Index-based reward calculation"""
        glucose_next = np.clip(glucose_next, 1e-6, None)
        log_term = np.log(glucose_next) ** 1.084
        risk_index = 10 * (1.509 * (log_term - 5.381)) ** 2
        rewards = -np.clip(risk_index / 100.0, 0, 1)
        rewards[glucose_next <= 39] = -15.0  # Severe hypoglycemia penalty
        return rewards.astype(np.float32)

    def _sanitize_transitions(self):
        """Remove invalid transitions and align array lengths"""
        valid_mask = np.ones(len(self.states), dtype=bool)
        valid_mask[-1] = False  # Remove last transition
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
# Neural Network Components
# --------------------------

class SACAgent(nn.Module):
    """Simplified SAC agent for diabetes management"""
    
    def __init__(self, state_dim=8, action_dim=2):
        super().__init__()
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)
        
        # Twin Q-networks
        self.q1 = self._create_q_network(state_dim, action_dim)
        self.q2 = self._create_q_network(state_dim, action_dim)
        self.q1_target = self._create_q_network(state_dim, action_dim)
        self.q2_target = self._create_q_network(state_dim, action_dim)
        
        # Initialize target networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': 3e-4},
            {'params': self.mean.parameters(), 'lr': 3e-4},
            {'params': self.log_std.parameters(), 'lr': 3e-4},
            {'params': self.q1.parameters(), 'lr': 3e-4},
            {'params': self.q2.parameters(), 'lr': 3e-4}
        ])

    def _create_q_network(self, state_dim, action_dim):
        """Create Q-network"""
        return nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        """Action selection with entropy regularization"""
        hidden = self.actor(state)
        mean = self.mean(hidden)
        log_std = torch.clamp(self.log_std(hidden), -5, 2)  # Constrained log_std
        return mean, log_std

    def act(self, state, deterministic=False):
        """Safe action selection with clamping"""
        mean, log_std = self.forward(state)
        if deterministic:
            action = torch.tanh(mean)
        else:
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            action = torch.tanh(dist.rsample())
        return action

    def update_targets(self, tau=0.005):
        """Soft target network updates"""
        with torch.no_grad():
            for target, source in zip(self.q1_target.parameters(), self.q1.parameters()):
                target.data.copy_(tau * source.data + (1 - tau) * target.data)
            for target, source in zip(self.q2_target.parameters(), self.q2.parameters()):
                target.data.copy_(tau * source.data + (1 - tau) * target.data)

# --------------------------
# Training Core
# --------------------------

def train_sac(dataset_path, epochs=200, batch_size=256, save_path='sac_model.pth'):
    """Simplified training loop for SAC agent"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize components
    dataset = DiabetesDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    agent = SACAgent().to(device)
    
    with tqdm(range(epochs), desc="Training") as pbar:
        for epoch in pbar:
            epoch_loss = 0
            
            for batch in dataloader:
                # Prepare batch
                states = batch['state'].to(device)
                actions = batch['action'].to(device)
                rewards = batch['reward'].to(device)
                next_states = batch['next_state'].to(device)
                dones = batch['done'].to(device)
                
                # Critic update
                with torch.no_grad():
                    next_actions = agent.act(next_states)
                    q1_next = agent.q1_target(torch.cat([next_states, next_actions], 1))
                    q2_next = agent.q2_target(torch.cat([next_states, next_actions], 1))
                    target_q = rewards + 0.99 * (1 - dones) * torch.min(q1_next, q2_next)
                
                # Current Q estimates
                current_q1 = agent.q1(torch.cat([states, actions], 1))
                current_q2 = agent.q2(torch.cat([states, actions], 1))
                
                # TD loss
                critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
                
                # Actor update
                mean, log_std = agent.forward(states)
                std = log_std.exp()
                dist = torch.distributions.Normal(mean, std)
                action_samples = torch.tanh(dist.rsample())
                
                # Q-values for policy
                q1_policy = agent.q1(torch.cat([states, action_samples], 1))
                actor_loss = -q1_policy.mean()
                
                # Combined update
                total_loss = critic_loss + actor_loss
                agent.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                agent.optimizer.step()
                
                # Update target networks
                agent.update_targets()
                
                epoch_loss += total_loss.item()
            
            # Update progress bar
            pbar.set_postfix({'Loss': f"{epoch_loss/len(dataloader):.3f}"})
            
            # Save checkpoint
            if (epoch+1) % 50 == 0:
                checkpoint_path = f"{os.path.splitext(save_path)[0]}_epoch{epoch+1}.pth"
                torch.save(agent.state_dict(), checkpoint_path)
    
    # Save final model
    torch.save(agent.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")
    return agent

if __name__ == "__main__":
    # Example usage
    agent = train_sac(
        dataset_path="datasets/processed/563-train.csv",
        epochs=200,
        save_path="sac_model.pth"
    )
