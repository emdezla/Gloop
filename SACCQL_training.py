"""
SAC-CQL Offline Reinforcement Learning Implementation
for Diabetes Management Policy Optimization

Key components:
- Soft Actor-Critic (SAC) core algorithm
- Conservative Q-Learning (CQL) regularization
- Stabilization techniques for medical applications
- Comprehensive logging and safety checks
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import csv
from tqdm import tqdm
from collections import defaultdict

# --------------------------
# Neural Network Components
# --------------------------

class Mish(nn.Module):
    """Mish activation function for smooth gradient flow"""
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))

class SACCQL(nn.Module):
    """Stabilized SAC agent with CQL regularization"""
    
    def __init__(self, state_dim=8, action_dim=2, action_scale=1.0):
        super().__init__()
        self.action_scale = action_scale
        
        # Actor network with layer normalization
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            Mish(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            Mish()
        )
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)
        
        # Twin Q-networks with conservative architecture
        self.q1 = self._create_q_network(state_dim, action_dim)
        self.q2 = self._create_q_network(state_dim, action_dim)
        self.q1_target = self._create_q_network(state_dim, action_dim)
        self.q2_target = self._create_q_network(state_dim, action_dim)
        
        # Entropy temperature parameter
        self.log_alpha = torch.nn.Parameter(torch.tensor([0.0]))
        
        # Initialize targets and weights
        self._initialize_networks()

    def _create_q_network(self, state_dim, action_dim):
        """Create stabilized Q-network with layer normalization"""
        return nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.LayerNorm(256),
            Mish(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            Mish(),
            nn.Linear(128, 1)
        )

    def _initialize_networks(self):
        """Careful weight initialization for stability"""
        # Actor weights
        nn.init.uniform_(self.mean.weight, -0.003, 0.003)
        nn.init.constant_(self.log_std.bias, -1.0)
        
        # Initialize target networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

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
        return torch.clamp(action * self.action_scale, -self.action_scale, self.action_scale)

    def update_targets(self, tau=0.005):
        """Conservative target network updates"""
        with torch.no_grad():
            for target, source in zip(self.q1_target.parameters(), self.q1.parameters()):
                target.data.copy_(tau * source.data + (1 - tau) * target.data)
            for target, source in zip(self.q2_target.parameters(), self.q2.parameters()):
                target.data.copy_(tau * source.data + (1 - tau) * target.data)

# --------------------------
# Data Handling
# --------------------------

class DiabetesDataset(Dataset):
    """Processed diabetes management dataset"""
    
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file).ffill().bfill()
        
        # State features (8 dimensions)
        self.states = df[["glu", "glu_d", "glu_t", "hr", "hr_d", "hr_t", "iob", "hour"]].values
        self.states = (self.states - self.states.mean(0)) / (self.states.std(0) + 1e-8)  # Normalized
        
        # Actions (normalized to [-1, 1] range)
        self.actions = df[["basal", "bolus"]].values
        self.actions = np.clip((self.actions - self.actions.mean(0)) / (self.actions.std(0) + 1e-6), -1, 1)
        
        # Rewards computed from next glucose values
        glucose_next = torch.tensor(df["glu_raw"].values, dtype=torch.float32)
        self.rewards = self._compute_rewards(glucose_next).numpy() / 5.0  # Scaled
        
        # Transition handling
        self.next_states = np.roll(self.states, -1, axis=0)
        self.dones = df["done"].values.astype(np.float32)
        self._sanitize_transitions()

    def _compute_rewards(self, glucose_next):
        """Risk Index-based reward calculation"""
        glucose_next = torch.clamp(glucose_next, 1e-6)
        log_term = torch.log(glucose_next) ** 1.084
        risk_index = 10 * (1.509 * (log_term - 5.381)) ** 2
        rewards = -torch.clamp(risk_index / 100.0, 0, 1)
        rewards[glucose_next <= 39] = -15.0  # Severe hypoglycemia penalty
        return rewards

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
# Training Core
# --------------------------

class SACCQLTrainer:
    """End-to-end training manager for SAC-CQL"""
    
    def __init__(self, dataset_path, device='auto'):
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
            if device == 'auto' else torch.device(device)
        
        # Initialize components
        self.dataset = DiabetesDataset(dataset_path)
        self.model = SACCQL().to(self.device)
        self.optimizers = self._create_optimizers()
        
        # Training parameters
        self.batch_size = 256
        self.target_entropy = -torch.prod(torch.Tensor([2.0])).item()  # For 2D action space
        self.cql_weight = 0.1
        self.grad_clip = 0.5

    def _create_optimizers(self):
        """Configure optimizers with stabilization settings"""
        return {
            'actor': optim.AdamW(
                list(self.model.actor.parameters()) + 
                list(self.model.mean.parameters()) + 
                list(self.model.log_std.parameters()), 
                lr=1e-4, weight_decay=1e-4),
            'critic': optim.AdamW(
                list(self.model.q1.parameters()) + 
                list(self.model.q2.parameters()),
                lr=3e-4, weight_decay=1e-4),
            'alpha': optim.AdamW([self.model.log_alpha], lr=1e-5)
        }

    def _compute_cql_penalty(self, states, actions):
        """Conservative Q-Learning regularization term"""
        # Current policy actions
        with torch.no_grad():
            policy_actions = self.model.act(states)
        
        # Combine dataset and policy actions
        all_actions = torch.cat([actions, policy_actions], dim=0)
        states_repeated = states.repeat(2, 1)
        
        # Compute Q-values
        q1_all = self.model.q1(torch.cat([states_repeated, all_actions], 1))
        q2_all = self.model.q2(torch.cat([states_repeated, all_actions], 1))
        
        # Normalized CQL penalty
        q1_data = q1_all[:len(actions)]
        q2_data = q2_all[:len(actions)]
        penalty = (torch.logsumexp(q1_all, 0) + torch.logsumexp(q2_all, 0) - 
                 (q1_data.mean() + q2_data.mean()))
        return torch.clamp(penalty, -1.0, 5.0) * 0.1

    def train_epoch(self, dataloader):
        """Single training epoch with stabilization"""
        metrics = defaultdict(float)
        
        for batch in dataloader:
            # Prepare batch with normalization
            states = batch['state'].to(self.device)
            actions = batch['action'].to(self.device)
            rewards = batch['reward'].to(self.device)
            next_states = batch['next_state'].to(self.device)
            dones = batch['done'].to(self.device)

            # --------------------------
            # Critic Update
            # --------------------------
            with torch.no_grad():
                # Target Q-values with noise regularization
                next_actions = self.model.act(next_states)
                q1_next = self.model.q1_target(torch.cat([next_states, next_actions], 1))
                q2_next = self.model.q2_target(torch.cat([next_states, next_actions], 1))
                target_q = rewards + 0.99 * (1 - dones) * torch.min(q1_next, q2_next)
                target_q = torch.clamp(target_q, -50, 0)  # Value clipping

            # Current Q estimates
            current_q1 = self.model.q1(torch.cat([states, actions], 1))
            current_q2 = self.model.q2(torch.cat([states, actions], 1))
            
            # TD loss with Huber loss
            td_loss = nn.HuberLoss()(current_q1, target_q) + nn.HuberLoss()(current_q2, target_q)
            
            # CQL regularization
            cql_penalty = self._compute_cql_penalty(states, actions)
            critic_loss = td_loss + self.cql_weight * cql_penalty
            
            # Update critics
            self.optimizers['critic'].zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.model.q1.parameters(), self.grad_clip)
            nn.utils.clip_grad_norm_(self.model.q2.parameters(), self.grad_clip)
            self.optimizers['critic'].step()

            # --------------------------
            # Actor and Alpha Update
            # --------------------------
            # Policy loss
            mean, log_std = self.model(states)
            dist = torch.distributions.Normal(mean, log_std.exp())
            action_samples = torch.tanh(dist.rsample())
            log_probs = dist.log_prob(dist.rsample()) - torch.log(1 - action_samples.pow(2) + 1e-6)
            log_probs = log_probs.sum(1, keepdim=True)
            
            # Q-values for policy
            q1_policy = self.model.q1(torch.cat([states, action_samples], 1))
            q2_policy = self.model.q2(torch.cat([states, action_samples], 1))
            actor_loss = (self.model.log_alpha.exp() * log_probs - torch.min(q1_policy, q2_policy)).mean()
            
            # Update actor
            self.optimizers['actor'].zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.model.actor.parameters(), self.grad_clip)
            self.optimizers['actor'].step()

            # Temperature (alpha) update
            alpha_loss = -(self.model.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
            self.optimizers['alpha'].zero_grad()
            alpha_loss.backward()
            nn.utils.clip_grad_norm_([self.model.log_alpha], 0.1)
            self.optimizers['alpha'].step()
            
            # --------------------------
            # Post-Update Processing
            # --------------------------
            self.model.update_targets()
            
            # Log metrics
            metrics['critic_loss'] += critic_loss.item()
            metrics['actor_loss'] += actor_loss.item()
            metrics['alpha'] = self.model.log_alpha.exp().item()
            metrics['q_values'] += (current_q1.mean() + current_q2.mean()).item() / 2

        return {k: v/len(dataloader) for k, v in metrics.items()}

    def train(self, epochs=200, save_path='sac_cql_final.pth', csv_file=None):
        """Full training loop with progress tracking"""
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        
        # Setup CSV logging if requested
        if csv_file:
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', 'Critic_Loss', 'Actor_Loss', 'Alpha', 'Q_Values'])
        
        with tqdm(range(epochs), desc="Training") as pbar:
            for epoch in pbar:
                metrics = self.train_epoch(dataloader)
                
                # Update progress bar
                pbar.set_postfix({
                    'Critic Loss': f"{metrics['critic_loss']:.3f}",
                    'Actor Loss': f"{metrics['actor_loss']:.3f}",
                    'Alpha': f"{metrics['alpha']:.3f}",
                    'Q Values': f"{metrics['q_values']:.3f}"
                })
                
                # Log to CSV if requested
                if csv_file:
                    with open(csv_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            epoch + 1,
                            metrics['critic_loss'],
                            metrics['actor_loss'],
                            metrics['alpha'],
                            metrics['q_values']
                        ])
                
                # Save checkpoint
                if (epoch+1) % 50 == 0:
                    checkpoint_path = f"{os.path.splitext(save_path)[0]}_epoch{epoch+1}.pth"
                    torch.save(self.model.state_dict(), checkpoint_path)
        
        # Save final model
        torch.save(self.model.state_dict(), save_path)
        print(f"Training complete. Model saved to {save_path}")
        return self.model

# --------------------------
# Main Execution
# --------------------------

def train_saccql(dataset_path, epochs=200, batch_size=256, save_path='sac_cql_final.pth', 
                csv_file=None, device='auto', cql_weight=0.1):
    """Convenience function for training a SAC-CQL model"""
    trainer = SACCQLTrainer(dataset_path, device)
    trainer.batch_size = batch_size
    trainer.cql_weight = cql_weight
    
    return trainer.train(epochs=epochs, save_path=save_path, csv_file=csv_file)

if __name__ == "__main__":
    # Example usage
    model = train_saccql(
        dataset_path="datasets/processed/563-train.csv",
        epochs=200,
        save_path="sac_cql_final.pth",
        csv_file="saccql_training_stats.csv"
    )
