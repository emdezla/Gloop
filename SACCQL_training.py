"""
Simplified SAC-CQL Training for Diabetes Management
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import csv
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# --------------------------
# Data Handling
# --------------------------

class DiabetesDataset(Dataset):
    """Processed diabetes management dataset"""
    
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        
        # Handle missing values by forward-filling and backward-filling
        df = df.ffill().bfill()
        
        # Verify no remaining NaNs
        if df[["glu", "glu_d", "glu_t", "hr", "hr_d", "hr_t", "iob", "hour"]].isna().any().any():
            raise ValueError("Dataset contains NaN values after preprocessing")
        
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
        """Improved reward scaling"""
        glucose_next = np.clip(glucose_next, 40, 400)
        with np.errstate(invalid='ignore'):
            log_term = np.log(glucose_next/180.0)
            risk_index = 10 * (1.509 * (log_term**1.084 - 1.861)**2)
        
        # Better reward scaling using sigmoid instead of tanh
        rewards = -1 / (1 + np.exp(-risk_index/50))  # Scaled to (-1, 0)
        rewards[glucose_next < 54] = -5.0  # Stronger hypo penalty
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
        
        # More stable actor with bounded outputs
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.Tanh()
        )
        self.mean = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))  # Start from neutral
        self.log_alpha = nn.Parameter(torch.tensor([1.0]))  # Start with higher alpha
        self.target_entropy = -torch.prod(torch.Tensor([2.0])).item()  # -2.0 for 2D actions
        self.action_scale = 1.0
        
        # Initialize weights properly
        for layer in self.actor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
        # Initialize outputs with small weights for stability
        nn.init.uniform_(self.mean.weight, -3e-3, 3e-3)
        nn.init.constant_(self.mean.bias, 0)
        
        # Twin Q-networks
        self.q1 = self._create_q_network(state_dim, action_dim)
        self.q2 = self._create_q_network(state_dim, action_dim)
        self.q1_target = self._create_q_network(state_dim, action_dim)
        self.q2_target = self._create_q_network(state_dim, action_dim)
        
        # Initialize target networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Different learning rates for different components
        self.optimizer = optim.AdamW([
            {'params': self.actor.parameters(), 'lr': 1e-5},  # Lower actor LR
            {'params': self.mean.parameters(), 'lr': 1e-5},
            {'params': self.log_std, 'lr': 1e-5},
            {'params': self.log_alpha, 'lr': 1e-4},  # Faster alpha updates
            {'params': self.q1.parameters(), 'lr': 3e-4},  # Faster critics
            {'params': self.q2.parameters(), 'lr': 3e-4}
        ], weight_decay=1e-4)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=5, factor=0.5)

    def _create_q_network(self, state_dim, action_dim):
        """Create Q-network with simpler architecture"""
        return nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        """Action selection with entropy regularization"""
        hidden = self.actor(state)
        mean = self.mean(hidden)
        log_std = torch.clamp(self.log_std, min=-2, max=0)  # Tighter bounds (std between 0.13 and 1.0)
        return mean, log_std

    def act(self, state, deterministic=False):
        """Safe action selection with clamping"""
        mean, log_std = self.forward(state)
        if deterministic:
            action = torch.tanh(mean) * self.action_scale
        else:
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            action = torch.tanh(dist.rsample()) * self.action_scale
        return action

    def update_targets(self, tau=0.05):  # Changed from 0.005 to 0.05
        """Soft target network updates"""
        with torch.no_grad():
            for target, source in zip(self.q1_target.parameters(), self.q1.parameters()):
                target.data.copy_(tau * source.data + (1 - tau) * target.data)
            for target, source in zip(self.q2_target.parameters(), self.q2.parameters()):
                target.data.copy_(tau * source.data + (1 - tau) * target.data)

# --------------------------
# Experience Replay Buffer
# --------------------------

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = []
        self.capacity = capacity
        
    def add(self, transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)
            
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size)
        return [self.buffer[i] for i in indices]

# --------------------------
# Training Core
# --------------------------

def train_sac(dataset_path, epochs=500, batch_size=512, save_path='pure_sac.pth', log_dir="sac_logs"):
    """Enhanced training loop with detailed logging"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / "training_log.csv"
    
    # Initialize components
    dataset = DiabetesDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    agent = SACAgent().to(device)
    
    # Initialize CSV logger
    fieldnames = [
        'epoch', 'critic_loss', 'actor_loss', 
        'q1_value', 'q2_value', 'action_mean', 
        'action_std', 'entropy', 'grad_norm',
        'log_std_mean', 'alpha', 'lr'
    ]
    
    with open(log_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        with tqdm(range(epochs), desc="Training") as pbar:
            for epoch in pbar:
                epoch_metrics = {
                    'critic_loss': 0.0,
                    'actor_loss': 0.0,
                    'q1_value': 0.0,
                    'q2_value': 0.0,
                    'action_mean': 0.0,
                    'action_std': 0.0,
                    'entropy': 0.0,
                    'grad_norm': 0.0,
                }
                
                for batch in dataloader:
                    # Prepare batch with normalization for stability
                    states_raw = batch['state'].to(device)
                    states = (states_raw - states_raw.mean(0)) / (states_raw.std(0) + 1e-8)
                    actions = batch['action'].to(device)
                    rewards = batch['reward'].to(device)
                    next_states_raw = batch['next_state'].to(device)
                    next_states = (next_states_raw - next_states_raw.mean(0)) / (next_states_raw.std(0) + 1e-8)
                    dones = batch['done'].to(device)
                    
                    # Critic update
                    # Current Q estimates
                    current_q1 = agent.q1(torch.cat([states, actions], 1))
                    current_q2 = agent.q2(torch.cat([states, actions], 1))
                    
                    with torch.no_grad():
                        next_actions = agent.act(next_states)
                        q1_next = agent.q1_target(torch.cat([next_states, next_actions], 1))
                        q2_next = agent.q2_target(torch.cat([next_states, next_actions], 1))
                        # Change from hard clipping to relative clipping
                        q_next = torch.min(q1_next, q2_next)
                        target_q = rewards + 0.99 * (1 - dones) * q_next
                        target_q = torch.clamp(target_q, -5.0, current_q1.detach().mean() + 5.0)  # Dynamic range
                    
                    # MSE loss for critic (pure SAC)
                    q1_loss = F.mse_loss(current_q1, target_q)
                    q2_loss = F.mse_loss(current_q2, target_q)
                    critic_loss = q1_loss + q2_loss
                    
                    # Actor update
                    mean, log_std = agent.forward(states)
                    std = log_std.exp()
                    dist = torch.distributions.Normal(mean, std)
                    action_samples = torch.tanh(dist.rsample()) * agent.action_scale  # Add action scaling
                    
                    # Proper Gaussian entropy calculation
                    entropy = 0.5 * (1.0 + torch.log(2 * torch.tensor(np.pi)) + 2 * log_std).mean()
                    
                    # Use adaptive entropy regularization
                    alpha = torch.exp(agent.log_alpha).detach()
                    
                    # Q-values for policy with entropy regularization
                    q1_policy = agent.q1(torch.cat([states, action_samples], 1))
                    actor_loss = -q1_policy.mean() + 0.5 * (entropy - agent.target_entropy).mean()  # Stronger entropy bonus
                    
                    # Combined update
                    total_loss = critic_loss + actor_loss
                    
                    # Check for NaN in loss
                    if torch.isnan(total_loss).any():
                        print("NaN detected in loss, skipping update")
                        continue
                        
                    agent.optimizer.zero_grad()
                    total_loss.backward()
                    
                    # Gradient monitoring and clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                    
                    # Additional gradient clipping for Q-networks
                    for param in agent.q1.parameters():
                        param.grad.data.clamp_(-1, 1)
                    for param in agent.q2.parameters():
                        param.grad.data.clamp_(-1, 1)
                    
                    # Check for NaN in gradients
                    has_nan_grad = False
                    for param in agent.parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            has_nan_grad = True
                            break
                    
                    if has_nan_grad:
                        print("NaN detected in gradients, skipping update")
                        continue
                        
                    agent.optimizer.step()
                    agent.update_targets()
                    
                    # Accumulate metrics
                    epoch_metrics['critic_loss'] += critic_loss.item()
                    epoch_metrics['actor_loss'] += actor_loss.item()
                    epoch_metrics['q1_value'] += current_q1.mean().item()
                    epoch_metrics['q2_value'] += current_q2.mean().item()
                    epoch_metrics['action_mean'] += action_samples.mean().item()
                    epoch_metrics['action_std'] += action_samples.std().item()
                    epoch_metrics['entropy'] += entropy.item()
                    epoch_metrics['grad_norm'] += grad_norm.item()
                
                # Average metrics over batches
                num_batches = len(dataloader)
                log_entry = {
                    'epoch': epoch + 1,
                    'critic_loss': epoch_metrics['critic_loss'] / num_batches,
                    'actor_loss': epoch_metrics['actor_loss'] / num_batches,
                    'q1_value': epoch_metrics['q1_value'] / num_batches,
                    'q2_value': epoch_metrics['q2_value'] / num_batches,
                    'action_mean': epoch_metrics['action_mean'] / num_batches,
                    'action_std': epoch_metrics['action_std'] / num_batches,
                    'entropy': epoch_metrics['entropy'] / num_batches,
                    'grad_norm': epoch_metrics['grad_norm'] / num_batches,
                    'log_std_mean': agent.log_std.mean().item(),
                    'alpha': torch.exp(agent.log_std).mean().item(),
                    'lr': agent.optimizer.param_groups[0]['lr']
                }
                
                writer.writerow(log_entry)
                
                # Update progress bar
                pbar.set_postfix({
                    'Critic Loss': f"{log_entry['critic_loss']:.3f}",
                    'Actor Loss': f"{log_entry['actor_loss']:.3f}",
                    'Q Values': f"{(log_entry['q1_value'] + log_entry['q2_value'])/2:.3f}"
                })
                
                # Update learning rate scheduler
                agent.scheduler.step(log_entry['critic_loss'])
                
                # Save checkpoint
                if (epoch+1) % 50 == 0:
                    checkpoint_path = f"{os.path.splitext(save_path)[0]}_epoch{epoch+1}.pth"
                    torch.save(agent.state_dict(), checkpoint_path)
    
    # Save final model
    torch.save(agent.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")
    return agent

def analyze_training_log(log_path="training_logs/training_log.csv", output_dir="training_analysis"):
    """Analyze training log and generate visualizations"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Read log data
    df = pd.read_csv(log_path)
    
    # Create dynamic subplot grid
    metrics = [col for col in df.columns if col != 'epoch']
    n_metrics = len(metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(18, 4*n_rows))
    fig.suptitle('Training Metrics Analysis', fontsize=16)
    
    # Flatten axes array for easier iteration
    axs = axs.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axs[idx]
        ax.plot(df['epoch'], df[metric])
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xlabel('Epoch')
        ax.grid(True)
    
    # Hide empty subplots
    for idx in range(n_metrics, len(axs)):
        axs[idx].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(Path(output_dir) / "training_metrics.png")
    plt.close()
    
    print(f"Analysis plots saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SAC-CQL agent for diabetes management')
    parser.add_argument('--dataset', type=str, default="datasets/processed/563-train.csv", 
                        help='Path to the training dataset')
    parser.add_argument('--epochs', type=int, default=500, 
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for training')
    parser.add_argument('--save_path', type=str, default="sac_model.pth", 
                        help='Path to save the trained model')
    parser.add_argument('--log_dir', type=str, default="training_logs", 
                        help='Directory to save training logs')
    
    args = parser.parse_args()
    
    # Example usage with command line arguments
    agent = train_sac(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_path=args.save_path,
        log_dir=args.log_dir
    )
    
    # Analyze training results
    analyze_training_log(
        log_path="training_logs/training_log.csv",
        output_dir="training_analysis"
    )
