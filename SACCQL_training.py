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
from datetime import datetime

# --------------------------
# Data Handling
# --------------------------

class DiabetesDataset(Dataset):
    """Processed diabetes management dataset"""
    
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        
        # Add explicit glu_raw validation
        if df["glu_raw"].isna().any():
            print("NaN values in glu_raw - filling with forward/backward fill")
            df["glu_raw"] = df["glu_raw"].ffill().bfill()
            if df["glu_raw"].isna().any():
                raise ValueError("glu_raw contains NaNs that couldn't be filled")

        # Add bounds check for glu_raw
        glu_raw = df["glu_raw"].values
        if np.any(glu_raw < 40) or np.any(glu_raw > 400):
            print("Warning: glu_raw contains values outside clinical range 40-400 mg/dL")
            print(f"Min: {np.nanmin(glu_raw)}, Max: {np.nanmax(glu_raw)}")
            
        # Handle missing values by forward-filling and backward-filling
        df = df.ffill().bfill()
        
        # Verify no remaining NaNs
        if df[["glu", "glu_d", "glu_t", "hr", "hr_d", "hr_t", "iob", "hour"]].isna().any().any():
            raise ValueError("Dataset contains NaN values after preprocessing")
        
        # Add data validation
        assert df["action"].between(-1, 1).all(), "Actions must be between -1 and 1"
        
        # State features (8 dimensions)
        self.states = df[["glu", "glu_d", "glu_t", "hr", "hr_d", "hr_t", "iob", "hour"]].values.astype(np.float32)
        
        # Single action dimension (already in [-1, 1] range)
        self.actions = df["action"].values.astype(np.float32).reshape(-1, 1)  # No scaling needed
        
        # Rewards computed from next glucose values
        self.rewards = self._compute_rewards(df["glu_raw"].values)
        
        # Transition handling
        self.next_states = np.roll(self.states, -1, axis=0)
        self.dones = df["done"].values.astype(np.float32)
        
        # Remove last invalid transition
        self._sanitize_transitions()

    def _compute_rewards(self, glucose_next):
        """Robust reward calculation with numerical safeguards"""
        # Add debug output for first 5 values
        print(f"Sample glu_raw inputs: {glucose_next[:5]}")
        
        # Convert to numpy array explicitly
        glucose_next = np.asarray(glucose_next, dtype=np.float32)
        
        # Handle remaining NaNs if any
        if np.isnan(glucose_next).any():
            print("Warning: NaN in glucose_next - replacing with 180 (nominal)")
            glucose_next = np.nan_to_num(glucose_next, nan=180.0)

        # Safer clipping with type preservation
        glucose_next = np.clip(glucose_next.astype(np.float32), 40, 400)
        
        # Stable log calculation
        safe_ratio = np.where(
            glucose_next > 0,
            glucose_next / 180.0,
            1e-4  # Avoid zero division
        )
        log_term = np.log(safe_ratio + 1e-8)
        
        # Vectorized risk calculation
        with np.errstate(invalid='ignore'):
            risk_index = 10 * (1.509 * (log_term**1.084 - 1.861)**2)
            risk_index = np.nan_to_num(risk_index, nan=50.0, posinf=50.0, neginf=0.0)
        
        # Reward calculation with failsafes
        rewards = -risk_index / (np.clip(risk_index, 1e-8, None) + 50)  # Prevent div by zero
        rewards = np.clip(rewards, -5.0, 0.0).astype(np.float32)
        
        # Hypoglycemia penalty with index check
        hypo_mask = glucose_next < 54
        rewards[hypo_mask] = -5.0
        
        # Add debug output for first 5 rewards
        print(f"Sample rewards: {rewards[:5]}")
        
        # Final validation
        if np.isnan(rewards).any():
            nan_indices = np.where(np.isnan(rewards))[0]
            raise ValueError(f"NaN rewards at indices: {nan_indices}")
            
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
# Neural Network Components
# --------------------------

class SACAgent(nn.Module):
    """Simplified SAC Agent"""
    
    def __init__(self, state_dim=8, action_dim=1):
        super().__init__()
        
        # Actor Network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        
        # Initialize weights properly
        for layer in self.actor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
        
        # Twin Q-networks
        self.q1 = self._create_q_network(state_dim, action_dim)
        self.q2 = self._create_q_network(state_dim, action_dim)
        self.q1_target = self._create_q_network(state_dim, action_dim)
        self.q2_target = self._create_q_network(state_dim, action_dim)
        
        # Initialize target networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Entropy regularization
        self.target_entropy = -torch.prod(torch.Tensor([1])).item()  # For continuous action space
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        
        # Separate optimizers for actor and critic
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=3e-5)
        self.critic_optim = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=3e-5)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=1e-4)

    def _create_q_network(self, state_dim, action_dim):
        """Create more stable Q-network with gradient protections"""
        net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.LayerNorm(128),  # Add layer normalization
            nn.LeakyReLU(0.01),  # Safer than ReLU
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1)
        )
        # Safer initialization
        for layer in net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('leaky_relu', 0.01))
                nn.init.constant_(layer.bias, 0)
        return net

    def act(self, state):
        """Direct tanh output without scaling"""
        return torch.tanh(self.actor(state))

    def update_targets(self, tau=0.01):  # Changed from 0.05 to 0.01
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

def train_sac(dataset_path, epochs=500, batch_size=512, save_path='models'):
    """Simplified training loop for SAC
    
    Args:
        dataset_path: Path to the training dataset CSV
        epochs: Number of training epochs
        batch_size: Batch size for training
        save_path: Path to save the trained model
        
    Returns:
        Trained SAC agent
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Initialize components
    dataset = DiabetesDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    agent = SACAgent().to(device)
    
    # Add logging setup
    log_dir = Path("training_logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "training_log.csv"
    
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'critic_loss', 'actor_loss', 'alpha_loss'])
    
    with tqdm(range(epochs), desc="Training") as pbar:
        for epoch in pbar:
            epoch_critic_loss = 0.0
            epoch_actor_loss = 0.0
            
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
                    q_next = torch.min(q1_next, q2_next)
                    target_q = rewards + 0.99 * (1 - dones) * q_next
                    target_q = torch.clamp(target_q, -10.0, 10.0)
                
                current_q1 = agent.q1(torch.cat([states, actions], 1))
                current_q2 = agent.q2(torch.cat([states, actions], 1))
                
                # MSE loss for critic
                q1_loss = F.mse_loss(current_q1, target_q)
                q2_loss = F.mse_loss(current_q2, target_q)
                critic_loss = q1_loss + q2_loss
                
                agent.critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(agent.q1.parameters()) + list(agent.q2.parameters()), 
                    1.0
                )
                agent.critic_optim.step()
                
                # Actor update with entropy regularization
                pred_actions = agent.act(states)
                q1_pred = agent.q1(torch.cat([states, pred_actions], 1))
                q2_pred = agent.q2(torch.cat([states, pred_actions], 1))
                
                # Get current alpha value
                alpha = agent.log_alpha.exp().detach()
                
                # Actor loss with entropy term
                actor_loss = -(torch.min(q1_pred, q2_pred)).mean()
                
                agent.actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
                agent.actor_optim.step()
                
                # Alpha (temperature) update
                alpha_loss = -(agent.log_alpha * (agent.target_entropy + 0.1).detach()).mean()
                
                agent.alpha_optim.zero_grad()
                alpha_loss.backward()
                agent.alpha_optim.step()
                
                # Update target networks
                agent.update_targets()
                
                # Track losses
                epoch_critic_loss += critic_loss.item()
                epoch_actor_loss += actor_loss.item()
                epoch_alpha_loss = alpha_loss.item() if 'alpha_loss' in locals() else 0.0
            
            # Average losses
            num_batches = len(dataloader)
            epoch_critic_loss /= num_batches
            epoch_actor_loss /= num_batches
            
            # Update progress bar
            pbar.set_postfix({
                'Critic Loss': f"{epoch_critic_loss:.3f}",
                'Actor Loss': f"{epoch_actor_loss:.3f}",
                'Alpha': f"{agent.log_alpha.exp().item():.3f}"
            })
            
            # Log metrics after each epoch
            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, epoch_critic_loss, epoch_actor_loss, epoch_alpha_loss])
            
            # Save checkpoint every 50 epochs
            if (epoch+1) % 50 == 0:
                Path(save_path).mkdir(parents=True, exist_ok=True)
                checkpoint_path = os.path.join(save_path, f"sac_checkpoint_epoch{epoch+1}.pth")
                torch.save(agent.state_dict(), checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final model
    Path(save_path).mkdir(parents=True, exist_ok=True)
    final_model_path = os.path.join(save_path, "sac_final_model.pth")
    torch.save(agent.state_dict(), final_model_path)
    print(f"Training complete. Model saved to {final_model_path}")
    return agent


# --------------------------
# Analysis & Reporting
# --------------------------

def analyze_training_log(log_path="training_logs/training_log.csv", output_dir="training_analysis"):
    """Analyze training log and generate visualizations"""
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
    
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
    
    parser = argparse.ArgumentParser(description='Train SAC agent for diabetes management')
    parser.add_argument('--dataset', type=str, default="datasets/processed/563-training.csv", 
                        help='Path to the training dataset')
    parser.add_argument('--epochs', type=int, default=500, 
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for training')
    parser.add_argument('--save_path', type=str, default="models", 
                        help='Directory to save the trained model')
    
    args = parser.parse_args()
    
    # Train the agent
    agent = train_sac(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_path=args.save_path
    )
    
    # Run analysis on training logs
    analyze_training_log()
