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
        
        # Handle missing values by forward-filling and backward-filling
        df = df.ffill().bfill()
        
        # Verify no remaining NaNs
        if df[["glu", "glu_d", "glu_t", "hr", "hr_d", "hr_t", "iob", "hour"]].isna().any().any():
            raise ValueError("Dataset contains NaN values after preprocessing")
        
        # Add data validation
        assert df["action"].between(0, 5).all(), "Actions must be between 0-5 units"
        assert df["glu"].between(40, 400).all(), "Invalid glucose values"
        
        # State features (8 dimensions)
        self.states = df[["glu", "glu_d", "glu_t", "hr", "hr_d", "hr_t", "iob", "hour"]].values.astype(np.float32)
        
        # Single action dimension
        self.actions = df["action"].values.astype(np.float32).reshape(-1, 1)  # Changed from 2 columns
        
        # Rewards computed from next glucose values
        self.rewards = self._compute_rewards(df["glu_raw"].values)
        
        # Transition handling
        self.next_states = np.roll(self.states, -1, axis=0)
        self.dones = df["done"].values.astype(np.float32)
        
        # Remove last invalid transition
        self._sanitize_transitions()

    def _compute_rewards(self, glucose_next):
        """Safer reward scaling"""
        glucose_next = np.clip(glucose_next, 40, 400)
        with np.errstate(invalid='ignore'):
            log_term = np.log(glucose_next/180.0 + 1e-8)  # Add epsilon to prevent log(0)
            risk_index = 10 * (1.509 * (log_term**1.084 - 1.861)**2)
        
        # More stable reward calculation
        rewards = -risk_index / (risk_index + 50)  # Range [-1, 0]
        rewards = np.clip(rewards, -5.0, 0.0)  # Hard clip
        rewards[glucose_next < 54] = -5.0  # Stronger hypo penalty
        
        # Add reward validation
        if np.isnan(rewards).any():
            nan_indices = np.where(np.isnan(rewards))[0]
            raise ValueError(f"NaN rewards at indices: {nan_indices}")
            
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
    
    def __init__(self, state_dim=8, action_dim=1):  # Changed action_dim to 1
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
        self.log_alpha = nn.Parameter(torch.tensor([0.0]))  # Start from 0.0 instead of 1.0
        self.target_entropy = -action_dim  # Should be -1 for 1D action
        self.action_scale = nn.Parameter(torch.tensor([1.0]), requires_grad=False)  # Changed to Parameter
        
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
        
        # Different learning rates for different components with reduced critic LR
        self.optimizer = optim.AdamW([
            {'params': self.actor.parameters(), 'lr': 1e-5},  # Lower actor LR
            {'params': self.mean.parameters(), 'lr': 1e-5},
            {'params': self.log_std, 'lr': 1e-5},
            {'params': self.log_alpha, 'lr': 1e-5},  # Reduced from 1e-4
            {'params': self.q1.parameters(), 'lr': 1e-4},  # Reduced from 3e-4
            {'params': self.q2.parameters(), 'lr': 1e-4}   # Reduced from 3e-4
        ], weight_decay=1e-4)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=5, factor=0.5)

    def _create_q_network(self, state_dim, action_dim):
        """Create more stable Q-network with gradient protections"""
        print(f"Creating Q-network with state_dim:{state_dim} action_dim:{action_dim}")
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
        print(f"Q-network structure: {net}")
        return net

    def forward(self, state):
        """Action selection with entropy regularization"""
        mean, log_std = self.actor_forward(state)
        print(f"Mean shape: {mean.shape}, Log_std shape: {log_std.shape}")  # Should be [batch,1]
        return mean, log_std
        
    def actor_forward(self, state):
        """Separate actor forward function for clarity"""
        hidden = self.actor(state)
        mean = self.mean(hidden)
        log_std = torch.clamp(self.log_std, min=-5, max=2)  # Adjusted bounds
        return mean, log_std

    def act(self, state, deterministic=False):
        """Proper action scaling"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        
        if deterministic:
            action = torch.tanh(mean) * self.action_scale
        else:
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

def train_sac(dataset_path, epochs=500, batch_size=512, save_path='models', log_dir="logs"):
    """Enhanced training loop with detailed logging
    
    Args:
        dataset_path: Path to the training dataset CSV
        epochs: Number of training epochs
        batch_size: Batch size for training
        save_path: Path to save the trained model directory
        log_dir: Directory to save training logs
        
    Returns:
        Trained SAC agent
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Create unique timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"sac_model_{timestamp}"
    
    # Update paths to use timestamped directory
    model_dir = Path(save_path) / run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path(log_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    final_model_path = model_dir / f"{run_name}.pth"
    log_path = log_dir / "training_log.csv"
    
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
                    
                    # Add dimension checks
                    print(f"State shape: {states.shape}")  # Should be [batch,8]
                    print(f"Action shape: {actions.shape}")  # Should be [batch,1]
                    
                    # Add input value checks
                    print(f"State stats: mean={states.mean().item():.2f} ±{states.std().item():.2f}")
                    print(f"Action stats: mean={actions.mean().item():.2f} ±{actions.std().item():.2f}")
                    
                    # Verify tensor ranges
                    print(f"State range: {states.min().item():.1f} to {states.max().item():.1f}")
                    print(f"Action range: {actions.min().item():.1f} to {actions.max().item():.1f}")
                    
                    # Add numerical stability checks
                    if torch.any(torch.isnan(states)) or torch.any(torch.isinf(states)):
                        print("Invalid states detected:")
                        print(states)
                        raise ValueError("NaN/Inf in states")

                    if torch.any(actions < 0) or torch.any(actions > 5):
                        print("Invalid actions detected:")
                        print(actions)
                        raise ValueError("Actions outside [0,5] range")
                    
                    # Add NaN checks
                    if torch.isnan(states).any() or torch.isnan(actions).any():
                        print("NaN detected in input data!")
                        continue
                    
                    # Critic update
                    # Current Q estimates
                    state_action = torch.cat([states, actions], 1)
                    print(f"State-action input shape: {state_action.shape}")  # Debug
                    
                    # Register hooks to detect NaN gradients
                    def grad_hook(name):
                        def hook(grad):
                            if torch.isnan(grad).any():
                                print(f"NaN gradient in {name}!")
                        return hook
                        
                    for name, param in agent.q1.named_parameters():
                        param.register_hook(grad_hook(f'q1.{name}'))
                    for name, param in agent.q2.named_parameters():
                        param.register_hook(grad_hook(f'q2.{name}'))
                    
                    current_q1 = agent.q1(state_action)
                    current_q2 = agent.q2(state_action)
                    
                    # Add output value clamping
                    current_q1 = torch.clamp(current_q1, -10, 10)
                    current_q2 = torch.clamp(current_q2, -10, 10)
                    
                    print(f"Q1 values: min={current_q1.min().item():.4f}, max={current_q1.max().item():.4f}, mean={current_q1.mean().item():.4f}")
                    
                    with torch.no_grad():
                        next_actions = agent.act(next_states)
                        q1_next = agent.q1_target(torch.cat([next_states, next_actions], 1))
                        q2_next = agent.q2_target(torch.cat([next_states, next_actions], 1))
                        q_next = torch.min(q1_next, q2_next)
                        target_q = rewards + 0.99 * (1 - dones) * q_next
                        target_q = torch.clamp(target_q, -10.0, 10.0)  # Absolute bounds
                        
                        # Add target value sanitization
                        target_q = torch.nan_to_num(target_q, nan=0.0, posinf=10.0, neginf=-10.0)
                    
                    # Add target value checks
                    print(f"Target Q stats: min={target_q.min().item():.2f}, max={target_q.max().item():.2f}")
                    
                    # Add action/value logging
                    print(f"Q1 outputs: {current_q1.detach().cpu().numpy()[:5]}")  # Show first 5 values
                    print(f"Target Q: {target_q.detach().cpu().numpy()[:5]}")
                    print(f"Rewards: {rewards.detach().cpu().numpy()[:5]}")
                    print(f"Dones: {dones.detach().cpu().numpy()[:5]}")
                    
                    # Huber loss for critic (more robust than MSE)
                    q1_loss = F.huber_loss(current_q1, target_q, delta=1.0)  # Reduced delta
                    q2_loss = F.huber_loss(current_q2, target_q, delta=1.0)
                    critic_loss = (q1_loss + q2_loss) * 0.5  # Added averaging
                    
                    # Add gradient scaling for critic
                    scale_gradients = lambda x: x * 0.5  # Scale gradients before clipping
                    critic_loss = critic_loss * scale_gradients(critic_loss.detach())
                    
                    # Actor update
                    mean, log_std = agent.forward(states)
                    std = log_std.exp()
                    dist = torch.distributions.Normal(mean, std)
                    action_samples = torch.tanh(dist.rsample()) * agent.action_scale  # Add action scaling
                    
                    # Proper Gaussian entropy calculation
                    entropy = 0.5 * (1.0 + torch.log(2 * torch.tensor(np.pi).to(device)) + log_std).mean()
                    print(f"Entropy: {entropy.item():.4f}, Target entropy: {agent.target_entropy:.4f}")
                    
                    # Use adaptive entropy regularization
                    alpha = torch.exp(agent.log_alpha).detach()
                    print(f"Alpha: {alpha.item():.4f}")
                    
                    # Q-values for policy with entropy regularization
                    q1_policy = agent.q1(torch.cat([states, action_samples], 1))
                    
                    # Proper SAC actor loss formulation
                    actor_loss = (alpha * entropy - q1_policy).mean()
                    
                    # Add alpha loss calculation
                    alpha_loss = -(agent.log_alpha * (entropy - agent.target_entropy).detach()).mean()
                    print(f"Alpha loss: {alpha_loss.item():.4f}")
                    
                    # Helper function to check gradients
                    def check_grads(parameters):
                        for p in parameters:
                            if p.grad is not None and torch.isnan(p.grad).any():
                                print(f"NaN gradients in {[n for n, param in agent.named_parameters() if param is p]}")
                                return True
                        return False
                    
                    # Separate updates for better stability
                    # 1. Critic update
                    agent.optimizer.zero_grad()
                    critic_loss.backward()
                    
                    # Gradient monitoring and clipping for critic
                    critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                        list(agent.q1.parameters()) + list(agent.q2.parameters()), 
                        1.0
                    )
                    torch.nn.utils.clip_grad_value_(
                        list(agent.q1.parameters()) + list(agent.q2.parameters()),
                        1.0
                    )
                    
                    # Check for NaN in critic gradients
                    if not check_grads(list(agent.q1.parameters()) + list(agent.q2.parameters())):
                        agent.optimizer.step()
                    else:
                        print("Skipping critic update due to NaN gradients")
                        agent.optimizer.zero_grad()
                    
                    # 2. Actor update
                    agent.optimizer.zero_grad()
                    actor_loss.backward()
                    
                    # Gradient monitoring and clipping for actor
                    actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                        list(agent.actor.parameters()) + [agent.log_std],
                        1.0
                    )
                    torch.nn.utils.clip_grad_value_(
                        list(agent.actor.parameters()) + [agent.log_std],
                        1.0
                    )
                    
                    # Check for NaN in actor gradients using the helper function
                    if not check_grads(list(agent.actor.parameters()) + [agent.log_std]):
                        agent.optimizer.step()
                    else:
                        print("Skipping actor update due to NaN gradients")
                        agent.optimizer.zero_grad()
                    
                    # 3. Alpha update
                    agent.optimizer.zero_grad()
                    alpha_loss.backward()
                    
                    # Gradient clipping for alpha
                    alpha_grad_norm = torch.nn.utils.clip_grad_norm_([agent.log_alpha], 1.0)
                    
                    # Check for NaN in alpha gradient using the helper function
                    if not check_grads([agent.log_alpha]):
                        agent.optimizer.step()
                    else:
                        print("Skipping alpha update due to NaN gradients")
                        agent.optimizer.zero_grad()
                    
                    # Total gradient norm for logging
                    grad_norm = critic_grad_norm + actor_grad_norm + alpha_grad_norm
                    
                    # Print gradient norms for debugging
                    print(f"Gradient norms - Critic: {critic_grad_norm:.4f}, Actor: {actor_grad_norm:.4f}, Alpha: {alpha_grad_norm:.4f}")
                    
                    # Update target networks
                    agent.update_targets()
                    
                    # Monitor parameter values for debugging
                    with torch.no_grad():
                        for name, param in agent.named_parameters():
                            if torch.isnan(param).any():
                                print(f"NaN detected in parameter {name}")
                    
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
                    checkpoint_dir = model_dir / "checkpoints"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint_path = checkpoint_dir / f"checkpoint_epoch{epoch+1}.pth"
                    
                    # Save both model weights and training metadata
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': agent.state_dict(),
                        'optimizer_state_dict': agent.optimizer.state_dict(),
                        'critic_loss': log_entry['critic_loss'],
                        'actor_loss': log_entry['actor_loss'],
                    }, str(checkpoint_path))
                    print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final model with metadata
    torch.save({
        'epoch': epochs,
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'model_type': 'SAC',
        'state_dim': 8,
        'action_dim': 1,
        'training_dataset': os.path.basename(dataset_path),
    }, str(final_model_path))
    print(f"Training complete. Model saved to {final_model_path}")
    return agent

def analyze_training_log(log_path="logs/sac/training_log.csv", output_dir="logs/sac/analysis"):
    """Analyze training log and generate visualizations
    
    Args:
        log_path: Path to the training log CSV file
        output_dir: Directory to save analysis visualizations
        
    Returns:
        None, but saves visualization files to output_dir
    """
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
    parser.add_argument('--dataset', type=str, default="datasets/processed/563-training.csv", 
                        help='Path to the training dataset')
    parser.add_argument('--epochs', type=int, default=500, 
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for training')
    parser.add_argument('--save_path', type=str, default="models", 
                        help='Directory to save the trained model')
    parser.add_argument('--log_dir', type=str, default="logs", 
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
        log_path=os.path.join(args.log_dir, "training_log.csv"),
        output_dir=os.path.join(args.log_dir, "analysis")
    )
