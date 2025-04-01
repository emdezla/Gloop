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
from torch import nn

# Add Mish activation function for better gradient flow
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

# Register Mish as a module
nn.Mish = Mish

# --------------------------
# Data Handling
# --------------------------

class DiabetesDataset(Dataset):
    """Processed diabetes management dataset"""
    
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        
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
        """Target-centered reward function with safe zones"""
        glucose = np.clip(glucose_next.astype(np.float32), 40, 400)
        
        # Base penalty (quadratic around target 100)
        target = 100.0
        dev_penalty = ((glucose - target) ** 2) / (target ** 2)  # Normalized
        
        # Hypoglycemia penalty (starts below 70)
        hypo_penalty = np.where( glucose < 70, (70 - glucose) * 0.03,  0.0)
        
        # Hyperglycemia penalty (starts above 180)
        hyper_penalty = np.where(glucose > 180, (glucose - 180) * 0.02,  0.0)
        
        # Combined reward with Q-value scaling control
        rewards = -(dev_penalty + hypo_penalty + hyper_penalty) * 0.5  # Critical scaling factor
        rewards = np.clip(rewards, -2.0, 0.0).astype(np.float32)
        
        # Add small time penalty to encourage control
        rewards -= 0.01  # Adjustable agent "effort" penalty
        
        return np.clip(rewards, -2.0, 0.0)

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
        
        # Actor Network with improved architecture
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.Mish(),  # Better than Tanh for gradient flow
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.Mish(),
            nn.Linear(128, action_dim)
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
        self.target_entropy = -torch.prod(torch.Tensor([1])).item() * 1.5  # Reduced from 2 to 1.5
        self.log_alpha = torch.tensor([1.0], requires_grad=True)  # Start with higher entropy
        
        # Separate optimizers for actor and critic with adjusted learning rates
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=1e-5)  # Increased from 5e-6
        self.critic_optim = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=1e-4)  # Reduced from 2e-4
        self.alpha_optim = optim.Adam([self.log_alpha], lr=1e-5)  # Reduced from 3e-5

    def _create_q_network(self, state_dim, action_dim):
        """Create more robust Q-network"""
        net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),  # Increased from 128
            nn.LayerNorm(256),
            nn.Dropout(0.2),  # Increased dropout
            nn.LeakyReLU(0.01),
            nn.Linear(256, 256),  # Additional layer
            nn.LayerNorm(256),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 1)
        )
        # Safer initialization
        for layer in net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('leaky_relu', 0.01))
                nn.init.constant_(layer.bias, 0)
        
        # Initialize final layer to smaller values
        nn.init.uniform_(net[-1].weight, -3e-3, 3e-3)
        return net

    def act(self, state):
        """Direct tanh output without scaling"""
        return torch.tanh(self.actor(state))

    def update_targets(self, tau=0.005):  # Reduced from 0.01 to 0.005
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

def train_sac(dataset_path, epochs=500, batch_size=512, save_path='models', lr_warmup_epochs=50,
              use_cql=False, cql_alpha=1.0):
    """Simplified training loop for SAC
    
    Args:
        dataset_path: Path to the training dataset CSV
        epochs: Number of training epochs
        batch_size: Batch size for training
        save_path: Path to save the trained model
        lr_warmup_epochs: Number of epochs for learning rate warmup
        use_cql: Whether to use Conservative Q-Learning penalty
        cql_alpha: Weight for CQL penalty term
        
    Returns:
        Trained SAC agent
    """

    
    # Generate timestamp once at start of training
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_path, timestamp)
    
    # Initialize components
    dataset = DiabetesDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    agent = SACAgent().to(device)
    
    # Compute dataset-wide mean/std for state normalization
    all_states = []
    for i in range(len(dataset)):
        all_states.append(dataset[i]['state'].numpy())
    all_states = np.vstack(all_states)
    dataset_mean = torch.FloatTensor(all_states.mean(axis=0)).to(device)
    dataset_std = torch.FloatTensor(all_states.std(axis=0)).to(device)
    dataset_std[dataset_std < 1e-4] = 1.0
    
    # Store initial learning rates for warmup
    initial_actor_lr = 5e-6
    initial_critic_lr = 2e-4
    initial_alpha_lr = 1e-5
    
    # Add logging setup
    log_dir = Path("training_logs") / timestamp
    log_dir.mkdir(exist_ok=True, parents=True)
    log_path = log_dir / "training_log.csv"
    
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'critic_loss', 'actor_loss', 'alpha_loss',
            'q1_value', 'q2_value', 'action_mean', 'action_std',
            'entropy', 'grad_norm'
        ])
    
    with tqdm(range(epochs), desc="Training") as pbar:
        for epoch in pbar:
            epoch_critic_loss = 0.0
            epoch_actor_loss = 0.0
            epoch_q1_value = 0.0
            epoch_q2_value = 0.0
            epoch_action_mean = 0.0
            epoch_action_std = 0.0
            epoch_entropy = 0.0
            epoch_grad_norm = 0.0
            
            # Learning rate warmup
            if epoch < lr_warmup_epochs:
                warmup_factor = (epoch + 1) / lr_warmup_epochs
                for param_group in agent.actor_optim.param_groups:
                    param_group['lr'] = initial_actor_lr * warmup_factor
                for param_group in agent.critic_optim.param_groups:
                    param_group['lr'] = initial_critic_lr * warmup_factor
                for param_group in agent.alpha_optim.param_groups:
                    param_group['lr'] = initial_alpha_lr * warmup_factor
            
            for batch in dataloader:
                # Prepare batch
                states = batch['state'].to(device)
                actions = batch['action'].to(device)
                rewards = batch['reward'].to(device)
                next_states = batch['next_state'].to(device)
                dones = batch['done'].to(device)
                
                # Normalize states
                states = (states - dataset_mean) / (dataset_std + 1e-8)
                next_states = (next_states - dataset_mean) / (dataset_std + 1e-8)
                
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
                
                # Conservative Q-Learning (CQL) penalty if enabled
                if use_cql:
                    # Sample random actions for CQL
                    random_actions = 2.0 * torch.rand_like(actions) - 1.0
                    q1_rand = agent.q1(torch.cat([states, random_actions.detach()], 1))
                    q2_rand = agent.q2(torch.cat([states, random_actions.detach()], 1))
                    # Conservative term: log-sum-exp minus the Q on real actions
                    cql_term = (torch.logsumexp(q1_rand, dim=0).mean() + torch.logsumexp(q2_rand, dim=0).mean()) \
                              - torch.min(current_q1, current_q2).mean()
                    critic_loss += cql_alpha * cql_term
                
                # Add gradient penalty to prevent critic collapse
                states.requires_grad_(True)
                q_penalty = agent.q1(torch.cat([states, actions.detach()], 1))
                grad_penalty = torch.autograd.grad(
                    outputs=q_penalty.mean(),
                    inputs=states,
                    create_graph=True
                )[0].pow(2).mean()
                critic_loss += 0.5 * grad_penalty
                states.requires_grad_(False)
                
                # Capture Q-values
                current_q1_mean = current_q1.mean().item()
                current_q2_mean = current_q2.mean().item()
                epoch_q1_value += current_q1_mean
                epoch_q2_value += current_q2_mean
                
                # Scale rewards for more stable learning
                scaled_rewards = rewards / 10.0  # Changed from 5.0 to 10.0
                
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
                
                # Actor loss with entropy term and gradient stabilization
                actor_loss = -(torch.min(q1_pred, q2_pred)).mean()
                
                # Add L2 regularization to actor loss
                l2_reg = 0.0005  # Reduced from 0.001
                for param in agent.actor.parameters():
                    actor_loss += l2_reg * param.pow(2).sum()
                
                # Capture action statistics
                with torch.no_grad():
                    action_mean = pred_actions.mean().item()
                    action_std = pred_actions.std().item()
                epoch_action_mean += action_mean
                epoch_action_std += action_std
                
                # Capture entropy
                current_alpha = agent.log_alpha.exp().item()
                epoch_entropy += current_alpha
                
                agent.actor_optim.zero_grad()
                actor_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    agent.actor.parameters(), 1.0
                ).item()
                epoch_grad_norm += grad_norm
                agent.actor_optim.step()
                
                # Alpha (temperature) update with more stable target
                alpha_loss = -(agent.log_alpha * (agent.target_entropy + torch.tensor(0.5))).mean()
                
                agent.alpha_optim.zero_grad()
                alpha_loss.backward()
                agent.alpha_optim.step()
                
                # Update target networks less frequently to stabilize learning
                if epoch % 5 == 0:  # Changed from every 2 epochs to every 5
                    agent.update_targets()
                
                # Track losses
                epoch_critic_loss += critic_loss.item()
                epoch_actor_loss += actor_loss.item()
                epoch_alpha_loss = alpha_loss.item() if 'alpha_loss' in locals() else 0.0
            
            # Average losses
            num_batches = len(dataloader)
            epoch_critic_loss /= num_batches
            epoch_actor_loss /= num_batches
            epoch_q1_value /= num_batches
            epoch_q2_value /= num_batches
            epoch_action_mean /= num_batches
            epoch_action_std /= num_batches
            epoch_entropy /= num_batches
            epoch_grad_norm /= num_batches
            
            # Update progress bar
            pbar.set_postfix({
                'Critic': f"{epoch_critic_loss:.3f}",
                'Actor': f"{epoch_actor_loss:.3f}",
                'Q1': f"{epoch_q1_value:.3f}",
                'Q2': f"{epoch_q2_value:.3f}",
                'α': f"{epoch_entropy:.3f}",
                '∇': f"{epoch_grad_norm:.2f}"
            })
            
            # Log metrics after each epoch
            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, 
                    epoch_critic_loss, 
                    epoch_actor_loss, 
                    epoch_alpha_loss,
                    epoch_q1_value,
                    epoch_q2_value,
                    epoch_action_mean,
                    epoch_action_std,
                    epoch_entropy,
                    epoch_grad_norm
                ])
            
            # Save checkpoint every 50 epochs
            if (epoch+1) % 50 == 0:
                Path(save_path).mkdir(parents=True, exist_ok=True)
                checkpoint_path = os.path.join(save_path, f"sac_checkpoint_epoch{epoch+1}_{timestamp}.pth")
                torch.save(agent.state_dict(), checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final model
    Path(save_path).mkdir(parents=True, exist_ok=True)
    final_model_path = os.path.join(save_path, f"sac_final_model_{timestamp}.pth")
    torch.save(agent.state_dict(), final_model_path)
    print(f"Training complete. Model saved to {final_model_path}")
    return agent


# --------------------------
# Analysis & Reporting
# --------------------------

def analyze_training_log(log_path, output_dir="training_analysis"):
    """Analyze training log and generate visualizations with clinical insights"""
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
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
    
    # Define healthy ranges for key metrics
    healthy_ranges = {
        'critic_loss': (0.1, 1.0),
        'actor_loss': (1.5, 3.5),
        'alpha_loss': (-0.5, 0.5),
        'q1_value': (-4.5, -0.5),
        'q2_value': (-4.5, -0.5),
        'action_mean': (-0.2, 0.2),
        'action_std': (0.3, 0.6),
        'entropy': (0.1, 0.5),
        'grad_norm': (0.1, 5.0)
    }
    
    for idx, metric in enumerate(metrics):
        ax = axs[idx]
        ax.plot(df['epoch'], df[metric])
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xlabel('Epoch')
        ax.grid(True)
        
        # Add healthy range shading if defined for this metric
        if metric in healthy_ranges:
            low, high = healthy_ranges[metric]
            ax.axhspan(low, high, alpha=0.2, color='green')
            
            # Add warning annotations for values outside healthy range
            last_value = df[metric].iloc[-1]
            if last_value < low or last_value > high:
                ax.annotate('⚠️', xy=(df['epoch'].iloc[-1], last_value), 
                           xytext=(5, 0), textcoords='offset points',
                           fontsize=12, color='red')
    
    # Hide empty subplots
    for idx in range(n_metrics, len(axs)):
        axs[idx].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(Path(output_dir) / "training_metrics.png")
    plt.close()
    
    # Create a separate plot for Q-value convergence
    if 'q1_value' in df.columns and 'q2_value' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df['epoch'], df['q1_value'], label='Q1')
        plt.plot(df['epoch'], df['q2_value'], label='Q2')
        plt.fill_between(df['epoch'], 
                         df['q1_value'] - 0.1 * np.abs(df['q1_value']),
                         df['q1_value'] + 0.1 * np.abs(df['q1_value']),
                         alpha=0.2, color='blue')
        plt.title('Q-Network Convergence')
        plt.xlabel('Epoch')
        plt.ylabel('Q-Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(Path(output_dir) / "q_convergence.png")
        plt.close()
    
    # Generate model readiness report
    generate_readiness_report(df, output_dir)
    
    print(f"Analysis plots saved to {output_dir}")

def generate_readiness_report(df, output_dir):
    """Generate a model readiness report based on training metrics"""
    from pathlib import Path
    
    # Calculate key indicators
    last_100_epochs = df.iloc[-100:] if len(df) >= 100 else df
    
    # Check if critic loss has stabilized
    critic_loss_std = last_100_epochs['critic_loss'].std()
    critic_stable = critic_loss_std < 0.1
    
    # Check if alpha is maintained above threshold
    final_20pct = df.iloc[int(len(df)*0.8):]
    alpha_maintained = (final_20pct['entropy'] > 0.1).all()
    
    # Check action std in healthy range
    action_std_healthy = 0.2 <= df['action_std'].iloc[-1] <= 0.6
    
    # Check Q-values in healthy range
    q_values_healthy = (-4.5 <= df['q1_value'].iloc[-1] <= -0.5 and 
                        -4.5 <= df['q2_value'].iloc[-1] <= -0.5)
    
    # Check Q1/Q2 convergence
    q_diff = abs(df['q1_value'].iloc[-1] - df['q2_value'].iloc[-1])
    q_avg = abs((df['q1_value'].iloc[-1] + df['q2_value'].iloc[-1]) / 2)
    q_convergence = (q_diff / q_avg) < 0.01 if q_avg != 0 else False
    
    # Generate report
    report = [
        "# Model Readiness Report",
        "",
        f"## Training Summary",
        f"- Total Epochs: {len(df)}",
        f"- Final Critic Loss: {df['critic_loss'].iloc[-1]:.4f}",
        f"- Final Actor Loss: {df['actor_loss'].iloc[-1]:.4f}",
        f"- Final Entropy (alpha): {df['entropy'].iloc[-1]:.4f}",
        "",
        f"## Readiness Checklist",
        f"- {'✅' if critic_stable else '❌'} Critic loss stabilized for last 100 epochs (std={critic_loss_std:.4f})",
        f"- {'✅' if alpha_maintained else '❌'} alpha > 0.1 maintained in final 20% of training",
        f"- {'✅' if action_std_healthy else '❌'} Action std between 0.2-0.6 (current={df['action_std'].iloc[-1]:.4f})",
        f"- {'✅' if q_values_healthy else '❌'} Q-values within [-4.5, -0.5] range (Q1={df['q1_value'].iloc[-1]:.4f}, Q2={df['q2_value'].iloc[-1]:.4f})",
        f"- {'✅' if q_convergence else '❌'} <1% difference between final Q1/Q2 values ({100*q_diff/q_avg if q_avg != 0 else 'N/A'}%)",
        "",
        "## Recommendations",
    ]
    
    # Add recommendations based on checks
    if not critic_stable:
        report.append("- Continue training until critic loss stabilizes")
    
    if not alpha_maintained:
        report.append("- Adjust target entropy to maintain exploration")
    
    if not action_std_healthy:
        if df['action_std'].iloc[-1] < 0.2:
            report.append("- Increase exploration by raising target entropy")
        else:
            report.append("- Decrease exploration by lowering target entropy")
    
    if not q_values_healthy:
        report.append("- Adjust reward scaling or critic network architecture")
    
    if not q_convergence:
        report.append("- Harmonize Q-networks or increase target network update frequency")
    
    # Write report to file
    with open(Path(output_dir) / "model_readiness.md", 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SAC agent for diabetes management')
    parser.add_argument('--dataset', type=str, default="datasets/processed/full-training.csv", 
                        help='Path to the training dataset')
    parser.add_argument('--epochs', type=int, default=500, 
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for training')
    parser.add_argument('--save_path', type=str, default="models", 
                        help='Directory to save the trained model')
    parser.add_argument('--lr_warmup', type=int, default=50,
                        help='Number of epochs for learning rate warmup')
    parser.add_argument('--use_cql', action='store_true',
                        help='Enable Conservative Q-Learning penalty')
    parser.add_argument('--cql_alpha', type=float, default=1.0,
                        help='Weight for CQL penalty term')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Train the agent
    agent = train_sac(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_path=args.save_path,
        lr_warmup_epochs=args.lr_warmup,
        use_cql=args.use_cql,
        cql_alpha=args.cql_alpha
    )

    # Get the timestamp from the most recent directory in training_logs
    training_logs_dir = Path("training_logs")
    if training_logs_dir.exists():
        # Find the most recent timestamp directory
        timestamp_dirs = [d for d in training_logs_dir.iterdir() if d.is_dir()]
        if timestamp_dirs:
            latest_dir = max(timestamp_dirs, key=lambda x: x.stat().st_mtime)
            # Run analysis on training logs
            analyze_training_log(log_path=latest_dir / "training_log.csv")
        else:
            print("No training logs found to analyze")
    else:
        print("Training logs directory not found")
