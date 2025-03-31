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
# Training Loop
# --------------------------
def train_sac(dataset_path, epochs=500, batch_size=512, save_path='models', lr_warmup_epochs=50,
              use_cql=False, cql_alpha=1.0):
    """
    Simplified training loop for SAC.
    
    Args:
        dataset_path (str): Path to the CSV training dataset.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for each training step.
        save_path (str): Directory to save model checkpoints.
        lr_warmup_epochs (int): Number of epochs for learning rate warmup.
        use_cql (bool): Whether to include the Conservative Q-Learning penalty.
        cql_alpha (float): Weight for the CQL penalty.
    
    Returns:
        SACAgent: The trained SAC agent.
    """
    
    # Setup timestamp and model saving directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_path, timestamp)
    
    # Load dataset and create DataLoader
    dataset = DiabetesDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Instantiate agent and move it to the chosen device
    agent = SACAgent().to(device)
    
    # Compute normalization statistics (vectorized)
    all_states = dataset.states  # (N, 8)
    dataset_mean = torch.FloatTensor(all_states.mean(axis=0)).to(device)
    dataset_std = torch.FloatTensor(all_states.std(axis=0)).to(device)
    dataset_std[dataset_std < 1e-4] = 1.0  # Prevent division by nearly zero
    
    # Final learning rates for warmup (these match those in SACAgent)
    final_actor_lr = FINAL_ACTOR_LR
    final_critic_lr = FINAL_CRITIC_LR
    final_alpha_lr = FINAL_ALPHA_LR
    
    # Setup logging for training metrics
    log_dir = Path("training_logs") / timestamp
    log_dir.mkdir(exist_ok=True, parents=True)
    log_path = log_dir / "training_log.csv"
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'critic_loss', 'actor_loss', 'alpha_loss',
                         'q1_value', 'q2_value', 'action_mean', 'action_std',
                         'entropy', 'grad_norm'])
    
    # Main training loop
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
            
            # Warmup: Linearly increase learning rates over initial epochs
            if epoch < lr_warmup_epochs:
                warmup_factor = (epoch + 1) / lr_warmup_epochs
                for param_group in agent.actor_optim.param_groups:
                    param_group['lr'] = final_actor_lr * warmup_factor
                for param_group in agent.critic_optim.param_groups:
                    param_group['lr'] = final_critic_lr * warmup_factor
                for param_group in agent.alpha_optim.param_groups:
                    param_group['lr'] = final_alpha_lr * warmup_factor
            
            for batch in dataloader:
                # Load batch data to device
                states = batch['state'].to(device)
                actions = batch['action'].to(device)
                rewards = batch['reward'].to(device)
                next_states = batch['next_state'].to(device)
                dones = batch['done'].to(device)
                
                # Normalize states and next states
                states = (states - dataset_mean) / (dataset_std + 1e-8)
                next_states = (next_states - dataset_mean) / (dataset_std + 1e-8)
                
                # ---------------------
                # Critic Update
                # ---------------------
                with torch.no_grad():
                    next_actions = agent.act(next_states)
                    q1_next = agent.q1_target(torch.cat([next_states, next_actions], dim=1))
                    q2_next = agent.q2_target(torch.cat([next_states, next_actions], dim=1))
                    q_next = torch.min(q1_next, q2_next)
                    target_q = rewards + 0.99 * (1 - dones) * q_next
                    target_q = torch.clamp(target_q, -10.0, 10.0)
                
                current_q1 = agent.q1(torch.cat([states, actions], dim=1))
                current_q2 = agent.q2(torch.cat([states, actions], dim=1))
                
                q1_loss = F.mse_loss(current_q1, target_q)
                q2_loss = F.mse_loss(current_q2, target_q)
                critic_loss = q1_loss + q2_loss
                
                # Optional CQL penalty
                if use_cql:
                    random_actions = 2.0 * torch.rand_like(actions) - 1.0
                    q1_rand = agent.q1(torch.cat([states, random_actions.detach()], dim=1))
                    q2_rand = agent.q2(torch.cat([states, random_actions.detach()], dim=1))
                    cql_term = (torch.logsumexp(q1_rand, dim=0).mean() +
                                torch.logsumexp(q2_rand, dim=0).mean()) - torch.min(current_q1, current_q2).mean()
                    critic_loss += cql_alpha * cql_term
                
                # Apply gradient penalty to both Q-networks
                grad_penalty = 0.0
                for q_net in [agent.q1, agent.q2]:
                    states.requires_grad_(True)
                    q_val = q_net(torch.cat([states, actions.detach()], dim=1))
                    grad = torch.autograd.grad(outputs=q_val.mean(), inputs=states, create_graph=True)[0]
                    grad_penalty += grad.pow(2).mean()
                    states.requires_grad_(False)
                grad_penalty /= 2.0
                critic_loss += 0.5 * grad_penalty
                
                # Record Q-values for logging
                epoch_q1_value += current_q1.mean().item()
                epoch_q2_value += current_q2.mean().item()
                
                agent.critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(agent.q1.parameters()) + list(agent.q2.parameters()), 1.0)
                agent.critic_optim.step()
                
                # ---------------------
                # Actor Update
                # ---------------------
                pred_actions = agent.act(states)
                q1_pred = agent.q1(torch.cat([states, pred_actions], dim=1))
                q2_pred = agent.q2(torch.cat([states, pred_actions], dim=1))
                actor_loss = -(torch.min(q1_pred, q2_pred)).mean()
                
                # L2 regularization for actor weights
                l2_reg = 0.0005
                for param in agent.actor.parameters():
                    actor_loss += l2_reg * torch.sum(param.pow(2))
                
                # Log action statistics
                with torch.no_grad():
                    epoch_action_mean += pred_actions.mean().item()
                    epoch_action_std += pred_actions.std().item()
                    epoch_entropy += agent.log_alpha.exp().item()
                
                agent.actor_optim.zero_grad()
                actor_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0).item()
                epoch_grad_norm += grad_norm
                agent.actor_optim.step()
                
                # ---------------------
                # Temperature (Alpha) Update
                # ---------------------
                # Note: This alpha update is a simplified version and does not follow the full SAC formulation.
                alpha_loss = -(agent.log_alpha * (agent.target_entropy + 0.5)).mean()
                agent.alpha_optim.zero_grad()
                alpha_loss.backward()
                agent.alpha_optim.step()
                
                # Update target networks every few epochs for stability
                if epoch % 5 == 0:
                    agent.update_targets()
                
                epoch_critic_loss += critic_loss.item()
                epoch_actor_loss += actor_loss.item()
                epoch_alpha_loss += alpha_loss.item()
            
            # Average metrics over batches
            num_batches = len(dataloader)
            epoch_critic_loss /= num_batches
            epoch_actor_loss /= num_batches
            epoch_q1_value /= num_batches
            epoch_q2_value /= num_batches
            epoch_action_mean /= num_batches
            epoch_action_std /= num_batches
            epoch_entropy /= num_batches
            epoch_grad_norm /= num_batches
            epoch_alpha_loss /= num_batches
            
            pbar.set_postfix({
                'Critic': f"{epoch_critic_loss:.3f}",
                'Actor': f"{epoch_actor_loss:.3f}",
                'Q1': f"{epoch_q1_value:.3f}",
                'Q2': f"{epoch_q2_value:.3f}",
                'α': f"{epoch_entropy:.3f}",
                '∇': f"{epoch_grad_norm:.2f}"
            })
            
            # Log metrics to CSV
            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, epoch_critic_loss, epoch_actor_loss, epoch_alpha_loss,
                                 epoch_q1_value, epoch_q2_value, epoch_action_mean, epoch_action_std,
                                 epoch_entropy, epoch_grad_norm])
            
            # Save a checkpoint every 50 epochs
            if (epoch + 1) % 50 == 0:
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
    training_logs_dir = Path("training_analysis")
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