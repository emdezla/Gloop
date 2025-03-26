import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import os
import csv
import pandas as pd
from datetime import datetime
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # For progress bar
from collections import defaultdict



device = "cuda" if torch.cuda.is_available() else "cpu"
state_dim = 8
action_dim = 2  
alpha = 0.2  
cql_weight = 0.1  # Reduced from 0.25 to 0.1 for less conservative Q-values
batch_size = 256


class DiabetesDataset(Dataset):
    def __init__(self, csv_file):
        # Load and clean CSV data
        self.df = pd.read_csv(csv_file)
        self.df = self.df.ffill().bfill()

        # Extract state features (8 dimensions)
        self.states = self.df[[
            "glu", "glu_d", "glu_t",
            "hr", "hr_d", "hr_t",
            "iob", "hour"
        ]].values.astype(np.float32)

        # Extract action features (2 dimensions)
        self.actions = self.df[["basal", "bolus"]].values.astype(np.float32)

        # Extract done flags
        self.dones = self.df["done"].values.astype(np.float32)

        # Compute rewards based on glu_raw at t+1
        glucose_next_tensor = torch.tensor(self.df["glu_raw"].values, dtype=torch.float32)
        self.rewards = compute_reward_torch(glucose_next_tensor) / 15.0  # Normalize if needed

        # Compute next_states using vectorized roll
        self.next_states = np.roll(self.states, shift=-1, axis=0)

        # Prevent transitions across episode boundaries
        self.next_states[self.dones == 1] = self.states[self.dones == 1]

        # Slice to make all arrays align: remove last step (no next state), and align reward with t

        self.states      = self.states[:-2]
        self.actions     = self.actions[:-2]
        self.rewards     = self.rewards[1:-1]
        self.next_states = self.next_states[:-2]
        self.dones       = self.dones[:-2]
        self.dones       = torch.tensor(self.dones, dtype=torch.float32)

        # Sanity check
        L = len(self.states)
        assert all(len(arr) == L for arr in [self.actions, self.rewards, self.next_states, self.dones]), \
            f"Inconsistent lengths in dataset components: {L}"

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            "state":      torch.from_numpy(self.states[idx]).float(),
            "action":     torch.from_numpy(self.actions[idx]).float(),
            "reward":     self.rewards[idx].float(),
            "next_state": torch.from_numpy(self.next_states[idx]).float(),
            "done":       self.dones[idx]
        }

class SACCQL(nn.Module):
    def __init__(self): 
        super().__init__()
        self.action_scale = 1.0  # Adjust to your insulin range (e.g., 0-5 units)
        # Add entropy temperature parameter with higher initialization
        self.log_alpha = torch.nn.Parameter(torch.tensor([2.0], requires_grad=True))

        # Stochastic Actor (Gaussian policy)
        self.actor = nn.Sequential(
            nn.Linear(8, 256),
            nn.LayerNorm(256),  # Added for stability
            nn.ReLU(),
            nn.Dropout(0.1),  # Added dropout to prevent overfitting
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),  # Added dropout
        )
        self.mean = nn.Linear(256, 2)
        self.log_std = nn.Linear(256, 2)
        # Initialize with larger weights for better exploration
        nn.init.uniform_(self.mean.weight, -0.1, 0.1)
        nn.init.constant_(self.log_std.weight, -1.0)  # Safer initialization

        # Twin Critics with CQL
        def create_q():
            return nn.Sequential(
                nn.Linear(10, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
        
        self.q1 = create_q()
        self.q2 = create_q()
        self.q1_target = create_q()
        self.q2_target = create_q()
        
        # Initialize targets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

    def forward(self, state):
        """For action sampling with entropy"""
        hidden = self.actor(state)
        mean = self.mean(hidden)
        log_std = self.log_std(hidden)
        log_std = torch.clamp(log_std, min=-5, max=0.5)  # Changed from (-20, 2) for better stability
        return mean, log_std

    def act(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        if deterministic:
            action = torch.tanh(mean) * self.action_scale
        else:
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()  # Reparameterization
            action = torch.tanh(x_t) * self.action_scale
        return action

    def update_targets(self, tau=0.005):
        # Soft update: target = tau * main + (1-tau) * target
        with torch.no_grad():
            for t, m in zip(self.q1_target.parameters(), self.q1.parameters()):
                t.data.copy_(tau * m.data + (1 - tau) * t.data)
            for t, m in zip(self.q2_target.parameters(), self.q2.parameters()):
                t.data.copy_(tau * m.data + (1 - tau) * t.data)

def compute_reward_torch(glucose_next):
    """
    Compute RI-based reward in PyTorch.
    """
    glucose_next = torch.clamp(glucose_next, min=1e-6)
    log_term = torch.log(glucose_next) ** 1.084
    f = 1.509 * (log_term - 5.381)
    ri = 10 * f ** 2

    reward = -torch.clamp(ri / 100.0, 0, 1)
    reward[glucose_next <= 39.0] = -15.0
    return reward

def debug_tensor(tensor, name="", check_grad=False, threshold=1e6):
    """
    Prints diagnostic information about a tensor.
    
    Args:
        tensor (torch.Tensor): The tensor to check.
        name (str): Optional name for logging.
        check_grad (bool): Also check gradients if available.
        threshold (float): Warn if values exceed this.
    """
    try:
        t_min = tensor.min().item()
        t_max = tensor.max().item()
        t_mean = tensor.mean().item()
        t_std = tensor.std().item()
    except Exception as e:
        print(f"⚠️ Could not extract stats for {name}: {e}")
        return

    print(f"🧪 [{name}] Shape: {tuple(tensor.shape)} | min: {t_min:.4f}, max: {t_max:.4f}, mean: {t_mean:.4f}, std: {t_std:.4f}")

    if torch.isnan(tensor).any():
        print(f"❌ NaNs detected in {name}")
    if torch.isinf(tensor).any():
        print(f"❌ Infs detected in {name}")
    if abs(t_min) > threshold or abs(t_max) > threshold:
        print(f"⚠️ Extreme values detected in {name}: values exceed ±{threshold}")

    if check_grad and tensor.requires_grad and tensor.grad is not None:
        grad = tensor.grad
        print(f"🔁 [{name}.grad] norm: {grad.norm().item():.4f}")
        if torch.isnan(grad).any():
            print(f"❌ NaNs in gradient of {name}")
        if torch.isinf(grad).any():
            print(f"❌ Infs in gradient of {name}")

def compute_cql_penalty(states, dataset_actions, model, num_action_samples=10):
    """
    Fixed CQL penalty calculation
    Args:
        states: Current states from batch (batch_size, state_dim)
        dataset_actions: Actions taken in the dataset (batch_size, action_dim)
        model: Reference to the agent's networks
    """
    batch_size = states.shape[0]
    
    # 1. Get policy-generated actions (from current actor)
    with torch.no_grad():
        mean, log_std = model(states)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        policy_actions = y_t * model.action_scale
    
    # 2. Combine dataset actions + policy actions
    # Shape: (2*batch_size, action_dim)
    all_actions = torch.cat([dataset_actions, policy_actions], dim=0)
    
    # 3. Expand states to match action candidates
    # Shape: (2*batch_size, state_dim)
    expanded_states = states.repeat(2, 1)
    
    # 4. Compute Q-values for ALL action candidates
    q1_all = model.q1(torch.cat([expanded_states, all_actions], dim=1))
    q2_all = model.q2(torch.cat([expanded_states, all_actions], dim=1))
    
    # 5. Compute Q-values for DATASET actions (original batch)
    q1_data = model.q1(torch.cat([states, dataset_actions], dim=1))
    q2_data = model.q2(torch.cat([states, dataset_actions], dim=1))
    
    # 6. Revised CQL penalty calculation
    logsumexp_q1 = torch.logsumexp(q1_all, dim=0).mean()
    logsumexp_q2 = torch.logsumexp(q2_all, dim=0).mean()
    dataset_q_mean = 0.5 * (q1_data.mean() + q2_data.mean())
    
    cql_penalty = (logsumexp_q1 + logsumexp_q2 - 2 * dataset_q_mean)
    
    # Dynamic margin based on Q-values
    margin = torch.clamp(5.0 / (torch.abs(dataset_q_mean) + 1e-6), 0.1, 10.0)
    return torch.clamp(cql_penalty, min=-margin, max=margin)  # Clamping to prevent Q-value collapse

def train_offline(dataset_path, model, csv_file='training_stats.csv', 
                 epochs=1000, batch_size=256, print_interval=100,
                 device="cuda", alpha=0.2, cql_weight=0.25, tau=0.005):
    """Main training loop for offline CQL-based SAC."""
    # Dataset setup
    dataset = DiabetesDataset(csv_file=dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = model.to(device)
    
    # Optimizers
    optimizer_actor = optim.Adam(model.actor.parameters(), lr=3e-4)
    optimizer_critic = optim.Adam(
        list(model.q1.parameters()) + list(model.q2.parameters()), 
        lr=3e-4,
        weight_decay=1e-4  # Added weight decay
    )
    optimizer_alpha = optim.Adam([model.log_alpha], lr=1e-4)  # Increased from 3e-5 for better adaptation
    target_entropy = -torch.tensor(action_dim).to(device)  # Target entropy = -action_dim
    
    # Logging
    writer = SummaryWriter()
    torch.autograd.set_detect_anomaly(False)  # Disable for performance
    
    # Initialize CSV
    with open(csv_file, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['Epoch', 'Iteration', 'TD Loss', 'CQL Penalty', 
                           'Critic Loss', 'Actor Loss', 'Q1 Value', 'Q2 Value',
                           'Action_Mean', 'Action_Std', 'Entropy', 'Alpha', 'Alpha_Loss',
                           'Q1_Grad', 'Q2_Grad', 'Actor_Grad'])

    # Training loop
    global_step = 0
    for epoch in tqdm(range(epochs), desc="Training"):
        metrics = defaultdict(float)
        batch_count = 0
        
        for i, batch in enumerate(dataloader):
            # Device transfer
            states = batch["state"].to(device)
            dataset_actions = batch["action"].to(device)
            rewards = batch["reward"].to(device).unsqueeze(1)
            next_states = batch["next_state"].to(device)
            dones = batch["done"].to(device).unsqueeze(1)

            # --- Critic Update ---
            with torch.no_grad():
                next_mean, next_log_std = model(next_states)
                next_std = next_log_std.exp()
                next_normal = torch.distributions.Normal(next_mean, next_std)
                next_actions = torch.tanh(next_normal.rsample()) * model.action_scale
                
                q1_next = model.q1_target(torch.cat([next_states, next_actions], 1))
                q2_next = model.q2_target(torch.cat([next_states, next_actions], 1))
                target_q = rewards + (1 - dones) * 0.99 * torch.min(q1_next, q2_next)

            current_q1 = model.q1(torch.cat([states, dataset_actions], 1))
            current_q2 = model.q2(torch.cat([states, dataset_actions], 1))
            
            td_loss = F.smooth_l1_loss(current_q1, target_q) + F.smooth_l1_loss(current_q2, target_q)
            cql_penalty = compute_cql_penalty(states, dataset_actions, model)
            # Apply cql_weight here instead of in the compute_cql_penalty function
            critic_loss = td_loss + cql_weight * cql_penalty

            optimizer_critic.zero_grad()
            critic_loss.backward()
            q1_grad_norm = torch.nn.utils.clip_grad_norm_(model.q1.parameters(), 1.0).item()
            q2_grad_norm = torch.nn.utils.clip_grad_norm_(model.q2.parameters(), 1.0).item()
            optimizer_critic.step()

            # --- Actor Update ---
            with torch.no_grad():
                alpha = model.log_alpha.exp().detach()
                
            mean, log_std = model(states)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            pred_actions = y_t * model.action_scale
            
            log_probs = normal.log_prob(x_t).sum(1)
            log_probs -= torch.log(1 - y_t.pow(2) + 1e-6).sum(1)
            entropy = log_probs.mean()  # Removed negative sign to fix entropy calculation
            
            # Add entropy regularization scaling
            entropy_scale = torch.clamp(1.0 / (entropy.detach() + 1e-6), 0.1, 10.0)

            q1_pred = model.q1(torch.cat([states, pred_actions], 1))
            q2_pred = model.q2(torch.cat([states, pred_actions], 1))
            actor_loss = -torch.min(q1_pred, q2_pred).mean() * 0.5 + alpha * entropy * entropy_scale

            optimizer_actor.zero_grad()
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 0.5).item()
            optimizer_actor.step()
            
            # Alpha optimization
            alpha_loss = -(model.log_alpha * (entropy - target_entropy).detach()).mean()
            optimizer_alpha.zero_grad()
            alpha_loss.backward()
            torch.nn.utils.clip_grad_norm_([model.log_alpha], 0.5)  # Increased clipping threshold
            optimizer_alpha.step()

            # --- Target Update ---
            model.update_targets(tau)

            # --- Metrics ---
            metrics['td'] += td_loss.item()
            metrics['cql'] += cql_penalty.item()
            metrics['critic'] += critic_loss.item()
            metrics['actor'] += actor_loss.item()
            metrics['q1'] += q1_pred.mean().item()
            metrics['q2'] += q2_pred.mean().item()
            metrics['action_mean'] += pred_actions.mean().item()
            metrics['action_std'] += pred_actions.std().item()
            metrics['entropy'] += entropy.item()
            metrics['alpha'] = alpha.item()
            metrics['alpha_loss'] = alpha_loss.item()
            metrics['q1_grad'] += q1_grad_norm
            metrics['q2_grad'] += q2_grad_norm
            metrics['actor_grad'] += actor_grad_norm
            batch_count += 1
            global_step += 1

            # --- Logging ---
            if i % print_interval == 0 and batch_count > 0:
                avg_metrics = {k: v/batch_count for k, v in metrics.items()}
                
                # TensorBoard
                for key, value in avg_metrics.items():
                    writer.add_scalar(f'{key.capitalize()}', value, global_step)
                
                # CSV
                with open(csv_file, 'a', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([
                        epoch, i,
                        avg_metrics['td'], avg_metrics['cql'],
                        avg_metrics['critic'], avg_metrics['actor'],
                        avg_metrics['q1'], avg_metrics['q2'],
                        avg_metrics['action_mean'], avg_metrics['action_std'],
                        avg_metrics['entropy'], avg_metrics['alpha'], avg_metrics['alpha_loss'],
                        avg_metrics.get('q1_grad', 0), avg_metrics.get('q2_grad', 0), avg_metrics.get('actor_grad', 0)
                    ])
                
                metrics = defaultdict(float)
                batch_count = 0

    writer.close()
    return model
