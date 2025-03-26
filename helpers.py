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
        self.rewards = compute_reward_torch(glucose_next_tensor) / 5.0  # Reduced normalization factor from 15.0

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
        # Remove Q-value normalization layer for better gradient flow

        # Improved Stochastic Actor (Gaussian policy)
        self.actor = nn.Sequential(
            nn.Linear(8, 512),
            nn.LayerNorm(512),  # Changed from BatchNorm for more stability
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),  # Changed from BatchNorm
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.Dropout(0.1),  # Reduced dropout rate
        )
        
        # Advantage estimation head
        self.adv_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )
        
        self.mean = nn.Linear(256, 2)
        self.log_std = nn.Linear(256, 2)
        # Initialize with larger weights for better exploration
        nn.init.uniform_(self.mean.weight, -0.1, 0.1)
        nn.init.constant_(self.log_std.weight, -1.0)  # Safer initialization

        # Twin Critics with CQL - Revised architecture for better gradient flow
        def create_q():
            return nn.Sequential(
                nn.Linear(10, 512),
                nn.SiLU(),  # Changed from ReLU
                nn.Linear(512, 512),  # Deeper network
                nn.LeakyReLU(0.1),  # Better gradient flow than SiLU
                nn.Linear(512, 1)
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

def compute_cql_penalty(states, dataset_actions, model, num_action_samples=10, global_step=0, epochs=100, dataloader_len=100):
    """
    Adaptive CQL penalty calculation with gradient regularization
    Args:
        states: Current states from batch (batch_size, state_dim)
        dataset_actions: Actions taken in the dataset (batch_size, action_dim)
        model: Reference to the agent's networks
        global_step: Current training step for dynamic margin adjustment
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
    
    # Adaptive CQL weight based on Q-value spread - less aggressive scaling
    q_spread = torch.std(q1_all) + torch.std(q2_all)
    adaptive_factor = 1.0 + 0.5*torch.tanh(q_spread/3.0)
    
    # New dynamic margin based on training progress - faster decay
    progress = min(global_step / (epochs * dataloader_len), 1.0)  # 0→1 scale
    margin = 2.0 * (1 - 0.98*progress)  # Start at 2.0, decay to 0.04
    
    # Add gradient regularization to prevent Q-value collapse
    q_penalty = torch.clamp(cql_penalty * adaptive_factor, min=-margin, max=margin)
    
    return q_penalty

def train_offline(dataset_path, model, csv_file='training_stats.csv', 
                 epochs=1000, batch_size=256, print_interval=100,
                 device="cuda", alpha=0.2, cql_weight=0.25, tau=0.01):  # Increased tau from 0.005 to 0.01
    """Main training loop for offline CQL-based SAC."""
    # Dataset setup
    dataset = DiabetesDataset(csv_file=dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = model.to(device)
    
    # Optimizers with different learning rates
    optimizer_actor = optim.Adam(
        list(model.actor.parameters()) + 
        list(model.mean.parameters()) + 
        list(model.log_std.parameters()) +
        list(model.adv_head.parameters()), 
        lr=3e-4
    )
    optimizer_critic = optim.Adam(
        list(model.q1.parameters()) + list(model.q2.parameters()), 
        lr=1e-4,  # Reduced from 3e-4
        weight_decay=1e-4
    )
    optimizer_alpha = optim.Adam([model.log_alpha], lr=1e-4)
    
    # Learning rate schedulers
    scheduler_critic = optim.lr_scheduler.OneCycleLR(
        optimizer_critic, 
        max_lr=1e-4,
        total_steps=epochs*len(dataloader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    scheduler_actor = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_actor,
        T_max=10*len(dataloader)
    )
    
    # Adjusted target entropy for better exploration-exploitation balance
    target_entropy = -torch.tensor(action_dim * 0.8).to(device)  # Reduced from 1.5 to prevent entropy collapse
    
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
            cql_penalty = compute_cql_penalty(states, dataset_actions, model, global_step=global_step, 
                                             epochs=epochs, dataloader_len=len(dataloader))
            # Apply cql_weight here instead of in the compute_cql_penalty function
            # Add gradient penalty to prevent Q-value collapse
            grad_pen = 0.001 * (current_q1.pow(2).mean() + current_q2.pow(2).mean())
            critic_loss = td_loss + cql_weight * cql_penalty + grad_pen

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
            
            # Policy smoothing regularization
            noise = torch.randn_like(pred_actions) * 0.1
            smooth_actions = torch.clamp(pred_actions + noise, -model.action_scale, model.action_scale)
            
            log_probs = normal.log_prob(x_t).sum(1)
            log_probs -= torch.log(1 - y_t.pow(2) + 1e-6).sum(1)
            # Shift entropy to positive range to prevent entropy collapse
            entropy = torch.clamp(log_probs.mean() + 2.0, min=0.1, max=2.0)
            
            # Add entropy regularization scaling
            entropy_scale = torch.clamp(1.0 / (entropy.detach() + 1e-6), 0.1, 10.0)

            # Compute advantage using the advantage head
            adv_value = model.adv_head(model.actor(states)).squeeze()
            
            q1_pred = model.q1(torch.cat([states, pred_actions], 1))
            q2_pred = model.q2(torch.cat([states, pred_actions], 1))
            q1_smooth = model.q1(torch.cat([states, smooth_actions], 1))
            q2_smooth = model.q2(torch.cat([states, smooth_actions], 1))
            
            # Add policy smoothing penalty
            smooth_penalty = F.mse_loss(q1_pred, q1_smooth) + F.mse_loss(q2_pred, q2_smooth)
            
            # Revised conservative actor loss to prevent over-optimization
            q_min = torch.min(q1_pred, q2_pred)
            actor_loss = -(q_min - 0.1 * q_min.std()).mean() + alpha * entropy + 0.1 * smooth_penalty

            optimizer_actor.zero_grad()
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 0.5).item()
            optimizer_actor.step()
            
            # Alpha optimization with detached entropy
            alpha_loss = -(model.log_alpha * (entropy.detach() - target_entropy)).mean()
            optimizer_alpha.zero_grad()
            alpha_loss.backward()
            torch.nn.utils.clip_grad_norm_([model.log_alpha], 0.5)
            optimizer_alpha.step()
            
            # Clamp alpha parameter to prevent extreme values
            with torch.no_grad():
                torch.clamp_(model.log_alpha, min=-4.0, max=5.0)

            # --- Target Update ---
            model.update_targets(tau)
            
            # Update learning rate schedulers
            scheduler_critic.step()
            scheduler_actor.step()

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
