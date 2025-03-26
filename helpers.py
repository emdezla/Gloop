import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
import os
import csv
import pandas as pd
from datetime import datetime
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # For progress bar

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

        # Stochastic Actor (Gaussian policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),  # Added for stability
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

        # Twin Critics with CQL
        def create_q():
            return nn.Sequential(
                nn.Linear(state_dim + action_dim, 256),
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
        log_std = torch.clamp(log_std, min=-20, max=2)  # Stability
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
        policy_actions = model.actor(states)  # (batch_size, action_dim)
    
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
    
    # 6. CQL Penalty = logsumexp(Q_all) - mean(Q_dataset)
    logsumexp = torch.logsumexp(
        torch.cat([q1_all, q2_all], dim=1),  # Shape: (2*batch_size, 2)
        dim=0
    ).mean()
    
    dataset_q_mean = 0.5 * (q1_data.mean() + q2_data.mean())
    
    return logsumexp - dataset_q_mean