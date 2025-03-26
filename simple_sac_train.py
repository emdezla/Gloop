import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Risk Index calculation
def calculate_risk_index(glucose_mgdl):
    """
    Calculate risk index based on blood glucose level.
    Reference: Kovatchev et al. (2006)
    """
    # Convert mg/dL to BGRI scale
    f_glucose = 1.509 * ((np.log(glucose_mgdl))**1.084 - 5.381)
    
    # Calculate risk
    if glucose_mgdl <= 112.5:  # Hypoglycemia risk
        risk = 10 * (f_glucose)**2
    else:  # Hyperglycemia risk
        risk = 10 * (f_glucose)**2
    
    return risk

# Reward function
def calculate_reward(next_glucose_mgdl):
    """Calculate reward based on next glucose value."""
    if next_glucose_mgdl <= 39:
        return -15.0  # Severe hypoglycemia penalty
    else:
        ri = calculate_risk_index(next_glucose_mgdl)
        return -1.0 * ri  # Negative risk as reward

# Action to insulin conversion
def action_to_insulin(a):
    """Maps normalized action [-1, 1] to insulin rate [0, 5] U/min."""
    I_max = 5.0
    eta = 4.0
    return I_max * torch.exp(eta * (a - 1))

# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size=100_000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return (
            torch.FloatTensor(states).to(device),
            torch.FloatTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(next_states).to(device),
            torch.FloatTensor(dones).to(device)
        )
    
    def __len__(self):
        return len(self.buffer)

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim=8, action_dim=2, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)
            
    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)  # Constrain log_std for numerical stability
        return mean, log_std
    
    def sample(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        if deterministic:
            action = mean
        else:
            std = log_std.exp()
            normal = torch.randn_like(mean)
            action = mean + normal * std
            
        action_squashed = torch.tanh(action)
        return action_squashed

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim=8, action_dim=2, hidden_dim=32):
        super().__init__()
        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)
            
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q1(x), self.q2(x)
    
    def q1_forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q1(x)

# SAC Agent
class SACAgent:
    def __init__(self, state_dim=8, action_dim=2, hidden_dim=32, lr=3e-4, gamma=0.997, tau=0.005, alpha=0.1):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha  # Entropy coefficient
        
        # Initialize networks
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        
        # Copy critic parameters to target
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
            
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Loss function
        self.mse_loss = nn.MSELoss()
        
    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = self.actor.sample(state, deterministic)
        return action.cpu().numpy().flatten()
    
    def update(self, replay_buffer, batch_size):
        # Sample batch
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        # Critic update
        with torch.no_grad():
            next_actions = self.actor.sample(next_states)
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2)
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q
            
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = self.mse_loss(current_q1, target_q) + self.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        actions_sampled = self.actor.sample(states)
        q1, q2 = self.critic(states, actions_sampled)
        q = torch.min(q1, q2)
        
        # Calculate log_probs for entropy
        mean, log_std = self.actor(states)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        
        # Get log probability of actions_sampled before tanh
        x = torch.atanh(actions_sampled)  # Inverse of tanh
        log_probs = normal.log_prob(x)
        log_probs -= torch.log(1 - actions_sampled.pow(2) + 1e-6)  # Correction for tanh
        log_probs = log_probs.sum(1, keepdim=True)
        
        actor_loss = -(q - self.alpha * log_probs).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return critic_loss.item(), actor_loss.item()
    
    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict()
        }, path)
        print(f"Model saved to {path}")
        
    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        print(f"Model loaded from {path}")

def train_sac(train_csv_path, epochs=3000, batch_size=256, save_interval=500, model_dir="models"):
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Load and preprocess data
    print(f"Loading data from {train_csv_path}...")
    df = pd.read_csv(train_csv_path)
    df = df.ffill().bfill()  # Fill missing values
    
    # Define state and action columns
    state_columns = ['glu', 'glu_d', 'glu_t', 'hr', 'hr_d', 'hr_t', 'iob', 'hour']
    action_columns = ['basal', 'bolus']
    
    # Extract states and actions
    states = df[state_columns].values
    actions = df[action_columns].values
    
    # Compute rewards
    rewards = []
    for i in range(len(df) - 1):
        next_glucose = df['glu_raw'].iloc[i + 1]
        reward = calculate_reward(next_glucose)
        rewards.append(reward)
    rewards.append(0.0)  # Last state has no next state
    
    # Create done flags (1 for end of episode, 0 otherwise)
    # For simplicity, we'll consider each day as an episode
    dones = np.zeros(len(df))
    if 'day' in df.columns:
        day_changes = df['day'].diff().fillna(0) != 0
        dones[day_changes] = 1.0
    
    # Initialize replay buffer
    buffer = ReplayBuffer(max_size=len(df))
    
    # Fill buffer with transitions
    print("Filling replay buffer...")
    for i in range(len(df) - 1):
        buffer.add(states[i], actions[i], rewards[i], states[i + 1], dones[i])
    
    # Initialize SAC agent
    state_dim = len(state_columns)
    action_dim = len(action_columns)
    agent = SACAgent(state_dim=state_dim, action_dim=action_dim)
    
    # Training loop
    print(f"Starting training for {epochs} epochs...")
    losses = []
    
    for epoch in tqdm(range(epochs)):
        critic_loss, actor_loss = agent.update(buffer, batch_size)
        losses.append((critic_loss, actor_loss))
        
        # Save model periodically
        if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
            save_path = os.path.join(model_dir, f"sac_model_epoch_{epoch+1}.pt")
            agent.save(save_path)
    
    # Save final model
    final_path = os.path.join(model_dir, "sac_model_final.pt")
    agent.save(final_path)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot([x[0] for x in losses])
    plt.title('Critic Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot([x[1] for x in losses])
    plt.title('Actor Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_curves.png'))
    plt.close()
    
    print(f"Training completed. Final model saved to {final_path}")
    return agent

if __name__ == "__main__":
    # Use the first available training dataset
    train_files = [f for f in os.listdir("datasets/processed") if f.endswith("-train.csv")]
    if train_files:
        train_csv_path = os.path.join("datasets/processed", train_files[0])
        print(f"Using training dataset: {train_csv_path}")
        train_sac(train_csv_path)
    else:
        print("No training datasets found in datasets/processed directory.")
