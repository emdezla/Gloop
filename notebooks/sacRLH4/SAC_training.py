import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
from tqdm import tqdm
from helpers import DiabetesDataset, debug_tensor, compute_reward_torch
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.FloatTensor(np.array(state)).to(device),
            torch.FloatTensor(np.array(action)).to(device),
            torch.FloatTensor(np.array(reward)).unsqueeze(-1).to(device),
            torch.FloatTensor(np.array(next_state)).to(device),
            torch.FloatTensor(np.array(done)).unsqueeze(-1).to(device)
        )
    
    def __len__(self):
        return len(self.buffer)

class LSTMActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=16, action_scale=1.0):
        super().__init__()
        self.action_scale = action_scale
        self.lstm = nn.LSTM(state_dim, hidden_size, batch_first=True)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU()
        )
        self.mean = nn.Linear(32, action_dim)
        self.log_std = nn.Linear(32, action_dim)
        
        # Initialize with small weights
        nn.init.xavier_uniform_(self.mean.weight, gain=0.01)
        nn.init.constant_(self.log_std.weight, -1.0)
        
    def forward(self, state):
        # state shape: [batch_size, seq_len, state_dim]
        lstm_out, _ = self.lstm(state)
        features = self.net(lstm_out[:, -1, :])  # Use last timestep
        mean = self.mean(features)
        log_std = torch.clamp(self.log_std(features), -5, 2)  # Constrain log_std
        return mean, log_std
        
    def sample(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        
        if deterministic:
            action = torch.tanh(mean) * self.action_scale
            return action, None
            
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        
        # Squash and scale action
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale
        
        # Calculate log probability with correction for tanh squashing
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob

class LSTMCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=16):
        super().__init__()
        self.lstm = nn.LSTM(state_dim, hidden_size, batch_first=True)
        self.net = nn.Sequential(
            nn.Linear(hidden_size + action_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, state, action):
        # state shape: [batch_size, seq_len, state_dim]
        lstm_out, _ = self.lstm(state)
        state_feat = lstm_out[:, -1, :]  # Use last timestep
        x = torch.cat([state_feat, action], dim=1)
        return self.net(x)

class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_size=16, action_scale=1.0):
        self.gamma = 0.997  # Match the gamma in the existing SAC implementation
        self.tau = 0.005    # Soft update parameter
        self.batch_size = 256
        self.action_scale = action_scale
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Networks
        self.actor = LSTMActor(state_dim, action_dim, hidden_size, action_scale).to(device)
        self.critic1 = LSTMCritic(state_dim, action_dim, hidden_size).to(device)
        self.critic2 = LSTMCritic(state_dim, action_dim, hidden_size).to(device)
        self.target_critic1 = LSTMCritic(state_dim, action_dim, hidden_size).to(device)
        self.target_critic2 = LSTMCritic(state_dim, action_dim, hidden_size).to(device)
        
        # Initialize targets
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optim = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 
            lr=3e-4
        )
        
        # Entropy tuning
        self.target_entropy = -torch.prod(torch.Tensor([action_dim])).item()  # -dim(A)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = torch.exp(self.log_alpha)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=3e-4)
        
        self.replay_buffer = ReplayBuffer(100000)
        
    def act(self, state, deterministic=False):
        """Get action for a single state"""
        with torch.no_grad():
            # Reshape state to [1, 1, state_dim] for single timestep
            if len(state.shape) == 1:
                state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
            elif len(state.shape) == 2:
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
            else:
                state = torch.FloatTensor(state).to(device)
                
            action, _ = self.actor.sample(state, deterministic)
        return action.cpu().numpy().squeeze()
        
    def update(self):
        """Perform one update step on the agent"""
        if len(self.replay_buffer) < self.batch_size:
            return {
                'critic_loss': 0,
                'actor_loss': 0,
                'alpha_loss': 0,
                'alpha': self.alpha.item()
            }
            
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Update critics
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next = self.target_critic1(next_states, next_actions)
            q2_next = self.target_critic2(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * q_next
            
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        
        critic_loss = nn.MSELoss()(q1, target_q) + nn.MSELoss()(q2, target_q)
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic_optim.step()
        
        # Update actor
        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optim.step()
        
        # Update alpha (temperature parameter)
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
        
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = torch.exp(self.log_alpha)
        
        # Soft update target networks
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item()
        }
    
    def save(self, path):
        """Save model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
            'log_alpha': self.log_alpha,
        }, path)
        
    def load(self, path):
        """Load model from disk"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha = torch.exp(self.log_alpha)

def train_sac_offline(dataset_path, epochs=1000, batch_size=256, save_path=None):
    """Train SAC agent using offline data from a dataset"""
    # Load dataset
    dataset = DiabetesDataset(csv_file=dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create agent
    state_dim = 8  # From the dataset
    action_dim = 2  # basal, bolus
    agent = SACAgent(state_dim, action_dim, hidden_size=16, action_scale=1.0)
    
    # Fill replay buffer with dataset
    print("Filling replay buffer with dataset...")
    for batch in tqdm(dataloader):
        states = batch["state"].numpy()
        actions = batch["action"].numpy()
        rewards = batch["reward"].numpy()
        next_states = batch["next_state"].numpy()
        dones = batch["done"].numpy()
        
        for i in range(len(states)):
            # Reshape state to sequence format [seq_len, state_dim]
            state_seq = np.expand_dims(states[i], axis=0)
            next_state_seq = np.expand_dims(next_states[i], axis=0)
            
            agent.replay_buffer.push(
                state_seq,
                actions[i],
                rewards[i],
                next_state_seq,
                dones[i]
            )
    
    # Training loop
    print(f"Training SAC agent for {epochs} epochs...")
    metrics = {
        'critic_loss': [],
        'actor_loss': [],
        'alpha_loss': [],
        'alpha': []
    }
    
    for epoch in tqdm(range(epochs)):
        update_info = agent.update()
        
        # Record metrics
        for k, v in update_info.items():
            metrics[k].append(v)
        
        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Critic Loss: {np.mean(metrics['critic_loss'][-100:]):.4f}")
            print(f"  Actor Loss: {np.mean(metrics['actor_loss'][-100:]):.4f}")
            print(f"  Alpha: {metrics['alpha'][-1]:.4f}")
            
            # Save checkpoint
            if save_path:
                agent.save(f"{save_path}_epoch{epoch+1}.pt")
    
    # Save final model
    if save_path:
        agent.save(f"{save_path}_final.pt")
        
    return agent

def evaluate_agent(agent, dataset_path, num_episodes=10):
    """Evaluate trained agent on test data"""
    dataset = DiabetesDataset(csv_file=dataset_path)
    
    total_reward = 0
    episode_rewards = []
    
    # Group dataset into episodes based on done flag
    episode_indices = []
    current_episode = []
    
    for i in range(len(dataset)):
        current_episode.append(i)
        if dataset[i]['done']:
            episode_indices.append(current_episode)
            current_episode = []
    
    # Add last episode if not done
    if current_episode:
        episode_indices.append(current_episode)
    
    # Select random episodes to evaluate
    selected_episodes = random.sample(episode_indices, min(num_episodes, len(episode_indices)))
    
    for episode in selected_episodes:
        episode_reward = 0
        
        for idx in episode:
            sample = dataset[idx]
            state = sample['state'].numpy()
            
            # Get action from agent
            state_seq = np.expand_dims(state, axis=0)  # [1, state_dim]
            action = agent.act(state_seq, deterministic=True)
            
            # Use reward from dataset
            reward = sample['reward'].item()
            episode_reward += reward
            
        episode_rewards.append(episode_reward)
        total_reward += episode_reward
    
    avg_reward = total_reward / len(selected_episodes)
    print(f"Evaluation over {len(selected_episodes)} episodes:")
    print(f"  Average Reward: {avg_reward:.4f}")
    print(f"  Min/Max Reward: {min(episode_rewards):.4f}/{max(episode_rewards):.4f}")
    
    return avg_reward, episode_rewards
