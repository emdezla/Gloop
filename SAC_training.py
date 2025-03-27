import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
from helpers import DiabetesDataset, debug_tensor, Mish
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SACAgent(nn.Module):
    def __init__(self, state_dim=8, action_dim=2, hidden_dim=256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = 0.99
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), Mish(),
            nn.Linear(hidden_dim, hidden_dim), Mish(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh()
        )
        
        # Twin Q-networks
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), Mish(),
            nn.Linear(hidden_dim, hidden_dim), Mish(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), Mish(),
            nn.Linear(hidden_dim, hidden_dim), Mish(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Target networks
        self.q1_target = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), Mish(),
            nn.Linear(hidden_dim, hidden_dim), Mish(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2_target = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), Mish(),
            nn.Linear(hidden_dim, hidden_dim), Mish(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Copy parameters to target networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), 
            lr=3e-4
        )
        
        # Move to device
        self.to(device)
        
    def act(self, state, deterministic=False):
        """Get action for a state"""
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(device)
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            action = self.actor(state)
        return action.cpu().numpy()
        
    def update(self, batch):
        """Perform one update step using a batch of data"""
        # Unpack batch
        states = batch["state"].to(device)
        actions = batch["action"].to(device)
        rewards = batch["reward"].to(device).unsqueeze(1)
        next_states = batch["next_state"].to(device)
        dones = batch["done"].to(device).unsqueeze(1)
        
        # Update critics
        with torch.no_grad():
            next_actions = self.actor(next_states)
            q1_next = self.q1_target(torch.cat([next_states, next_actions], 1))
            q2_next = self.q2_target(torch.cat([next_states, next_actions], 1))
            q_next = torch.min(q1_next, q2_next)
            target_q = rewards + (1 - dones) * self.gamma * q_next
            
        current_q1 = self.q1(torch.cat([states, actions], 1))
        current_q2 = self.q2(torch.cat([states, actions], 1))
        
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 1.0)
        self.critic_optimizer.step()
        
        # Update actor
        policy_actions = self.actor(states)
        q1_policy = self.q1(torch.cat([states, policy_actions], 1))
        
        actor_loss = -q1_policy.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update_targets()
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'q_value': q1_policy.mean().item()
        }
    
    def _soft_update_targets(self, tau=0.005):
        """Soft update target networks"""
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def save(self, path):
        """Save model to disk"""
        if path:
            save_dir = os.path.dirname(path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            torch.save({
                'actor': self.actor.state_dict(),
                'q1': self.q1.state_dict(),
                'q2': self.q2.state_dict(),
                'q1_target': self.q1_target.state_dict(),
                'q2_target': self.q2_target.state_dict(),
            }, path)
        
    def load(self, path):
        """Load model from disk"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.q1.load_state_dict(checkpoint['q1'])
        self.q2.load_state_dict(checkpoint['q2'])
        self.q1_target.load_state_dict(checkpoint['q1_target'])
        self.q2_target.load_state_dict(checkpoint['q2_target'])

def train_sac_offline(dataset_path, epochs=1000, batch_size=256, save_path=None):
    """Train SAC agent using offline data from a dataset"""
    # Load dataset
    dataset = DiabetesDataset(csv_file=dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create agent
    agent = SACAgent(state_dim=8, action_dim=2, hidden_dim=256)
    
    # Training loop
    print(f"Training SAC agent for {epochs} epochs...")
    metrics = {
        'critic_loss': [],
        'actor_loss': [],
        'q_value': []
    }
    
    for epoch in tqdm(range(epochs)):
        epoch_metrics = {
            'critic_loss': 0,
            'actor_loss': 0,
            'q_value': 0
        }
        batch_count = 0
        
        for batch in dataloader:
            update_info = agent.update(batch)
            
            # Record metrics
            for k, v in update_info.items():
                epoch_metrics[k] += v
            batch_count += 1
        
        # Average metrics for this epoch
        for k in epoch_metrics:
            avg_value = epoch_metrics[k] / batch_count
            metrics[k].append(avg_value)
        
        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Critic Loss: {metrics['critic_loss'][-1]:.4f}")
            print(f"  Actor Loss: {metrics['actor_loss'][-1]:.4f}")
            print(f"  Q Value: {metrics['q_value'][-1]:.4f}")
            
            # Save checkpoint
            if save_path and (epoch + 1) % 100 == 0:
                checkpoint_path = f"{os.path.splitext(save_path)[0]}_epoch{epoch+1}.pt"
                agent.save(checkpoint_path)
    
    # Save final model
    if save_path:
        final_path = f"{os.path.splitext(save_path)[0]}_final.pt"
        agent.save(final_path)
        
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
    import random
    selected_episodes = random.sample(episode_indices, min(num_episodes, len(episode_indices)))
    
    for episode in selected_episodes:
        episode_reward = 0
        
        for idx in episode:
            sample = dataset[idx]
            state = sample['state'].numpy()
            
            # Get action from agent
            action = agent.act(state, deterministic=True)
            
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
