import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.stack(state), action, reward, np.stack(next_state), done

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
         mean, log_std = self.forward(state)
         std = log_std.exp()
         normal = torch.distributions.Normal(mean, std)
         x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
         action = torch.tanh(x_t)
         log_prob = normal.log_prob(x_t)
         # Enforcing Action Bound
         log_prob -= torch.log(1 - action.pow(2) + 1e-6)
         log_prob = log_prob.sum(1, keepdim=True)
         return action, log_prob, mean

class SACAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.q1_net = QNetwork(state_dim, action_dim)
        self.q2_net = QNetwork(state_dim, action_dim)
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.target_q1_net = QNetwork(state_dim, action_dim)
        self.target_q2_net = QNetwork(state_dim, action_dim)

        self.target_q1_net.load_state_dict(self.q1_net.state_dict())
        self.target_q2_net.load_state_dict(self.q2_net.state_dict())

        self.q1_optimizer = optim.Adam(self.q1_net.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2_net.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = self.policy_net.sample(state)
        return action.cpu().numpy()[0]

    def update_parameters(self, memory, batch_size):
      if len(memory) < batch_size:
          return
      state, action, reward, next_state, done = memory.sample(batch_size)
      state = torch.FloatTensor(state)
      next_state = torch.FloatTensor(next_state)
      action = torch.FloatTensor(action)
      reward = torch.FloatTensor(reward).unsqueeze(1)
      done = torch.FloatTensor(done).unsqueeze(1)

      with torch.no_grad():
          next_action, next_log_prob, _ = self.policy_net.sample(next_state)
          target_q1_value = self.target_q1_net(next_state, next_action)
          target_q2_value = self.target_q2_net(next_state, next_action)
          target_min_q = torch.min(target_q1_value, target_q2_value)
          target_q = reward + (1 - done) * self.gamma * (target_min_q - self.alpha * next_log_prob)

      current_q1 = self.q1_net(state, action)
      current_q2 = self.q2_net(state, action)

      q1_loss = F.mse_loss(current_q1, target_q)
      q2_loss = F.mse_loss(current_q2, target_q)

      self.q1_optimizer.zero_grad()
      q1_loss.backward()
      self.q1_optimizer.step()

      self.q2_optimizer.zero_grad()
      q2_loss.backward()
      self.q2_optimizer.step()

      new_action, log_prob, mean = self.policy_net.sample(state)
      q1_value = self.q1_net(state, new_action)
      q2_value = self.q2_net(state, new_action)
      min_q_new_actions = torch.min(q1_value, q2_value)

      policy_loss = (self.alpha * log_prob - min_q_new_actions).mean()

      self.policy_optimizer.zero_grad()
      policy_loss.backward()
      self.policy_optimizer.step()

      for target_param, param in zip(self.target_q1_net.parameters(), self.q1_net.parameters()):
          target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

      for target_param, param in zip(self.target_q2_net.parameters(), self.q2_net.parameters()):
          target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))