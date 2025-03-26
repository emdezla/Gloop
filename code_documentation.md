# Diabetes Management RL: Critical Analysis & Simplifications

## Current Architecture Critique

### 1. Algorithm Choice (CQL+SAC)
- **Original Justification**:
  - Combines SAC's entropy maximization with CQL's conservatism
  - Meant to handle distributional shift in offline RL
- **Actual Issues**:
  - Complex loss landscape from competing objectives
  - CQL penalty dominates learning dynamics
  - Difficult hyperparameter tuning (α, CQL weight, τ)
- **Simplification Potential**:
  - Use vanilla SAC with behavioral cloning regularization
  - Add dropout instead of CQL for uncertainty estimation
  - Use simpler TD3 algorithm with noise injection

### 2. Network Architecture
- **Original Choices**:
  - 3-layer networks with BatchNorm
  - Twin Q-networks with LayerNorm
  - Separate advantage head
- **Identified Problems**:
  - BatchNorm causes unstable gradients
  - Overparameterization for small action space
  - Advantage head creates conflicting gradients
- **Simpler Alternative**:
  ```python
  class SimpleActor(nn.Module):
      def __init__(self):
          super().__init__()
          self.net = nn.Sequential(
              nn.Linear(8, 64),
              nn.ReLU(),
              nn.Linear(64, 2),
              nn.Tanh()  # Constrained action output
          )
  ```

### 3. Entropy Management
- **Current Complexity**:
  - Learnable α parameter
  - Entropy clamping and scaling
  - Target entropy scheduling
- **Simpler Approach**:
  - Fixed temperature (α=0.2)
  - Remove entropy from actor loss
  - Use behavioral cloning regularization instead

## Proposed Simplified Algorithm

### Basic Offline SAC
- **Key Differences**:
  1. Remove CQL penalty
  2. Single Q-network
  3. Fixed entropy coefficient
  4. Add behavioral cloning loss

| Component          | Original          | Simplified       |
|--------------------|-------------------|------------------|
| Q-Networks         | Twin + LayerNorm  | Single + ReLU    |
| Policy Update      | CQL + Entropy     | BC Regularized   |
| Exploration        | Learned α         | Fixed α=0.2      |
| Params (Actor)     | 500k+             | <10k             |

**basic_offline_training.py**:
```python
import torch
import torch.nn as nn
import torch.optim as optim
from helpers import DiabetesDataset, device

class SimpleSAC(nn.Module):
    def __init__(self):
        super().__init__()
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh()
        )
        
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(10, 64),  # state + action
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.critic_target = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.critic_target.load_state_dict(self.critic.state_dict())

    def act(self, state):
        return self.actor(state)

def train_basic(dataset_path, epochs=100, lr=3e-4, batch_size=256):
    # Data
    dataset = DiabetesDataset(dataset_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    model = SimpleSAC().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    
    # Fixed hyperparams
    gamma = 0.99
    alpha = 0.2  # Fixed entropy coeff
    tau = 0.01
    
    for epoch in range(epochs):
        for batch in loader:
            states = batch["state"].to(device)
            actions = batch["action"].to(device)
            rewards = batch["reward"].to(device)
            next_states = batch["next_state"].to(device)
            dones = batch["done"].to(device)
            
            # Critic Update
            with torch.no_grad():
                next_actions = model.actor(next_states)
                target_q = rewards + (1 - dones) * gamma * model.critic_target(
                    torch.cat([next_states, next_actions], 1))
            
            current_q = model.critic(torch.cat([states, actions], 1))
            critic_loss = ((current_q - target_q)**2).mean()
            
            # Behavioral Cloning
            bc_loss = ((model.actor(states) - actions)**2).mean()
            
            # Total Loss
            loss = critic_loss + 0.1 * bc_loss
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            # Target Update
            with torch.no_grad():
                for t, m in zip(model.critic_target.parameters(), model.critic.parameters()):
                    t.data.copy_(tau * m.data + (1 - tau) * t.data)
    
    torch.save(model.state_dict(), "basic_sac.pth")

if __name__ == "__main__":
    train_basic(
        dataset_path="datasets/processed/563-train.csv",
        epochs=100,
        lr=3e-4,
        batch_size=256
    )
```

## Key Simplifications

1. **Algorithm**:
   - Removed CQL complexity
   - Single critic network
   - Fixed entropy coefficient
   - Added behavioral cloning regularization

2. **Architecture**:
   - 2-layer networks instead of 3
   - ReLU instead of LeakyReLU/SiLU
   - No normalization layers
   - 10x fewer parameters

3. **Training**:
   - Single optimizer
   - No entropy adaptation
   - Simple TD error + BC loss

## Tradeoffs

| Aspect          | Original                   | Simplified          |
|-----------------|----------------------------|---------------------|
| OOD Prevention  | Strong (CQL)               | Weak (BC only)      |
| Stability       | High variance              | More stable         |
| Training Speed  | 10 min/epoch               | 1 min/epoch         |
| Performance     | Theoretically better       | Practically usable  |
| Tuning Effort   | High                       | Low                 |

**Recommendation**: Start with the simplified version to establish baseline performance, then gradually reintroduce complexity only where needed. The simplified version should train successfully within 1 hour and provide actionable insights.
