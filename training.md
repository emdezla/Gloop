# Reinforcement Learning for Diabetes Management

This document explains the architecture and training process of our Soft Actor-Critic (SAC) reinforcement learning agent for diabetes management.

## Architecture Overview

Our implementation uses a Soft Actor-Critic (SAC) algorithm, which is an off-policy actor-critic deep RL algorithm that maximizes both the expected return and entropy. This approach encourages exploration and prevents premature convergence to suboptimal policies.

### Key Components

1. **Actor Network (Policy)**: A Gaussian policy network that outputs a distribution over actions
2. **Critic Networks**: Twin Q-networks to reduce overestimation bias
3. **Temperature Parameter (Alpha)**: Automatically adjusted to maintain target entropy

## Network Architectures

### Gaussian Actor

The actor network maps states to a distribution over actions:

```python
class GaussianActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units=32):
        super().__init__()
        # Three dense layers with 32 units each
        self.fc1 = nn.Linear(state_dim, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, hidden_units)
        # Separate output heads for mean and log_std
        self.mean_head = nn.Linear(hidden_units, action_dim)
        self.log_std_head = nn.Linear(hidden_units, action_dim)
```

- **Input**: State vector (8 dimensions by default)
- **Hidden Layers**: 3 fully-connected layers with 32 units each and ReLU activation
- **Output**: Two heads producing the mean and log standard deviation of a Gaussian distribution
- **Action Sampling**: Uses the reparameterization trick and tanh squashing to bound actions to [-1, 1]

### Q-Networks (Critics)

Twin Q-networks that estimate the Q-value (expected return) for state-action pairs:

```python
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units=32):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, hidden_units)
        self.out = nn.Linear(hidden_units, 1)
```

- **Input**: Concatenated state and action vectors
- **Hidden Layers**: 3 fully-connected layers with 32 units each and ReLU activation
- **Output**: Single Q-value estimate

## State and Action Spaces

### State Space (8 dimensions)
- Glucose level
- Glucose derivative (rate of change)
- Glucose trend
- Heart rate
- Heart rate derivative
- Heart rate trend
- Insulin on Board (IOB)
- Hour of day

### Action Space (1 dimension)
- Insulin dose (normalized to [-1, 1])

## Training Process

The agent is trained using the `train_saccql` function, which implements offline reinforcement learning with an optional Conservative Q-Learning (CQL) penalty.

### Training Steps

1. **Data Loading**: Load transitions from a diabetes dataset
2. **Batch Processing**: Process mini-batches of (state, action, reward, next_state, done) tuples
3. **Critic Update**:
   - Compute target Q-values using the target networks
   - Update critics to minimize the TD error
   - Apply CQL penalty if enabled to prevent overestimation on out-of-distribution actions
4. **Actor Update**:
   - Update the policy to maximize the expected Q-value minus the entropy term
   - Apply L2 regularization to prevent overfitting
5. **Temperature Update**:
   - Adjust the temperature parameter to maintain the target entropy
6. **Target Network Update**:
   - Soft-update the target networks using polyak averaging

### Conservative Q-Learning (CQL)

When enabled, CQL adds a regularization term to the critic loss:

```python
if use_cql:
    # Generate random actions
    random_actions = torch.empty_like(actions).uniform_(-1.0, 1.0)

    # Get policy actions from the actor
    policy_output = agent.actor(states)
    policy_actions = policy_output[0] if isinstance(policy_output, tuple) else policy_output

    # Compute Q-values for policy and random actions
    q1_policy = agent.q1(states, policy_actions)
    q2_policy = agent.q2(states, policy_actions)
    q1_rand = agent.q1(states, random_actions)
    q2_rand = agent.q2(states, random_actions)
    q1_data = current_q1  
    q2_data = current_q2

    # Concatenate Q-values for log-sum-exp computation
    q1_cat = torch.cat([q1_rand, q1_policy, q1_data], dim=0)
    q2_cat = torch.cat([q2_rand, q2_policy, q2_data], dim=0)
    q1_lse = torch.logsumexp(q1_cat, dim=0)
    q2_lse = torch.logsumexp(q2_cat, dim=0)

    # Compute and add the CQL penalty to the critic loss
    cql_penalty = ((q1_lse.mean() - q1_data.mean()) + (q2_lse.mean() - q2_data.mean()))
    critic_loss += cql_alpha * cql_penalty
```

This penalty helps prevent the Q-function from overestimating values for out-of-distribution actions, which is crucial for offline RL where the agent cannot explore the environment.

## Hyperparameters

- **Discount Factor (gamma)**: 0.997
- **Target Network Update Rate (tau)**: 0.005
- **Learning Rates**: 3e-4 for actor, critic, and temperature
- **Batch Size**: 512
- **CQL Penalty Weight**: 0.1 (when enabled)
- **Target Entropy**: -1 (default)

## Model Evaluation

The agent is evaluated using several metrics:

1. **Action Distribution Comparison**: Comparing the distribution of predicted actions to actual actions
2. **Mean Squared Error (MSE)**: Between predicted and actual actions
3. **Pearson Correlation**: Measuring the linear correlation between predicted and actual actions
4. **R² Score**: Coefficient of determination
5. **Feature Sensitivity Analysis**: Measuring the influence of each state feature on the predicted actions

## Variants

Two main model variants are implemented:

1. **Full Model (HR)**: Uses all 8 state dimensions including heart rate features
2. **Reduced Model (no-HR)**: Uses only 5 state dimensions, excluding heart rate features

## Training Workflow

1. **Data Preparation**: Process raw diabetes data into state-action-reward transitions
2. **Offline Training**: Train the agent on collected data without environment interaction
3. **Model Checkpointing**: Save model weights every 50 epochs
4. **Evaluation**: Assess model performance on test data
5. **Comparison**: Compare different model variants (with/without heart rate)

## Visualization

The training process and model evaluation include various visualizations:

1. **Training Metrics**: Loss curves, temperature values, and action statistics
2. **Action Distributions**: Histograms comparing predicted vs. actual actions
3. **Scatter Plots**: Actual vs. predicted actions with regression lines
4. **Residual Analysis**: Histograms and plots of prediction errors
5. **Heatmaps**: Visualizing the policy's response to different state features
6. **Feature Sensitivity**: Bar charts showing the relative importance of each state feature
