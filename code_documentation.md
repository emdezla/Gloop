# Diabetes Management Reinforcement Learning System

This documentation explains the reinforcement learning (RL) system implemented for diabetes management, specifically focusing on insulin dosing optimization.

## Overview

The system implements a Conservative Q-Learning (CQL) algorithm with Soft Actor-Critic (SAC) components for offline reinforcement learning. This approach is designed to learn optimal insulin dosing policies from historical diabetes management data without requiring active interaction with patients.

## Data Structure

The system uses the OhioT1DM dataset, which contains continuous glucose monitoring (CGM) data and insulin delivery records from patients with Type 1 Diabetes.

### State Space (8 dimensions):
- `glu`: Current glucose level
- `glu_d`: Rate of change in glucose
- `glu_t`: Acceleration of glucose
- `hr`: Heart rate
- `hr_d`: Rate of change in heart rate
- `hr_t`: Acceleration of heart rate
- `iob`: Insulin on board (active insulin)
- `hour`: Time of day (circadian rhythm)

### Action Space (2 dimensions):
- `basal`: Continuous basal insulin rate
- `bolus`: Insulin bolus amount

### Reward Function:
The reward is based on a Risk Index (RI) derived from blood glucose levels:
- Penalizes both hyperglycemia and hypoglycemia
- Severe hypoglycemia (glucose ≤ 39 mg/dL) receives a large negative reward (-15)
- Optimal glucose levels receive rewards closer to 0

## Neural Network Architecture

### SACCQL (Soft Actor-Critic with Conservative Q-Learning)

#### Actor Network (Policy):
- Input: State vector (8 dimensions)
- Hidden layers: 2 fully connected layers with 256 neurons each
- Layer normalization and ReLU activations
- Output: Mean and log standard deviation of a Gaussian distribution
- Action sampling: Uses reparameterization trick with tanh squashing

#### Critic Networks (Twin Q-networks):
- Input: Concatenated state and action vectors (10 dimensions)
- Hidden layers: 2 fully connected layers with 256 neurons each
- Layer normalization and ReLU activations
- Output: Q-value (expected return)

## Algorithm: Conservative Q-Learning with SAC

The algorithm combines elements from:
1. **Soft Actor-Critic (SAC)**: Maximizes expected reward while also maximizing action entropy
2. **Conservative Q-Learning (CQL)**: Prevents overestimation of Q-values for out-of-distribution actions

### Training Process:

1. **Critic Update**:
   - Compute target Q-values using next states and policy actions
   - Calculate TD error between current and target Q-values
   - Add CQL penalty to prevent overestimation
   - Update critic networks to minimize the combined loss

2. **Actor Update**:
   - Sample actions from the current policy
   - Evaluate these actions using critic networks
   - Update actor to maximize Q-values and entropy

3. **Target Network Update**:
   - Soft update of target networks for stability

### CQL Penalty Calculation:
- Samples actions from both the dataset and current policy
- Computes Q-values for all action candidates
- Penalty = logsumexp(Q_all) - mean(Q_dataset)
- This penalizes overestimation of values for out-of-distribution actions

## Hyperparameters

- Learning rate: 3e-4
- Discount factor (gamma): 0.99
- Entropy coefficient (alpha): 0.2
- CQL weight: 0.25
- Batch size: 256
- Target network update rate (tau): 0.005

## Monitoring and Evaluation

The training process is monitored using:
- TensorBoard for real-time visualization
- CSV logging of key metrics
- Progress tracking with tqdm

Key metrics tracked:
- TD Loss
- CQL Penalty
- Critic Loss
- Actor Loss
- Q-values
- Action statistics
- Policy entropy

## Implementation Details

The implementation uses PyTorch and includes:
- Custom dataset class for diabetes data
- Tensor-based reward computation
- Debugging utilities for monitoring training stability
- Soft target network updates for stable learning
