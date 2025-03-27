# SAC-CQL for Diabetes Management

## Overview

This repository contains an implementation of Soft Actor-Critic (SAC) with Conservative Q-Learning (CQL) for automated insulin delivery in diabetes management. The system learns insulin dosing strategies from historical data to maintain blood glucose levels within a healthy range.

## Model Architecture

The implementation uses a Soft Actor-Critic (SAC) architecture with the following components:

- **Actor Network**: Produces insulin dosing actions (basal rates and boluses)
- **Twin Q-Networks**: Estimate action values with reduced overestimation bias
- **Entropy Regularization**: Encourages exploration during training

### State Space (8 dimensions)
- Current glucose level (mg/dL)
- Glucose rate of change (mg/dL/min)
- Glucose acceleration (mg/dL/min²)
- Heart rate (bpm)
- Heart rate derivative (bpm/min)
- Heart rate acceleration (bpm/min²)
- Insulin on board (IOB) (units)
- Hour of day (0-23)

### Action Space (2 dimensions)
- Basal insulin rate (U/hr)
- Bolus insulin dose (U)

### Reward Function
The reward function is based on the blood glucose risk index, which penalizes both hyperglycemia and hypoglycemia, with stronger penalties for dangerous hypoglycemic events.

## Training Process

The training process uses historical diabetes management data to learn optimal insulin dosing strategies:

1. **Data Preprocessing**: Handles missing values and computes derived features
2. **Batch Training**: Uses mini-batch gradient descent with experience replay
3. **Soft Target Updates**: Gradually updates target networks for stability
4. **Entropy Regularization**: Automatically tunes exploration-exploitation balance

### Hyperparameters

Key hyperparameters include:
- Learning rates: 1e-5 (actor), 3e-4 (critics)
- Batch size: 512
- Discount factor: 0.99
- Target network update rate: 0.05
- Weight decay: 1e-4

## Evaluation Metrics

The model is evaluated on several metrics:

- **Time in Range (TIR)**: Percentage of time glucose is between 70-180 mg/dL
- **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual insulin doses
- **Root Mean Square Error (RMSE)**: Root of the mean squared difference between predicted and actual insulin doses
- **Hypoglycemia Rate**: Percentage of time glucose is below 70 mg/dL
- **Severe Hypoglycemia Rate**: Percentage of time glucose is below 54 mg/dL

## Usage

### Training

```bash
python SACCQL_training.py --dataset datasets/processed/563-train.csv --epochs 500 --batch_size 512 --save_path models/sac_model.pth --log_dir logs/sac
```

### Testing

```bash
python SACCQL_testing.py --model models/sac_model.pth --test_data datasets/processed/563-test.csv --output_dir logs/evaluation
```

## Results Analysis

The training and evaluation processes generate various visualizations:

- Training metrics over time
- Action distribution comparisons
- Glucose distribution analysis
- State-action heatmaps
- Reward distribution

These visualizations help understand model behavior and identify potential improvements.

## Implementation Details

The implementation includes several techniques for improved stability and performance:

- Gradient clipping to prevent exploding gradients
- Layer normalization in the actor network
- Dynamic learning rate scheduling
- Twin Q-networks to reduce overestimation bias
- Proper entropy calculation and regularization
