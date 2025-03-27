"""
Testing module for SAC-CQL diabetes management models
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import json
from tqdm import tqdm
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import the model architecture from training module
from SACCQL_training import SACAgent, DiabetesDataset

class DiabetesTestDataset(DiabetesDataset):
    """Extended dataset class for testing with additional metrics"""
    
    def __init__(self, csv_file):
        super().__init__(csv_file)
        # Store raw glucose values for evaluation
        df = pd.read_csv(csv_file)
        self.glucose_raw = df["glu_raw"].values.astype(np.float32)
        
        # Calculate time in range metrics
        self.time_in_range = self._calculate_time_in_range(self.glucose_raw)
        
    def _calculate_time_in_range(self, glucose_values):
        """Calculate time in range metrics
        
        Returns:
            Dictionary with percentage of time in different glucose ranges
        """
        total_readings = len(glucose_values)
        
        # Count readings in different ranges
        severe_hypo = np.sum(glucose_values < 54)
        hypo = np.sum((glucose_values >= 54) & (glucose_values < 70))
        normal = np.sum((glucose_values >= 70) & (glucose_values <= 180))
        hyper = np.sum((glucose_values > 180) & (glucose_values <= 250))
        severe_hyper = np.sum(glucose_values > 250)
        
        # Calculate percentages and convert numpy floats to Python floats
        return {
            "severe_hypo_percent": float(100 * severe_hypo / total_readings),
            "hypo_percent": float(100 * hypo / total_readings),
            "normal_percent": float(100 * normal / total_readings),
            "hyper_percent": float(100 * hyper / total_readings),
            "severe_hyper_percent": float(100 * severe_hyper / total_readings),
            "time_in_range": float(100 * normal / total_readings)
        }

def load_model(model_path):
    """Load a trained model from checkpoint
    
    Args:
        model_path: Path to the saved model checkpoint
        
    Returns:
        Loaded model instance
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model metadata
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # New format with metadata
        state_dict = checkpoint['model_state_dict']
        model_type = checkpoint.get('model_type', 'SAC')
        state_dim = checkpoint.get('state_dim', 8)
        action_dim = checkpoint.get('action_dim', 2)
        print(f"Loading {model_type} model (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        # Old format (just state dict)
        state_dict = checkpoint
        state_dim = 8
        action_dim = 2
        print("Loading model (legacy format)")
    
    # Create model instance
    model = SACAgent(state_dim=state_dim, action_dim=action_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()  # Set to evaluation mode
    
    return model

def evaluate_model(model, test_dataset, output_dir="logs/evaluation"):
    """Evaluate model performance on test dataset
    
    Args:
        model: Trained model instance
        test_dataset: Test dataset instance
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary with evaluation metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Helper function to convert numpy types to native Python types for JSON
    def convert_numpy(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj
    
    # Initialize metrics
    metrics = {
        "rmse": 0.0,
        "mae": 0.0,
        "mean_reward": 0.0,
        "action_mean": 0.0,
        "action_std": 0.0,
    }
    
    # Prepare for prediction collection
    all_states = []
    all_actions_true = []
    all_actions_pred = []
    all_glucose = []
    all_rewards = []
    
    # Evaluate in batches
    batch_size = 128
    n_samples = len(test_dataset)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(n_batches), desc="Evaluating"):
            # Get batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            batch_states = []
            batch_actions_true = []
            batch_rewards = []
            
            for j in range(start_idx, end_idx):
                sample = test_dataset[j]
                batch_states.append(sample['state'])
                batch_actions_true.append(sample['action'])
                batch_rewards.append(sample['reward'])
            
            # Convert to tensors
            states = torch.stack(batch_states).to(device)
            actions_true = torch.stack(batch_actions_true).to(device)
            rewards = torch.stack(batch_rewards).to(device)
            
            # Normalize states for model input
            states_norm = (states - states.mean(0)) / (states.std(0) + 1e-8)
            
            # Get model predictions
            actions_pred = model.act(states_norm, deterministic=True)
            
            # Store for later analysis
            all_states.extend(states.cpu().numpy())
            all_actions_true.extend(actions_true.cpu().numpy())
            all_actions_pred.extend(actions_pred.cpu().numpy())
            all_rewards.extend(rewards.cpu().numpy())
            all_glucose.extend(test_dataset.glucose_raw[start_idx:end_idx])
    
    # Convert to numpy arrays
    all_states = np.array(all_states)
    all_actions_true = np.array(all_actions_true)
    all_actions_pred = np.array(all_actions_pred)
    all_rewards = np.array(all_rewards)
    all_glucose = np.array(all_glucose)
    
    # Calculate metrics
    metrics["rmse"] = np.sqrt(mean_squared_error(all_actions_true, all_actions_pred))
    metrics["mae"] = mean_absolute_error(all_actions_true, all_actions_pred)
    metrics["mean_reward"] = np.mean(all_rewards)
    metrics["action_mean"] = np.mean(all_actions_pred)
    metrics["action_std"] = np.std(all_actions_pred)
    
    # Add time in range metrics
    metrics.update(test_dataset.time_in_range)
    
    # Save metrics to JSON with numpy type conversion
    with open(os.path.join(output_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(convert_numpy(metrics), f, indent=4)
    
    # Generate visualizations
    generate_evaluation_plots(
        all_states, all_actions_true, all_actions_pred, 
        all_glucose, all_rewards, output_dir
    )
    
    return metrics

def generate_evaluation_plots(states, actions_true, actions_pred, glucose, rewards, output_dir):
    """Generate evaluation visualizations
    
    Args:
        states: Array of state vectors
        actions_true: Array of true actions
        actions_pred: Array of predicted actions
        glucose: Array of glucose values
        rewards: Array of rewards
        output_dir: Directory to save plots
    """
    # Set style
    sns.set(style="whitegrid")
    
    # 1. Action comparison plot
    plt.figure(figsize=(12, 6))
    
    # Basal comparison
    plt.subplot(1, 2, 1)
    plt.scatter(actions_true[:, 0], actions_pred[:, 0], alpha=0.3)
    plt.plot([0, 1], [0, 1], 'r--')  # Diagonal line
    plt.xlabel('True Basal')
    plt.ylabel('Predicted Basal')
    plt.title('Basal Rate Comparison')
    
    # Bolus comparison
    plt.subplot(1, 2, 2)
    plt.scatter(actions_true[:, 1], actions_pred[:, 1], alpha=0.3)
    plt.plot([0, 1], [0, 1], 'r--')  # Diagonal line
    plt.xlabel('True Bolus')
    plt.ylabel('Predicted Bolus')
    plt.title('Bolus Comparison')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "action_comparison.png"))
    plt.close()
    
    # 2. Glucose distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(glucose, bins=50, kde=True)
    
    # Add vertical lines for range boundaries
    plt.axvline(x=54, color='r', linestyle='--', label='Severe Hypo (<54)')
    plt.axvline(x=70, color='orange', linestyle='--', label='Hypo (54-70)')
    plt.axvline(x=180, color='orange', linestyle='--', label='Hyper (180-250)')
    plt.axvline(x=250, color='r', linestyle='--', label='Severe Hyper (>250)')
    
    plt.xlabel('Glucose (mg/dL)')
    plt.ylabel('Frequency')
    plt.title('Glucose Distribution')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "glucose_distribution.png"))
    plt.close()
    
    # 3. Action distribution
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(actions_pred[:, 0], bins=30, kde=True, color='blue', label='Predicted')
    sns.histplot(actions_true[:, 0], bins=30, kde=True, color='green', alpha=0.6, label='True')
    plt.xlabel('Basal Rate')
    plt.ylabel('Frequency')
    plt.title('Basal Rate Distribution')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    sns.histplot(actions_pred[:, 1], bins=30, kde=True, color='blue', label='Predicted')
    sns.histplot(actions_true[:, 1], bins=30, kde=True, color='green', alpha=0.6, label='True')
    plt.xlabel('Bolus')
    plt.ylabel('Frequency')
    plt.title('Bolus Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "action_distribution.png"))
    plt.close()
    
    # 4. Reward distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(rewards, bins=30, kde=True)
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    plt.savefig(os.path.join(output_dir, "reward_distribution.png"))
    plt.close()
    
    # 5. State-action heatmap for key features
    plt.figure(figsize=(15, 10))
    
    # Glucose vs Basal
    plt.subplot(2, 3, 1)
    glucose_values = states[:, 0]  # First state dimension is glucose
    plt.hexbin(glucose_values, actions_pred[:, 0], gridsize=30, cmap='viridis')
    plt.xlabel('Glucose')
    plt.ylabel('Predicted Basal')
    plt.title('Glucose vs Basal')
    plt.colorbar(label='Count')
    
    # Glucose vs Bolus
    plt.subplot(2, 3, 2)
    plt.hexbin(glucose_values, actions_pred[:, 1], gridsize=30, cmap='viridis')
    plt.xlabel('Glucose')
    plt.ylabel('Predicted Bolus')
    plt.title('Glucose vs Bolus')
    plt.colorbar(label='Count')
    
    # IOB vs Bolus
    plt.subplot(2, 3, 3)
    iob_values = states[:, 6]  # 7th state dimension is IOB
    plt.hexbin(iob_values, actions_pred[:, 1], gridsize=30, cmap='viridis')
    plt.xlabel('IOB')
    plt.ylabel('Predicted Bolus')
    plt.title('IOB vs Bolus')
    plt.colorbar(label='Count')
    
    # Hour vs Basal
    plt.subplot(2, 3, 4)
    hour_values = states[:, 7]  # 8th state dimension is hour
    plt.hexbin(hour_values, actions_pred[:, 0], gridsize=30, cmap='viridis')
    plt.xlabel('Hour')
    plt.ylabel('Predicted Basal')
    plt.title('Hour vs Basal')
    plt.colorbar(label='Count')
    
    # Hour vs Bolus
    plt.subplot(2, 3, 5)
    plt.hexbin(hour_values, actions_pred[:, 1], gridsize=30, cmap='viridis')
    plt.xlabel('Hour')
    plt.ylabel('Predicted Bolus')
    plt.title('Hour vs Bolus')
    plt.colorbar(label='Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "state_action_heatmap.png"))
    plt.close()

def main():
    """Main function for model evaluation"""
    parser = argparse.ArgumentParser(description='Evaluate SAC-CQL agent for diabetes management')
    parser.add_argument('--model', type=str, default="models/sac_model.pth", 
                        help='Path to the trained model')
    parser.add_argument('--test_data', type=str, default="datasets/processed/563-train.csv", 
                        help='Path to the test dataset')
    parser.add_argument('--output_dir', type=str, default="logs/evaluation", 
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model)
    
    # Load test dataset
    test_dataset = DiabetesTestDataset(args.test_data)
    
    # Evaluate model
    metrics = evaluate_model(model, test_dataset, args.output_dir)
    
    # Print summary
    print("\nEvaluation Results:")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"Time in Range: {metrics['time_in_range']:.2f}%")
    print(f"Mean Reward: {metrics['mean_reward']:.4f}")
    print(f"Full results saved to {args.output_dir}/evaluation_metrics.json")

if __name__ == "__main__":
    main()
