"""
Testing module for SAC-CQL diabetes management models

This module provides comprehensive evaluation tools for diabetes management models,
including metrics calculation, visualization generation, and result analysis.
See testing_doc.md for detailed documentation on interpreting results.
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
import time
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import the model architecture from training module
from SACCQL_training import SACAgent, DiabetesDataset

class DiabetesTestDataset(DiabetesDataset):
    """Extended dataset class for testing with additional metrics"""
    
    def __init__(self, csv_file):
        super().__init__(csv_file)
        # Store raw glucose values for evaluation
        df = pd.read_csv(csv_file)
        self.glucose_raw = df["glu_raw"].values.astype(np.float32)
        
        # Store additional data for advanced analysis
        self.timestamps = df["timestamp"].values if "timestamp" in df.columns else np.arange(len(self.glucose_raw))
        self.meal_data = df["meal"].values.astype(np.float32) if "meal" in df.columns else None
        self.patient_id = os.path.basename(csv_file).split('-')[0] if '-' in os.path.basename(csv_file) else "unknown"
        
        # Calculate time in range metrics
        self.time_in_range = self._calculate_time_in_range(self.glucose_raw)
        
        # Calculate glycemic variability metrics
        self.glycemic_metrics = self._calculate_glycemic_variability(self.glucose_raw)
        
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
        
    def _calculate_glycemic_variability(self, glucose_values):
        """Calculate glycemic variability metrics
        
        Returns:
            Dictionary with glycemic variability metrics
        """
        # Calculate coefficient of variation (CV)
        cv = float(100 * np.std(glucose_values) / np.mean(glucose_values))
        
        # Calculate mean amplitude of glycemic excursions (MAGE)
        # Simplified version - standard deviation works as a proxy
        mage = float(np.std(glucose_values))
        
        # Calculate glucose management indicator (GMI)
        # GMI = 3.31 + 0.02392 × mean glucose (mg/dL)
        gmi = float(3.31 + 0.02392 * np.mean(glucose_values))
        
        return {
            "cv_percent": cv,
            "mage": mage,
            "gmi": gmi,
            "mean_glucose": float(np.mean(glucose_values)),
            "median_glucose": float(np.median(glucose_values)),
            "min_glucose": float(np.min(glucose_values)),
            "max_glucose": float(np.max(glucose_values)),
            "std_glucose": float(np.std(glucose_values))
        }

def load_model(model_path):
    """Load a trained model from checkpoint
    
    Args:
        model_path: Path to the saved model checkpoint
        
    Returns:
        Loaded model instance and model metadata
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract model metadata
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # New format with metadata
            state_dict = checkpoint['model_state_dict']
            model_type = checkpoint.get('model_type', 'SAC')
            state_dim = checkpoint.get('state_dim', 8)
            action_dim = checkpoint.get('action_dim', 2)
            training_dataset = checkpoint.get('training_dataset', 'unknown')
            epoch = checkpoint.get('epoch', 'unknown')
            
            print(f"Loading {model_type} model (epoch {epoch})")
            print(f"Trained on: {training_dataset}")
            print(f"State dim: {state_dim}, Action dim: {action_dim}")
            
            metadata = {
                'model_type': model_type,
                'state_dim': state_dim,
                'action_dim': action_dim,
                'epoch': epoch,
                'training_dataset': training_dataset
            }
        else:
            # Old format (just state dict)
            state_dict = checkpoint
            state_dim = 8
            action_dim = 2
            print("Loading model (legacy format)")
            metadata = {
                'model_type': 'SAC',
                'state_dim': state_dim,
                'action_dim': action_dim,
                'epoch': 'unknown',
                'training_dataset': 'unknown'
            }
        
        # Create model instance
        model = SACAgent(state_dim=state_dim, action_dim=action_dim).to(device)
        model.load_state_dict(state_dict)
        model.eval()  # Set to evaluation mode
        
        return model, metadata
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def evaluate_model(model, test_dataset, model_metadata=None, output_dir="logs/evaluation", noise_test=False):
    """Evaluate model performance on test dataset
    
    Args:
        model: Trained model instance
        test_dataset: Test dataset instance
        model_metadata: Dictionary with model metadata
        output_dir: Directory to save evaluation results
        noise_test: Whether to test model robustness to noise
        
    Returns:
        Dictionary with evaluation metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create timestamped subdirectory for this evaluation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = os.path.join(output_dir, f"eval_{timestamp}")
    Path(eval_dir).mkdir(parents=True, exist_ok=True)
    
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
        "r2_score": 0.0,
        "mean_reward": 0.0,
        "action_mean": 0.0,
        "action_std": 0.0,
        "basal_rmse": 0.0,
        "bolus_rmse": 0.0,
        "evaluation_timestamp": timestamp,
        "patient_id": test_dataset.patient_id,
    }
    
    # Add model metadata if available
    if model_metadata:
        metrics["model_metadata"] = model_metadata
    
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
    
    # Measure inference time
    start_time = time.time()
    
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
    
    # Calculate inference time
    inference_time = time.time() - start_time
    metrics["inference_time_seconds"] = float(inference_time)
    metrics["inference_time_per_sample"] = float(inference_time / n_samples)
    
    # Convert to numpy arrays
    all_states = np.array(all_states)
    all_actions_true = np.array(all_actions_true)
    all_actions_pred = np.array(all_actions_pred)
    all_rewards = np.array(all_rewards)
    all_glucose = np.array(all_glucose)
    
    # Save raw predictions for further analysis
    np.savez(
        os.path.join(eval_dir, "raw_predictions.npz"),
        states=all_states,
        actions_true=all_actions_true,
        actions_pred=all_actions_pred,
        glucose=all_glucose,
        rewards=all_rewards
    )
    
    # Calculate metrics
    metrics["rmse"] = float(np.sqrt(mean_squared_error(all_actions_true, all_actions_pred)))
    metrics["mae"] = float(mean_absolute_error(all_actions_true, all_actions_pred))
    metrics["r2_score"] = float(r2_score(all_actions_true.flatten(), all_actions_pred.flatten()))
    metrics["mean_reward"] = float(np.mean(all_rewards))
    metrics["action_mean"] = float(np.mean(all_actions_pred))
    metrics["action_std"] = float(np.std(all_actions_pred))
    
    # Calculate per-component metrics
    metrics["basal_rmse"] = float(np.sqrt(mean_squared_error(all_actions_true[:, 0], all_actions_pred[:, 0])))
    metrics["bolus_rmse"] = float(np.sqrt(mean_squared_error(all_actions_true[:, 1], all_actions_pred[:, 1])))
    metrics["basal_mae"] = float(mean_absolute_error(all_actions_true[:, 0], all_actions_pred[:, 0]))
    metrics["bolus_mae"] = float(mean_absolute_error(all_actions_true[:, 1], all_actions_pred[:, 1]))
    
    # Add time in range metrics
    metrics.update(test_dataset.time_in_range)
    
    # Add glycemic variability metrics
    metrics.update(test_dataset.glycemic_metrics)
    
    # Test noise robustness if requested
    if noise_test:
        noise_metrics = test_noise_robustness(model, all_states, all_actions_true, device)
        metrics.update(noise_metrics)
    
    # Save metrics to JSON with numpy type conversion
    with open(os.path.join(eval_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(convert_numpy(metrics), f, indent=4)
    
    # Also save to the parent directory for easy access
    with open(os.path.join(output_dir, "latest_metrics.json"), "w") as f:
        json.dump(convert_numpy(metrics), f, indent=4)
    
    # Generate visualizations
    generate_evaluation_plots(
        all_states, all_actions_true, all_actions_pred, 
        all_glucose, all_rewards, eval_dir
    )
    
    # Generate clinical report
    generate_clinical_report(metrics, eval_dir)
    
    return metrics

def test_noise_robustness(model, states, actions_true, device, noise_levels=[0.01, 0.05, 0.1, 0.2]):
    """Test model robustness to input noise
    
    Args:
        model: Trained model instance
        states: Array of state vectors
        actions_true: Array of true actions
        device: Computation device
        noise_levels: List of noise standard deviations to test
        
    Returns:
        Dictionary with noise robustness metrics
    """
    print("Testing noise robustness...")
    noise_metrics = {}
    
    with torch.no_grad():
        for noise_level in noise_levels:
            rmse_values = []
            
            # Run 5 trials for each noise level
            for trial in range(5):
                # Add Gaussian noise to states
                noisy_states = states + np.random.normal(0, noise_level, states.shape)
                
                # Convert to tensor and normalize
                states_tensor = torch.FloatTensor(noisy_states).to(device)
                states_norm = (states_tensor - states_tensor.mean(0)) / (states_tensor.std(0) + 1e-8)
                
                # Get model predictions
                actions_pred = model.act(states_norm, deterministic=True).cpu().numpy()
                
                # Calculate RMSE
                rmse = np.sqrt(mean_squared_error(actions_true, actions_pred))
                rmse_values.append(rmse)
            
            # Average RMSE across trials
            noise_metrics[f"noise_robustness_{noise_level}"] = float(np.mean(rmse_values))
    
    return noise_metrics

def generate_clinical_report(metrics, output_dir):
    """Generate a clinical summary report from evaluation metrics
    
    Args:
        metrics: Dictionary with evaluation metrics
        output_dir: Directory to save the report
    """
    report_path = os.path.join(output_dir, "clinical_report.md")
    
    # Clinical risk assessment based on metrics
    risk_level = "LOW"
    if metrics["severe_hypo_percent"] > 1 or metrics["time_in_range"] < 60:
        risk_level = "MODERATE"
    if metrics["severe_hypo_percent"] > 3 or metrics["time_in_range"] < 50:
        risk_level = "HIGH"
    
    # Format metrics for report
    tir = f"{metrics['time_in_range']:.1f}%"
    hypo = f"{metrics['hypo_percent'] + metrics['severe_hypo_percent']:.1f}%"
    hyper = f"{metrics['hyper_percent'] + metrics['severe_hyper_percent']:.1f}%"
    severe_hypo = f"{metrics['severe_hypo_percent']:.1f}%"
    mean_glucose = f"{metrics['mean_glucose']:.1f} mg/dL"
    gmi = f"{metrics['gmi']:.1f}%"
    cv = f"{metrics['cv_percent']:.1f}%"
    
    # Create report content
    report = f"""# Clinical Evaluation Report

## Summary
- **Patient ID**: {metrics['patient_id']}
- **Evaluation Date**: {datetime.now().strftime("%Y-%m-%d %H:%M")}
- **Clinical Risk Assessment**: {risk_level}

## Glycemic Outcomes
- **Time in Range (70-180 mg/dL)**: {tir}
- **Time in Hypoglycemia (<70 mg/dL)**: {hypo}
- **Time in Hyperglycemia (>180 mg/dL)**: {hyper}
- **Time in Severe Hypoglycemia (<54 mg/dL)**: {severe_hypo}
- **Mean Glucose**: {mean_glucose}
- **Glucose Management Indicator**: {gmi}
- **Coefficient of Variation**: {cv}

## Model Performance
- **Basal Insulin RMSE**: {metrics['basal_rmse']:.4f}
- **Bolus Insulin RMSE**: {metrics['bolus_rmse']:.4f}
- **Overall R² Score**: {metrics['r2_score']:.4f}

## Clinical Recommendations
"""
    
    # Add recommendations based on metrics
    if metrics['time_in_range'] < 70:
        report += "- Consider adjusting insulin sensitivity factors to improve time in range\n"
    
    if metrics['severe_hypo_percent'] > 1:
        report += "- Review basal rates to reduce severe hypoglycemia risk\n"
    
    if metrics['hyper_percent'] > 30:
        report += "- Evaluate insulin-to-carb ratios to address hyperglycemia\n"
    
    if metrics['cv_percent'] > 36:
        report += "- High glycemic variability detected, consider more frequent monitoring\n"
    
    # Write report to file
    with open(report_path, "w") as f:
        f.write(report)
    
    print(f"Clinical report saved to {report_path}")

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
    
    # Add regression line
    z = np.polyfit(actions_true[:, 0], actions_pred[:, 0], 1)
    p = np.poly1d(z)
    plt.plot(np.linspace(0, 1, 100), p(np.linspace(0, 1, 100)), 'b-', alpha=0.7)
    
    # Add correlation coefficient
    corr = np.corrcoef(actions_true[:, 0], actions_pred[:, 0])[0, 1]
    plt.text(0.05, 0.95, f'r = {corr:.3f}', transform=plt.gca().transAxes)
    
    plt.xlabel('True Basal')
    plt.ylabel('Predicted Basal')
    plt.title('Basal Rate Comparison')
    
    # Bolus comparison
    plt.subplot(1, 2, 2)
    plt.scatter(actions_true[:, 1], actions_pred[:, 1], alpha=0.3)
    plt.plot([0, 1], [0, 1], 'r--')  # Diagonal line
    
    # Add regression line
    z = np.polyfit(actions_true[:, 1], actions_pred[:, 1], 1)
    p = np.poly1d(z)
    plt.plot(np.linspace(0, 1, 100), p(np.linspace(0, 1, 100)), 'b-', alpha=0.7)
    
    # Add correlation coefficient
    corr = np.corrcoef(actions_true[:, 1], actions_pred[:, 1])[0, 1]
    plt.text(0.05, 0.95, f'r = {corr:.3f}', transform=plt.gca().transAxes)
    
    plt.xlabel('True Bolus')
    plt.ylabel('Predicted Bolus')
    plt.title('Bolus Comparison')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "action_comparison.png"), dpi=300)
    plt.close()
    
    # 2. Glucose distribution with clinical ranges
    plt.figure(figsize=(10, 6))
    
    # Create histogram with KDE
    ax = sns.histplot(glucose, bins=50, kde=True)
    
    # Add colored regions for different glucose ranges
    xmin, xmax = ax.get_xlim()
    
    # Fill regions with colors
    ax.axvspan(xmin, 54, alpha=0.2, color='red', label='Severe Hypoglycemia')
    ax.axvspan(54, 70, alpha=0.2, color='orange', label='Hypoglycemia')
    ax.axvspan(70, 180, alpha=0.2, color='green', label='Target Range')
    ax.axvspan(180, 250, alpha=0.2, color='orange', label='Hyperglycemia')
    ax.axvspan(250, xmax, alpha=0.2, color='red', label='Severe Hyperglycemia')
    
    # Add vertical lines for range boundaries
    plt.axvline(x=54, color='r', linestyle='--')
    plt.axvline(x=70, color='orange', linestyle='--')
    plt.axvline(x=180, color='orange', linestyle='--')
    plt.axvline(x=250, color='r', linestyle='--')
    
    # Add mean and median lines
    plt.axvline(x=np.mean(glucose), color='black', linestyle='-', label=f'Mean: {np.mean(glucose):.1f} mg/dL')
    plt.axvline(x=np.median(glucose), color='black', linestyle=':', label=f'Median: {np.median(glucose):.1f} mg/dL')
    
    plt.xlabel('Glucose (mg/dL)')
    plt.ylabel('Frequency')
    plt.title('Glucose Distribution with Clinical Ranges')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(output_dir, "glucose_distribution.png"), dpi=300)
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
    plt.savefig(os.path.join(output_dir, "action_distribution.png"), dpi=300)
    plt.close()
    
    # 4. Reward distribution with clinical interpretation
    plt.figure(figsize=(10, 6))
    
    # Create histogram with KDE
    sns.histplot(rewards, bins=30, kde=True)
    
    # Add mean line
    plt.axvline(x=np.mean(rewards), color='red', linestyle='--', 
                label=f'Mean Reward: {np.mean(rewards):.3f}')
    
    # Add percentile lines
    p10 = np.percentile(rewards, 10)
    p90 = np.percentile(rewards, 90)
    plt.axvline(x=p10, color='orange', linestyle=':', 
                label=f'10th Percentile: {p10:.3f}')
    plt.axvline(x=p90, color='green', linestyle=':', 
                label=f'90th Percentile: {p90:.3f}')
    
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "reward_distribution.png"), dpi=300)
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
    plt.savefig(os.path.join(output_dir, "state_action_heatmap.png"), dpi=300)
    plt.close()
    
    # 6. Error analysis plot (new)
    plt.figure(figsize=(12, 10))
    
    # Basal error vs glucose
    plt.subplot(2, 2, 1)
    basal_errors = actions_pred[:, 0] - actions_true[:, 0]
    plt.scatter(glucose_values, basal_errors, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Glucose (mg/dL)')
    plt.ylabel('Basal Error (Predicted - True)')
    plt.title('Basal Error vs Glucose')
    
    # Bolus error vs glucose
    plt.subplot(2, 2, 2)
    bolus_errors = actions_pred[:, 1] - actions_true[:, 1]
    plt.scatter(glucose_values, bolus_errors, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Glucose (mg/dL)')
    plt.ylabel('Bolus Error (Predicted - True)')
    plt.title('Bolus Error vs Glucose')
    
    # Basal error vs hour
    plt.subplot(2, 2, 3)
    plt.scatter(hour_values, basal_errors, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Hour of Day')
    plt.ylabel('Basal Error (Predicted - True)')
    plt.title('Basal Error vs Time of Day')
    
    # Bolus error vs IOB
    plt.subplot(2, 2, 4)
    plt.scatter(iob_values, bolus_errors, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Insulin on Board (IOB)')
    plt.ylabel('Bolus Error (Predicted - True)')
    plt.title('Bolus Error vs IOB')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_analysis.png"), dpi=300)
    plt.close()

def main():
    """Main function for model evaluation"""
    parser = argparse.ArgumentParser(description='Evaluate SAC-CQL agent for diabetes management')
    parser.add_argument('--model', type=str, default="models/sac_model.pth", 
                        help='Path to the trained model')
    parser.add_argument('--test_data', type=str, default="datasets/processed/563-test.csv", 
                        help='Path to the test dataset')
    parser.add_argument('--output_dir', type=str, default="logs/evaluation", 
                        help='Directory to save evaluation results')
    parser.add_argument('--noise_test', action='store_true',
                        help='Test model robustness to input noise')
    parser.add_argument('--compare', type=str, default=None,
                        help='Path to another model for comparison')
    parser.add_argument('--doc', action='store_true',
                        help='Show documentation on interpreting results')
    
    args = parser.parse_args()
    
    # Show documentation if requested
    if args.doc:
        try:
            with open("testing_doc.md", "r") as f:
                print(f.read())
            return
        except FileNotFoundError:
            print("Documentation file not found. Continuing with evaluation.")
    
    # Load model
    model, model_metadata = load_model(args.model)
    
    # Load test dataset
    test_dataset = DiabetesTestDataset(args.test_data)
    
    # Evaluate model
    metrics = evaluate_model(
        model, 
        test_dataset, 
        model_metadata=model_metadata,
        output_dir=args.output_dir,
        noise_test=args.noise_test
    )
    
    # Print summary with color formatting
    print("\n" + "="*50)
    print("EVALUATION RESULTS SUMMARY")
    print("="*50)
    
    # Clinical metrics
    print("\nCLINICAL METRICS:")
    print(f"Time in Range (70-180 mg/dL): {metrics['time_in_range']:.2f}%")
    print(f"Hypoglycemia (<70 mg/dL): {metrics['hypo_percent'] + metrics['severe_hypo_percent']:.2f}%")
    print(f"Severe Hypoglycemia (<54 mg/dL): {metrics['severe_hypo_percent']:.2f}%")
    print(f"Mean Glucose: {metrics['mean_glucose']:.1f} mg/dL")
    print(f"Glycemic Variability (CV): {metrics['cv_percent']:.1f}%")
    
    # Technical metrics
    print("\nMODEL PERFORMANCE:")
    print(f"Overall RMSE: {metrics['rmse']:.4f}")
    print(f"Basal RMSE: {metrics['basal_rmse']:.4f}")
    print(f"Bolus RMSE: {metrics['bolus_rmse']:.4f}")
    print(f"R² Score: {metrics['r2_score']:.4f}")
    print(f"Mean Reward: {metrics['mean_reward']:.4f}")
    
    # Inference metrics
    print(f"\nInference Time: {metrics['inference_time_seconds']:.2f} seconds")
    print(f"Inference Time per Sample: {metrics['inference_time_per_sample']*1000:.2f} ms")
    
    # Compare with another model if requested
    if args.compare:
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        
        try:
            compare_model, compare_metadata = load_model(args.compare)
            compare_metrics = evaluate_model(
                compare_model, 
                test_dataset,
                model_metadata=compare_metadata,
                output_dir=os.path.join(args.output_dir, "comparison"),
                noise_test=args.noise_test
            )
            
            # Print comparison
            print("\nMETRIC          | CURRENT MODEL | COMPARISON MODEL | DIFFERENCE")
            print("-"*60)
            
            for key in ['rmse', 'time_in_range', 'mean_reward', 'r2_score']:
                if key in metrics and key in compare_metrics:
                    diff = metrics[key] - compare_metrics[key]
                    diff_str = f"{diff:+.4f}" if key != 'time_in_range' else f"{diff:+.2f}%"
                    
                    if key == 'time_in_range':
                        print(f"Time in Range   | {metrics[key]:12.2f}% | {compare_metrics[key]:16.2f}% | {diff_str}")
                    else:
                        print(f"{key.upper():14} | {metrics[key]:12.4f} | {compare_metrics[key]:16.4f} | {diff_str}")
            
        except Exception as e:
            print(f"Error comparing models: {e}")
    
    print("\nFull results saved to:")
    print(f"  {args.output_dir}/latest_metrics.json")
    print(f"  {args.output_dir}/eval_{metrics['evaluation_timestamp']}/")

if __name__ == "__main__":
    main()
