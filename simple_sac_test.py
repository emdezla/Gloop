import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from simple_sac_train import SACAgent, action_to_insulin, calculate_reward, calculate_risk_index

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def evaluate_agent(agent, test_csv_path, output_dir="results"):
    """
    Evaluate a trained SAC agent on test data.
    
    Args:
        agent: Trained SAC agent
        test_csv_path: Path to test CSV file
        output_dir: Directory to save evaluation results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test data
    print(f"Loading test data from {test_csv_path}...")
    df = pd.read_csv(test_csv_path)
    df = df.ffill().bfill()  # Fill missing values
    
    # Define state columns
    state_columns = ['glu', 'glu_d', 'glu_t', 'hr', 'hr_d', 'hr_t', 'iob', 'hour']
    
    # Extract states
    states = df[state_columns].values
    
    # Initialize arrays to store results
    actions_pred = np.zeros((len(df), 2))  # Predicted actions (basal, bolus)
    insulin_rates = np.zeros((len(df), 2))  # Converted insulin rates
    rewards = np.zeros(len(df))
    
    # Evaluate agent on each state
    print("Evaluating agent...")
    for i in tqdm(range(len(df))):
        state = states[i]
        
        # Get agent's action
        action = agent.select_action(state, deterministic=True)
        actions_pred[i] = action
        
        # Convert action to insulin rate
        insulin = action_to_insulin(torch.FloatTensor(action)).numpy()
        insulin_rates[i] = insulin
        
        # Calculate reward if not the last state
        if i < len(df) - 1:
            next_glucose = df['glu_raw'].iloc[i + 1]
            rewards[i] = calculate_reward(next_glucose)
    
    # Create results dataframe
    results_df = df.copy()
    results_df['pred_basal'] = actions_pred[:, 0]
    results_df['pred_bolus'] = actions_pred[:, 1]
    results_df['insulin_basal'] = insulin_rates[:, 0]
    results_df['insulin_bolus'] = insulin_rates[:, 1]
    results_df['reward'] = rewards
    
    # Save results to CSV
    results_path = os.path.join(output_dir, f"evaluation_results_{os.path.basename(test_csv_path)}")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    # Calculate metrics
    total_reward = np.sum(rewards)
    mean_reward = np.mean(rewards)
    
    # Calculate time in range metrics
    in_range = ((df['glu_raw'] >= 70) & (df['glu_raw'] <= 180)).mean() * 100
    hypo = (df['glu_raw'] < 70).mean() * 100
    hyper = (df['glu_raw'] > 180).mean() * 100
    severe_hypo = (df['glu_raw'] < 54).mean() * 100
    
    # Print metrics
    print("\nEvaluation Metrics:")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Mean Reward: {mean_reward:.2f}")
    print(f"Time in Range (70-180 mg/dL): {in_range:.2f}%")
    print(f"Time in Hypoglycemia (<70 mg/dL): {hypo:.2f}%")
    print(f"Time in Hyperglycemia (>180 mg/dL): {hyper:.2f}%")
    print(f"Time in Severe Hypoglycemia (<54 mg/dL): {severe_hypo:.2f}%")
    
    # Save metrics to file
    metrics = {
        'total_reward': total_reward,
        'mean_reward': mean_reward,
        'time_in_range': in_range,
        'time_in_hypo': hypo,
        'time_in_hyper': hyper,
        'time_in_severe_hypo': severe_hypo
    }
    
    metrics_path = os.path.join(output_dir, f"metrics_{os.path.basename(test_csv_path).replace('.csv', '.txt')}")
    with open(metrics_path, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    
    # Plot results
    plot_evaluation_results(df, actions_pred, insulin_rates, rewards, output_dir)
    
    return metrics

def plot_evaluation_results(df, actions_pred, insulin_rates, rewards, output_dir):
    """Plot evaluation results."""
    # Create time index
    if 'timestamp' in df.columns:
        time_index = pd.to_datetime(df['timestamp'])
    else:
        time_index = pd.RangeIndex(len(df))
    
    # Plot glucose and insulin
    plt.figure(figsize=(15, 10))
    
    # Glucose plot
    plt.subplot(3, 1, 1)
    plt.plot(time_index, df['glu_raw'], 'b-', label='Glucose (mg/dL)')
    plt.axhline(y=70, color='g', linestyle='--', alpha=0.7, label='Lower Bound (70 mg/dL)')
    plt.axhline(y=180, color='r', linestyle='--', alpha=0.7, label='Upper Bound (180 mg/dL)')
    plt.title('Glucose Levels')
    plt.ylabel('Glucose (mg/dL)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Actions plot
    plt.subplot(3, 1, 2)
    plt.plot(time_index, actions_pred[:, 0], 'g-', label='Predicted Basal Action')
    plt.plot(time_index, actions_pred[:, 1], 'r-', label='Predicted Bolus Action')
    if 'basal' in df.columns and 'bolus' in df.columns:
        plt.plot(time_index, df['basal'], 'g--', alpha=0.5, label='Historical Basal')
        plt.plot(time_index, df['bolus'], 'r--', alpha=0.5, label='Historical Bolus')
    plt.title('Actions (Normalized)')
    plt.ylabel('Action [-1, 1]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Insulin rates plot
    plt.subplot(3, 1, 3)
    plt.plot(time_index, insulin_rates[:, 0], 'g-', label='Basal Insulin Rate')
    plt.plot(time_index, insulin_rates[:, 1], 'r-', label='Bolus Insulin Rate')
    plt.title('Insulin Rates')
    plt.ylabel('Insulin Rate (U/min)')
    plt.xlabel('Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'glucose_insulin_plot.png'))
    plt.close()
    
    # Plot rewards
    plt.figure(figsize=(15, 5))
    plt.plot(time_index, rewards)
    plt.title('Rewards')
    plt.ylabel('Reward')
    plt.xlabel('Time')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'rewards_plot.png'))
    plt.close()

def test_agent(model_path="models/sac_model_final.pt"):
    """
    Load a trained agent and evaluate it on test data.
    
    Args:
        model_path: Path to the trained model
    """
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return
    
    # Define state and action dimensions
    state_dim = 8  # glu, glu_d, glu_t, hr, hr_d, hr_t, iob, hour
    action_dim = 2  # basal, bolus
    
    # Initialize agent
    agent = SACAgent(state_dim=state_dim, action_dim=action_dim)
    
    # Load trained model
    agent.load(model_path)
    
    # Find test datasets
    test_files = [f for f in os.listdir("datasets/processed") if f.endswith("-test.csv")]
    
    if not test_files:
        print("No test datasets found in datasets/processed directory.")
        return
    
    # Evaluate on each test dataset
    all_metrics = {}
    for test_file in test_files:
        test_csv_path = os.path.join("datasets/processed", test_file)
        print(f"\nEvaluating on {test_file}...")
        metrics = evaluate_agent(agent, test_csv_path)
        all_metrics[test_file] = metrics
    
    # Print summary of all evaluations
    print("\n===== Evaluation Summary =====")
    for test_file, metrics in all_metrics.items():
        print(f"\n{test_file}:")
        print(f"  Mean Reward: {metrics['mean_reward']:.2f}")
        print(f"  Time in Range: {metrics['time_in_range']:.2f}%")
        print(f"  Time in Hypoglycemia: {metrics['time_in_hypo']:.2f}%")

if __name__ == "__main__":
    test_agent()
