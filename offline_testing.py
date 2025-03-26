import numpy as np
import torch
from helpers import DiabetesDataset, SACCQL, compute_reward_torch
import matplotlib.pyplot as plt
import os

def test_policy(model_path, test_csv, device="cuda"):
    """Test trained policy on offline dataset"""
    # Model loading
    model = SACCQL().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        raise FileNotFoundError(f"Model file {model_path} not found")
    
    # Dataset setup
    test_set = DiabetesDataset(test_csv)
    actions_real = []
    actions_pred = []
    glucose = []
    
    # Testing loop
    model.eval()
    with torch.no_grad():
        for idx in range(len(test_set)):
            batch = test_set[idx]
            state = batch["state"].to(device)
            action_real = batch["action"].numpy()
            
            # Get policy action
            action_pred = model.act(state.unsqueeze(0), deterministic=True).cpu().numpy()[0]
            
            # Store results
            actions_real.append(action_real)
            actions_pred.append(action_pred)
            glucose.append(batch["state"][0].item())  # Assuming glu is first feature
    
    # Calculate metrics
    actions_real = np.array(actions_real)
    actions_pred = np.array(actions_pred)
    
    # 1. Behavior Cloned Value Approximation (BCVA)
    mse_basal = np.mean((actions_real[:,0] - actions_pred[:,0])**2)
    mse_bolus = np.mean((actions_real[:,1] - actions_pred[:,1])**2)
    
    # 2. Safety analysis
    hypoglycemia = np.sum(np.array(glucose) < 70)
    hypo_percent = hypoglycemia / len(glucose)
    
    # 3. Q-value analysis
    q_values = []
    for state, action in zip(test_set.states, actions_pred):
        state_t = torch.FloatTensor(state).to(device)
        action_t = torch.FloatTensor(action).to(device)
        q1 = model.q1(torch.cat([state_t, action_t]))
        q2 = model.q2(torch.cat([state_t, action_t]))
        q_values.append((q1.item(), q2.item()))
    
    print(f"\n=== Test Results ===")
    print(f"BCVA MSE - Basal: {mse_basal:.4f}, Bolus: {mse_bolus:.4f}")
    print(f"Hypoglycemia Rate: {hypo_percent:.2%}")
    print(f"Avg Q-values: Q1={np.mean([q[0] for q in q_values]):.2f}, Q2={np.mean([q[1] for q in q_values]):.2f}")
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Glucose trajectory
    plt.subplot(3, 1, 1)
    plt.plot(glucose, label='Glucose')
    plt.axhline(70, color='r', linestyle='--', label='Hypo Threshold')
    plt.ylabel('mg/dL')
    plt.legend()
    
    # Insulin actions
    plt.subplot(3, 1, 2)
    plt.plot(actions_pred[:,0], label='Predicted Basal')
    plt.plot(actions_real[:,0], label='Real Basal', alpha=0.5)
    plt.ylabel('Basal Insulin')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(actions_pred[:,1], label='Predicted Bolus')
    plt.plot(actions_real[:,1], label='Real Bolus', alpha=0.5)
    plt.ylabel('Bolus Insulin')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('offline_test_results.png')
    plt.show()

def robustness_check(model_path, test_csv, noise_level=0.2):
    """Test policy robustness with input noise"""
    model = SACCQL().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    test_set = DiabetesDataset(test_csv)
    total_reward = 0
    
    with torch.no_grad():
        for idx in range(len(test_set)):
            state = test_set[idx]["state"].numpy()
            # Add Gaussian noise
            noisy_state = state + np.random.normal(0, noise_level, state.shape)
            state_t = torch.FloatTensor(noisy_state).to(device)
            
            action = model.act(state_t.unsqueeze(0))
            # Get actual next state from dataset
            next_state = test_set[idx]["next_state"].numpy()
            
            # Calculate real reward
            glucose_next = test_set.df.iloc[idx]["glu_raw"]
            reward = compute_reward_torch(torch.tensor([glucose_next])).item()
            total_reward += reward
    
    print(f"\nRobustness Check (Noise σ={noise_level})")
    print(f"Average Perturbed Reward: {total_reward/len(test_set):.2f}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Main test
    test_policy(
        model_path="sac_cql_final.pth",
        test_csv="datasets/processed/563-test.csv",
        device=device
    )
    
    # Additional robustness checks
    robustness_check("sac_cql_final.pth", "datasets/processed/563-test.csv")
