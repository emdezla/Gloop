import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

class DiabetesDataset(Dataset):
    """Processed diabetes management dataset"""
    
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        
        # Handle missing values by forward-filling and backward-filling
        df = df.ffill().bfill()
        
        # Verify no remaining NaNs
        if df[["glu", "glu_d", "glu_t", "hr", "hr_d", "hr_t", "iob", "hour"]].isna().any().any():
            raise ValueError("Dataset contains NaN values after preprocessing")
        
        # State features (8 dimensions)
        self.states = df[["glu", "glu_d", "glu_t", "hr", "hr_d", "hr_t", "iob", "hour"]].values.astype(np.float32)
        
        # Actions (2 dimensions)
        self.actions = df[["basal", "bolus"]].values.astype(np.float32)
        
        # Rewards computed from next glucose values
        self.rewards = self._compute_rewards(df["glu_raw"].values)
        
        # Transition handling
        self.next_states = np.roll(self.states, -1, axis=0)
        self.dones = df["done"].values.astype(np.float32)
        
        # Remove last invalid transition
        self._sanitize_transitions()

    def _compute_rewards(self, glucose_next):
        """Improved reward scaling"""
        glucose_next = np.clip(glucose_next, 40, 400)
        with np.errstate(invalid='ignore'):
            log_term = np.log(glucose_next/180.0)
            risk_index = 10 * (1.509 * (log_term**1.084 - 1.861)**2)
        
        # Better reward scaling using sigmoid instead of tanh
        rewards = -1 / (1 + np.exp(-risk_index/50))  # Scaled to (-1, 0)
        rewards[glucose_next < 54] = -5.0  # Stronger hypo penalty
        return rewards.astype(np.float32)

    def _sanitize_transitions(self):
        """Remove invalid transitions and align array lengths"""
        valid_mask = np.ones(len(self.states), dtype=bool)
        valid_mask[-1] = False  # Remove last transition
        self.states = self.states[valid_mask]
        self.actions = self.actions[valid_mask]
        self.rewards = self.rewards[valid_mask]
        self.next_states = self.next_states[valid_mask]
        self.dones = self.dones[valid_mask]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            'state': torch.FloatTensor(self.states[idx]),
            'action': torch.FloatTensor(self.actions[idx]),
            'reward': torch.FloatTensor([self.rewards[idx]]),
            'next_state': torch.FloatTensor(self.next_states[idx]),
            'done': torch.FloatTensor([self.dones[idx]])
        }
        
def evaluate_dataset_coverage(dataset, n_samples=1000):
    """Evaluate state and action coverage of DiabetesDataset"""
    # Convert to pandas DataFrame for easier analysis
    df = pd.DataFrame({
        'state': [s.numpy() for s in dataset.states],
        'action': [a.numpy() for a in dataset.actions],
        'reward': dataset.rewards.numpy(),
        'done': dataset.dones.numpy()
    })
    
    # Random sample for visualization
    sample_idx = np.random.choice(len(df), size=min(n_samples, len(df)), replace=False)
    states_sample = np.array(df['state'].iloc[sample_idx].tolist())
    actions_sample = np.array(df['action'].iloc[sample_idx].tolist())

    # Create subplots
    fig, axs = plt.subplots(3, 2, figsize=(16, 18))
    
    # --------------------------
    # 1. State Space Analysis
    # --------------------------
    # PCA projection for state visualization
    pca = PCA(n_components=2)
    states_2d = pca.fit_transform(states_sample)
    
    axs[0,0].scatter(states_2d[:, 0], states_2d[:, 1], alpha=0.6)
    axs[0,0].set_title(f'State Space PCA (Explained Variance: {pca.explained_variance_ratio_.sum():.2f})')
    axs[0,0].set_xlabel('PC1')
    axs[0,0].set_ylabel('PC2')
    
    # State feature distributions
    state_df = pd.DataFrame(states_sample, columns=[
        "glu", "glu_d", "glu_t", "hr", "hr_d", "hr_t", "iob", "hour"
    ])
    sns.violinplot(data=state_df, ax=axs[0,1])
    axs[0,1].set_title('State Feature Distributions')
    axs[0,1].tick_params(axis='x', rotation=45)

    # --------------------------
    # 2. Action Space Analysis
    # --------------------------
    # Action distribution scatter plot
    action_df = pd.DataFrame(actions_sample, columns=["basal", "bolus"])
    sns.scatterplot(data=action_df, x="basal", y="bolus", alpha=0.6, ax=axs[1,0])
    axs[1,0].set_title('Action Space Coverage')
    axs[1,0].set_xlim(-1.1, 1.1)
    axs[1,0].set_ylim(-1.1, 1.1)
    
    # Action histogram
    sns.histplot(data=action_df, bins=50, ax=axs[1,1])
    axs[1,1].set_title('Action Value Distribution')
    axs[1,1].set_xlim(-1.1, 1.1)

    # --------------------------
    # 3. Temporal Analysis
    # --------------------------
    # Reward distribution
    sns.histplot(df['reward'], bins=50, ax=axs[2,0])
    axs[2,0].set_title('Reward Distribution')
    
    # Episode length analysis
    done_indices = np.where(df['done'] == 1)[0]
    episode_lengths = np.diff(np.concatenate(([0], done_indices+1)))
    sns.histplot(episode_lengths, bins=50, ax=axs[2,1])
    axs[2,1].set_title(f'Episode Lengths (Mean: {np.mean(episode_lengths):.1f})')

    plt.tight_layout()
    plt.show()

    # --------------------------
    # 4. Coverage Metrics
    # --------------------------
    print("\n=== Coverage Metrics ===")
    
    # State coverage
    state_ranges = state_df.agg(['min', 'max', 'mean', 'std'])
    print("\nState Ranges:")
    print(state_ranges)
    
    # Action coverage
    action_coverage = action_df.agg(['min', 'max', 'mean', 'std'])
    print("\nAction Coverage:")
    print(action_coverage)
    
    # Reward balance
    reward_stats = pd.DataFrame(df['reward'].describe()).T
    print("\nReward Statistics:")
    print(reward_stats)
    
    # Missing transitions check
    print(f"\nMissing States: {state_df.isnull().sum().sum()}")
    print(f"Missing Actions: {action_df.isnull().sum().sum()}")
    
    # Action-state correlation
    corr_matrix = pd.concat([state_df, action_df], axis=1).corr()
    print("\nTop State-Action Correlations:")
    print(corr_matrix[['basal', 'bolus']].abs().max().sort_values(ascending=False))