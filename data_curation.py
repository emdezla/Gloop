import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader

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
        self.actions = df["action"].values.astype(np.float32)
        
        # Rewards computed from next glucose values
        self.rewards = self._compute_rewards(df["glu_raw"].values)
        self.glu_raw = df["glu_raw"].values.astype(np.float32)
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
    """Comprehensive evaluation of diabetes management dataset"""
    # Convert to pandas DataFrame with proper feature extraction
    df = pd.DataFrame({
        'state': dataset.states.tolist(),
        'action': dataset.actions.squeeze().tolist(),  # Ensure 1D actions
        'reward': dataset.rewards.tolist(),
        'done': dataset.dones.tolist(),
        'hour': [s[-1] for s in dataset.states]  # Extract hour from all states
    })
    
    # Add raw glucose values if available (modify your DiabetesDataset to include this)
    if hasattr(dataset, 'glu_raw'):
        df['glu_raw'] = dataset.glu_raw.tolist()
    
    # Random sample for visualization
    sample_idx = np.random.choice(len(df), size=min(n_samples, len(df)), replace=False)
    states_sample = np.array(df['state'].iloc[sample_idx].tolist())
    actions_sample = np.array(df['action'].iloc[sample_idx])

    # Create comprehensive visual layout
    fig = plt.figure(figsize=(18, 24))
    gs = fig.add_gridspec(4, 2)
    axs = [
        fig.add_subplot(gs[0, 0]),  # State PCA
        fig.add_subplot(gs[0, 1]),  # State distributions
        fig.add_subplot(gs[1, 0]),  # Action vs Glucose
        fig.add_subplot(gs[1, 1]),  # Action histogram
        fig.add_subplot(gs[2, 0]),  # Reward dist
        fig.add_subplot(gs[2, 1]),  # Episode lengths
        fig.add_subplot(gs[3, :])   # Hourly actions (full width)
    ]

    # --------------------------
    # 1. State Space Analysis
    # --------------------------
    # PCA projection
    pca = PCA(n_components=2)
    states_2d = pca.fit_transform(states_sample)
    axs[0].scatter(states_2d[:, 0], states_2d[:, 1], alpha=0.6)
    axs[0].set_title(f'State Space PCA (Explained Variance: {pca.explained_variance_ratio_.sum():.2f})')
    
    # Feature distributions
    state_df = pd.DataFrame(states_sample, columns=[
        "glu", "glu_d", "glu_t", "hr", "hr_d", "hr_t", "iob", "hour"
    ])
    sns.violinplot(data=state_df, ax=axs[1])
    axs[1].set_title('State Feature Distributions')
    axs[1].tick_params(axis='x', rotation=45)

    # --------------------------
    # 2. Action Space Analysis
    # --------------------------
    # Action vs Glucose
    action_glu_df = pd.DataFrame({
        'glu': states_sample[:, 0],
        'action': actions_sample
    })
    sns.scatterplot(data=action_glu_df, x='glu', y='action', alpha=0.6, ax=axs[2])
    axs[2].set_title('Action vs. Glucose Level')
    
    # Action distribution
    sns.histplot(data=actions_sample, bins=50, ax=axs[3], kde=True)
    axs[3].set_title('Action Value Distribution')
    axs[3].set_xlabel('Normalized Action Value')

    # --------------------------
    # 3. Temporal Analysis
    # --------------------------
    # Reward distribution
    sns.histplot(df['reward'], bins=50, ax=axs[4], kde=True)
    axs[4].set_title('Reward Distribution')
    
    # Episode lengths
    done_indices = np.where(df['done'] == 1)[0]
    episode_lengths = np.diff(np.concatenate(([0], done_indices+1)))
    sns.histplot(episode_lengths, bins=50, ax=axs[5], kde=True)
    axs[5].set_title(f'Episode Lengths (Mean: {np.mean(episode_lengths):.1f})')

    # --------------------------
    # 4. Circadian Analysis
    # --------------------------
    # Hourly action distribution
    sns.boxplot(data=df, x='hour', y='action', ax=axs[6], showfliers=False)
    axs[6].set_title('Action Distribution by Hour of Day')
    axs[6].set_xticks(np.arange(0, 24, 3))
    axs[6].set_xlabel('Hour of Day')
    axs[6].set_ylabel('Action Value')

    plt.tight_layout()
    plt.show()

    # --------------------------
    # 5. Quantitative Metrics
    # --------------------------
    print("\n=== Critical Metrics ===")
    
    # Action bounds analysis
    action_percentiles = np.percentile(actions_sample, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    print("\nAction Value Percentiles (0, 1, 5, 25, 50, 75, 95, 99, 100):")
    print(np.round(action_percentiles, 3))

    # Hypo/hyper analysis
    if 'glu_raw' in df:
        hypo_rate = (df['glu_raw'] < 54).mean()
        hyper_rate = (df['glu_raw'] > 180).mean()
        print(f"\nHypoglycemia Rate (<54 mg/dL): {hypo_rate:.2%}")
        print(f"Hyperglycemia Rate (>180 mg/dL): {hyper_rate:.2%}")
    else:
        print("\n⚠️ Raw glucose values not available for hypo/hyper analysis")

    # Action correlations
    corr_matrix = pd.concat([state_df, pd.DataFrame({'action': actions_sample})], axis=1).corr()
    print("\nTop State-Action Correlations:")
    print(corr_matrix['action'].abs().sort_values(ascending=False).head(5))

    # Safety metrics
    risky_actions = (actions_sample > 0.8).mean()
    print(f"\nPotentially Dangerous Actions (>0.8): {risky_actions:.2%}")

    # Data quality checks
    print(f"\nMissing States: {state_df.isnull().sum().sum()}")
    print(f"Missing Actions: {pd.isna(actions_sample).sum()}")

    # Temporal consistency
    hour_coverage = df.groupby('hour')['action'].count()
    print(f"\nWeakest Hourly Coverage: {hour_coverage.min()} samples at hour {hour_coverage.idxmin()}")


"""Evaluate state and action coverage of DiabetesDataset
def evaluate_dataset_coverage(dataset, n_samples=1000):
    
    # Convert to pandas DataFrame for easier analysis
    df = pd.DataFrame({
        'state': dataset.states.tolist(),
        'action': dataset.actions.tolist(),  # Now single column
        'reward': dataset.rewards.tolist(),
        'done': dataset.dones.tolist()
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
    # Create DataFrames for analysis
    action_df = pd.DataFrame(actions_sample, columns=["action"])  # Single column
    action_glu_df = pd.DataFrame({
        'glu': states_sample[:, 0],  # Glucose values from state
        'action': actions_sample
    })
    
    # Action vs. Glucose Level
    sns.scatterplot(data=action_glu_df, x='glu', y='action', alpha=0.6, ax=axs[1,0])
    axs[1,0].set_title('Action vs. Glucose Level')
    
    # Action histogram
    sns.histplot(data=action_df, bins=50, ax=axs[1,1])
    axs[1,1].set_title('Action Value Distribution')
    axs[1,1].set_xlim(action_df.min().min()-0.1, action_df.max().max()+0.1)

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
    print(corr_matrix['action'].abs().sort_values(ascending=False))

    print("Action bounds:", np.percentile(action_df, [0, 1, 5, 95, 99, 100]))

    hypo_rate = (df['glu'] < 54).mean()  # Using raw glucose values
    print(f"Hypoglycemia rate: {hypo_rate:.2%}")

    df['hour'] = states_sample[:, -1]  # Last state feature is hour
    sns.boxplot(data=df, x='hour', y='action') """

"""Evaluate state and action coverage of DiabetesDataset with BOLUS and BASAL
def evaluate_dataset_coverage(dataset, n_samples=1000):
    
    # Convert to pandas DataFrame for easier analysis
    df = pd.DataFrame({
        'state': dataset.states.tolist(),
        'action': dataset.actions.tolist(),
        'reward': dataset.rewards.tolist(),
        'done': dataset.dones.tolist()
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
"""