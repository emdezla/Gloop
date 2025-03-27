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
    """Evaluate state and action coverage of DiabetesDataset"""
    # Convert to pandas DataFrame with proper dimension handling
    df = pd.DataFrame({
        'state': [s.tolist() for s in dataset.states],
        'action': dataset.actions.squeeze().tolist(),  # Handle (N,1) -> (N) array
        'reward': dataset.rewards.tolist(),
        'done': dataset.dones.tolist(),
        'hour': [s[-1] for s in dataset.states]  # Extract hour from state
    })
    
    # Create state feature DataFrame
    state_df = pd.DataFrame(dataset.states, columns=[
        "glu", "glu_d", "glu_t", "hr", "hr_d", "hr_t", "iob", "hour"
    ])
    
    # Create subplots
    fig, axs = plt.subplots(4, 2, figsize=(18, 24))
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    
    # --------------------------
    # 1. State Space Analysis
    # --------------------------
    # PCA projection
    pca = PCA(n_components=2)
    states_2d = pca.fit_transform(dataset.states)
    axs[0,0].scatter(states_2d[:, 0], states_2d[:, 1], alpha=0.6)
    axs[0,0].set_title(f'State Space PCA (Explained: {pca.explained_variance_ratio_.sum():.2f})')
    
    # Feature distributions
    sns.violinplot(data=state_df, ax=axs[0,1])
    axs[0,1].set_title('State Feature Distributions')
    axs[0,1].tick_params(axis='x', rotation=45)

    # --------------------------
    # 2. Action Space Analysis
    # --------------------------
    # Action vs Glucose
    axs[1,0].scatter(state_df['glu'], df['action'], alpha=0.3)
    axs[1,0].set_title('Action vs. Glucose Level')
    axs[1,0].set_xlabel('Normalized Glucose')
    axs[1,0].set_ylabel('Action')
    
    # Action histogram
    sns.histplot(df['action'], bins=50, ax=axs[1,1], kde=True)
    axs[1,1].set_title('Action Distribution')
    axs[1,1].set_xlim(-1.1, 1.1)

    # --------------------------
    # 3. Temporal Analysis
    # --------------------------
    # Reward distribution
    sns.histplot(df['reward'], bins=50, ax=axs[2,0], kde=True)
    axs[2,0].set_title('Reward Distribution')
    
    # Episode lengths
    done_indices = np.where(df['done'] == 1)[0]
    episode_lengths = np.diff(np.concatenate(([0], done_indices+1)))
    sns.histplot(episode_lengths, bins=50, ax=axs[2,1], kde=True)
    axs[2,1].set_title(f'Episode Lengths (Mean: {np.mean(episode_lengths):.1f})')

    # --------------------------
    # 4. Clinical Analysis
    # --------------------------
    # Hourly action patterns
    sns.boxplot(x='hour', y='action', data=df, ax=axs[3,0], showfliers=False)
    axs[3,0].set_title('Hourly Action Distribution')
    axs[3,0].set_xticks(np.arange(0, 24, 3))
    
    # Action-IoB relationship
    sns.scatterplot(x=state_df['iob'], y=df['action'], ax=axs[3,1], alpha=0.3)
    axs[3,1].set_title('Action vs. Insulin On Board')
    axs[3,1].set_xlabel('Normalized IoB')
    axs[3,1].set_ylabel('Action')

    plt.tight_layout()
    plt.show()

    # --------------------------
    # 5. Quantitative Metrics
    # --------------------------
    print("\n=== Critical Metrics ===")
    
    # Action statistics
    action_stats = pd.DataFrame({
        'min': [np.min(dataset.actions)],
        'max': [np.max(dataset.actions)],
        'mean': [np.mean(dataset.actions)],
        'std': [np.std(dataset.actions)],
        '5th %ile': [np.percentile(dataset.actions, 5)],
        '95th %ile': [np.percentile(dataset.actions, 95)]
    })
    print("\nAction Space Statistics:")
    print(action_stats.T)
    
    # State-action correlations
    corr_matrix = pd.concat([state_df, df[['action']]], axis=1).corr()
    print("\nTop State-Action Correlations:")
    print(corr_matrix['action'].abs().sort_values(ascending=False).head(5))

    # Safety metrics
    print(f"\nEpisodes with Extreme Actions (<-0.9): {(df['action'] < -0.9).mean():.2%}")
    print(f"Episodes with No Action (>-0.1): {(df['action'] > -0.1).mean():.2%}")

    # Temporal coverage
    print(f"\nWeakest Hourly Coverage: {df.groupby('hour').size().min()} samples")

    # Data quality
    print(f"\nMissing States: {state_df.isnull().sum().sum()}")
    print(f"Missing Actions: {pd.isna(df['action']).sum()}")

    # Add to quantitative metrics
    print("\nAction Value Brackets:")
    action_bins = pd.cut(df['action'], bins=[-1, -0.5, -0.1, 0, 0.5, 1], 
                    labels=['<-0.5', '-0.5:-0.1', '-0.1:0', '0:0.5', '>0.5'])
    print(action_bins.value_counts(normalize=True).sort_index())

    # Add to temporal analysis
    action_changes = np.diff(df['action'])
    plt.figure(figsize=(12,4))
    sns.histplot(action_changes, bins=50, kde=True)
    plt.title('Insulin Dose Change Distribution')
    plt.xlabel('Δ Action Between Steps')
    plt.show()

    print(f"\nInsulin Change Stats:")
    print(f"Mean Δ: {np.mean(action_changes):.3f}")
    print(f"Max Increase: {np.max(action_changes):.3f}")
    print(f"Max Decrease: {np.min(action_changes):.3f}")

    # Add to clinical analysis
    glu_bins = pd.cut(state_df['glu'], bins=5)
    action_heatmap = pd.pivot_table(
    df,
        values='action',
        index=glu_bins,
        columns=pd.cut(state_df['iob'], bins=5),
        aggfunc=np.mean
    )

    plt.figure(figsize=(10,6))
    sns.heatmap(action_heatmap, annot=True, cmap='coolwarm', center=0)
    plt.title('Mean Action by Glucose and IOB Levels')
    plt.xlabel('IOB Bins')
    plt.ylabel('Glucose Bins')
    plt.show()

        # Add to safety metrics
    conservative_ratio = (df['action'] < np.percentile(df['action'], 25)).mean()
    print(f"\nConservative Actions (Bottom 25%): {conservative_ratio:.2%}")
    print(f"Action Range Utilization: {(df['action'].max() - df['action'].min())/2:.1%} of possible [-1,1] range")

    # Add to reward analysis
    reward_breakdown = pd.DataFrame({
        'Hypoglycemia': [df.loc[state_df['glu'] < -1.0, 'reward'].mean()],  # Assuming normalized glu < 70mg/dL
        'Target Range': [df.loc[(state_df['glu'] >= -1.0) & (state_df['glu'] <= 1.0), 'reward'].mean()],
        'Hyperglycemia': [df.loc[state_df['glu'] > 1.0, 'reward'].mean()]  # Assuming normalized glu > 180mg/dL
    })

    print("\nReward Breakdown by Glucose Range:")
    print(reward_breakdown.T)

    high_iob_mask = state_df['iob'] > np.percentile(state_df['iob'], 75)
    print(f"\nHigh IOB Actions (Top 25%): Mean={df.loc[high_iob_mask, 'action'].mean():.3f}")

    print("\nHourly Coverage (% of total):")
    print(df.groupby('hour').size() / len(df) * 100)

    success_mask = (state_df['glu'] > -0.5) & (state_df['glu'] < 0.5)  # Normalized target range
    print(f"\nSuccess Rate in Target Range: {success_mask.mean():.2%}")
    print(f"Average Action in Target Range: {df.loc[success_mask, 'action'].mean():.3f}")
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