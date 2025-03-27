# Gloop: Glucose-Insulin Loop Reinforcement Learning
Expanding the glucose-insulin loop of artificial pancreas systems using SAC-CQL

## 📁 Dataset Characteristics

### OhioT1DM Processed Data
**State Space (8 dimensions):**
- `glu`: Normalized glucose level (40-400 mg/dL → [-1,1])
- `glu_d`: Rate of change (mg/dL/min, normalized)
- `glu_t`: 30-min trend slope (normalized)
- `hr`: Heart rate (normalized per patient)
- `hr_d`: HR change rate (bpm/min, normalized)
- `hr_t`: 30-min HR trend slope
- `iob`: Insulin on Board (0-5U, normalized)
- `hour`: Time of day (0-1 scaled from 0-24h)

**Action Space (Continuous):**  
Single action value ∈ [-1,1] mapping to insulin pump rate:
- -1: Minimum insulin (0 U/hr)
- 1: Maximum insulin (5 U/hr)
- Nonlinear transformation via `tia_action()` function

**Key Features:**
- 5-minute resolution temporal data
- Patient-specific normalization scalers
- Episode boundaries at day transitions
- 6 patient datasets with train/test splits

## 🏗️ Code Structure

```
.
├── SACCQL_training.py       # Main training loop with SAC-CQL
├── SACCQL_testing.py        # Model evaluation & clinical reports
├── dataset_creation.py      # XML→CSV processing pipeline
├── dataset_analysis.py      # Data quality/coverage metrics
├── models/                  # Saved model checkpoints
├── datasets/
│   ├── processed/           # Normalized CSVs
│   └── raw/                 # Original OhioT1DM XMLs
└── training_logs/           # Training metrics & visualizations
```

## 🧠 Model Architecture (SAC-CQL)

**Actor Network:**
```python
nn.Sequential(
    nn.Linear(8, 256),
    nn.LayerNorm(256),
    nn.Mish(),
    nn.Linear(256, 128), 
    nn.LayerNorm(128),
    nn.Mish(),
    nn.Linear(128, 1)
)
```

**Critic Networks (Twin Q):**
```python
nn.Sequential(
    nn.Linear(8+1, 128),
    nn.LayerNorm(128),
    nn.Dropout(0.1),
    nn.LeakyReLU(0.01),
    nn.Linear(128, 128),
    nn.LayerNorm(128),
    nn.Dropout(0.1),
    nn.LeakyReLU(0.01),
    nn.Linear(128, 1)
)
```

**Key Features:**
- Mish activation for smooth gradient flow
- Layer normalization + dropout for stability
- Twin Q-networks with delayed targets
- Entropy regularization (α = 0.2-0.5)
- Xavier initialization with leaky ReLU slopes

## 🔄 RL Algorithm (SAC-CQL Hybrid)

**Soft Actor-Critic with:**
- Entropy-regularized policy optimization
- Conservative Q-Learning (CQL) penalty
- Experience replay (100k capacity)
- Learning rate warmup (50 epochs)
- Gradient clipping (max norm=1.0)

**Key Hyperparameters:**
```python
{
  "gamma": 0.99,       # Discount factor
  "tau": 0.01,         # Target network update
  "alpha": 0.2,        # Entropy coefficient
  "batch_size": 512,   # Training batch size
  "epochs": 500,       # Max training epochs
  "l2_reg": 0.001      # Actor L2 regularization
}
```

## 📈 Evaluation Metrics

### Dataset Coverage
- State space PCA projections
- Action-value distributions
- Temporal coverage analysis
- Feature correlation matrices
- Insulin change statistics
- Hourly action patterns

### Training Metrics
- Critic/Actor loss curves
- Q-value convergence
- Action statistics (mean/std)
- Entropy coefficient (α)
- Gradient norms
- Overfitting/underfitting risk scores

### Testing Metrics
**Technical:**
- RMSE/MAE of actions
- R² score (policy similarity)
- Inference latency (<10ms/sample)

**Clinical:**
- Time-in-Range (70-180 mg/dL)
- Hypo/Hyperglycemia % 
- Glucose CV (% variability)
- GMI (Glucose Management Indicator)
- Model readiness risk assessment

## 🚀 Quick Start

1. **Preprocess Data:**
```python
python dataset_creation.py datasets/raw/OhioT1DM/559-ws-training.xml
```

2. **Train Model:**
```python
python SACCQL_training.py --dataset datasets/processed/559-train.csv
```

3. **Evaluate:**
```python
python SACCQL_testing.py --model models/sac_final_model.pth --test_data datasets/processed/559-test.csv
```

4. **Analyze:**
```python
python dataset_analysis.py datasets/processed/559-train.csv
```

## 📚 Documentation
- `training_doc.md`: Full training protocol
- `testing_doc.md`: Evaluation methodology
- `requirements.txt`: Python dependencies
```
