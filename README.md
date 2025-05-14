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

