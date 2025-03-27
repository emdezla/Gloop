# Diabetes Management RL Training Metrics Guide

## Core Metrics Interpretation

### 1. Critic Loss (Q-loss)
**Ideal Behavior:**
```plaintext
[Initial] 2.0 → [Mid] 0.8 → [Final] 0.3 (gradual decrease then stabilization)
```
**Healthy Signs:**
- Smooth exponential decay pattern
- Final value between 0.1-1.0
- Q1 and Q2 losses remain close (<10% difference)

**Warning Signs:**
- ████▁▁▁▁ (Collapse to near-zero - overfitting)
- ▁▁▁▁████ (Monotonic increase - divergence)
- Q1/Q2 divergence >20%

**Clinical Relevance:**  
Stable Q-values ensure consistent insulin dosing decisions.

---

### 2. Actor Loss (Policy Loss)
**Ideal Behavior:**
```plaintext
[Initial] 3.0 → [Mid] 2.2 → [Final] 1.8 (oscillating decrease)
```
**Healthy Signs:**
- Moderate oscillations (±15%)
- Final range: 1.5-3.5
- Correlated with entropy decay

**Warning Signs:**
- ████████ (Continuous rise - policy collapse)
- ▔▔▔▔▔▔▔▔ (Near-zero values - no exploration)

**Clinical Relevance:**  
Balanced policy learns both correction and maintenance doses.

---

### 3. Alpha Loss (Entropy)
**Ideal Behavior:**
```plaintext
[Initial] 1.0 → [Mid] 0.5 → [Final] 0.2 (slow decay)
```
**Healthy Signs:**
- Maintains α > 0.1 through 75% of training
- Final α between 0.1-0.5
- Smooth decay curve

**Warning Signs:**
- ▁▁▁▁▁▁▁▁ (Fast collapse <0.01)
- ████████ (Stays >1.0 - no policy focus)

**Clinical Relevance:**  
Maintains exploration for unexpected glucose scenarios.

---

### 4. Q-values
**Ideal Behavior:**
```plaintext
Q1: [-4.0 → -1.5] | Q2: [-4.2 → -1.6] (converging trajectories)
```
**Healthy Signs:**
- Stay within reward bounds (-5 to 0)
- Difference <10% between Q1/Q2
- Follow reward distribution shape

**Warning Signs:**
- Values < -4.5 (over-pessimism)
- Values > -0.5 (over-optimism)
- Growing divergence

**Clinical Relevance:**  
Accurate value estimation prevents dangerous over/under-dosing.

---

## Performance Prediction Framework

### Action Statistics
```plaintext
          | Good              | Bad
----------|-------------------|-------------------
Mean      | -0.2 to +0.2      | < -0.5 or > +0.5
Std       | 0.3-0.6           | < 0.1 (rigid) or > 1.0 (erratic)
```

### Gradient Norms
**Healthy:**  
```plaintext
[Initial] 50 → [Mid] 5 → [Final] 0.5 (steady decrease)
```

**Unhealthy Patterns:**  
- Spike >100 (exploding gradients)
- Flatline at 0 (dead network)

---

## Clinical Validation Thresholds

1. **Time-in-Range (70-180 mg/dL):**
   - Train/Test gap <5%
   - Absolute value >65%

2. **Hypoglycemia (<54 mg/dL):**
   - <2% of readings
   - No consecutive episodes

3. **Action Consistency:**
   - Nighttime std < daytime std
   - Meal-time actions > basal by 30-50%

---

## Troubleshooting Guide

| Symptom                | Likely Causes               | Fixes                         |
|------------------------|-----------------------------|-------------------------------|
| Rising actor loss      | 1. Over-optimistic Q-values | 1. Increase CQL weight        |
|                        | 2. Entropy collapse         | 2. Adjust α target entropy    |
| Q-value divergence     | 1. Different initialization | 1. Harmonize Q-networks       |
|                        | 2. Asymmetric updates       | 2. Use same optimizer params  |
| Low action variance    | 1. Excessive α decay        | 1. Add action noise           |
|                        | 2. Reward hacking           | 2. Penalize action repetition |

## Model Readiness Checklist

✅ Critic loss stabilized for 100+ epochs  
✅ α > 0.1 maintained in final 20% training  
✅ Test-time action std between 0.2-0.6  
✅ Q-values within [-4.5, -0.5] range  
✅ <1% difference between final Q1/Q2 values
