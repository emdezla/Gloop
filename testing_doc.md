# SAC-CQL Diabetes Model Testing Documentation

## Testing Workflow

### 1. Model Loading
- Loads the trained SAC model from checkpoint
- Handles both new format (with metadata) and legacy models
- Automatically detects GPU availability

### 2. Data Preparation
- Loads processed test dataset CSV
- Calculates baseline metrics:
  - Time in Range (70-180 mg/dL)
  - Hypo/Hyperglycemia percentages
  - Severe event thresholds (<54, >250 mg/dL)

### 3. Evaluation Process
- Batched evaluation (128 samples/batch)
- Computes key metrics:
  - **RMSE/MAE** - Action prediction accuracy
  - **Mean Reward** - Overall policy performance
  - **Action Statistics** - Policy confidence analysis
  - **Glucose Control** - Clinical safety metrics

### 4. Result Generation
- Saves JSON file with all metrics
- Produces 5 diagnostic plots
- Stores raw predictions for further analysis

---

## Graph Interpretations

### 1. Action Comparison Plot
![Action Comparison](action_comparison.png)
- **Left (Basal):** Ideal points follow red diagonal
- **Right (Bolus):** Tight cluster = good prediction
- **Analysis:**
  - Points above diagonal = under-dosing
  - Points below diagonal = over-dosing
  - Vertical/horizontal spreads indicate systematic errors

### 2. Glucose Distribution
![Glucose Distribution](glucose_distribution.png)
- **Key Ranges:**
  - Green (70-180 mg/dL): Ideal range
  - Yellow (54-70, 180-250): Warning zones
  - Red (<54, >250): Danger zones
- **Analysis:**
  - Tall middle peak = good control
  - Right skew = hyperglycemia risk
  - Left skew = hypoglycemia risk

### 3. Action Distribution
![Action Distribution](action_distribution.png)
- **Basal (Left):**
  - Should match circadian rhythm patterns
  - Sharp peaks may indicate insufficient adaptation
- **Bolus (Right):**
  - Expected multi-modal distribution
  - Compare predicted vs true meal responses
- **Analysis:**
  - Matching distributions = good policy learning
  - Divergence in peaks = missed meal patterns

### 4. Reward Distribution
![Reward Distribution](reward_distribution.png)
- **Ideal Profile:**
  - Left-skewed (most rewards near 0)
  - Few extreme negative values
- **Analysis:**
  - Wide spread = unstable control
  - Secondary modes = recurring dangerous events
  - Tight cluster = conservative policy

### 5. State-Action Heatmaps
![Heatmaps](state_action_heatmap.png)
- **Glucose vs Basal (Top-Left):**
  - Expected: Smooth basal reduction as glucose drops
- **Glucose vs Bolus (Top-Middle):**
  - Should show meal response curve
- **IOB vs Bolus (Top-Right):**
  - Check insulin stacking avoidance
- **Time-Based Patterns (Bottom):**
  - Basal: Circadian rhythm visible?
  - Bolus: Meal timing peaks present?

### 6. Error Analysis
![Error Analysis](error_analysis.png)
- **Error vs Glucose:**
  - Systematic errors at certain glucose levels?
- **Error vs Time:**
  - Time-dependent prediction issues?
- **Error vs IOB:**
  - Insulin stacking handling issues?
- **Analysis:**
  - Clusters away from zero line indicate model bias
  - Wider spread at extremes indicates poor generalization

---

## Key Metrics

| Metric | Ideal Range | Interpretation |
|--------|-------------|-----------------|
| **RMSE** | <0.2 | Action prediction accuracy |
| **Time in Range** | >70% | Clinical effectiveness |
| **Hypo Events** | <4% | Patient safety |
| **Action STD** | 0.1-0.3 | Policy confidence level |
| **Mean Reward** | >-0.5 | Overall performance |
| **Noise Robustness** | <0.15 | Model stability |

---

## Running the Tests

```bash
python SACCQL_testing.py \
  --model models/sac_model_20240315_143256.pth \
  --test_data datasets/processed/563-test.csv \
  --output_dir results/latest_run
```

**Output Structure:**
```
results/
  latest_run/
    latest_metrics.json
    eval_20240315_143256/
      evaluation_metrics.json
      clinical_report.md
      action_comparison.png
      glucose_distribution.png
      action_distribution.png
      reward_distribution.png
      state_action_heatmap.png
      error_analysis.png
      raw_predictions.npz
```

**Additional Options:**
```bash
# Test model robustness to input noise
python SACCQL_testing.py --model model.pth --noise_test

# Compare two models
python SACCQL_testing.py --model model1.pth --compare model2.pth

# Show this documentation
python SACCQL_testing.py --doc
```

---

## Troubleshooting Guide

1. **JSON Serialization Errors**
   - Ensure all metrics are native Python types
   - Run `convert_numpy()` helper before saving

2. **Missing Glucose Data**
   - Verify CSV contains "glu_raw" column
   - Check for NaN values in preprocessing

3. **Model Version Mismatch**
   - Confirm state/action dimensions match
   - Check metadata in model checkpoint

4. **Flatlined Predictions**
   - Investigate gradient clipping values
   - Check for over-regularization
