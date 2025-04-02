# Model Evaluation

This document explains the metrics used to evaluate our reinforcement learning models for diabetes management and discusses the results of both the full model (with heart rate features) and the reduced model (without heart rate features).

## Evaluation Metrics

We use several metrics to evaluate the performance of our models:

1. **Average Q-Value**: The average expected return estimated by the critic networks. Higher values generally indicate the model believes its actions will lead to better outcomes.

2. **Mean Squared Error (MSE)**: Measures the average squared difference between the predicted insulin doses and the actual doses in the dataset. Lower values indicate better prediction accuracy.

3. **Pearson Correlation**: Measures the linear correlation between predicted and actual insulin doses. Values range from -1 to 1, with values closer to 1 indicating stronger positive correlation.

4. **R² Score (Coefficient of Determination)**: Indicates the proportion of variance in the actual insulin doses that is predictable from the model. Values range from 0 to 1, with higher values indicating better fit.

5. **Regression Line Parameters**: The slope and intercept of the best-fit line between actual and predicted actions. A slope closer to 1 and intercept closer to 0 would indicate perfect prediction.

6. **Feature Sensitivity Analysis**: Measures the influence of each state feature on the predicted insulin dose, helping us understand which features the model relies on most heavily.

## Comparison of Models

We trained two variants of our Soft Actor-Critic (SAC) model:

1. **Full Model (HR)**: Uses all 8 state dimensions including glucose level, glucose derivative, glucose trend, heart rate, heart rate derivative, heart rate trend, insulin on board, and hour of day.

2. **Reduced Model (no-HR)**: Uses only 5 state dimensions, excluding the heart rate features.

### Quantitative Results

The table below summarizes the evaluation metrics for both models:

| Metric | Full Model (HR) | Reduced Model (no-HR) |
|--------|----------------|----------------------|
| Avg Q-Value | -9.9713 | -9.9678 |
| MSE | 0.0175 | 0.0178 |
| Pearson Correlation | 0.7830 | 0.7856 |
| R² Score | 0.5947 | 0.5875 |
| Regression Slope | 0.5847 | 0.5884 |
| Regression Intercept | -0.1551 | -0.1618 |

At first glance, the metrics appear very similar between the two models, with the reduced model showing slightly better correlation but slightly worse MSE and R² score.

### Feature Sensitivity Analysis

The feature sensitivity analysis reveals important differences between the models:

- In the **Full Model (HR)**, Insulin on Board has the highest influence (49.2%), followed by Glucose (15.9%) and Hour of Day (15.9%). Heart rate and its derivatives together account for about 11.5% of the model's decision-making.

- In the **Reduced Model (no-HR)**, Insulin on Board dominates even more strongly (61.0%), followed by Hour of Day (26.4%), with the glucose-related features having much less influence (around 4% each).

### Clinical Correctness

The most significant difference between the models becomes apparent when examining the State-Action Sensitivity graph, which shows how insulin dosing changes with glucose levels:

- The **Full Model (HR)** correctly increases insulin dosing as glucose levels rise, especially at higher glucose values (>2.0 normalized). This aligns with clinical expectations: higher blood glucose should trigger higher insulin doses.

- The **Reduced Model (no-HR)** initially follows a similar pattern but then incorrectly decreases insulin dosing at higher glucose levels. This behavior contradicts clinical knowledge and could be dangerous in practice.

## Conclusion

While the standard evaluation metrics suggest similar performance between the two models, the clinically incorrect behavior of the reduced model at high glucose levels reveals a critical flaw. The heart rate features, though accounting for only about 11.5% of the model's decision-making, appear to provide important context that helps the model learn a more clinically appropriate policy.

Therefore, we conclude that the full model with heart rate features is superior for this application, demonstrating that physiological signals beyond glucose can contribute meaningfully to diabetes management algorithms.

This finding supports the value of multimodal sensing in automated insulin delivery systems and suggests that incorporating additional physiological signals may improve the safety and effectiveness of artificial pancreas systems.
