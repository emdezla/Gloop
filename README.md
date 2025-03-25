# Gloop
Expanding the glucose-insulin loop of artificial pancreas systems

(Project repository created for the MIT course "How to AI almost anything" (MAS.S60))


## 📁 Dataset: OhioT1DM

The dataset OhioT1DM is processed to obtain csv files with the following structure:

---

| Column       | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `time`       | Timestamp of the measurement (5-minute intervals).                          |
| **STATE**    |                                                                             |
| `glu`        | Glucose level (normalized).                                                 |
| `glu_d`      | Glucose derivative (change from previous value, normalized).                |
| `glu_t`      | Glucose trend (slope over past 30 minutes, normalized).                     |
| `hr`         | Heart rate (normalized).                                                    |
| `hr_d`       | Heart rate derivative (change from previous value, normalized).             |
| `hr_t`       | Heart rate trend (slope over past 30 minutes, normalized).                  |
| `iob`        | Insulin on Board — active insulin in the body (normalized).                 |
| `hour_norm`  | Time of day encoded as hour / 24 (e.g., 8:00 AM → 0.33).                    |
| **ACTION**   |                                                                             |
| `basal`      | Continuous background insulin rate (normalized).                            |
| `bol`        | Discrete insulin bolus dose (normalized).                                   |
| **OTHER**    |                                                                             |
| `done`       | Episode boundary flag (1 = end of episode/day, 0 = otherwise).              |

---

## 🧠 Intended Use (RL Environment)

- **State vector** (shape: 8):  
  `[glu, glu_d, glu_t, hr, hr_d, hr_t, iob, hour_norm]`

- **Action vector** (shape: 2):  
  `[basal, bol]`