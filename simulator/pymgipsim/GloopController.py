import numpy as np
import matplotlib.pyplot as plt
import os, sys, torch, torch.nn as nn
from collections import deque

class Controller:
    name = "GloopController"

    def __init__(self, scenario_instance):
        try:
            print("CHO intake fraction:", scenario_instance.input_generation.fraction_cho_intake)
        except AttributeError:
            print("CHO intake fraction: not available (input_generation is likely a class, not a dict)")

        ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        GLOOP_PATH = os.path.join(ROOT_PATH, "Gloop")
        if GLOOP_PATH not in sys.path:
            sys.path.append(GLOOP_PATH)

        from model.sac_cql import SACAgent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SACAgent().to(self.device)

        checkpoint_path = os.path.join(ROOT_PATH, "Gloop/checkpoints/saccql_trained.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"❌ Checkpoint not found at {checkpoint_path}")
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.train()

        self.replay_buffer = deque(maxlen=10000)
        self.train_every = 8
        self.batch_size = 128
        self.alpha = 0.2
        self.loss_fn = nn.MSELoss()
        self.gamma = 0.99
        self.tau = 0.005

        self.logged_dose = [[] for _ in range(20)]
        self.logged_glucose = [[] for _ in range(20)]
        self.logged_reward = [[] for _ in range(20)]

    def run(self, measurements, inputs, states, sample):
        import random

        if sample < 0:
            return

        time_dim = states.shape[2]
        num_patients = states.shape[0]

        for patient_idx in range(num_patients):
            state_vector = states[patient_idx, :, sample]
            glucose = float(states[patient_idx, 6, sample])

            s_t = np.array([
                glucose,
                float(state_vector[1]),
                0.0,
                float(state_vector[2]),
                float(state_vector[3]),
                0.0,
                float(state_vector[4]),
                (sample % 1440) / 60.0
            ], dtype=np.float32)

            if sample < time_dim - 1:
                state_vector_next = states[patient_idx, :, sample + 1]
                glucose_next = float(state_vector_next[0]) if state_vector_next[0] > 0 else float(states[patient_idx, 6, sample + 1])
                s_tp1 = np.array([
                    glucose_next,
                    float(state_vector_next[1]),
                    0.0,
                    float(state_vector_next[2]),
                    float(state_vector_next[3]),
                    0.0,
                    float(state_vector_next[4]),
                    ((sample + 1) % 1440) / 60.0
                ], dtype=np.float32)
            else:
                s_tp1 = s_t

            with torch.no_grad():
                state_tensor = torch.tensor(s_t, dtype=torch.float32).to(self.device).unsqueeze(0)
                output = self.model.actor(state_tensor)
                if isinstance(output, tuple):
                    _, action_tensor = output
                else:
                    action_tensor = output
                action = action_tensor.squeeze(0).cpu()

            if glucose > 300:
                action = torch.tensor([0.5])
            elif glucose < 50:
                action = torch.tensor([-0.5])

            dose = self.convert_to_dose(action)
            reward = self.compute_reward(glucose)
            self.replay_buffer.append((s_t, action.item(), reward, s_tp1))

            self.logged_dose[patient_idx].append(dose)
            self.logged_glucose[patient_idx].append(glucose)
            self.logged_reward[patient_idx].append(reward)

            print(f"[step={sample}, patient={patient_idx}] action={action.item():.2f}, glucose={glucose:.1f}, dose={dose:.2f}, reward={reward:.2f}")

            if isinstance(inputs, dict):
                for key in ["uInsulin", "u_insulin"]:
                    if key in inputs and hasattr(inputs[key], "sampled_signal"):
                        inputs[key].sampled_signal[sample, patient_idx] = dose
            elif isinstance(inputs, np.ndarray) and inputs.ndim == 3:
                inputs[patient_idx, 0, sample] = dose

    def convert_to_dose(self, x, min_dose=0.1, max_dose=5.0):
        scaled = (x + 1) / 2 * (max_dose - min_dose) + min_dose
        return float(np.clip(scaled, min_dose, max_dose))

    def compute_reward(self, glucose):
        target = 120.0
        return -abs(glucose - target)

    def plot_results(self):
        # Histogram of actions
        if hasattr(self, 'replay_buffer') and len(self.replay_buffer) > 0:
            actions = [a for (_, a, _, _) in self.replay_buffer]
            plt.figure(figsize=(6, 3))
            plt.hist(actions, bins=30, color='purple', alpha=0.7)
            plt.title("Histogram of Sampled Actions")
            plt.xlabel("Action Value")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("results/action_distribution.png")
            plt.close()

        time = np.linspace(0, 1440, len(self.logged_glucose[0])) / 60

        # Glucose plot
        fig, ax = plt.subplots(figsize=(12, 4))
        for pid in range(len(self.logged_glucose)):
            glucose_arr = np.array(self.logged_glucose[pid])
            glucose_arr[glucose_arr == 0] = np.nan
            ax.plot(time, glucose_arr, alpha=0.4)
        ax.axhspan(70, 180, color="green", alpha=0.1, label="Target Range")
        ax.axhline(180, color="red", linestyle="--", linewidth=1, label="Upper Limit (180)")
        ax.set_ylim(50, 500)
        ax.set_xlabel("Time (hrs)")
        ax.set_ylabel("CGM [mg/dL]")
        ax.set_title("Glucose Levels for All Patients")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.savefig("results/glucose_all_patients.png")
        plt.show()
        plt.close()

        # Insulin plot
        plt.figure(figsize=(12, 4))
        for pid in range(len(self.logged_dose)):
            plt.plot(time, self.logged_dose[pid], alpha=0.4)
        plt.xlabel("Time (hrs)")
        plt.ylabel("Insulin Dose [U/hr]")
        plt.title("Insulin Doses for All Patients")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("results/insulin_all_patients.png")
        plt.close()

        # Reward plot
        plt.figure(figsize=(12, 6))
        for pid in range(len(self.logged_reward)):
            plt.plot(time, self.logged_reward[pid], alpha=0.4)
        plt.xlabel("Time (hrs)")
        plt.ylabel("Reward")
        plt.title("Reward Over Time for All Patients")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("results/reward_all_patients.png")
        plt.close()

        # Avg insulin
        avg_dose = np.mean(self.logged_dose, axis=0)
        plt.figure(figsize=(10, 3))
        plt.plot(time, avg_dose, color='black', label='Average Dose')
        plt.xlabel("Time (hrs)")
        plt.ylabel("Insulin Dose [U/hr]")
        plt.title("Average Insulin Dose")
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        plt.savefig("results/insulin_avg.png")
        plt.show()
        plt.close()

        # Avg glucose
        avg_glucose = np.mean(self.logged_glucose, axis=0)
        plt.figure(figsize=(10, 3))
        plt.plot(time, avg_glucose, color='black', label='Average Glucose')
        plt.axhline(180, color="red", linestyle="--", linewidth=1)
        plt.ylim(bottom=0, top=500)
        plt.xlabel("Time (hrs)")
        plt.ylabel("CGM [mg/dL]")
        plt.title("Average Glucose Level")
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        plt.savefig("results/glucose_avg.png")
        plt.show()
        plt.close()
