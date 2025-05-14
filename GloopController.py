import numpy as np
import torch
import sys
import os

# ➕ Dynamically add Gloop repo to path
GLOOP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Gloop"))
if GLOOP_PATH not in sys.path:
    sys.path.append(GLOOP_PATH)

# 📌 Import your trained model
from model.sac_cql import SACCQL

class Controller:
    name = "GloopController"

    def __init__(self, scenario_instance):
        print(">> GloopController initialized")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 🔁 Load SAC-CQL model
        self.model = SACCQL().to(self.device)
        checkpoint_path = os.path.join(GLOOP_PATH, "checkpoints/saccql_trained.pt")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"❌ Checkpoint not found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        print("✅ SAC-CQL model loaded")

    def convert_to_dose(self, x, min_dose=0.0, max_dose=10.0):
        """Scale [-1, 1] model output to actual insulin range."""
        return float(np.clip((x + 1) / 2 * max_dose, min_dose, max_dose))

    def run(self, measurements, states, inputs, sample):
        print(f"[step={sample}] GloopController running...")

        if sample >= states.shape[0]:
            print(f"[step={sample}] sample out of bounds")
            return

        try:
            # ⚙️ Build 8D input vector (match model expectations)
            state = np.array([
                states[sample, 0],  # glucose
                states[sample, 1],  # glucose_derivative
                               # glucose_trend (placeholder)
                states[sample, 2],  # heart_rate
                states[sample, 3],  # hr_derivative
                0.0,                # heart_rate_trend (placeholder)
                states[sample, 4],  # insulin_on_board
                (sample % 1440) / 60.0  # hour of day
            ], dtype=np.float32)

            state_tensor = torch.tensor(state).unsqueeze(0).to(self.device)

            # 🧠 Inference
            with torch.no_grad():
                action = self.model.act(state_tensor)[0]
            dose = self.convert_to_dose(action)

        except Exception as e:
            print(f"[step={sample}] ❌ Model inference error: {e}")
            dose = 0.0

        try:
            # 💉 Inject into simulator
            if isinstance(inputs, dict):
                target = None
                if "u_insulin" in inputs and hasattr(inputs["u_insulin"], "sampled_signal"):
                    target = inputs["u_insulin"]
                elif "uInsulin" in inputs and hasattr(inputs["uInsulin"], "sampled_signal"):
                    target = inputs["uInsulin"]

                if target:
                    target.sampled_signal[sample, 0] = dose
                    print(f"[step={sample}] ✅ Dose injected: {dose:.2f} U/hr")
                else:
                    print(f"[step={sample}] ❌ No insulin input signal found in dict")

            elif isinstance(inputs, np.ndarray):
                inputs[sample, 0] = dose
                print(f"[step={sample}] ✅ Dose injected via array: {dose:.2f} U/hr")

            else:
                print(f"[step={sample}] ❌ Unknown input structure")

        except Exception as e:
            print(f"[step={sample}] ❌ Injection failed: {e}")