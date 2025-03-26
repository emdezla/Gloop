import os
import torch
from SAC_training import SACAgent, train_sac_offline
from helpers import DiabetesDataset

def main():
    # Hardcoded configuration
    config = {
        "dataset_path": "datasets/processed/563-train.csv",  # Default dataset
        "epochs": 1000,
        "batch_size": 256,
        "save_path": "sac_final.pth",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "checkpoint_freq": 100,
        "hidden_size": 16,
        "action_scale": 1.0
    }

    print(f"\n🚀 Starting SAC training with configuration:")
    for k, v in config.items():
        print(f"  {k:15}: {v}")

    # Validate dataset path
    if not os.path.exists(config["dataset_path"]):
        raise FileNotFoundError(f"Dataset file {config['dataset_path']} not found")

    # Create save directory if needed
    if config["save_path"]:
        os.makedirs(os.path.dirname(config["save_path"]), exist_ok=True)

    try:
        # Train the model
        trained_agent = train_sac_offline(
            dataset_path=config["dataset_path"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            save_path=os.path.splitext(config["save_path"])[0]  # Remove extension for checkpoints
        )

        # Save final model
        if config["save_path"]:
            trained_agent.save(config["save_path"])
            print(f"\n✅ Training complete! Final model saved to {config['save_path']}")

    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
