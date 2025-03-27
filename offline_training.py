from helpers import *
from SAC_training import SACAgent, train_sac_offline

if __name__ == "__main__":
    # Configuration for SAC model
    sac_config = {
        "dataset_path": "datasets/processed/563-train.csv",
        "epochs": 200,
        "batch_size": 256,
        "save_path": "sac_final.pth",
        "device": device
    }
    
    # Train the SAC model
    print("Training SAC model...")
    trained_model = train_sac_offline(
        dataset_path=sac_config["dataset_path"],
        epochs=sac_config["epochs"],
        batch_size=sac_config["batch_size"],
        save_path=sac_config["save_path"]
    )
    print(f"Model saved to {sac_config['save_path']}")
