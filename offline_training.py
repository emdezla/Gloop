from helpers import *

if __name__ == "__main__":
    # Configuration
    config = {
        "dataset_path": "datasets/processed/563-train.csv",
        "csv_file": "training_stats.csv",
        "epochs": 60,
        "batch_size": 512,  # Increased from 256
        "device": device,
        "cql_weight": 0.1,  # Reduced from 0.25
        "alpha": 0.2,
        "tau": 0.01  # Added target network update rate
    }

    # Initialize and train
    model = SACCQL()
    trained_model = train_offline(
        dataset_path=config["dataset_path"],
        model=model,
        csv_file=config["csv_file"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        device=config["device"],
        alpha=config["alpha"],
        cql_weight=config["cql_weight"],
        tau=config["tau"]
    )
    
    # Save final model
    torch.save(trained_model.state_dict(), "sac_cql_final.pth")
