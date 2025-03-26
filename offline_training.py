from helpers import *

if __name__ == "__main__":
    # Configuration
    config = {
        "dataset_path": "datasets/processed/563-train.csv",
        "csv_file": "training_stats.csv",
        "epochs": 100,  # Increased from 60 for longer training
        "batch_size": 512,
        "device": device,
        "cql_weight": 0.02,  # Further reduced from 0.03 to prevent Q-value collapse
        "alpha": 0.2,
        "tau": 0.02  # Increased from 0.01 for faster target updates
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
