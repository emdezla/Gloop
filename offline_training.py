from helpers import *

if __name__ == "__main__":
    # Configuration
    config = {
        "dataset_path": "datasets/processed/563-train.csv",
        "csv_file": "training_stats.csv",
        "epochs": 200,  # Increased for longer training
        "batch_size": 2048,  # Larger batches to reduce variance
        "device": device,
        "cql_weight": 0.008,  # Further reduced to prevent Q-value collapse
        "alpha": 0.2,
        "tau": 0.05  # Increased for faster target updates
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
