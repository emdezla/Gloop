from helpers import *
from basic_offline_training import SimpleSAC, train_basic
from SAC_training import SACAgent, train_sac_offline

if __name__ == "__main__":
    # Configuration for original complex model
    complex_config = {
        "dataset_path": "datasets/processed/563-train.csv",
        "csv_file": "complex_training_stats.csv",
        "epochs": 200,
        "batch_size": 2048,
        "device": device,
        "cql_weight": 0.008,
        "alpha": 0.2,
        "tau": 0.05
    }
    
    # Configuration for simplified model
    simple_config = {
        "dataset_path": "datasets/processed/563-train.csv",
        "csv_file": "simple_training_stats.csv",
        "epochs": 100,
        "batch_size": 256,
        "device": device
    }
    
    # Configuration for LSTM-based SAC model
    lstm_sac_config = {
        "dataset_path": "datasets/processed/563-train.csv",
        "epochs": 1000,
        "batch_size": 256,
        "save_path": "lstm_sac_final.pth",
        "device": device
    }

    # Choose which model to train (uncomment one)
    
    # Option 1: Train the simplified model (recommended first)
    print("Training simplified SAC model...")
    simple_model = train_basic(
        dataset_path=simple_config["dataset_path"],
        csv_file=simple_config["csv_file"],
        epochs=simple_config["epochs"],
        batch_size=simple_config["batch_size"]
    )
    
    # Option 2: Train the complex model (if needed)
    """
    print("Training complex SACCQL model...")
    complex_model = SACCQL()
    trained_complex_model = train_offline(
        dataset_path=complex_config["dataset_path"],
        model=complex_model,
        csv_file=complex_config["csv_file"],
        epochs=complex_config["epochs"],
        batch_size=complex_config["batch_size"],
        device=complex_config["device"],
        alpha=complex_config["alpha"],
        cql_weight=complex_config["cql_weight"],
        tau=complex_config["tau"]
    )
    torch.save(trained_complex_model.state_dict(), "sac_cql_final.pth")
    """
    
    # Option 3: Train the LSTM-based SAC model
    """
    print("Training LSTM-based SAC model...")
    trained_lstm_sac = train_sac_offline(
        dataset_path=lstm_sac_config["dataset_path"],
        epochs=lstm_sac_config["epochs"],
        batch_size=lstm_sac_config["batch_size"],
        save_path=os.path.splitext(lstm_sac_config["save_path"])[0]
    )
    trained_lstm_sac.save(lstm_sac_config["save_path"])
    print(f"LSTM SAC model saved to {lstm_sac_config['save_path']}")
    """
