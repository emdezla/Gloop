from helpers import *
from basic_offline_training import SimpleSAC, train_basic
from SAC_training import SACAgent, train_sac_offline
from SACCQL_training import SACCQL, train_saccql

if __name__ == "__main__":
    # Configuration for consolidated SAC-CQL model
    saccql_config = {
        "dataset_path": "datasets/processed/563-train.csv",
        "csv_file": "saccql_training_stats.csv",
        "epochs": 200,
        "batch_size": 256,
        "device": device,
        "cql_weight": 0.1,
        "save_path": "sac_cql_final.pth"
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
    """
    print("Training simplified SAC model...")
    simple_model = train_basic(
        dataset_path=simple_config["dataset_path"],
        csv_file=simple_config["csv_file"],
        epochs=simple_config["epochs"],
        batch_size=simple_config["batch_size"]
    )
    """
    
    # Option 2: Train the consolidated SAC-CQL model
    print("Training consolidated SAC-CQL model...")
    trained_model = train_saccql(
        dataset_path=saccql_config["dataset_path"],
        epochs=saccql_config["epochs"],
        batch_size=saccql_config["batch_size"],
        save_path=saccql_config["save_path"],
        csv_file=saccql_config["csv_file"],
        device=saccql_config["device"],
        cql_weight=saccql_config["cql_weight"]
    )
    print(f"Model saved to {saccql_config['save_path']}")
    
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
