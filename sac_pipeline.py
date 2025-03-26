import argparse
import os
import torch
from SAC_training import SACAgent, train_sac_offline
from helpers import DiabetesDataset, compute_reward_torch

def main():
    parser = argparse.ArgumentParser(description='Train SAC agent on diabetes management dataset')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to training CSV file')
    parser.add_argument('--epochs', type=int, default=1000,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Training batch size')
    parser.add_argument('--save_path', type=str, default='sac_final.pth',
                       help='Path to save final model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training (cuda/cpu)')
    parser.add_argument('--checkpoint_freq', type=int, default=100,
                       help='Frequency (in epochs) for saving checkpoints')
    parser.add_argument('--hidden_size', type=int, default=16,
                       help='LSTM hidden state size')
    parser.add_argument('--action_scale', type=float, default=1.0,
                       help='Scaling factor for action range')

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset file {args.dataset} not found")

    print(f"\n🚀 Starting SAC training with:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device: {args.device}")
    print(f"  Checkpoint frequency: every {args.checkpoint_freq} epochs\n")

    # Create save directory if needed
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    try:
        # Train the model
        trained_agent = train_sac_offline(
            dataset_path=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_path=os.path.splitext(args.save_path)[0]  # Remove extension for checkpoints
        )

        # Save final model
        if args.save_path:
            trained_agent.save(args.save_path)
            print(f"\n✅ Training complete! Final model saved to {args.save_path}")

    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
