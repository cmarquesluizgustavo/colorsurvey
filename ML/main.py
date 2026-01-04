import argparse
import os
import sys

# Enable MPS fallback before importing torch
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ML.data_prep import DataLoader
from ML.trainers import TrainerFactory
from ML.utils import load_config, save_config, validate_config, set_seed, ExperimentLogger, plot_training_metrics


def main():
    """Unified entry point for all ML training."""
    parser = argparse.ArgumentParser(description="Run ML Training Experiment")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate training plots after completion"
    )
    args = parser.parse_args()
    
    # Load and validate config
    print(f"Loading config from {args.config}...")
    config = load_config(args.config)
    validate_config(config)
    
    # Set seed
    set_seed(config["seed"])
    
    # Create logger
    logger = ExperimentLogger(
        "experiments/runs",
        config.get("experiment_name")
    )
    
    # Save config for reproducibility
    save_config(config, os.path.join(logger.log_dir, "config.json"))
    
    # Load data
    print(f"\nTrainer type: {config['trainer_type']}")
    data_loader = DataLoader(config)
    data_bundle = data_loader.load()
    
    # Create and run trainer
    trainer = TrainerFactory.create(
        config["trainer_type"],
        config,
        data_bundle,
        logger
    )
    
    try:
        trainer.train()
    finally:
        logger.close()
    
    print(f"\n✅ Training complete! Results saved to {logger.log_dir}")
    
    # Generate plots if requested
    if args.plot:
        print("\nGenerating training plots...")
        plot_path = plot_training_metrics(logger.log_dir)
        if plot_path:
            print(f"📊 Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
