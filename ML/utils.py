import json
import random
import numpy as np
import torch
import os
import csv
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime, timezone
from pathlib import Path


def set_seed(seed):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def save_config(config, save_path):
    """Save configuration to JSON file."""
    with open(save_path, "w") as f:
        json.dump(config, f, indent=4)


def validate_config(config):
    """Validate configuration has required fields."""
    required = ["seed", "data", "model", "training"]
    for field in required:
        if field not in config:
            raise ValueError(f"Config missing required field: {field}")
    
    # Validate trainer_type
    if "trainer_type" not in config:
        raise ValueError("Config missing 'trainer_type' field")
    
    if config["trainer_type"] not in ["metric_learning", "xgboost"]:
        raise ValueError(f"Invalid trainer_type: {config['trainer_type']}")
    
    # Validate data config
    data_required = ["csv_path", "top_n_colors", "test_size"]
    for field in data_required:
        if field not in config["data"]:
            raise ValueError(f"Config data section missing: {field}")
    
    return True


class ExperimentLogger:
    """Logger for experiments with TensorBoard and CSV support."""
    
    def __init__(self, log_dir, experiment_name=None):
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d-%H%M%S")

        experiment_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        
        run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(experiment_dir, f"run_{run_timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Fixed CSV structure with all possible fields
        self.csv_path = os.path.join(self.log_dir, "metrics.csv")
        self.csv_file = open(self.csv_path, "w", newline="")
        self.fieldnames = [
            "timestamp", "global_step", "cycle", "step_type", 
            "epoch_in_step", "step_time", 
            "ce_loss", "triplet_loss", "total_loss", "accuracy", "train_accuracy"
        ]
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames, 
                                         extrasaction='ignore')
        self.csv_writer.writeheader()

    def log_metrics(self, global_step, cycle, metrics):
        """Log metrics to TensorBoard and CSV."""
        # TensorBoard
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                self.writer.add_scalar(key, value, global_step)

        # CSV with fixed structure
        row = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S") + " UTC",
            "global_step": global_step,
            "cycle": cycle,
        }
        # Add all metrics, missing ones will be empty
        for key in self.fieldnames[3:]:  # Skip timestamp, global_step, cycle
            row[key] = metrics.get(key, "")
        
        self.csv_writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        """Close logger resources."""
        self.writer.close()
        self.csv_file.close()


def plot_training_metrics(run_dir):
    """
    Plot training metrics from a run directory.
    
    Args:
        run_dir: Path to run directory (string or Path object)
    
    Returns:
        Path to saved plot or None if failed
    """
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: pandas or matplotlib not available, skipping plot")
        return None
    
    run_dir = Path(run_dir)
    csv_path = run_dir / 'metrics.csv'
    
    if not csv_path.exists():
        print(f"Warning: metrics.csv not found in {run_dir}")
        return None
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Determine trainer type and plot
    if 'step_type' in df.columns and df['step_type'].str.contains('Step').any():
        return _plot_metric_learning(df, run_dir)
    else:
        return _plot_xgboost(df, run_dir)


def _plot_metric_learning(df, save_dir):
    """Plot metric learning training curves."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Metric Learning Training Progress', fontsize=16, fontweight='bold')
    
    # Separate M-Step and E-Step
    m_step = df[df['step_type'] == 'M-Step'].copy()
    e_step = df[df['step_type'] == 'E-Step'].copy()
    
    # Plot 1: Total Loss (both steps)
    ax = axes[0, 0]
    if not m_step.empty and 'total_loss' in m_step.columns:
        ax.plot(m_step['global_step'], m_step['total_loss'], 
                marker='o', label='M-Step', linewidth=2, markersize=8)
    if not e_step.empty and 'total_loss' in e_step.columns:
        ax.plot(e_step['global_step'], e_step['total_loss'], 
                marker='s', label='E-Step', linewidth=2, alpha=0.7, markersize=8)
    ax.set_xlabel('Global Step', fontsize=11)
    ax.set_ylabel('Total Loss', fontsize=11)
    ax.set_title('Total Loss Over Training', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: CE Loss (both steps)
    ax = axes[0, 1]
    if not m_step.empty and 'ce_loss' in m_step.columns:
        ax.plot(m_step['global_step'], m_step['ce_loss'], 
                marker='o', label='M-Step', linewidth=2, markersize=8, color='steelblue')
    if not e_step.empty and 'ce_loss' in e_step.columns:
        ax.plot(e_step['global_step'], e_step['ce_loss'], 
                marker='s', label='E-Step', linewidth=2, alpha=0.7, markersize=8, color='coral')
    ax.set_xlabel('Global Step', fontsize=11)
    ax.set_ylabel('Cross-Entropy Loss', fontsize=11)
    ax.set_title('CE Loss Over Training', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Triplet Loss (E-Step only)
    ax = axes[1, 0]
    if not e_step.empty and 'triplet_loss' in e_step.columns:
        ax.plot(e_step['global_step'], e_step['triplet_loss'], 
                marker='s', color='darkgreen', linewidth=2, markersize=8)
        ax.set_xlabel('Global Step', fontsize=11)
        ax.set_ylabel('Triplet Loss', fontsize=11)
        ax.set_title('Triplet Loss (E-Step)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Train vs Test Accuracy
    ax = axes[1, 1]
    test_data = df[df['accuracy'].notna()].copy()
    if not test_data.empty:
        ax.plot(test_data['global_step'], test_data['accuracy'], 
                marker='o', label='Test Acc', linewidth=2, markersize=8, color='navy')
    
    train_data = df[df['train_accuracy'].notna()].copy()
    if not train_data.empty:
        ax.plot(train_data['global_step'], train_data['train_accuracy'], 
                marker='s', label='Train Acc', linewidth=2, markersize=8, 
                color='coral', alpha=0.7, linestyle='--')
    
    ax.set_xlabel('Global Step', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Train vs Test Accuracy', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    output_path = save_dir / 'training_curves.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path


def _plot_xgboost(df, save_dir):
    """Plot XGBoost training curves."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('XGBoost Training Progress', fontsize=16, fontweight='bold')
    
    # Plot 1: Accuracy over epochs
    ax = axes[0]
    if 'accuracy' in df.columns:
        ax.plot(df['global_step'], df['accuracy'], 
                marker='o', label='Test Accuracy', linewidth=2)
    if 'train_accuracy' in df.columns:
        ax.plot(df['global_step'], df['train_accuracy'], 
                marker='s', label='Train Accuracy', linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Plot 2: Training time
    ax = axes[1]
    if 'step_time' in df.columns:
        ax.bar(df['global_step'], df['step_time'], color='steelblue', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Training Time per Epoch')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = save_dir / 'training_curves.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path

