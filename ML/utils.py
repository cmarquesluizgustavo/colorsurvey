import json
import random
import numpy as np
import torch
import os
import csv
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def set_seed(seed):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def save_config(config, save_path):
    with open(save_path, "w") as f:
        json.dump(config, f, indent=4)


class ExperimentLogger:
    def __init__(self, log_dir, experiment_name=None):
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Create experiment folder
        experiment_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Create a unique run folder within the experiment
        run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(experiment_dir, f"run_{run_timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)

        # TensorBoard Writer
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # CSV Writer
        self.csv_file = open(os.path.join(self.log_dir, "metrics.csv"), "w", newline="")
        self.csv_writer = csv.DictWriter(
            self.csv_file,
            fieldnames=[
                "timestamp", "global_step", "cycle", "step_type", "epoch_in_step",
                "step_time", "ce_loss", "triplet_loss", "total_loss", "accuracy"
            ]
        )
        self.csv_writer.writeheader()

    def log_metrics(self, global_step, cycle, metrics):
        # Log to TensorBoard
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                self.writer.add_scalar(key, value, global_step)

        # Log to CSV with all metrics
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "global_step": global_step,
            "cycle": cycle,
            "step_type": metrics.get("step_type", ""),
            "epoch_in_step": metrics.get("m_step_epoch", metrics.get("e_step_epoch", "")),
            "step_time": metrics.get("step_time", ""),
            "ce_loss": metrics.get("ce_loss", ""),
            "triplet_loss": metrics.get("triplet_loss", ""),
            "total_loss": metrics.get("total_loss", ""),
            "accuracy": metrics.get("accuracy", "")
        }
        self.csv_writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.writer.close()
        self.csv_file.close()
