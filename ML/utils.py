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

        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)

        # TensorBoard Writer
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # CSV Writer
        self.csv_file = open(os.path.join(self.log_dir, "metrics.csv"), "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["epoch", "cycle", "train_loss", "val_accuracy"])

    def log_metrics(self, epoch, cycle, metrics):
        # Log to TensorBoard
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                self.writer.add_scalar(key, value, epoch)
            else:
                # Optionally log text or ignore
                pass

        # Log to CSV (assuming specific keys for simplicity, or just dumping all)
        # We'll log the main ones to CSV for easy pandas reading later
        self.csv_writer.writerow(
            [epoch, cycle, metrics.get("total_loss", 0), metrics.get("accuracy", 0)]
        )
        self.csv_file.flush()

    def close(self):
        self.writer.close()
        self.csv_file.close()
