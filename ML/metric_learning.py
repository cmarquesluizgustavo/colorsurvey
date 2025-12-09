import os
# Enable MPS fallback for operations not implemented in MPS
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse
import sys

# Add project root to path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ML.data_prep import load_and_preprocess_data, ColorDataset
from ML.models.choice_models import BasicChoiceModel, MLPChoiceModel
from ML.models.embedding_models import EmbeddingModel
from ML.losses.triplet_losses import VanillaTripletLoss
from ML.utils import set_seed, load_config, save_config, ExperimentLogger


def get_choice_model(model_type, embedding_dim, num_classes, hidden_dim=64):
    if model_type == "basic":
        return BasicChoiceModel(embedding_dim=embedding_dim, num_classes=num_classes)
    elif model_type == "mlp":
        return MLPChoiceModel(
            embedding_dim=embedding_dim, num_classes=num_classes, hidden_dim=hidden_dim
        )
    else:
        raise NotImplementedError(f"Unknown choice model type: {model_type}")


def get_loss_function(loss_name, margin=1.0):
    if loss_name == "vanilla_triplet":
        return VanillaTripletLoss(margin=margin)
    else:
        raise NotImplementedError(f"Unknown loss function: {loss_name}")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Trainer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.device = get_device()

        # Data
        self.X, self.y, self.le = load_and_preprocess_data(
            config["data"]["csv_path"], top_n=config["data"]["top_n_colors"]
        )
        self.num_classes = len(self.le.classes_)

        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            test_size=config["data"]["test_size"],
            random_state=config["seed"],
            stratify=self.y,
        )

        self.train_loader = DataLoader(
            ColorDataset(X_train, y_train),
            batch_size=config["data"]["batch_size"],
            shuffle=True,
        )
        self.test_loader = DataLoader(
            ColorDataset(X_test, y_test),
            batch_size=config["data"]["batch_size"],
            shuffle=False,
        )

        # Models
        self.embedding_model = EmbeddingModel(
            embedding_dim=config["model"]["embedding_dim"]
        ).to(self.device)
        self.choice_model = get_choice_model(
            config["model"]["choice_model_type"],
            config["model"]["embedding_dim"],
            self.num_classes,
            config["model"].get("hidden_dim", 64),
        ).to(self.device)

        # Loss
        self.triplet_loss_fn = get_loss_function(
            config["training"]["loss_fn"], margin=config["training"]["margin"]
        ).to(self.device)
        self.ce_loss_fn = nn.CrossEntropyLoss().to(self.device)

        # Optimizers
        self.opt_theta = optim.Adam(
            self.embedding_model.parameters(), lr=config["training"]["lr_embedding"]
        )
        self.opt_phi = optim.Adam(
            self.choice_model.parameters(), lr=config["training"]["lr_classifier"]
        )

        # Config params
        self.lambda_val = config["training"]["lambda"]

        # Create models directory
        self.models_dir = os.path.join(logger.log_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)

    def evaluate(self, epoch):
        self.embedding_model.eval()
        self.choice_model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                z = self.embedding_model(batch_x)
                logits = self.choice_model(z)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.detach().cpu().numpy())
                all_labels.extend(batch_y.detach().cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        print(f"Evaluation Accuracy: {acc:.4f}")
        return acc

    def train(self):
        cycles = self.config["training"]["cycles"]
        epochs_per_step = self.config["training"]["epochs_per_step"]

        print(f"Starting training loop for {cycles} cycles...")
        global_step = 0

        for cycle in range(cycles):
            print(f"\n--- Cycle {cycle+1}/{cycles} ---")

            # --- M-Step: Train Classifier ---
            print("M-Step: Training Classifier...")
            # Freeze Embedding, Unfreeze Classifier
            for p in self.embedding_model.parameters():
                p.requires_grad = False
            for p in self.choice_model.parameters():
                p.requires_grad = True

            for epoch in range(epochs_per_step):
                global_step += 1
                total_loss_m = 0
                for batch_x, batch_y in self.train_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    self.opt_phi.zero_grad()

                    with torch.no_grad():
                        z = self.embedding_model(batch_x)

                    logits = self.choice_model(z)
                    loss = self.ce_loss_fn(logits, batch_y)
                    loss.backward()
                    self.opt_phi.step()
                    total_loss_m += loss.item()

                print("M-Step Epoch {} completed.".format(epoch + 1))
                # Log M-Step
                self.logger.log_metrics(
                    global_step,
                    cycle,
                    {
                        "step_type": "M-Step",
                        "m_step_epoch": epoch,
                        "ce_loss": total_loss_m / len(self.train_loader),
                    },
                )

            # --- E-Step: Train Embedding ---
            print("E-Step: Training Embedding...")
            # Unfreeze Embedding, Freeze Classifier
            for p in self.embedding_model.parameters():
                p.requires_grad = True
            for p in self.choice_model.parameters():
                p.requires_grad = False

            for epoch in range(epochs_per_step):
                global_step += 1
                total_loss_e = 0
                for batch_x, batch_y in self.train_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    self.opt_theta.zero_grad()

                    z = self.embedding_model(batch_x)

                    # Triplet Loss
                    loss_triplet = self.triplet_loss_fn(z, batch_y)

                    # Cross Entropy Loss
                    logits = self.choice_model(z)
                    loss_ce = self.ce_loss_fn(logits, batch_y)

                    # Combined Loss
                    loss = loss_triplet + self.lambda_val * loss_ce

                    loss.backward()
                    self.opt_theta.step()
                    total_loss_e += loss.item()

                # Evaluate and Log
                acc = self.evaluate(global_step)
                self.logger.log_metrics(
                    global_step,
                    cycle,
                    {
                        "step_type": "E-Step",
                        "e_step_epoch": epoch,
                        "total_loss": total_loss_e / len(self.train_loader),
                        "accuracy": acc,
                        "triplet_loss": loss_triplet.item(),
                        "ce_loss": loss_ce.item(),
                    },
                )
                print("E-Step Epoch {} completed.".format(epoch + 1))

        # Final Evaluation
        print("\nFinal Evaluation:")
        self.evaluate(global_step)

        # Save models
        torch.save(
            self.embedding_model.state_dict(),
            os.path.join(self.models_dir, "embedding_model.pth"),
        )
        torch.save(
            self.choice_model.state_dict(),
            os.path.join(self.models_dir, "choice_model.pth"),
        )
        print(f"Models saved to {self.models_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Metric Learning Experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/config.json",
        help="Path to config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    # Create logger
    logger = ExperimentLogger("experiments/runs", config.get("experiment_name"))

    # Save config to run dir for reproducibility
    save_config(config, os.path.join(logger.log_dir, "config.json"))

    trainer = Trainer(config, logger)
    trainer.train()
    logger.close()
