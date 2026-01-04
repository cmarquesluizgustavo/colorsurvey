import os
import sys
import time
import argparse

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

# Ensure project root is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ML.data_prep import load_and_preprocess_data
from ML.utils import set_seed, load_config, save_config, ExperimentLogger


class XGBoostTrainer:
    def __init__(self, config, logger):
        try:
            import xgboost as xgb
            self.xgb = xgb
        except ImportError:
            raise RuntimeError("xgboost is not installed. Please add it to requirements and install.")

        self.config = config
        self.logger = logger
        set_seed(config["seed"])

        # Load and split data
        X, y, le = load_and_preprocess_data(
            config["data"]["csv_path"], 
            top_n=config["data"]["top_n_colors"],
            balance_strategy=config["data"].get("balance_strategy", "none"),
            balance_ratio=config["data"].get("balance_ratio", 1.0),
            random_state=config["seed"]
        )
        if X is None or y is None:
            raise ValueError("No data loaded; aborting training.")

        self.num_classes = len(le.classes_)
        self.le = le

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=config["data"]["test_size"],
            random_state=config["seed"],
            stratify=y,
        )

        print(f"Train samples: {len(self.X_train):,}, Test samples: {len(self.X_test):,}")
        print(f"Number of classes: {self.num_classes}")

        # Create DMatrix objects for efficient XGBoost training
        self.dtrain = self.xgb.DMatrix(self.X_train, label=self.y_train)
        self.dtest = self.xgb.DMatrix(self.X_test, label=self.y_test)

        # Setup model config
        model_cfg = config.get("model", {})
        self.params = {
            "objective": model_cfg.get("objective", "multi:softmax"),
            "num_class": self.num_classes,
            "max_depth": model_cfg.get("max_depth", 6),
            "learning_rate": model_cfg.get("learning_rate", 0.1),
            "subsample": model_cfg.get("subsample", 0.8),
            "colsample_bytree": model_cfg.get("colsample_bytree", 0.8),
            "lambda": model_cfg.get("reg_lambda", 1.0),
            "tree_method": model_cfg.get("tree_method", "hist"),
            "seed": config["seed"],
            "verbosity": 0,  # Suppress default output
        }

        self.trees_per_epoch = config["training"].get("trees_per_epoch", 10)
        self.num_epochs = config["training"].get("num_epochs", 20)
        self.eval_every = config["training"].get("eval_every", 1)
        self.save_every = config["training"].get("save_every", 5)

        self.models_dir = os.path.join(logger.log_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)

        self.bst = None

    def evaluate(self):
        """Evaluate model on test set"""
        y_pred = self.bst.predict(self.dtest)
        acc = accuracy_score(self.y_test, y_pred)
        
        # Also compute train accuracy
        y_train_pred = self.bst.predict(self.dtrain)
        train_acc = accuracy_score(self.y_train, y_train_pred)
        
        return train_acc, acc

    def train(self):
        print(f"\nStarting training for {self.num_epochs} epochs ({self.trees_per_epoch} trees/epoch)...")
        print(f"Total boosting rounds: {self.num_epochs * self.trees_per_epoch}")
        
        global_step = 0
        total_trees = 0

        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            print(f"\n--- Epoch {epoch + 1}/{self.num_epochs} ---")

            # Train for trees_per_epoch boosting rounds
            self.bst = self.xgb.train(
                self.params,
                self.dtrain,
                num_boost_round=self.trees_per_epoch,
                xgb_model=self.bst,  # Continue from previous model
                verbose_eval=False,
            )
            
            total_trees += self.trees_per_epoch
            global_step += 1
            epoch_time = time.time() - epoch_start

            # Periodic evaluation
            if (epoch + 1) % self.eval_every == 0 or epoch == self.num_epochs - 1:
                eval_start = time.time()
                train_acc, test_acc = self.evaluate()
                eval_time = time.time() - eval_start
                
                print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s (eval: {eval_time:.2f}s)")
                print(f"  Trees so far: {total_trees} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

                # Log metrics
                self.logger.log_metrics(
                    global_step,
                    cycle=epoch,
                    metrics={
                        "step_type": "Train",
                        "epoch_in_step": epoch,
                        "step_time": epoch_time,
                        "accuracy": float(test_acc),
                        "train_accuracy": float(train_acc),
                        "total_trees": total_trees,
                    },
                )
            else:
                print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")

            # Periodic model saving
            if (epoch + 1) % self.save_every == 0 or epoch == self.num_epochs - 1:
                model_path = os.path.join(self.models_dir, f"xgb_model_epoch_{epoch+1}.json")
                self.bst.save_model(model_path)
                print(f"  Saved checkpoint: {model_path}")

        # Final evaluation
        print("\n=== Final Evaluation ===")
        train_acc, test_acc = self.evaluate()
        print(f"Final Train Accuracy: {train_acc:.4f}")
        print(f"Final Test Accuracy: {test_acc:.4f}")

        # Save final model
        final_model_path = os.path.join(self.models_dir, "xgb_model_final.json")
        self.bst.save_model(final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")

        # Save label classes
        classes_path = os.path.join(self.logger.log_dir, "label_classes.json")
        import json
        with open(classes_path, "w") as f:
            json.dump(self.le.classes_.tolist(), f, indent=2)
        print(f"Label classes saved to: {classes_path}")


def main():
    parser = argparse.ArgumentParser(description="Train vanilla XGBoost classifier on color survey data")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("experiments", "xgb_config.json"),
        help="Path to XGBoost config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Create logger with experiment name and persist the config
    logger = ExperimentLogger("experiments/runs", config.get("experiment_name"))
    save_config(config, os.path.join(logger.log_dir, "config.json"))

    trainer = XGBoostTrainer(config, logger)
    trainer.train()
    logger.close()


if __name__ == "__main__":
    main()
