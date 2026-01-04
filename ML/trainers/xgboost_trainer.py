import os
import time
import json

from ML.trainers.base import BaseTrainer, TrainerFactory
from ML.metrics import compute_metrics


class XGBoostTrainer(BaseTrainer):
    """Trainer for XGBoost gradient boosting classifier."""
    
    def __init__(self, config, data_bundle, logger):
        super().__init__(config, data_bundle, logger)
        try:
            import xgboost as xgb
            self.xgb = xgb
        except ImportError:
            raise RuntimeError("xgboost is not installed. Install it first.")
        self.setup()
    
    def setup(self):
        """Initialize XGBoost model and parameters."""
        # Extract data
        self.X_train = self.data_bundle["X_train"]
        self.X_test = self.data_bundle["X_test"]
        self.y_train = self.data_bundle["y_train"]
        self.y_test = self.data_bundle["y_test"]
        self.num_classes = self.data_bundle["num_classes"]
        self.le = self.data_bundle["label_encoder"]
        
        print(f"Train samples: {len(self.X_train):,}, Test samples: {len(self.X_test):,}")
        print(f"Number of classes: {self.num_classes}")
        
        # Create DMatrix objects
        self.dtrain = self.xgb.DMatrix(self.X_train, label=self.y_train)
        self.dtest = self.xgb.DMatrix(self.X_test, label=self.y_test)
        
        # Setup parameters
        model_cfg = self.config.get("model", {})
        self.params = {
            "objective": model_cfg.get("objective", "multi:softmax"),
            "num_class": self.num_classes,
            "max_depth": model_cfg.get("max_depth", 6),
            "learning_rate": model_cfg.get("learning_rate", 0.1),
            "subsample": model_cfg.get("subsample", 0.8),
            "colsample_bytree": model_cfg.get("colsample_bytree", 0.8),
            "lambda": model_cfg.get("reg_lambda", 1.0),
            "tree_method": model_cfg.get("tree_method", "hist"),
            "seed": self.config["seed"],
            "verbosity": 0
        }
        
        self.trees_per_epoch = self.config["training"].get("trees_per_epoch", 10)
        self.num_epochs = self.config["training"].get("num_epochs", 20)
        self.eval_every = self.config["training"].get("eval_every", 1)
        self.save_every = self.config["training"].get("save_every", 5)
        
        self.models_dir = os.path.join(self.logger.log_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        self.bst = None
    
    def evaluate(self, per_class=False):
        """Evaluate model on train and test sets."""
        y_pred = self.bst.predict(self.dtest)
        test_metrics = compute_metrics(self.y_test, y_pred, self.num_classes, per_class)
        
        y_train_pred = self.bst.predict(self.dtrain)
        train_metrics = compute_metrics(self.y_train, y_train_pred, self.num_classes, per_class)
        
        return train_metrics, test_metrics
    
    def train(self):
        """Execute XGBoost training loop."""
        print(f"\nStarting training for {self.num_epochs} epochs ({self.trees_per_epoch} trees/epoch)...")
        print(f"Total boosting rounds: {self.num_epochs * self.trees_per_epoch}")
        
        global_step = 0
        total_trees = 0
        
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            print(f"\n--- Epoch {epoch+1}/{self.num_epochs} ---")
            
            # Train for trees_per_epoch boosting rounds
            self.bst = self.xgb.train(
                self.params,
                self.dtrain,
                num_boost_round=self.trees_per_epoch,
                xgb_model=self.bst,
                verbose_eval=False
            )
            
            total_trees += self.trees_per_epoch
            global_step += 1
            epoch_time = time.time() - epoch_start
            
            # Periodic evaluation
            if (epoch + 1) % self.eval_every == 0 or epoch == self.num_epochs - 1:
                eval_start = time.time()
                train_metrics, test_metrics = self.evaluate()
                eval_time = time.time() - eval_start
                
                print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s (eval: {eval_time:.2f}s)")
                print(f"  Trees: {total_trees} | Train Acc: {train_metrics['accuracy']:.4f} (J: {train_metrics['youdens_j']:.4f}) | "
                      f"Test Acc: {test_metrics['accuracy']:.4f} (J: {test_metrics['youdens_j']:.4f})")
                
                self.logger.log_metrics(global_step, epoch, {
                    "step_type": "Train",
                    "epoch_in_step": epoch,
                    "step_time": epoch_time,
                    "accuracy": float(test_metrics['accuracy']),
                    "youdens_j": float(test_metrics['youdens_j']),
                    "train_accuracy": float(train_metrics['accuracy']),
                    "train_youdens_j": float(train_metrics['youdens_j'])
                })
            else:
                print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
            
            # Periodic model saving
            if (epoch + 1) % self.save_every == 0 or epoch == self.num_epochs - 1:
                model_path = os.path.join(self.models_dir, f"xgb_model_epoch_{epoch+1}.json")
                self.bst.save_model(model_path)
                print(f"  Saved checkpoint: {model_path}")
        
        print("\n=== Final Evaluation ===")
        train_metrics, test_metrics = self.evaluate(per_class=True)
        print(f"Final Train Accuracy: {train_metrics['accuracy']:.4f} | Youden's J: {train_metrics['youdens_j']:.4f}")
        print(f"Final Test Accuracy: {test_metrics['accuracy']:.4f} | Youden's J: {test_metrics['youdens_j']:.4f}")
        if 'per_class_recall' in test_metrics:
            print(f"Test per-class recall: mean={test_metrics['per_class_recall'].mean():.4f}, "
                  f"min={test_metrics['per_class_recall'].min():.4f}, max={test_metrics['per_class_recall'].max():.4f}")
        
        self.save_model()
    
    def save_model(self):
        """Save final model and label classes."""
        final_model_path = os.path.join(self.models_dir, "xgb_model_final.json")
        self.bst.save_model(final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")
        
        classes_path = os.path.join(self.logger.log_dir, "label_classes.json")
        with open(classes_path, "w") as f:
            json.dump(self.le.classes_.tolist(), f, indent=2)
        print(f"Label classes saved to: {classes_path}")


# Register trainer
TrainerFactory.register("xgboost", XGBoostTrainer)
