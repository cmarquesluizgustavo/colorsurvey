import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from ML.trainers.base import BaseTrainer, TrainerFactory
from ML.metrics import compute_metrics
from ML.models.choice_models import BasicChoiceModel, MLPChoiceModel
from ML.models.embedding_models import EmbeddingModel
from ML.losses.triplet_losses import VanillaTripletLoss, ConditionalTripletLoss, SoftNearestNeighborLoss
from ML.utils import get_device


class MetricLearningTrainer(BaseTrainer):
    """Trainer for metric learning with EM-style alternating optimization."""
    
    def __init__(self, config, data_bundle, logger):
        super().__init__(config, data_bundle, logger)
        self.device = get_device()
        self.setup()
    
    def setup(self):
        """Initialize models, optimizers, and loss functions."""
        # Extract data
        self.train_loader = self.data_bundle["train_loader"]
        self.test_loader = self.data_bundle["test_loader"]
        self.num_classes = self.data_bundle["num_classes"]
        
        # Models
        self.embedding_model = EmbeddingModel(
            embedding_dim=self.config["model"]["embedding_dim"]
        ).to(self.device)
        
        self.choice_model = self._get_choice_model(
            self.config["model"]["choice_model_type"],
            self.config["model"]["embedding_dim"],
            self.num_classes,
            self.config["model"]["hidden_dim"]
        ).to(self.device)
        
        # Loss functions
        self.triplet_loss_fn = self._get_loss_function(
            self.config["training"]["loss_fn"],
            self.choice_model,
            self.config["training"].get("margin"),
            self.config["training"].get("temperature")
        ).to(self.device)
        
        self.ce_loss_fn = nn.CrossEntropyLoss().to(self.device)
        
        # Optimizers
        self.opt_theta = optim.Adam(
            self.embedding_model.parameters(),
            lr=self.config["training"]["lr_embedding"]
        )
        self.opt_phi = optim.Adam(
            self.choice_model.parameters(),
            lr=self.config["training"]["lr_classifier"]
        )
        
        self.lambda_val = self.config["training"]["lambda"]
        self.models_dir = os.path.join(self.logger.log_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Early stopping configuration
        es_config = self.config["training"].get("early_stopping", {})
        self.stop_metric = es_config.get("metric", "youdens_j")
        self.patience = es_config.get("patience", 5)
        self.min_delta = es_config.get("min_delta", 0.0)
        self.stop_at_cycle = es_config.get("at_cycle_end", True)
        
        # Early stopping state
        self.best_metric = -float('inf')
        self.patience_counter = 0
        self.best_epoch_info = None
    
    def _get_choice_model(self, model_type, embedding_dim, num_classes, hidden_dim):
        """Get choice model by type."""
        if model_type == "basic":
            return BasicChoiceModel(embedding_dim, num_classes)
        elif model_type == "mlp":
            return MLPChoiceModel(embedding_dim, num_classes, hidden_dim)
        raise ValueError(f"Unknown choice model type: {model_type}")
    
    def _get_loss_function(self, loss_name, choice_model, margin, temperature):
        """Get loss function by name."""
        if loss_name == "vanilla_triplet":
            return VanillaTripletLoss(margin=margin)
        elif loss_name == "conditional_triplet":
            return ConditionalTripletLoss(margin=margin, choice_model=choice_model)
        elif loss_name == "snnl":
            return SoftNearestNeighborLoss(temperature=temperature)
        raise ValueError(f"Unknown loss function: {loss_name}")
    
    def _check_early_stopping(self, test_metrics, global_step, cycle):
        """Check early stopping condition and save best models to disk."""
        current_value = test_metrics.get(self.stop_metric, 0)
        
        if current_value > self.best_metric + self.min_delta:
            self.best_metric = current_value
            self.patience_counter = 0
            self.best_epoch_info = {"step": global_step, "cycle": cycle, "metric": current_value}
            
            # Save best models to disk
            torch.save(
                self.embedding_model.state_dict(),
                os.path.join(self.models_dir, "best_embedding.pth")
            )
            torch.save(
                self.choice_model.state_dict(),
                os.path.join(self.models_dir, "best_choice.pth")
            )
            print(f"  [✓ Best] {self.stop_metric}: {current_value:.4f} (saved to disk)")
            return False
        else:
            self.patience_counter += 1
            print(f"  [Patience {self.patience_counter}/{self.patience}] Best {self.stop_metric}: {self.best_metric:.4f}")
            return self.patience_counter >= self.patience
    
    def evaluate(self, data_loader=None, per_class=False):
        """Evaluate model on given data loader (defaults to test set)."""
        if data_loader is None:
            data_loader = self.test_loader
        
        self.embedding_model.eval()
        self.choice_model.eval()
        
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                z = self.embedding_model(batch_x)
                logits = self.choice_model(z)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        return compute_metrics(all_labels, all_preds, self.num_classes, per_class)
    
    def _run_m_step(self, cycle, epochs_per_step, global_step):
        """Execute M-Step: train classifier with frozen embeddings."""
        print("M-Step: Training Classifier...")
        for p in self.embedding_model.parameters():
            p.requires_grad = False
        for p in self.choice_model.parameters():
            p.requires_grad = True
        
        for epoch in range(epochs_per_step):
            epoch_start = time.time()
            global_step += 1
            total_loss = 0
            
            for batch_idx, (batch_x, batch_y) in enumerate(self.train_loader, 1):
                if batch_idx % 1000 == 0 or batch_idx == len(self.train_loader):
                    print(f"\rM-Step Epoch {epoch+1} - Batch {batch_idx}/{len(self.train_loader)}", end="", flush=True)
                
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.opt_phi.zero_grad()
                
                with torch.no_grad():
                    z = self.embedding_model(batch_x)
                
                logits = self.choice_model(z)
                loss = self.ce_loss_fn(logits, batch_y)
                loss.backward()
                self.opt_phi.step()
                total_loss += loss.item()
            
            epoch_time = time.time() - epoch_start
            avg_loss = total_loss / len(self.train_loader)
            train_metrics = self.evaluate(self.train_loader)
            test_metrics = self.evaluate(self.test_loader)
            
            print(f"\rM-Step Epoch {epoch+1} completed in {epoch_time:.2f}s | Loss: {avg_loss:.4f} | "
                  f"Train Acc: {train_metrics['accuracy']:.4f} (J: {train_metrics['youdens_j']:.4f}) | "
                  f"Test Acc: {test_metrics['accuracy']:.4f} (J: {test_metrics['youdens_j']:.4f})")
            
            self.logger.log_metrics(global_step, cycle, {
                "step_type": "M-Step",
                "epoch_in_step": epoch,
                "step_time": epoch_time,
                "ce_loss": avg_loss,
                "total_loss": avg_loss,
                "train_accuracy": train_metrics['accuracy'],
                "train_youdens_j": train_metrics['youdens_j'],
                "accuracy": test_metrics['accuracy'],
                "youdens_j": test_metrics['youdens_j']
            })
            
            # Check early stopping after each epoch if not waiting for cycle end
            if not self.stop_at_cycle:
                if self._check_early_stopping(test_metrics, global_step, cycle):
                    return global_step, True
        
        return global_step, False
    
    def _run_e_step(self, cycle, epochs_per_step, global_step):
        """Execute E-Step: train embeddings with frozen classifier."""
        print("E-Step: Training Embedding...")
        for p in self.embedding_model.parameters():
            p.requires_grad = True
        for p in self.choice_model.parameters():
            p.requires_grad = False
        
        for epoch in range(epochs_per_step):
            epoch_start = time.time()
            global_step += 1
            total_loss, total_triplet, total_ce = 0, 0, 0
            
            for batch_idx, (batch_x, batch_y) in enumerate(self.train_loader, 1):
                if batch_idx % 1000 == 0 or batch_idx == len(self.train_loader):
                    print(f"\rE-Step Epoch {epoch+1} - Batch {batch_idx}/{len(self.train_loader)}", end="", flush=True)
                
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.opt_theta.zero_grad()
                
                z = self.embedding_model(batch_x)
                loss_triplet = self.triplet_loss_fn(z, batch_y)
                logits = self.choice_model(z)
                loss_ce = self.ce_loss_fn(logits, batch_y)
                loss = loss_triplet + self.lambda_val * loss_ce
                
                loss.backward()
                self.opt_theta.step()
                
                total_loss += loss.item()
                total_triplet += loss_triplet.item()
                total_ce += loss_ce.item()
            
            epoch_time = time.time() - epoch_start
            train_metrics = self.evaluate(self.train_loader)
            test_metrics = self.evaluate(self.test_loader)
            
            print(f"\rE-Step Epoch {epoch+1} completed in {epoch_time:.2f}s | "
                  f"Loss: {total_loss/len(self.train_loader):.4f} | "
                  f"Train Acc: {train_metrics['accuracy']:.4f} (J: {train_metrics['youdens_j']:.4f}) | "
                  f"Test Acc: {test_metrics['accuracy']:.4f} (J: {test_metrics['youdens_j']:.4f})")
            
            self.logger.log_metrics(global_step, cycle, {
                "step_type": "E-Step",
                "epoch_in_step": epoch,
                "step_time": epoch_time,
                "total_loss": total_loss / len(self.train_loader),
                "triplet_loss": total_triplet / len(self.train_loader),
                "ce_loss": total_ce / len(self.train_loader),
                "train_accuracy": train_metrics['accuracy'],
                "train_youdens_j": train_metrics['youdens_j'],
                "accuracy": test_metrics['accuracy'],
                "youdens_j": test_metrics['youdens_j']
            })
            
            # Check early stopping after each epoch if not waiting for cycle end
            if not self.stop_at_cycle:
                if self._check_early_stopping(test_metrics, global_step, cycle):
                    return global_step, True
        
        return global_step, False
    
    def _finalize_training(self):
        """Load best model and perform final evaluation."""
        # Load best models from disk if they exist
        best_embedding_path = os.path.join(self.models_dir, "best_embedding.pth")
        best_choice_path = os.path.join(self.models_dir, "best_choice.pth")
        
        if os.path.exists(best_embedding_path) and os.path.exists(best_choice_path):
            print(f"\nRestoring best model from step {self.best_epoch_info['step']} "
                  f"({self.stop_metric}: {self.best_metric:.4f})")
            self.embedding_model.load_state_dict(torch.load(best_embedding_path))
            self.choice_model.load_state_dict(torch.load(best_choice_path))
        
        print("\nFinal Evaluation:")
        final_metrics = self.evaluate(per_class=True)
        print(f"Final Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"Final Youden's J: {final_metrics['youdens_j']:.4f}")
        if 'per_class_recall' in final_metrics:
            print(f"Per-class recall: mean={final_metrics['per_class_recall'].mean():.4f}, "
                  f"min={final_metrics['per_class_recall'].min():.4f}, max={final_metrics['per_class_recall'].max():.4f}")
        
        self.save_model()
    
    def train(self):
        """Execute EM-style training loop with early stopping."""
        cycles = self.config["training"]["cycles"]
        epochs_per_step = self.config["training"]["epochs_per_step"]
        global_step = 0
        
        print(f"Starting training for {cycles} cycles...")
        if self.patience > 0:
            print(f"Early stopping: metric={self.stop_metric}, patience={self.patience}, "
                  f"min_delta={self.min_delta}, at_cycle_end={self.stop_at_cycle}")
        
        for cycle in range(cycles):
            print(f"\n--- Cycle {cycle+1}/{cycles} ---")
            
            # M-Step
            global_step, should_stop = self._run_m_step(cycle, epochs_per_step, global_step)
            if should_stop:
                print(f"\nEarly stopping triggered after M-Step (patience exceeded)")
                self._finalize_training()
                return
            
            # E-Step
            global_step, should_stop = self._run_e_step(cycle, epochs_per_step, global_step)
            if should_stop:
                print(f"\nEarly stopping triggered after E-Step (patience exceeded)")
                self._finalize_training()
                return
            
            # Check early stopping at end of cycle if configured
            if self.stop_at_cycle and self.patience > 0:
                test_metrics = self.evaluate(self.test_loader)
                if self._check_early_stopping(test_metrics, global_step, cycle):
                    print(f"\nEarly stopping triggered at end of cycle {cycle+1}")
                    self._finalize_training()
                    return
        
        self._finalize_training()
    
    def save_model(self):
        """Save final trained models."""
        torch.save(
            self.embedding_model.state_dict(),
            os.path.join(self.models_dir, "embedding_model.pth")
        )
        torch.save(
            self.choice_model.state_dict(),
            os.path.join(self.models_dir, "choice_model.pth")
        )
        print(f"Models saved to {self.models_dir}")


# Register trainer
TrainerFactory.register("metric_learning", MetricLearningTrainer)
