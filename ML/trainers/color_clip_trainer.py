import os
import time
import torch
import torch.optim as optim

from ML.trainers.base import BaseTrainer, TrainerFactory
from ML.models.clip_models import ColorCLIPModel
from ML.losses.clip_loss import CLIPInfoNCELoss
from ML.metrics import compute_clip_metrics
from ML.utils import get_device


class ColorCLIPTrainer(BaseTrainer):
    """
    Trainer for ColorCLIP: contrastive dual-encoder alignment of OKLCH
    colors and Bag-of-Words text descriptions via symmetric InfoNCE loss.
    """

    def __init__(self, config, data_bundle, logger):
        super().__init__(config, data_bundle, logger)
        self.device = get_device()
        self.setup()

    # ------------------------------------------------------------------
    # BaseTrainer interface
    # ------------------------------------------------------------------

    def setup(self):
        """Initialize model, optimizer, and loss."""
        self.train_loader = self.data_bundle["train_loader"]
        self.test_loader = self.data_bundle["test_loader"]
        vocab_size = self.data_bundle["vocab_size"]

        model_cfg = self.config["model"]
        self.model = ColorCLIPModel(
            vocab_size=vocab_size,
            embed_dim=model_cfg["embed_dim"],
        ).to(self.device)

        self.loss_fn = CLIPInfoNCELoss(
            temperature=self.config["training"].get("temperature", 0.07)
        ).to(self.device)

        # Single optimizer for model + learnable temperature
        lr = self.config["training"]["lr"]
        self.optimizer = optim.AdamW(
            list(self.model.parameters()) + list(self.loss_fn.parameters()),
            lr=lr,
            weight_decay=self.config["training"].get("weight_decay", 0.01),
        )

        self.epochs = self.config["training"]["epochs"]
        self.max_eval_samples = self.config["training"].get("max_eval_samples", 5000)
        self.models_dir = os.path.join(self.logger.log_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)

        # Early stopping state
        es_cfg = self.config["training"].get("early_stopping", {})
        self.patience = es_cfg.get("patience", 0)       # 0 = disabled
        self.min_delta = es_cfg.get("min_delta", 0.0)
        self.best_r1 = -1.0
        self.patience_counter = 0

    def train(self):
        """Standard epoch loop: forward, InfoNCE loss, backward, eval."""
        print(f"Starting ColorCLIP training for {self.epochs} epochs "
              f"on {self.device}...")

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(epoch)
            clip_metrics = self.evaluate()

            r1 = clip_metrics["r_at_1"]
            temp = self.loss_fn.log_temperature.exp().item()
            print(
                f"Epoch {epoch:3d}/{self.epochs} | loss: {train_loss:.4f} | "
                f"R@1: {r1:.4f}  R@5: {clip_metrics['r_at_5']:.4f}  "
                f"Med.Rank: {clip_metrics['median_rank']:.1f}  "
                f"ΔE: {clip_metrics['delta_e']:.4f}  τ: {temp:.4f}"
            )

            self.logger.log_metrics(epoch, epoch, {
                "step_type": "Train",
                "epoch_in_step": epoch,
                "total_loss": train_loss,
                "r_at_1": r1,
                "r_at_5": clip_metrics["r_at_5"],
                "r_at_10": clip_metrics["r_at_10"],
                "median_rank": clip_metrics["median_rank"],
                "delta_e": clip_metrics["delta_e"],
                "temperature": temp,
            })

            if self._check_early_stopping(r1, epoch):
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best R@1: {self.best_r1:.4f})")
                self._restore_best()
                break
        else:
            # No early stopping — save at end
            self.save_model()

        print("\nFinal evaluation:")
        self._print_metrics(self.evaluate())

    def evaluate(self, data_loader=None) -> dict:
        """
        Evaluate on a subsample of the test set (config: max_eval_samples,
        default 5000). Accumulates only that many embeddings on CPU, then
        computes the full N×N ranking in one shot.
        """
        if data_loader is None:
            data_loader = self.test_loader

        self.model.eval()
        all_color, all_text, all_raw = [], [], []
        collected = 0

        with torch.no_grad():
            for colors, bow, _ in data_loader:
                colors = colors.to(self.device)
                bow = bow.to(self.device)
                c_emb, t_emb = self.model(colors, bow)
                all_color.append(c_emb.cpu())
                all_text.append(t_emb.cpu())
                all_raw.append(colors.cpu())
                collected += colors.shape[0]
                if collected >= self.max_eval_samples:
                    break

        n = self.max_eval_samples
        color_embeds = torch.cat(all_color)[:n]
        text_embeds = torch.cat(all_text)[:n]
        raw_colors = torch.cat(all_raw)[:n]

        return compute_clip_metrics(color_embeds, text_embeds, raw_colors)

    def save_model(self):
        """Save model weights and config."""
        path = os.path.join(self.models_dir, "color_clip_model.pth")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "loss_fn_state_dict": self.loss_fn.state_dict(),
            "config": self.config,
        }, path)
        print(f"Model saved to {path}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> float:
        """Run one training epoch. Returns mean loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = len(self.train_loader)

        t0 = time.time()
        for batch_idx, (colors, bow, labels) in enumerate(self.train_loader, 1):
            colors = colors.to(self.device)
            bow = bow.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            color_embeds, text_embeds = self.model(colors, bow)
            loss = self.loss_fn(color_embeds, text_embeds, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % max(1, n_batches // 5) == 0 or batch_idx == n_batches:
                print(
                    f"\rEpoch {epoch} [{batch_idx}/{n_batches}] "
                    f"loss: {total_loss / batch_idx:.4f}  "
                    f"({time.time() - t0:.1f}s)",
                    end="",
                    flush=True,
                )

        print()
        return total_loss / n_batches

    def _check_early_stopping(self, r1: float, epoch: int) -> bool:
        """Returns True if training should stop."""
        if self.patience <= 0:
            return False

        if r1 > self.best_r1 + self.min_delta:
            self.best_r1 = r1
            self.patience_counter = 0
            # Save best checkpoint
            path = os.path.join(self.models_dir, "best_color_clip_model.pth")
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "loss_fn_state_dict": self.loss_fn.state_dict(),
                "config": self.config,
            }, path)
            print(f"  [✓ Best] R@1: {r1:.4f} (saved)")
            return False

        self.patience_counter += 1
        return self.patience_counter >= self.patience

    def _restore_best(self):
        """Load best checkpoint if it exists."""
        path = os.path.join(self.models_dir, "best_color_clip_model.pth")
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.loss_fn.load_state_dict(ckpt["loss_fn_state_dict"])
            print(f"Restored best model (R@1: {self.best_r1:.4f})")
        self.save_model()

    @staticmethod
    def _print_metrics(metrics: dict):
        print(
            f"  R@1: {metrics['r_at_1']:.4f}  R@5: {metrics['r_at_5']:.4f}  "
            f"R@10: {metrics['r_at_10']:.4f}  Med.Rank: {metrics['median_rank']:.1f}  "
            f"ΔE: {metrics['delta_e']:.4f}"
        )


# Self-register with the factory
TrainerFactory.register("color_clip", ColorCLIPTrainer)
