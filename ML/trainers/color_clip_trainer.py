import json
import os
import time
import numpy as np
import torch
import torch.optim as optim

from ML.trainers.base import BaseTrainer, TrainerFactory
from ML.models.clip_models import ColorCLIPModel
from ML.losses.clip_loss import (
    OriginalCLIPLoss, MaskedCLIPLoss, SupConCLIPLoss, PrototypeCLIPLoss,
)
from ML.data_prep import test_parts
from ML.metrics import compute_clip_class_metrics
from ML.utils import get_device

LOSS_REGISTRY = {
    "original": OriginalCLIPLoss,
    "masked": MaskedCLIPLoss,
    "supcon": SupConCLIPLoss,
    "prototype": PrototypeCLIPLoss,
}


class ColorCLIPTrainer(BaseTrainer):
    """
    Trainer for ColorCLIP: contrastive dual-encoder alignment of colors 
    and Bag-of-Words text descriptions via symmetric InfoNCE loss.
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
        self.bow_embedder = self.data_bundle["bow_embedder"]
        self.label_encoder = self.data_bundle["label_encoder"]
        vocab_size = self.data_bundle["vocab_size"]

        model_cfg = self.config["model"]
        self.model = ColorCLIPModel(
            vocab_size=vocab_size,
            embed_dim=model_cfg["embed_dim"],
            color_hidden_dims=model_cfg.get("color_hidden_dims"),
            text_hidden_dims=model_cfg.get("text_hidden_dims"),
            text_encoder_type=model_cfg.get("text_encoder", "bow"),  # or "learnable", see LearnablePrototypeTextEncoder
            num_classes=self.data_bundle["num_classes"],
        ).to(self.device)

        self.loss_type = self.config["training"].get("loss_type", "original")
        loss_kwargs = {
            "temperature": self.config["training"].get("temperature", 0.07),
            "t2c_eps": self.config["training"].get("t2c_eps", 0),
        }
        # The pixel tolerance is measured on the 0-255 RGB grid, recovered from the
        # model input; OKLCH input would need the raw RGB carried through the loader.
        if loss_kwargs["t2c_eps"] and self.config["data"].get("color_space", "rgb") != "rgb":
            raise ValueError("t2c_eps needs data.color_space = 'rgb'")

        # Prototype mode classifies each color against the K fixed class
        # prototypes, so cache the (K, vocab_size) class BoW table on device.
        if self.loss_type == "prototype":
            class_bow = self.bow_embedder.transform(self.label_encoder.classes_)
            self.class_bow_tensor = torch.FloatTensor(class_bow).to(self.device)
            loss_kwargs["t2c_weight"] = self.config["training"].get("t2c_weight", 1.0)
            if self.config["training"].get("class_weights"):
                if self.config["data"].get("sampler") == "class_balanced":
                    print("⚠️  class_weights AND class_balanced sampler are both on "
                          "— this double-corrects the imbalance. Use one, not both.")
                loss_kwargs["class_weights"] = self._compute_class_weights()

        loss_cls = LOSS_REGISTRY[self.loss_type]
        self.loss_fn = loss_cls(**loss_kwargs).to(self.device)

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
        self.es_metric = es_cfg.get("metric", "r_at_1")
        self.best_metric = -1.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.last_epoch = 0
        self.last_parts = None  # last epoch's scores, set by _save_last_epoch

    def train(self):
        """Standard epoch loop: forward, InfoNCE loss, backward, eval."""
        print(f"Starting ColorCLIP training for {self.epochs} epochs "
              f"on {self.device}...")

        for epoch in range(1, self.epochs + 1):
            self.last_epoch = epoch
            train_loss, train_components = self._train_epoch(epoch)
            clip_metrics = self.evaluate()

            r1 = clip_metrics["r_at_1"]
            temp = self.loss_fn.log_temperature.exp().item()
            print(
                f"Epoch {epoch:3d}/{self.epochs} | loss: {train_loss:.4f} | "
                f"R@1: {r1:.4f}  MRR: {clip_metrics['mrr']:.4f}  "
                f"R@5: {clip_metrics['r_at_5']:.4f}  "
                f"R@10: {clip_metrics['r_at_10']:.4f}  "
                f"Med.Rank: {clip_metrics['median_rank']:.1f}  τ: {temp:.4f}"
            )

            self.logger.log_metrics(epoch, epoch, {
                "step_type": "Train",
                "epoch_in_step": epoch,
                "total_loss": train_loss,
                **train_components,
                "r_at_1": r1,
                "r_at_5": clip_metrics["r_at_5"],
                "r_at_10": clip_metrics["r_at_10"],
                "median_rank": clip_metrics["median_rank"],
                "mrr": clip_metrics["mrr"],
                "avg_rank": clip_metrics["avg_rank"],
                "class_oriented_r_at_1": clip_metrics["class_oriented_r_at_1"],
                "class_oriented_r_at_5": clip_metrics["class_oriented_r_at_5"],
                "class_oriented_r_at_10": clip_metrics["class_oriented_r_at_10"],
                "youdens_j": clip_metrics["youdens_j"],
                "youdens_j_at_5": clip_metrics["youdens_j_at_5"],
                "mean_log_odds": clip_metrics["mean_log_odds"],
                "temperature": temp,
            })

            if self._check_early_stopping(clip_metrics, epoch):
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best {self.es_metric}: {self.best_metric:.4f})")
                self._save_last_epoch()
                self._restore_best()
                break
        else:
            # Ran every epoch. With early stopping on, the best epoch may be an earlier
            # one, so restore it as the early-stopped path does; otherwise keep the last.
            if self.patience > 0:
                self._save_last_epoch()
                self._restore_best()
            else:
                self.best_epoch = self.epochs
                self.save_model()

        print("\nFinal evaluation:")
        # Raw scores are only needed when we are asked to persist them.
        needs_logits = bool(self.config["training"].get("save_ranking")
                            or self.config["training"].get("save_scores"))
        parts = self.evaluate_parts(return_logits=needs_logits)
        self._print_metrics(parts["clean"])
        print(f"R@1 on the test parts: select {parts['select']['r_at_1']:.4f} | "
              f"A {parts['a']['r_at_1']:.4f} | B {parts['b']['r_at_1']:.4f} | "
              f"A+B {parts['clean']['r_at_1']:.4f}")
        self._save_eval_dump(parts["whole"])
        self._save_final_metrics(parts, self.last_parts)

    def evaluate(self, data_loader=None, max_samples=None, return_logits=False) -> dict:
        """
        Class-level evaluation: each test color embedding is ranked
        against K unique class text prototypes.
        """
        if data_loader is None:
            data_loader = self.test_loader
        if max_samples is None:
            max_samples = self.max_eval_samples

        class_text_embeds = self._compute_class_prototypes()
        color_embeds, labels = self._encode_colors(data_loader, max_samples)
        return compute_clip_class_metrics(color_embeds, class_text_embeds, labels,
                                          return_logits=return_logits)

    def evaluate_parts(self, return_logits=False) -> dict:
        """
        Score the current weights on the whole test set and on each part of the
        evaluation rule (data_prep.test_parts, research_plan.md note 10): `select`
        (the first max_eval_samples rows, which chose the epoch), `a`, `b`, and
        `clean` = a + b. Only `whole` carries the raw score matrix.
        """
        class_text_embeds = self._compute_class_prototypes()
        color_embeds, labels = self._encode_colors(self.test_loader, len(self.test_loader.dataset))
        parts = test_parts(labels.numpy(), self.max_eval_samples)
        parts["clean"] = np.sort(np.concatenate([parts["a"], parts["b"]]))
        # Rows per similarity chunk, sized so a chunk's (rows, K) scores stay near 200 MB:
        # 50,000 rows at 5,363 names took a rescoring job past 3.6 GB. Results do not change.
        chunk = min(50_000, max(1_000, 50_000_000 // len(class_text_embeds)))
        out = {"whole": compute_clip_class_metrics(color_embeds, class_text_embeds, labels,
                                                   return_logits=return_logits, chunk_size=chunk)}
        for name, rows in parts.items():
            idx = torch.from_numpy(rows)
            out[name] = compute_clip_class_metrics(color_embeds[idx], class_text_embeds, labels[idx],
                                                   chunk_size=chunk)
        return out

    def _encode_colors(self, data_loader, max_samples):
        """Colour embeddings and labels of the first max_samples rows, in loader order."""
        self.model.eval()
        all_color, all_labels = [], []
        collected = 0

        with torch.no_grad():
            for colors, bow, labels in data_loader:
                colors = colors.to(self.device)
                c_emb, _ = self.model(colors, bow.to(self.device))
                all_color.append(c_emb.cpu())
                all_labels.append(labels)
                collected += colors.shape[0]
                if collected >= max_samples:
                    break

        return torch.cat(all_color)[:max_samples], torch.cat(all_labels)[:max_samples]

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

    def _compute_class_weights(self) -> torch.Tensor:
        """Balanced inverse-frequency weights from training labels: w_k = N / (K * count_k)."""
        y_train = torch.as_tensor(self.data_bundle["y_train"], dtype=torch.long)
        k = len(self.label_encoder.classes_)
        counts = torch.bincount(y_train, minlength=k).float().clamp(min=1)
        return counts.sum() / (k * counts)

    # Scalar metrics of the final evaluation that collect_results.py reads.
    _FINAL_METRIC_KEYS = (
        "r_at_1", "r_at_5", "r_at_10", "median_rank", "avg_rank", "mrr",
        "class_oriented_r_at_1", "class_oriented_r_at_5", "class_oriented_r_at_10",
        "youdens_j", "youdens_j_at_5", "mean_log_odds",
    )

    def _scalars(self, metrics: dict) -> dict:
        out = {k: float(metrics[k]) for k in self._FINAL_METRIC_KEYS}
        out["n_samples"] = int(len(metrics["ranks"]))
        return out

    def _save_final_metrics(self, parts: dict, last_parts: dict = None):
        """
        Save the final evaluation of the best checkpoint to final_metrics.json next to
        metrics.csv (metrics.csv's rows are training epochs on the `select` rows only).

        Top level: the clean rows (A + B), which never chose anything. `parts` holds
        select (kept for checks; never reported), a, b and clean. The whole test set is
        not saved: it mixes the rows that chose the epoch into the score. `epoch` is the
        epoch the evaluated checkpoint comes from. With the last epoch's scores (`last_parts`, new runs
        only), `last` holds them and `last_beats_best` is True when the last epoch
        beats the best on A by more than twice the R@1 sampling error.
        """
        out = self._scalars(parts["clean"])
        out["temperature"] = self.loss_fn.log_temperature.exp().item()
        out["epoch"] = self.best_epoch
        out["parts"] = {name: self._scalars(parts[name]) for name in ("select", "a", "b", "clean")}
        if last_parts is not None:
            out["last"] = {"epoch": self.last_epoch,
                           "parts": {name: self._scalars(last_parts[name]) for name in ("a", "b", "clean")}}
            p, n = parts["a"]["r_at_1"], len(parts["a"]["ranks"])
            margin = 2 * np.sqrt(p * (1 - p) / max(n, 1))
            out["last_beats_best"] = bool(last_parts["a"]["r_at_1"] > p + margin)
        with open(os.path.join(self.logger.log_dir, "final_metrics.json"), "w") as f:
            json.dump(out, f, indent=2)

    def _save_eval_dump(self, metrics: dict):
        """
        Save per-sample evaluation results to eval_dump.npz, so any R@k,
        Youden@k, or a prior-correction sweep can be recomputed offline
        without re-running the model.

        Size = N samples x K classes, compressed (small integer arrays compress
        well; the near-random float scores barely do). Measured on our configs
        (final eval, full test set):
            14c (N=294,910, K=14):  base 1.2 MB | +ranking 2.6 MB  | +scores 10.1 MB
            96c (N=480,719, K=96):  base 3.4 MB | +ranking 40.8 MB | +scores 117.4 MB
        Beware large K: ranking/scores need the full (N, K) score matrix in
        memory and grow roughly linearly with K (797c and 5363c would run
        several hundred MB to a few GB per key, and the ranking's argsort costs
        twice that again). Leave both flags off for those runs; `ranks` alone
        still gives every R@k, and the base keys never need that matrix.
        """
        k = len(self.label_encoder.classes_)
        idx_dtype = np.int8 if k < 128 else np.int16

        dump = {
            "ranks": metrics["ranks"].numpy().astype(np.int16),
            "labels": metrics["labels"].numpy().astype(np.int16),
            "top10": metrics["top10_preds"].numpy().astype(idx_dtype),
            "j_at_1": metrics["per_class_j_at_1"].numpy().astype(np.float32),
            "j_at_5": metrics["per_class_j_at_5"].numpy().astype(np.float32),
        }
        if self.config["training"].get("save_ranking"):
            dump["ranking"] = metrics["logits"].argsort(dim=1, descending=True).numpy().astype(idx_dtype)
        if self.config["training"].get("save_scores"):
            dump["scores"] = metrics["logits"].numpy().astype(np.float16)

        np.savez_compressed(os.path.join(self.logger.log_dir, "eval_dump.npz"), **dump)

    def _compute_class_prototypes(self) -> torch.Tensor:
        """Encode all K unique class names into text embeddings (K, D)."""
        class_names = self.label_encoder.classes_
        bow_matrix = self.bow_embedder.transform(class_names)
        bow_tensor = torch.FloatTensor(bow_matrix).to(self.device)
        self.model.eval()
        with torch.no_grad():
            class_text_embeds = self.model.encode_text(bow_tensor)
        return class_text_embeds.cpu()

    def _train_epoch(self, epoch: int):
        """Run one training epoch. Returns (mean_loss, mean_loss_components)."""
        self.model.train()
        total_loss = 0.0
        component_sums = {}
        n_batches = len(self.train_loader)

        t0 = time.time()
        for batch_idx, (colors, bow, labels) in enumerate(self.train_loader, 1):
            colors = colors.to(self.device)
            bow = bow.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            if self.loss_type == "prototype":
                # Columns are the K class prototypes (encoded with grad each
                # step so the text tower still trains); the per-sample bow is
                # unused. Rows are the batch colors.
                color_embeds = self.model.encode_color(colors)
                text_embeds = self.model.encode_text(self.class_bow_tensor)
            else:
                color_embeds, text_embeds = self.model(colors, bow)
            # 0-255 pixels for t2c_eps (ignored when it is 0); colors are RGB/255
            # whenever t2c_eps is set, which setup() enforces.
            loss = self.loss_fn(color_embeds, text_embeds, labels,
                                rgb=(colors * 255.0).round())
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            for key, value in getattr(self.loss_fn, "last_components", {}).items():
                component_sums[key] = component_sums.get(key, 0.0) + value

            if batch_idx % max(1, n_batches // 5) == 0 or batch_idx == n_batches:
                print(
                    f"\rEpoch {epoch} [{batch_idx}/{n_batches}] "
                    f"loss: {total_loss / batch_idx:.4f}  "
                    f"({time.time() - t0:.1f}s)",
                    end="",
                    flush=True,
                )

        print()
        mean_components = {key: value / n_batches for key, value in component_sums.items()}
        return total_loss / n_batches, mean_components

    def _check_early_stopping(self, metrics: dict, epoch: int) -> bool:
        """Returns True if training should stop."""
        if self.patience <= 0:
            return False

        val = metrics[self.es_metric]
        if val > self.best_metric + self.min_delta:
            self.best_metric = val
            self.best_epoch = epoch
            self.patience_counter = 0
            # Save best checkpoint
            path = os.path.join(self.models_dir, "best_color_clip_model.pth")
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "loss_fn_state_dict": self.loss_fn.state_dict(),
                "config": self.config,
            }, path)
            print(f"  [✓ Best] {self.es_metric}: {val:.4f} (saved)")
            return False

        self.patience_counter += 1
        return self.patience_counter >= self.patience

    def _save_last_epoch(self):
        """Save the last epoch's weights and scores before the best checkpoint replaces them."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "loss_fn_state_dict": self.loss_fn.state_dict(),
            "config": self.config,
            "epoch": self.last_epoch,
        }, os.path.join(self.models_dir, "last_color_clip_model.pth"))
        self.last_parts = self.evaluate_parts()

    def _restore_best(self):
        """Load the best checkpoint if it exists."""
        path = os.path.join(self.models_dir, "best_color_clip_model.pth")
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.loss_fn.load_state_dict(ckpt["loss_fn_state_dict"])
            print(f"Restored best model ({self.es_metric}: {self.best_metric:.4f})")
        self.save_model()

    @staticmethod
    def _print_metrics(metrics: dict):
        pcr = metrics["per_class_rank"]
        pcc = metrics["per_class_cosine"]
        print(
            f"  R@1: {metrics['r_at_1']:.4f}  MRR: {metrics['mrr']:.4f}  "
            f"R@5: {metrics['r_at_5']:.4f}  "
            f"R@10: {metrics['r_at_10']:.4f}  Med.Rank: {metrics['median_rank']:.1f}\n"
            f"  Youden's J: {metrics['youdens_j']:.4f}  "
            f"Log-odds: {metrics['mean_log_odds']:.4f}\n"
            f"  Per-class rank  — mean: {pcr.mean():.1f}  "
            f"min: {pcr.min():.1f}  max: {pcr.max():.1f}\n"
            f"  Per-class cosine — mean: {pcc.mean():.4f}  "
            f"min: {pcc.min():.4f}  max: {pcc.max():.4f}"
        )


# Self-register with the factory
TrainerFactory.register("color_clip", ColorCLIPTrainer)
