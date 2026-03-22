"""Metrics for multiclass classification and CLIP-style retrieval."""
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def youdens_j(y_true, y_pred, num_classes):
    """
    Compute Youden's J statistic for multiclass classification.
    
    J = (1/(K-1)) * (K * balanced_accuracy - 1)
    where balanced_accuracy = (1/K) * sum(recall_i for i in classes)
    
    Returns:
        float: Youden's J in [0, 1], where 0=random, 1=perfect
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    per_class_recall = np.diag(cm) / (cm.sum(axis=1) + 1e-10)
    balanced_acc = per_class_recall.mean()
    j = (num_classes * balanced_acc - 1) / (num_classes - 1) if num_classes > 1 else balanced_acc
    return j


def compute_metrics(y_true, y_pred, num_classes, per_class=False):
    """
    Compute metrics for multiclass classification.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Number of classes
        per_class: If True, include per-class precision/recall/f1
        
    Returns:
        dict: Metrics including accuracy, youdens_j, and optionally per-class metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "youdens_j": youdens_j(y_true, y_pred, num_classes)
    }
    
    if per_class:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=range(num_classes), zero_division=0
        )
        metrics["per_class_precision"] = precision
        metrics["per_class_recall"] = recall
        metrics["per_class_f1"] = f1
    
    return metrics


def compute_clip_class_metrics(
    color_embeds: torch.Tensor,
    class_text_embeds: torch.Tensor,
    labels: torch.Tensor,
) -> dict:
    """
    Compute ColorCLIP retrieval metrics via class-level evaluation.

    Each test color embedding is ranked against K unique class text
    prototypes.

    Args:
        color_embeds:      (N, D) L2-normalized color embeddings (test samples)
        class_text_embeds: (K, D) L2-normalized text embeddings (one per class)
        labels:            (N,)   ground-truth class indices in [0, K)

    Returns:
        dict with scalar metrics and per-class vectors:
        - r_at_1, r_at_5, r_at_10, median_rank: scalar retrieval metrics
        - per_class_rank:   (K,) mean rank per class (label_encoder order)
        - per_class_cosine: (K,) mean cosine similarity per class
    """
    k = class_text_embeds.shape[0]

    # (N, K) similarity: row i = scores for test sample i against all classes
    logits = color_embeds @ class_text_embeds.t()

    gt_scores = logits[torch.arange(logits.shape[0]), labels]  # (N,)
    ranks = (logits >= gt_scores.unsqueeze(1)).sum(dim=1)      # 1-based rank

    # Per-class aggregation: mean rank and mean cosine similarity
    counts = torch.zeros(k).scatter_add_(0, labels, torch.ones_like(labels, dtype=torch.float))
    counts.clamp_(min=1)
    per_class_rank = torch.zeros(k).scatter_add_(0, labels, ranks.float()) / counts
    per_class_cosine = torch.zeros(k).scatter_add_(0, labels, gt_scores) / counts

    return {
        "r_at_1": (ranks == 1).float().mean().item(),
        "r_at_5": (ranks <= 5).float().mean().item(),
        "r_at_10": (ranks <= 10).float().mean().item(),
        "median_rank": ranks.float().median().item(),
        "per_class_rank": per_class_rank,
        "per_class_cosine": per_class_cosine,
    }
