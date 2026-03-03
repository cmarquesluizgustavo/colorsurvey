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


def compute_clip_metrics(
    color_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    raw_oklch_colors: torch.Tensor,
) -> dict:
    """
    Compute ColorCLIP retrieval metrics on a (sub-)set of test samples.

    Expects inputs already on CPU, with a manageable N (e.g. 5000).
    Builds the full N×N cosine similarity matrix (5k→~100 MB).

    Args:
        color_embeds:     (N, D) L2-normalized color embeddings
        text_embeds:      (N, D) L2-normalized text embeddings
        raw_oklch_colors: (N, 3) normalized OKLCH inputs (for Delta-E)

    Returns:
        dict with keys: r_at_1, r_at_5, r_at_10, median_rank, delta_e
    """
    n = text_embeds.shape[0]

    # Text -> Color similarity: row i = query for text[i]
    logits = text_embeds @ color_embeds.t()  # (N, N)

    # Ground truth: text[i] matches color[i] (the diagonal)
    gt_scores = logits.diag()                # (N,)
    ranks = (logits >= gt_scores.unsqueeze(1)).sum(dim=1)  # 1-based rank

    top1_indices = logits.argmax(dim=1)

    r1 = (ranks == 1).float().mean().item()
    r5 = (ranks <= 5).float().mean().item()
    r10 = (ranks <= 10).float().mean().item()
    median_rank = ranks.float().median().item()

    pred_colors = raw_oklch_colors[top1_indices]
    delta_e = torch.norm(raw_oklch_colors - pred_colors, dim=1).mean().item()

    return {
        "r_at_1": r1,
        "r_at_5": r5,
        "r_at_10": r10,
        "median_rank": median_rank,
        "delta_e": delta_e,
    }
