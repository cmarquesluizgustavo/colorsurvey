"""Metrics for multiclass classification and CLIP-style retrieval."""
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def youdens_j(y_true, y_pred, num_classes):
    """
    Compute Youden's J statistic for multiclass classification.
    
    J = (1/(K-1)) * (K * balanced_accuracy - 1)
    where balanced_accuracy = (1/K) * sum(recall_i for i in classes)
    
    Returns:
        float: Youden's J in [0, 1], where 0=random, 1=perfect
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    totals = np.bincount(y_true_arr, minlength=num_classes)
    hits = np.bincount(y_true_arr[y_true_arr == y_pred_arr], minlength=num_classes)
    per_class_recall = hits / (totals + 1e-10)
    balanced_acc = per_class_recall.mean()
    j = (num_classes * balanced_acc - 1) / (num_classes - 1) if num_classes > 1 else balanced_acc
    return j


def youdens_j_at_k(topk_preds: torch.Tensor, labels: torch.Tensor, k: int, num_classes: int):
    """
    Per-class Youden's J at k: a class c counts as a "positive prediction" for
    a sample when c appears anywhere in that sample's top-k, not just at rank 1.

        J_c@k = sensitivity_c@k - false_positive_rate_c@k
        sensitivity_c@k = fraction of c's own samples with c in their top-k
        FPR_c@k         = fraction of samples of OTHER classes with c in their top-k

    Args:
        topk_preds:  (N, >=k) predicted class indices per sample, best first
        labels:      (N,) ground-truth class indices
        k:           cutoff
        num_classes: K

    Returns:
        (per_class_j, mean_j): (K,) tensor and its scalar mean
    """
    topk = topk_preds[:, :k]
    n = labels.shape[0]

    counts = torch.zeros(num_classes).scatter_add_(0, labels, torch.ones(n))
    hits = (topk == labels.unsqueeze(1)).any(dim=1).float()  # true label in own top-k
    true_pos = torch.zeros(num_classes).scatter_add_(0, labels, hits)
    contains = torch.bincount(topk.flatten(), minlength=num_classes).float()  # any sample with c in its top-k

    sensitivity = true_pos / counts.clamp(min=1)
    false_pos = contains - true_pos
    fpr = false_pos / (n - counts).clamp(min=1)

    per_class_j = sensitivity - fpr
    # Mean over classes that have samples: an absent class has no sensitivity to measure.
    return per_class_j, per_class_j[counts > 0].mean().item()


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
    return_logits: bool = False,
    chunk_size: int = 50_000,
    temperature: Optional[float] = None,
) -> dict:
    """
    Compute ColorCLIP retrieval metrics via class-level evaluation.

    Each test color embedding is ranked against K unique class text
    prototypes.

    Args:
        color_embeds:      (N, D) L2-normalized color embeddings (test samples)
        class_text_embeds: (K, D) L2-normalized text embeddings (one per class)
        labels:            (N,)   ground-truth class indices in [0, K)
        return_logits:     also return the full (N, K) score matrix. Off by
            default because that matrix is the one thing here that does not
            fit at large K (~13 GB at N=600k, K=5363); only callers persisting
            raw scores/rankings need it.
        chunk_size:        rows per similarity chunk (memory/speed knob only;
            results are identical for any value)
        temperature:       optional positive softmax temperature. When provided,
            also computes held-out cross-entropy (log_loss and per_class_log_loss)
            inside the similarity chunk loop.

    Returns:
        dict with scalar metrics, per-class vectors, and per-sample vectors:
        - r_at_1, r_at_5, r_at_10, median_rank, avg_rank: scalar retrieval metrics
        - class_oriented_r_at_1/5/10: scalar, each class weighted equally
          regardless of size (mean of per-class R@k, not per-sample R@k)
        - per_class_counts: (K,) test sample counts per class
        - per_class_r_at_1: (K,) top-1 recall per class (NaN where count == 0)
        - per_class_rank:   (K,) mean rank per class (label_encoder order)
        - per_class_cosine: (K,) mean cosine similarity per class
        - mrr:              scalar Mean Reciprocal Rank
        - youdens_j:        scalar Youden's J for top-1 predictions
        - youdens_j_at_5:   scalar Youden's J@5 (a "hit" counts anywhere in the top-5)
        - per_class_j_at_1/5: (K,) per-class Youden's J@1 / @5
        - mean_log_odds:    scalar mean log-odds ratio (0 = perfect)
        - log_loss:         scalar mean negative log-likelihood (only with temperature)
        - per_class_log_loss: (K,) mean NLL per class, NaN where absent (only with temperature)
        - ranks:            (N,) 1-based rank of correct class per sample
        - top1_preds:       (N,) predicted class index per sample
        - top10_preds:      (N, min(10, K)) top predicted class indices, best first
        - labels:           (N,) ground-truth class indices (passed through)
        - logits:           (N, K) raw similarity scores — only when
          return_logits=True
    """
    if temperature is not None and temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    k = class_text_embeds.shape[0]
    n = color_embeds.shape[0]

    # Every metric below is an aggregate of small per-sample vectors, so the
    # (N, K) similarity matrix is built one row-chunk at a time and discarded.
    # Materializing it in full is what breaks at large K (~13 GB at K=5363).
    gt_chunks, rank_chunks, top1_chunks, topk_chunks, odds_chunks = [], [], [], [], []
    nll_chunks = []
    logit_chunks = []
    for i in range(0, n, chunk_size):
        chunk_logits = color_embeds[i:i + chunk_size] @ class_text_embeds.t()
        chunk_labels = labels[i:i + chunk_size]
        rows = torch.arange(chunk_logits.shape[0])

        gt = chunk_logits[rows, chunk_labels]
        pred = chunk_logits.argmax(dim=1)
        gt_chunks.append(gt)
        rank_chunks.append((chunk_logits >= gt.unsqueeze(1)).sum(dim=1))  # 1-based rank
        top1_chunks.append(pred)
        topk_chunks.append(chunk_logits.topk(min(10, k), dim=1).indices)
        # Log-odds log(p_pred / p_correct) reduces to the raw logit difference:
        # the softmax denominator is shared by both terms and cancels. Avoids a
        # second (N, K) allocation and is numerically exact (no underflow clamp).
        odds_chunks.append(chunk_logits[rows, pred] - gt)

        if return_logits:
            logit_chunks.append(chunk_logits)

        if temperature is not None:
            if return_logits:
                scaled = chunk_logits / temperature
                nll_chunks.append(scaled.logsumexp(dim=1) - scaled[rows, chunk_labels])
            else:
                chunk_logits.div_(temperature)
                nll_chunks.append(chunk_logits.logsumexp(dim=1) - (gt / temperature))

    gt_scores = torch.cat(gt_chunks)                # (N,)
    ranks = torch.cat(rank_chunks)                  # (N,)
    top1_preds = torch.cat(top1_chunks)             # (N,)
    top10_preds = torch.cat(topk_chunks)            # (N, min(10, K))
    mean_log_odds = torch.cat(odds_chunks).mean().item()

    # Scalar metrics
    mrr = (1.0 / ranks.float()).mean().item()
    j = youdens_j(labels.numpy(), top1_preds.numpy(), k)

    # Per-class aggregation: mean rank and mean cosine similarity
    raw_counts = torch.zeros(k).scatter_add_(0, labels, torch.ones_like(labels, dtype=torch.float))
    present = raw_counts > 0  # a subset of the test set can miss rare classes entirely
    counts = raw_counts.clamp(min=1)
    per_class_rank = torch.zeros(k).scatter_add_(0, labels, ranks.float()) / counts
    per_class_cosine = torch.zeros(k).scatter_add_(0, labels, gt_scores) / counts

    # Class-oriented R@k: per-class R@k first, then mean over classes, so a
    # rare class counts as much as a frequent one (unlike r_at_k above, which
    # is dominated by whichever classes have the most samples).
    class_oriented = {}
    per_class_r_at_1 = None
    for k_at in (1, 5, 10):
        hits = (ranks <= k_at).float()
        per_class_hits = torch.zeros(k).scatter_add_(0, labels, hits) / counts
        class_oriented[f"class_oriented_r_at_{k_at}"] = per_class_hits[present].mean().item()
        if k_at == 1:
            per_class_r_at_1 = torch.where(present, per_class_hits, torch.nan)

    per_class_j_at_1, _ = youdens_j_at_k(top10_preds, labels, 1, k)
    per_class_j_at_5, j_at_5_mean = youdens_j_at_k(top10_preds, labels, 5, k)

    result = {
        "r_at_1": (ranks == 1).float().mean().item(),
        "r_at_5": (ranks <= 5).float().mean().item(),
        "r_at_10": (ranks <= 10).float().mean().item(),
        "median_rank": ranks.float().median().item(),
        "avg_rank": ranks.float().mean().item(),
        **class_oriented,
        "mrr": mrr,
        "youdens_j": j,
        "youdens_j_at_5": j_at_5_mean,
        "per_class_j_at_1": per_class_j_at_1,
        "per_class_j_at_5": per_class_j_at_5,
        "per_class_counts": raw_counts.long(),
        "per_class_r_at_1": per_class_r_at_1,
        "mean_log_odds": mean_log_odds,
        "per_class_rank": per_class_rank,
        "per_class_cosine": per_class_cosine,
        "ranks": ranks,
        "top1_preds": top1_preds,
        "top10_preds": top10_preds,
        "labels": labels,
    }
    if temperature is not None:
        nll = torch.cat(nll_chunks)
        per_class_nll = torch.zeros(k).scatter_add_(0, labels, nll) / counts
        result["log_loss"] = nll.mean().item()
        result["per_class_log_loss"] = torch.where(present, per_class_nll, torch.nan)
    if return_logits:
        result["logits"] = torch.cat(logit_chunks)
    return result
