"""Metrics for multiclass classification."""
import numpy as np
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
