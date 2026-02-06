"""Metrics utilities for probes/eval (simple sklearn wrappers)."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    labels: Optional[List[int]] = None,
    label_names: Optional[Dict[int, str]] = None,
) -> Dict[str, object]:
    """Compute classification metrics.
    
    Args:
        y_true: Ground truth labels (mapped indices)
        y_pred: Predicted labels (mapped indices)
        labels: List of label indices to include
        label_names: Optional mapping from label index to string name
        
    Returns:
        Dict with global and per-class metrics
    """
    y_true_arr = np.asarray(list(y_true))
    y_pred_arr = np.asarray(list(y_pred))

    bal_acc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", labels=labels, zero_division=0))
    macro_precision = float(precision_score(y_true_arr, y_pred_arr, average="macro", labels=labels, zero_division=0))
    macro_recall = float(recall_score(y_true_arr, y_pred_arr, average="macro", labels=labels, zero_division=0))

    if labels is None:
        labels = sorted(np.unique(y_true_arr).tolist())
    
    # Per-class metrics
    per_class_f1_vals = f1_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)
    per_class_precision_vals = precision_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)
    per_class_recall_vals = recall_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)
    
    # Per-class accuracy: correct predictions / total samples for that class
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    per_class_correct = np.diag(cm)
    per_class_total = cm.sum(axis=1)
    per_class_acc_vals = np.divide(
        per_class_correct, per_class_total,
        out=np.zeros_like(per_class_correct, dtype=float),
        where=per_class_total > 0
    )
    
    # Build per-class dicts with string names if available
    per_class = {}
    for i, lbl in enumerate(labels):
        key = label_names.get(lbl, str(lbl)) if label_names else str(lbl)
        per_class[key] = {
            "f1": float(per_class_f1_vals[i]),
            "precision": float(per_class_precision_vals[i]),
            "recall": float(per_class_recall_vals[i]),
            "accuracy": float(per_class_acc_vals[i]),
            "support": int(per_class_total[i]),
        }

    # Label list with names
    label_list = [label_names.get(l, str(l)) if label_names else str(l) for l in labels]

    return {
        "bal_acc": bal_acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "per_class": per_class,
        "confusion": cm.tolist(),
        "labels": label_list,
    }


def format_metrics_txt(metrics: Dict[str, object], prefix: str = "") -> str:
    """Serialize metrics dict to key=value tokens for metrics.txt lines."""

    tokens = []
    pre = f"{prefix}" if prefix else ""

    def add(k: str, v: object):
        key = f"{pre}{k}" if pre else k
        tokens.append(f"{key}={v}")

    add("bal_acc", f"{metrics.get('bal_acc', 0.0):.6f}")
    add("macro_f1", f"{metrics.get('macro_f1', 0.0):.6f}")
    add("macro_precision", f"{metrics.get('macro_precision', 0.0):.6f}")
    add("macro_recall", f"{metrics.get('macro_recall', 0.0):.6f}")

    # Per-class metrics (new nested structure)
    per_class = metrics.get("per_class", {}) or {}
    for lbl, class_metrics in per_class.items():
        if isinstance(class_metrics, dict):
            add(f"{lbl}_f1", f"{class_metrics.get('f1', 0.0):.6f}")
            add(f"{lbl}_precision", f"{class_metrics.get('precision', 0.0):.6f}")
            add(f"{lbl}_recall", f"{class_metrics.get('recall', 0.0):.6f}")
            add(f"{lbl}_accuracy", f"{class_metrics.get('accuracy', 0.0):.6f}")
            add(f"{lbl}_support", class_metrics.get('support', 0))

    return " ".join(tokens)
