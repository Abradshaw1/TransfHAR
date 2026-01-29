"""Metrics utilities for probes/eval (simple sklearn wrappers)."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score


def compute_metrics(
    y_true: Iterable[int], y_pred: Iterable[int], labels: Optional[List[int]] = None
) -> Dict[str, object]:
    y_true_arr = np.asarray(list(y_true))
    y_pred_arr = np.asarray(list(y_pred))

    bal_acc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", labels=labels))

    if labels is None:
        labels = sorted(np.unique(y_true_arr).tolist())
    per_class_f1_vals = f1_score(y_true_arr, y_pred_arr, labels=labels, average=None)
    per_class_f1 = {int(lbl): float(val) for lbl, val in zip(labels, per_class_f1_vals)}

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)

    return {
        "bal_acc": bal_acc,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "confusion": cm.tolist(),
        "labels": [int(l) for l in labels],
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

    per_class = metrics.get("per_class_f1", {}) or {}
    for lbl, val in per_class.items():
        add(f"per_class_f1_{lbl}", f"{val:.6f}")

    return " ".join(tokens)
