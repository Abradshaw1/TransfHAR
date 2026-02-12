"""Evaluation utilities for probe heads."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from imu_lm.probe.io import select_mapped_batch
from imu_lm.utils.metrics import compute_metrics

logger = logging.getLogger(__name__)


def eval_head(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    loader,
    label_map: Dict[str, Any],
    device,
) -> Dict[str, Any]:
    """Evaluate encoder+head on a loader using stored label mapping."""

    raw_to_idx = {int(k): int(v) for k, v in label_map.get("raw_to_idx", {}).items()}
    unknown_id = label_map.get("unknown_id")
    label_names = label_map.get("label_names", None)

    encoder.eval()
    head.eval()

    y_true: Iterable[int] = []
    y_pred: Iterable[int] = []
    total_loss = 0.0
    n_samples = 0

    total_batches = len(loader) if hasattr(loader, '__len__') else None
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, 1):
            if batch is None:
                continue
            x, y_raw = batch
            x, y = select_mapped_batch(x, y_raw, raw_to_idx, device, unknown_raw_id=unknown_id)
            if x is None:
                continue
            feats = encoder(x)
            logits = head(feats)
            loss = F.cross_entropy(logits, y)

            preds = logits.argmax(dim=1)

            total_loss += float(loss.item() * y.shape[0])
            n_samples += y.shape[0]
            y_true = np.concatenate([np.asarray(y_true), y.cpu().numpy()]) if len(y_true) else y.cpu().numpy()
            y_pred = np.concatenate([np.asarray(y_pred), preds.cpu().numpy()]) if len(y_pred) else preds.cpu().numpy()

            if batch_idx % 5 == 0 or batch_idx == total_batches:
                logger.info("[eval] batch %d/%s samples=%d", batch_idx, total_batches or "?", n_samples)

    if n_samples == 0:
        return {"loss": 0.0, "acc": 0.0, "bal_acc": 0.0, "macro_f1": 0.0}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = compute_metrics(y_true, y_pred, label_names=label_names)
    acc = float((np.asarray(y_true) == np.asarray(y_pred)).mean())
    metrics.update({"loss": total_loss / n_samples, "acc": acc})
    return metrics
