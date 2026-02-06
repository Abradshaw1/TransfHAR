"""Evaluation utilities for probe heads."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from imu_lm.utils.metrics import compute_metrics

logger = logging.getLogger(__name__)


def _remap_labels(y: torch.Tensor, raw_to_idx: Dict[int, int]) -> torch.Tensor:
    mapped = [raw_to_idx[int(v)] for v in y.tolist() if int(v) in raw_to_idx]
    if len(mapped) == 0:
        return torch.empty(0, dtype=torch.long, device=y.device)
    return torch.tensor(mapped, dtype=torch.long, device=y.device)


def eval_head(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    loader,
    label_map: Dict[str, Any],
    device,
) -> Dict[str, Any]:
    """Evaluate encoder+head on a loader using stored label mapping."""

    raw_to_idx = {int(k): int(v) for k, v in label_map.get("raw_to_idx", {}).items()}
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
            y_raw = y_raw.to(device)
            idxs = [i for i, v in enumerate(y_raw.tolist()) if int(v) in raw_to_idx]
            if len(idxs) == 0:
                continue
            y = torch.tensor([raw_to_idx[int(y_raw[i])] for i in idxs], dtype=torch.long, device=device)

            x = torch.index_select(x.to(device), 0, torch.tensor(idxs, device=device))
            with torch.no_grad():
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

    metrics = compute_metrics(y_true, y_pred, label_names=label_names)
    acc = float((np.asarray(y_true) == np.asarray(y_pred)).mean())
    metrics.update({"loss": total_loss / n_samples, "acc": acc})
    return metrics
