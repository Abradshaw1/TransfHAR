"""Supervised classification objective for fully-supervised training.

Architecture-agnostic: works with any encoder that outputs [B, embed_dim].
All current encoders (CNN1D, ViT1D, TSTransformer1D, ViT2D) handle pooling
internally, so this objective just needs encoder output → linear head → loss.

Reuses LinearHead from probe for the classification head.
Reuses compute_metrics from utils/metrics.py for comprehensive evaluation.

Contract:
    forward_loss(batch, encoder, head, cfg) -> (loss, logs)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from imu_lm.utils.metrics import compute_metrics


def forward_loss(
    batch: Tuple[torch.Tensor, torch.Tensor],
    encoder: nn.Module,
    head: nn.Module,
    cfg: Any,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute supervised classification loss with comprehensive metrics.
    
    Uses centralized compute_metrics from utils/metrics.py (same as probe).
    
    Args:
        batch: (x, y) where x is input tensor, y is class labels [B]
        encoder: Any encoder that maps input → [B, embed_dim]
        head: Classification head that maps [B, embed_dim] → [B, num_classes]
        cfg: Config dict with objective.* params
        
    Returns:
        loss: Scalar cross-entropy loss
        logs: Dict with loss and metrics from compute_metrics
    """
    x, y = batch
    
    # Encode input → embedding
    z = encoder(x)  # [B, embed_dim]
    
    # Classify
    logits = head(z)  # [B, num_classes]
    
    # Cross-entropy loss
    loss = F.cross_entropy(logits, y)
    
    # Predictions
    preds = logits.argmax(dim=1)
    
    # Use centralized metrics computation (same as probe)
    y_true = y.detach().cpu().tolist()
    y_pred = preds.detach().cpu().tolist()
    metrics = compute_metrics(y_true, y_pred)
    
    # Return flat metrics for trainer logging (exclude per_class and confusion for brevity)
    return loss, {
        "loss": float(loss.detach().item()),
        "acc": float((y.detach().cpu() == preds.detach().cpu()).float().mean()),
        "bal_acc": metrics["bal_acc"],
        "macro_f1": metrics["macro_f1"],
        "macro_prec": metrics["macro_precision"],
        "macro_rec": metrics["macro_recall"],
    }
