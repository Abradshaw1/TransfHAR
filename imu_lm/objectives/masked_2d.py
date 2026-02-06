"""2D Masked Autoencoder objective for ViT2D (patch-based MAE on spectrograms).

Uses HuggingFace ViTMAEForPreTraining which handles patch masking internally.
The ViT2D encoder wraps ViTMAE and provides the mae_model attribute.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


def forward_loss(
    batch: Tuple[torch.Tensor, torch.Tensor],
    encoder: nn.Module,
    cfg: Any,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute 2D masked patch reconstruction loss (MAE-style).
    
    The ViT2D encoder contains mae_model (HuggingFace ViTMAEForPreTraining)
    which handles patch embedding, masking, and reconstruction internally.
    
    Args:
        batch: (x, y) where x is [B, C, H, W] spectrogram image
        encoder: ViT2D encoder with mae_model attribute
        cfg: Config dict (mask_ratio comes from model config)
        
    Returns:
        loss: Scalar reconstruction loss on masked patches
        logs: Dict with loss and mask_ratio
    """
    x, _ = batch  # y labels unused for reconstruction
    
    mask_ratio = encoder.mae_model.config.mask_ratio
    
    # Prepare input (resize to model's expected resolution)
    x_img = encoder._prepare(x)
    
    # Forward through ViTMAE - handles masking and computes loss internally
    outputs = encoder.mae_model(pixel_values=x_img, mask_ratio=mask_ratio)
    loss = outputs.loss
    
    return loss, {
        "loss": float(loss.detach().item()),
        "mask_ratio": mask_ratio,
    }
