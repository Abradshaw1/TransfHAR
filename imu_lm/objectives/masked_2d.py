"""2D Masked Autoencoder objective (MAE-style) - reconstruction on 2D inputs like spectrograms.

Architecture-agnostic objective for masked reconstruction pretraining on 2D data.
Works with any encoder/decoder pair that follows the interface:
- encoder: has mae_model with forward that returns loss (HuggingFace ViTMAE style)
- Or encoder/decoder pair for custom architectures

The decoder is architecture-specific and should live with its encoder
(e.g., ViT decoder in models/ViT/, CNN2D decoder in models/CNN2D/).

For ViT-MAE: Uses HuggingFace ViTMAEForPreTraining which handles masking internally.
For custom 2D models: Would need encoder/decoder passed in like masked_1d.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from imu_lm.utils.helpers import cfg_get


def forward_loss(
    batch: Tuple[torch.Tensor, torch.Tensor],
    encoder: nn.Module,
    cfg: Any,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute 2D masked reconstruction loss (MAE-style).
    
    Architecture-agnostic for 2D inputs. For ViT-MAE, the encoder contains
    the full mae_model which handles masking and reconstruction internally.
    
    Args:
        batch: (x, y) where x is [B, C, H, W] 2D input (e.g., spectrogram)
        encoder: Model with mae_model attribute (ViTMAE) or custom encoder
        cfg: Config dict with objective.* params
        
    Returns:
        loss: Scalar reconstruction loss
        logs: Dict with loss value and mask stats
    """
    x, _ = batch  # x: [B, C, H, W], y: labels (unused for reconstruction)
    
    # Get mask ratio from config or model
    obj_cfg = cfg_get(cfg, ["objective"], {}) or {}
    
    # Check if encoder has HuggingFace-style mae_model (ViT-MAE)
    if hasattr(encoder, 'mae_model'):
        # ViT-MAE style: model handles masking internally
        mask_ratio = encoder.mae_model.config.mask_ratio
        
        # Prepare input (resize if needed)
        x_img = encoder._prepare(x)
        
        # Forward through MAE model - returns loss directly
        outputs = encoder.mae_model(pixel_values=x_img, mask_ratio=mask_ratio)
        loss = outputs.loss
        
        return loss, {
            "loss": float(loss.detach().item()),
            "mask_ratio": mask_ratio,
        }
    
    else:
        # Custom encoder/decoder: would need decoder passed in
        # This path supports future custom 2D architectures
        raise NotImplementedError(
            "Custom 2D encoder without mae_model not yet supported. "
            "For custom architectures, pass encoder and decoder separately."
        )


def forward_loss_custom(
    batch: Tuple[torch.Tensor, torch.Tensor],
    encoder: nn.Module,
    decoder: nn.Module,
    cfg: Any,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute 2D masked reconstruction loss with custom encoder/decoder.
    
    For architectures that don't use HuggingFace ViTMAE.
    
    Args:
        batch: (x, y) where x is [B, C, H, W] 2D input
        encoder: Any encoder that maps [B, C, H, W] → [B, embed_dim] or [B, N, D]
        decoder: Any decoder that reconstructs [B, C, H, W]
        cfg: Config dict with objective.* params
        
    Returns:
        loss: Scalar MSE loss on masked positions
        logs: Dict with loss value and mask stats
    """
    x, _ = batch
    
    obj_cfg = cfg_get(cfg, ["objective"], {}) or {}
    mask_ratio = float(obj_cfg.get("mask_ratio", 0.75))
    
    # TODO: Implement custom 2D masking + reconstruction
    # This would follow similar pattern to masked_1d but for 2D patches
    raise NotImplementedError("Custom 2D masking not yet implemented")
