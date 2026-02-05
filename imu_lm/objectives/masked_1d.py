"""1D Masked Autoencoder objective (LSM-2 style) - MSE reconstruction on raw 1D signals.

Architecture-agnostic objective for masked reconstruction pretraining.
Works with any encoder/decoder pair that follows the interface:
- encoder(x) → [B, embed_dim]  
- decoder(z) → [B, C, T]

The decoder is architecture-specific and should live with its encoder
(e.g., CNN1DDecoder in models/CNN/, ViT1DDecoder in models/ViT1D/).

Masking strategies (from config):
- mask_random: fraction of timesteps to mask randomly
- mask_temporal: probability of masking contiguous time blocks  
- mask_signal: probability of masking entire channels
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from imu_lm.utils.helpers import cfg_get


def generate_mask(
    x: torch.Tensor,
    mask_random: float = 0.15,
    mask_temporal: float = 0.0,
    mask_signal: float = 0.0,
) -> torch.Tensor:
    """Generate a boolean mask for input tensor [B, C, T].
    
    Args:
        x: Input tensor [B, C, T]
        mask_random: Fraction of individual positions to mask randomly
        mask_temporal: Probability of masking contiguous time blocks
        mask_signal: Probability of masking entire channels
        
    Returns:
        Boolean mask [B, C, T] where True = masked (to be predicted)
    """
    B, C, T = x.shape
    device = x.device
    
    mask = torch.zeros(B, C, T, dtype=torch.bool, device=device)
    
    # Random masking: mask individual positions
    if mask_random > 0:
        rand_mask = torch.rand(B, C, T, device=device) < mask_random
        mask = mask | rand_mask
    
    # Temporal masking: mask contiguous time blocks
    if mask_temporal > 0:
        for b in range(B):
            if torch.rand(1).item() < mask_temporal:
                block_len = max(1, int(T * torch.rand(1).item() * 0.5))  # up to 50% of T
                start = int(torch.randint(0, max(1, T - block_len), (1,)).item())
                mask[b, :, start:start + block_len] = True
    
    # Signal/channel masking: mask entire channels
    if mask_signal > 0:
        for b in range(B):
            for c in range(C):
                if torch.rand(1).item() < mask_signal:
                    mask[b, c, :] = True
    
    # Ensure at least some positions are masked
    if not mask.any():
        fallback = torch.rand(B, C, T, device=device) < 0.1
        mask = mask | fallback
    
    return mask


def forward_loss(
    batch: Tuple[torch.Tensor, torch.Tensor],
    encoder: nn.Module,
    decoder: nn.Module,
    cfg: Any,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute 1D masked reconstruction loss (MSE on masked positions).
    
    Architecture-agnostic: works with any encoder/decoder pair that follows:
    - encoder(x) → [B, embed_dim]
    - decoder(z) → [B, C, T]
    
    Args:
        batch: (x, y) where x is [B, C, T] raw 1D signal
        encoder: Any encoder that maps [B, C, T] → [B, embed_dim]
        decoder: Any decoder that maps [B, embed_dim] → [B, C, T]
        cfg: Config dict with objective.* params
        
    Returns:
        loss: Scalar MSE loss on masked positions
        logs: Dict with loss value and mask stats
    """
    x, _ = batch  # x: [B, C, T], y: labels (unused for reconstruction)
    
    # Get masking config
    obj_cfg = cfg_get(cfg, ["objective"], {}) or {}
    mask_random = float(obj_cfg.get("mask_random", 0.15))
    mask_temporal = float(obj_cfg.get("mask_temporal", 0.0))
    mask_signal = float(obj_cfg.get("mask_signal", 0.0))
    skip_missing = bool(obj_cfg.get("skip_missing", True))
    
    # Generate mask
    mask = generate_mask(x, mask_random, mask_temporal, mask_signal)
    
    # Create masked input (zero out masked positions)
    x_masked = x.clone()
    x_masked[mask] = 0.0
    
    # Encode masked input → embedding
    z = encoder(x_masked)  # [B, embed_dim]
    
    # Decode → reconstruction
    reconstructed = decoder(z)  # [B, C, T]
    
    # MSE loss on masked positions only
    if skip_missing:
        masked_pred = reconstructed[mask]
        masked_target = x[mask]
        loss = F.mse_loss(masked_pred, masked_target)
    else:
        loss = F.mse_loss(reconstructed, x)
    
    mask_ratio = mask.float().mean().item()
    
    return loss, {
        "loss": float(loss.detach().item()),
        "mask_ratio": mask_ratio,
        "mse": float(loss.detach().item()),
    }
