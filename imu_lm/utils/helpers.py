"""
helpers.py
----------
YAML/load/merge stubs + shared helpers.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


# ############ ViT helpers ############
def resize_bilinear(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Resize BCHW float tensor to `size` using bilinear interpolation.

    Args:
        x: [B, C, H, W] float tensor.
        size: (H_out, W_out)
    Returns:
        Resized tensor [B, C, H_out, W_out].
    """

    if x.dim() != 4:
        raise ValueError(f"Expected x with shape [B,C,H,W], got {x.shape}")
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)
