from __future__ import annotations

"""Misc helpers.

Includes generic utilities and model-specific helpers (see section banners).
"""

from typing import Tuple

import torch
import torch.nn.functional as F


# ############ ViT helpers ############
def resize_bilinear(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Resize BCHW float tensor to `size` using bilinear interpolation."""

    if x.dim() != 4:
        raise ValueError(f"Expected x with shape [B,C,H,W], got {x.shape}")
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)
