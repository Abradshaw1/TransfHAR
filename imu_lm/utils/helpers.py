from __future__ import annotations

"""Misc helpers."""

from typing import Any, Iterable, Tuple

import torch
import torch.nn.functional as F


def cfg_get(cfg: Any, path: Iterable[str], default=None):
    """Navigate nested config (dict or object) by path."""
    cur = cfg
    for key in path:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, default)
        else:
            cur = getattr(cur, key, default)
    return cur if cur is not None else default


def resize_bilinear(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Resize BCHW float tensor to `size` using bilinear interpolation."""

    if x.dim() != 4:
        raise ValueError(f"Expected x with shape [B,C,H,W], got {x.shape}")
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)
