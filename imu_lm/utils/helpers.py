from __future__ import annotations

"""Misc helpers."""

from typing import Any, Dict, Iterable, Tuple

import yaml
import torch
import torch.nn.functional as F


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file and return as dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge upd into base, returning new dict."""
    out = dict(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


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
