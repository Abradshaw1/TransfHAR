"""STFT-based spectrogram encoding for raw IMU windows."""

from __future__ import annotations

from io import BytesIO
from typing import Any, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def _cfg_get(cfg: Any, path: Iterable[str], default=None):
    cur = cfg
    for key in path:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, default)
        else:
            cur = getattr(cur, key, default)
    return cur if cur is not None else default

def _spec_to_img(spec: torch.Tensor) -> torch.Tensor:
    """[C,F,TT] -> [3,F,TT] float32 in [0,1] (acc_x/y/z -> R/G/B)."""
    if spec.dim() != 3:
        raise ValueError(f"Expected spec shape [C, F, TT], got {spec.shape}")

    spec_np = spec.detach().cpu().numpy()
    spec_np = np.nan_to_num(spec_np, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if spec_np.shape[0] >= 3:
        base = spec_np[:3]
    elif spec_np.shape[0] == 1:
        base = np.repeat(spec_np[:1], 3, axis=0)
    else:
        base = np.concatenate([spec_np, np.zeros_like(spec_np[:1])], axis=0)

    base /= max(float(base.max()), 1e-12)  # global scale -> [0,1]
    return torch.from_numpy(base)  # [3,F,TT]


def stft_encode(x_ct: torch.Tensor, cfg: Any) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """
    x_ct: [C,T]
    Returns:
      spec [C,F,TT] if return_image False
      (spec, img) where img is [3,F,TT] if return_image True
    """
    if x_ct.dim() != 2:
        raise ValueError(f"Expected x_ct of shape [C, T], got {x_ct.shape}")

    scfg = _cfg_get(cfg, ["data", "augment", "spectrogram"], {}) or {}
    n_fft = int(scfg.get("n_fft", 64))
    win_length = int(scfg.get("win_length", n_fft))
    hop_length = int(scfg.get("hop_length", max(1, win_length // 4)))
    center = bool(scfg.get("center", False))
    return_image = bool(scfg.get("return_image", False))

    window = torch.hann_window(win_length, device=x_ct.device)

    X = torch.stft(
        x_ct,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        return_complex=True,
    )
    spec = torch.abs(X)  # [C,F,TT]

    if return_image:
        img = _spec_to_img(spec)  # [3,F,TT]
        return spec, img

    return spec
