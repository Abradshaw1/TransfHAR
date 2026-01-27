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


# def _spec_to_png(spec: torch.Tensor) -> bytes:
#     """Render [C, F, TT] spectrogram to a PNG (no axes/labels)."""

#     if spec.dim() != 3:
#         raise ValueError(f"Expected spec shape [C, F, TT], got {spec.shape}")

#     spec_np = spec.detach().cpu().numpy()
#     spec_np = np.nan_to_num(spec_np, nan=0.0, posinf=0.0, neginf=0.0)
#     # Log compress to avoid DC-line saturation.
#     spec_np = np.log1p(spec_np)

#     # Map channels to RGB (use first 3 channels; repeat/pad if fewer).
#     if spec_np.shape[0] >= 3:
#         base = spec_np[:3]
#     elif spec_np.shape[0] == 1:
#         base = np.repeat(spec_np[:1], 3, axis=0)
#     else:
#         pad = np.zeros_like(spec_np[:1])
#         base = np.concatenate([spec_np, pad], axis=0)

#     # Per-channel min-max normalize to 0..1.
#     img = np.zeros_like(base)
#     for c in range(3):
#         ch = base[c]
#         cmin, cmax = ch.min(), ch.max()
#         if cmax > cmin:
#             img[c] = (ch - cmin) / (cmax - cmin)

#     # Avoid all-black if flat.
#     if img.max() <= 0:
#         img = np.zeros_like(img)

#     fig, ax = plt.subplots(figsize=(3, 2), dpi=150)
#     ax.axis("off")
#     ax.imshow(img.transpose(1, 2, 0), aspect="auto", origin="lower")
#     fig.tight_layout(pad=0)

#     buf = BytesIO()
#     fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
#     plt.close(fig)
#     buf.seek(0)
#     return buf.getvalue()

def _spec_to_png(spec: torch.Tensor) -> bytes:
    """Render [C, F, TT] -> PNG by *only* packing channels to RGB (no log/minmax hacks)."""

    if spec.dim() != 3:
        raise ValueError(f"Expected spec shape [C, F, TT], got {spec.shape}")

    spec_np = spec.detach().cpu().numpy()
    spec_np = np.nan_to_num(spec_np, nan=0.0, posinf=0.0, neginf=0.0)

    # Take first 3 channels (pad/repeat if needed)
    if spec_np.shape[0] >= 3:
        base = spec_np[:3]
    elif spec_np.shape[0] == 1:
        base = np.repeat(spec_np[:1], 3, axis=0)
    else:
        pad = np.zeros_like(spec_np[:1])
        base = np.concatenate([spec_np, pad], axis=0)

    # IMPORTANT: no log, no per-channel minmax. If you want scaling/clipping, do it upstream.
    # If you want PNG output to be visually meaningful, ensure upstream produces roughly [0,1].
    img = base.transpose(1, 2, 0)  # [F, TT, 3]
    img = img / max(img.max(), 1e-12)

    fig, ax = plt.subplots(figsize=(3, 2), dpi=150)
    ax.axis("off")
    ax.imshow(img, aspect="auto", origin="lower")
    fig.tight_layout(pad=0)

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def stft_encode(x_ct: torch.Tensor, cfg: Any) -> Tuple[torch.Tensor, bytes] | torch.Tensor:
    """
    Compute magnitude spectrogram for a single window using torch.stft knobs.

    Args:
        x_ct: [C, T] float tensor (e.g., 3-axis accel).
        cfg: config object/dict with values under data.augment.spectrogram.

    Returns:
        spec [C, F, TT] if return_image is False (default).
        (spec, png_bytes) if return_image is True.
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

    spec = torch.abs(X)

    if return_image:
        png_bytes = _spec_to_png(spec)
        return spec, png_bytes

    return spec
