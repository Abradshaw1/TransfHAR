"""Label-agnostic augmentations for raw sensor windows.

All functions operate on numpy arrays shaped [T, C] (time first, channels second).
They are intended to be called *before* tensor conversion so they remain
representation-agnostic.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from imu_lm.utils.helpers import cfg_get


def add_gaussian_noise(x: np.ndarray, sigma: float) -> np.ndarray:
    """Additive sensor jitter: x <- x + N(0, sigma^2)."""

    noise = np.random.normal(0.0, sigma, size=x.shape).astype(x.dtype)
    return x + noise


def apply_gain_jitter(x: np.ndarray, min_scale: float, max_scale: float) -> np.ndarray:
    """Channel-wise gain jitter with a single scale per axis."""

    scale = np.random.uniform(min_scale, max_scale, size=(1, x.shape[1])).astype(x.dtype)
    return x * scale


def apply_axis_bias(x: np.ndarray, max_bias: float) -> np.ndarray:
    """Channel-wise bias offset jitter with a constant offset per axis."""

    bias = np.random.uniform(-max_bias, max_bias, size=(1, x.shape[1])).astype(x.dtype)
    return x + bias


def smooth_time_warp(x: np.ndarray, max_warp: float, sigma: int) -> np.ndarray:
    """Smooth, monotonic time warp (bounded by +/- max_warp of timeline).

    - Preserves ordering (monotonic warp).
    - Uses a Gaussian-smoothed noise curve to create the warp field.
    - Resamples each channel with 1D interpolation.
    """

    T = x.shape[0]
    if T < 2 or max_warp <= 0:
        return x

    t_orig = np.linspace(0.0, 1.0, T)
    noise = np.random.normal(0.0, 1.0, size=T)

    if sigma > 0:
        radius = int(3 * sigma)
        t = np.arange(-radius, radius + 1)
        kernel = np.exp(-(t ** 2) / (2 * (sigma ** 2)))
        kernel = kernel / kernel.sum()
        smooth = np.convolve(noise, kernel, mode="same")
    else:
        smooth = noise

    # Bound the warp displacement to +/- max_warp of normalized time.
    smooth = smooth / (np.max(np.abs(smooth)) + 1e-8) * max_warp

    warp = t_orig + smooth
    warp = np.clip(warp, 0.0, 1.0)
    warp[0] = 0.0
    warp[-1] = 1.0
    warp = np.maximum.accumulate(warp)

    x_warped = np.empty_like(x)
    for c in range(x.shape[1]):
        x_warped[:, c] = np.interp(t_orig, warp, x[:, c]).astype(x.dtype)
    return x_warped


def apply_augment(x: np.ndarray, cfg: Any) -> np.ndarray:
    """Apply configured augmentations to a raw window [T, C].
    
    Augmentations are at top-level in config (flat structure).
    Each augment has its own enabled flag.
    """
    out = x

    gn_cfg = cfg_get(cfg, ["gaussian_noise"], {}) or {}
    if gn_cfg.get("enabled", False):
        sigma = float(gn_cfg.get("sigma", 0.0))
        out = add_gaussian_noise(out, sigma)

    gj_cfg = cfg_get(cfg, ["gain_jitter"], {}) or {}
    if gj_cfg.get("enabled", False):
        lo = float(gj_cfg.get("min_scale", 1.0))
        hi = float(gj_cfg.get("max_scale", 1.0))
        out = apply_gain_jitter(out, lo, hi)

    ab_cfg = cfg_get(cfg, ["axis_bias"], {}) or {}
    if ab_cfg.get("enabled", False):
        max_bias = float(ab_cfg.get("max_bias", 0.0))
        out = apply_axis_bias(out, max_bias)

    tw_cfg = cfg_get(cfg, ["time_warp"], {}) or {}
    if tw_cfg.get("enabled", False):
        max_warp = float(tw_cfg.get("max_warp", 0.0))
        smooth_sigma = int(tw_cfg.get("smooth_sigma", 0))
        out = smooth_time_warp(out, max_warp=max_warp, sigma=smooth_sigma)

    return out

