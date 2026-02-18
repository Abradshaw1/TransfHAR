"""Window-level preprocessing: impute, filter, normalize, transpose to [C, T]."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from imu_lm.utils.helpers import cfg_get

logger = logging.getLogger(__name__)


try:  # optional heavy dependency; used for filtering/spline if available
    from scipy import signal, interpolate  # type: ignore
except Exception:  # pragma: no cover - optional
    signal = None
    interpolate = None


@dataclass
class PreprocessStats:
    windows_seen: int = 0
    windows_dropped_impute: int = 0
    windows_filtered: int = 0
    windows_normalized: int = 0


def impute_window(Xw: np.ndarray, cfg: Any, stats: Optional[PreprocessStats] = None) -> Optional[np.ndarray]:
    if not bool(cfg_get(cfg, ["preprocess", "impute", "enabled"], False)):
        return Xw

    method = cfg_get(cfg, ["preprocess", "impute", "method"], "linear")
    max_missing = float(cfg_get(cfg, ["preprocess", "impute", "max_missing_frac"], 0.0))

    missing_mask = np.isnan(Xw)
    missing_frac = missing_mask.mean() if Xw.size else 0.0
    if missing_frac > max_missing:
        if stats:
            stats.windows_dropped_impute += 1
        return None

    T = Xw.shape[0]
    t_idx = np.arange(T)
    for c in range(Xw.shape[1]):
        xc = Xw[:, c]
        missing = np.isnan(xc)
        if not missing.any():
            continue
        valid_idx = t_idx[~missing]
        valid_vals = xc[~missing]
        if valid_idx.size == 0:
            if stats:
                stats.windows_dropped_impute += 1
            return None
        if method == "spline" and interpolate is not None and valid_idx.size >= 4:
            spline = interpolate.CubicSpline(valid_idx, valid_vals, extrapolate=True)
            xc[missing] = spline(t_idx[missing])
        else:  # linear fallback
            # np.interp requires sorted valid_idx and at least one value; handle single value
            if valid_idx.size == 1:
                xc[missing] = valid_vals[0]
            else:
                xc[missing] = np.interp(t_idx[missing], valid_idx, valid_vals)
        Xw[:, c] = xc

    return Xw


def filter_window(Xw: np.ndarray, cfg: Any, stats: Optional[PreprocessStats] = None) -> np.ndarray:
    filt_cfg = cfg_get(cfg, ["preprocess", "filter"], {}) or {}
    if not filt_cfg.get("enabled", False):
        return Xw

    if signal is None:
        raise ImportError("scipy is required for filtering; install scipy to enable filter_window")

    ftype = filt_cfg.get("type", "bandpass")
    order = int(filt_cfg.get("order", 4))
    low = filt_cfg.get("low_hz")
    high = filt_cfg.get("high_hz")
    if (ftype in {"highpass", "bandpass"} and low is None) or (ftype in {"lowpass", "bandpass"} and high is None):
        raise ValueError("filter.low_hz/high_hz must be set for the chosen filter type")
    fs = 50.0  # canonical sample rate

    nyq = 0.5 * fs
    if ftype == "lowpass":
        Wn = high / nyq
        b, a = signal.butter(order, Wn, btype="low")
    elif ftype == "highpass":
        Wn = low / nyq
        b, a = signal.butter(order, Wn, btype="high")
    else:  # bandpass
        Wn = [low / nyq, high / nyq]
        b, a = signal.butter(order, Wn, btype="band")

    for c in range(Xw.shape[1]):
        Xw[:, c] = signal.filtfilt(b, a, Xw[:, c], method="gust")

    if stats:
        stats.windows_filtered += 1
    return Xw


def normalize_window(Xw: np.ndarray, cfg: Any, stats: Optional[PreprocessStats] = None) -> np.ndarray:
    norm_cfg = cfg_get(cfg, ["preprocess", "normalize"], {}) or {}
    if not norm_cfg.get("enabled", False):
        return Xw

    method = norm_cfg.get("method", "zscore")
    eps = float(norm_cfg.get("eps", 1e-6))

    Xn = np.empty_like(Xw, dtype=np.float32)
    for c in range(Xw.shape[1]):
        x = Xw[:, c].astype(np.float32)
        if method == "robust":
            q25, q75 = np.percentile(x, [25, 75])
            scale = (q75 - q25) + eps
            Xn[:, c] = (x - np.median(x)) / scale
        elif method == "minmax":
            xmin, xmax = x.min(), x.max()
            Xn[:, c] = (x - xmin) / ((xmax - xmin) + eps)
        else:  # zscore
            Xn[:, c] = (x - x.mean()) / (x.std() + eps)

    if stats:
        stats.windows_normalized += 1
    return Xn


def preprocess_window(Xw: np.ndarray, cfg: Any, stats: Optional[PreprocessStats] = None) -> Optional[np.ndarray]:
    """Apply impute -> filter -> normalize, returning [C, T] float32 or None if dropped."""

    if stats:
        stats.windows_seen += 1

    Xw = Xw.astype(np.float32)

    Xw = impute_window(Xw, cfg, stats)
    if Xw is None:
        return None

    Xw = filter_window(Xw, cfg, stats)
    Xw = normalize_window(Xw, cfg, stats)

    return Xw.T  # [T, C] -> [C, T]
