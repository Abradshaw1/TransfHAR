from __future__ import annotations

import logging
from typing import Any, Dict, Generator, Iterable, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


CANONICAL_SAMPLE_RATE_HZ = 50.0


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


def compute_T_and_hop(cfg: Any) -> Tuple[int, int]:
    """Compute window length (T) and hop in samples.

    Uses canonical 50Hz sample rate (no config rate field per spec).
    """

    window_seconds = float(_cfg_get(cfg, ["data", "windowing", "window_seconds"], 2.56))
    hop_ratio = float(_cfg_get(cfg, ["data", "windowing", "window_hop_ratio"], 0.5))
    T = int(round(window_seconds * CANONICAL_SAMPLE_RATE_HZ))
    hop = max(1, int(round(T * hop_ratio)))
    return T, hop


def resolve_window_label(yw: np.ndarray, cfg: Any) -> Optional[int]:
    """Resolve a window label according to policy.

    Returns None if the window should be skipped.
    """

    policy = _cfg_get(cfg, ["data", "windowing", "label_policy"], "pure")
    majority_threshold = float(_cfg_get(cfg, ["data", "windowing", "majority_threshold"], 0.8))
    unknown_label_id = _cfg_get(cfg, ["data", "loading", "unknown_label_id"], None)

    if yw.size == 0:
        return None

    if policy == "pure":
        if np.all(yw == yw[0]):
            return int(yw[0])
        return None

    if policy == "majority":
        vals, counts = np.unique(yw, return_counts=True)
        idx = counts.argmax()
        p = counts[idx] / float(len(yw))
        if p >= majority_threshold:
            return int(vals[idx])
        return None

    if policy == "center":
        return int(yw[len(yw) // 2])

    if policy == "unknown_if_mixed":
        if np.all(yw == yw[0]):
            return int(yw[0])
        return int(unknown_label_id) if unknown_label_id is not None else None

    raise ValueError(f"Unsupported label_policy={policy}")


def _detect_gaps(t: Optional[np.ndarray], max_gap_ms: float) -> np.ndarray:
    if t is None:
        return np.array([], dtype=int)
    dt = np.diff(t)
    gap_ns = max_gap_ms * 1e6
    return np.where(dt > gap_ns)[0]


def iter_windows_from_session(
    X: np.ndarray,
    y: np.ndarray,
    t: Optional[np.ndarray],
    cfg: Any,
) -> Generator[Tuple[int, int, Dict[str, Any]], None, None]:
    """Yield windows (start, label, gap_meta) using index-based sliding.

    Gap handling affects eligibility but not stride placement.
    gap_meta contains indices of gaps inside the window when interpolate is chosen.
    """

    T, hop = compute_T_and_hop(cfg)
    N = len(X)
    handle_gaps = bool(_cfg_get(cfg, ["data", "windowing", "handle_gaps"], False))
    gap_method = _cfg_get(cfg, ["data", "windowing", "gap_method"], "interpolate")
    max_gap_ms = float(_cfg_get(cfg, ["data", "windowing", "max_gap_ms"], 200.0))

    gaps = _detect_gaps(t, max_gap_ms) if handle_gaps else np.array([], dtype=int)

    yielded = 0
    skipped_label = 0
    dropped_gap = 0

    for start in range(0, N - T + 1, hop):
        end = start + T
        yw = y[start:end]
        label = resolve_window_label(yw, cfg)
        if label is None:
            skipped_label += 1
            continue

        gap_meta: Dict[str, Any] = {}
        if handle_gaps and gaps.size > 0:
            # gaps marks indices in dt, so gap between i and i+1 samples
            gap_positions = gaps[(gaps >= start) & (gaps < end - 1)]
            if gap_positions.size > 0:
                gap_meta["gap_positions"] = gap_positions - start
                if gap_method == "drop":
                    dropped_gap += 1
                    continue
                if gap_method == "split_segment":
                    # skip windows that cross any gap
                    dropped_gap += 1
                    continue
                # interpolate: keep window but record gaps
        yield (start, label, gap_meta)
        yielded += 1

    logger.debug(
        "windowing session_len=%d T=%d hop=%d yielded=%d skipped_label=%d dropped_gap=%d",
        N,
        T,
        hop,
        yielded,
        skipped_label,
        dropped_gap,
    )
