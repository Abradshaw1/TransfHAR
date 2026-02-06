from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:  # required for parquet IO
    import pyarrow.dataset as ds
except Exception as exc:  # pragma: no cover - hard fail
    raise ImportError("pyarrow is required for dataset loading") from exc

from imu_lm.data.splits import SessionKey, build_session_index, make_splits
from imu_lm.data.windowing import compute_T_and_hop, resolve_window_label
from imu_lm.data.augmentations.preprocess import PreprocessStats, preprocess_window
from imu_lm.data.augmentations.spectrogram import stft_encode
from imu_lm.data.augmentations.transform import apply_augment
from imu_lm.utils.helpers import cfg_get

logger = logging.getLogger(__name__)


def _load_session_df(parquet_path: str, key: SessionKey, cols: List[str], cfg: Any) -> pd.DataFrame:
    dataset_col = cfg_get(cfg, ["data", "dataset_column"], "dataset")
    subject_col = cfg_get(cfg, ["data", "subject_column"], "subject_id")
    session_col = cfg_get(cfg, ["data", "session_column"], "session_id")

    if ds is None:  # pragma: no cover - enforced above
        raise ImportError("pyarrow is required for dataset loading")

    dataset = ds.dataset(parquet_path, format="parquet")
    filt = (
        (ds.field(dataset_col) == key.dataset)
        & (ds.field(subject_col) == key.subject_id)
        & (ds.field(session_col) == key.session_id)
    )
    table = dataset.to_table(columns=cols, filter=filt)
    return table.to_pandas()


def _build_window_specs(
    session_keys: List[SessionKey], n_rows_map: Dict[SessionKey, int], cfg: Any
) -> Tuple[List[Tuple[SessionKey, int]], Dict[str, int]]:
    """Precompute (key, start_idx) using only lengths (no data reads)."""

    T, hop = compute_T_and_hop(cfg)
    window_specs: List[Tuple[SessionKey, int]] = []
    counters = {"sessions": 0, "windows": 0}

    for key in session_keys:
        N = n_rows_map.get(key, 0)
        if N < T:
            continue
        for start in range(0, N - T + 1, hop):
            window_specs.append((key, start))
            counters["windows"] += 1
        counters["sessions"] += 1

    return window_specs, counters


@dataclass
class _SessionCache:
    key: Optional[SessionKey] = None
    X: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    t: Optional[np.ndarray] = None


class WindowDataset(Dataset):
    def __init__(self, parquet_path: str, session_index: pd.DataFrame, session_keys: List[SessionKey], cfg: Any, split_name: str = "train"):
        self.parquet_path = parquet_path
        self.cfg = cfg
        self.split_name = split_name
        self.cache = _SessionCache()
        self.stats = PreprocessStats()

        max_gaps_per_session = int(cfg_get(cfg, ["windowing", "max_gaps_per_session"], 1_000_000_000))
        gap_counts = {
            SessionKey(r["dataset"], str(r["subject_id"]), str(r["session_id"])): int(r.get("gap_count", 0))
            for _, r in session_index.iterrows()
        }
        filtered_keys = [k for k in session_keys if gap_counts.get(k, 0) <= max_gaps_per_session]

        n_rows_map = {
            SessionKey(r["dataset"], str(r["subject_id"]), str(r["session_id"])): int(r["n_rows"])
            for _, r in session_index.iterrows()
        }

        self.window_specs, counters = _build_window_specs(filtered_keys, n_rows_map, cfg)

        logger.info(
            "WindowDataset split=%s sessions=%d windows=%d (filtered=%d)",
            split_name,
            counters.get("sessions", 0),
            counters.get("windows", 0),
            len(session_keys) - len(filtered_keys),
        )

    def __len__(self) -> int:
        return len(self.window_specs)

    def _load_session(self, key: SessionKey) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if self.cache.key == key and self.cache.X is not None:
            return self.cache.X, self.cache.y, self.cache.t

        dataset_col = cfg_get(self.cfg, ["data", "dataset_column"], "dataset")
        subject_col = cfg_get(self.cfg, ["data", "subject_column"], "subject_id")
        session_col = cfg_get(self.cfg, ["data", "session_column"], "session_id")
        label_col = cfg_get(self.cfg, ["data", "label_column"], "global_activity_id")
        time_col = cfg_get(self.cfg, ["data", "time_column"], None)
        sensor_cols = cfg_get(self.cfg, ["data", "sensor_columns"], []) or []
        drop_na = bool(cfg_get(self.cfg, ["data", "drop_na"], False))

        cols = list(sensor_cols) + [label_col]
        if time_col:
            cols.append(time_col)
        cols.extend([dataset_col, subject_col, session_col])

        df = _load_session_df(self.parquet_path, key, cols, self.cfg)
        if time_col:
            df = df.sort_values(time_col).reset_index(drop=True)
        if drop_na:
            df = df.dropna(subset=sensor_cols)

        X = df[sensor_cols].to_numpy(dtype=np.float32)
        y = df[label_col].to_numpy()
        t = df[time_col].to_numpy() if time_col else None

        self.cache = _SessionCache(key=key, X=X, y=y, t=t)
        return X, y, t

    def __getitem__(self, idx: int):
        key, start = self.window_specs[idx]
        X, y, t = self._load_session(key)
        T, _ = compute_T_and_hop(self.cfg)
        Xw = X[start : start + T]
        yw = y[start : start + T]

        label = resolve_window_label(yw, self.cfg)
        if label is None:
            return None

        # gap gating at fetch time
        handle_gaps = bool(cfg_get(self.cfg, ["windowing", "handle_gaps"], False))
        gap_method = cfg_get(self.cfg, ["windowing", "gap_method"], "interpolate")
        max_gap_ms = float(cfg_get(self.cfg, ["windowing", "max_gap_ms"], 200.0))
        if handle_gaps and t is not None:
            dt = np.diff(t[start : start + T])
            if np.any(dt > max_gap_ms * 1e6):
                if gap_method in {"drop", "split_segment"}:
                    return None

        Xproc = preprocess_window(Xw, self.cfg, self.stats)
        if Xproc is None:
            return None

        # time-domain augmentations (apply_augment expects [T, C])
        # Always apply if any augment is enabled (checked inside apply_augment)
        Xproc = apply_augment(Xproc.T, self.cfg).T

        # Default: raw 1D IMU [C, T]. If spectrogram.enabled, convert to [3, F, TT]
        spec_cfg = cfg_get(self.cfg, ["spectrogram"], {}) or {}
        if spec_cfg.get("enabled", False):
            out = stft_encode(torch.from_numpy(Xproc).float(), self.cfg)
            x_tensor = out[1] if isinstance(out, tuple) else out
            if x_tensor.dim() != 3:
                return None
        else:
            x_tensor = torch.from_numpy(Xproc).float()

        y_tensor = torch.tensor(label, dtype=torch.long)
        return x_tensor, y_tensor


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0)


def _make_loader(dataset: WindowDataset, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_skip_none,
    )


def make_loaders(cfg: Any, dataset_filter=None) -> Dict[str, DataLoader]:
    """Construct DataLoaders for train/val/probe splits.

    dataset_filter: optional list of dataset names to restrict session scan.
    """

    parquet_path = cfg_get(cfg, ["paths", "dataset_path"])
    batch_size = int(cfg_get(cfg, ["data", "batch_size"], 256))
    eval_batch_size = int(cfg_get(cfg, ["data", "eval_batch_size"], batch_size))
    num_workers = int(cfg_get(cfg, ["data", "num_workers"], 0))
    pin_memory = bool(cfg_get(cfg, ["data", "pin_memory"], False))

    mode = "probe" if dataset_filter else "pretrain"
    logger.info("[%s] build_session_index path=%s", mode, parquet_path)
    session_index = build_session_index(parquet_path, cfg, dataset_filter=dataset_filter)
    logger.info("[%s] sessions=%d", mode, len(session_index))
    splits = make_splits(session_index, cfg)

    loaders: Dict[str, DataLoader] = {}
    def add(name: str, keys: List[SessionKey], bs: int, shuf: bool):
        if not keys:
            return
        ds = WindowDataset(parquet_path, session_index, keys, cfg, split_name=name)
        loaders[f"{name}_loader"] = _make_loader(ds, bs, shuf, num_workers, pin_memory)
        logger.info("[%s] %s_loader: sessions=%d windows=%d batch_size=%d", mode, name, len(keys), len(ds), bs)

    if dataset_filter:
        add("probe_train", splits["probe_train_keys"], batch_size, True)
        add("probe_val", splits["probe_val_keys"], eval_batch_size, False)
        add("probe_test", splits["probe_test_keys"], eval_batch_size, False)
    else:
        add("train", splits["train_keys"], batch_size, True)
        add("val", splits["val_keys"], eval_batch_size, False)
        add("test", splits["test_keys"], eval_batch_size, False)

    return loaders
