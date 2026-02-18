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


def _build_window_specs(
    session_keys: List[SessionKey], n_rows_map: Dict[SessionKey, int], cfg: Any
) -> Tuple[List[SessionKey], np.ndarray, np.ndarray, Dict[str, int]]:
    """Precompute compact window specs using numpy arrays.

    Returns:
        kept_keys: deduplicated list of SessionKeys that have windows
        sess_idx:  int32 array, index into kept_keys for each window
        starts:    int32 array, start row for each window
        counters:  dict with sessions/windows counts
    """
    T, hop = compute_T_and_hop(cfg)
    kept_keys: List[SessionKey] = []

    # First pass: count total windows to preallocate arrays
    total_win = 0
    for key in session_keys:
        N = n_rows_map.get(key, 0)
        if N >= T:
            total_win += (N - T) // hop + 1

    sess_idx = np.empty(total_win, dtype=np.int32) #pre allocate arrays
    starts = np.empty(total_win, dtype=np.int32) #pre allocate arrays

    # Second pass: fill arrays
    offset = 0
    for key in session_keys:
        N = n_rows_map.get(key, 0)
        if N < T:
            continue
        ki = len(kept_keys)
        kept_keys.append(key)
        n_win = (N - T) // hop + 1
        sess_idx[offset : offset + n_win] = ki
        starts[offset : offset + n_win] = np.arange(0, N - T + 1, hop, dtype=np.int32)
        offset += n_win

    counters = {"sessions": len(kept_keys), "windows": total_win}
    return kept_keys, sess_idx, starts, counters


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
        self._pa_dataset = None  # lazily opened per-worker

        # Cache config values read on every __getitem__
        self._dataset_col = cfg_get(cfg, ["data", "dataset_column"], "dataset")
        self._subject_col = cfg_get(cfg, ["data", "subject_column"], "subject_id")
        self._session_col = cfg_get(cfg, ["data", "session_column"], "session_id")
        self._label_col = cfg_get(cfg, ["data", "label_column"], "dataset_activity_id")
        self._time_col = cfg_get(cfg, ["data", "time_column"], None)
        self._sensor_cols = cfg_get(cfg, ["data", "sensor_columns"], []) or []
        self._drop_na = bool(cfg_get(cfg, ["data", "drop_na"], False))
        self._T, self._hop = compute_T_and_hop(cfg)
        self._handle_gaps = bool(cfg_get(cfg, ["windowing", "handle_gaps"], False))
        self._gap_method = cfg_get(cfg, ["windowing", "gap_method"], "interpolate")
        self._max_gap_ns = float(cfg_get(cfg, ["windowing", "max_gap_ms"], 200.0)) * 1e6
        self._spec_enabled = bool((cfg_get(cfg, ["spectrogram"], {}) or {}).get("enabled", False))

        # Build load columns list once
        self._load_cols = list(self._sensor_cols) + [self._label_col]
        if self._time_col:
            self._load_cols.append(self._time_col)
        self._load_cols.extend([self._dataset_col, self._subject_col, self._session_col])

        # Single pass over session_index for both gap_counts and n_rows
        max_gaps_per_session = int(cfg_get(cfg, ["windowing", "max_gaps_per_session"], 1_000_000_000))
        gap_counts: Dict[SessionKey, int] = {}
        n_rows_map: Dict[SessionKey, int] = {}
        for _, r in session_index.iterrows():
            k = SessionKey(r["dataset"], str(r["subject_id"]), str(r["session_id"]))
            gap_counts[k] = int(r.get("gap_count", 0))
            n_rows_map[k] = int(r["n_rows"])
        filtered_keys = [k for k in session_keys if gap_counts.get(k, 0) <= max_gaps_per_session]

        self._keys, self._sess_idx, self._starts, counters = _build_window_specs(
            filtered_keys, n_rows_map, cfg
        )

        logger.info(
            "WindowDataset split=%s sessions=%d windows=%d (filtered=%d)",
            split_name,
            counters.get("sessions", 0),
            counters.get("windows", 0),
            len(session_keys) - len(filtered_keys),
        )

    def __len__(self) -> int:
        return len(self._starts)

    def _get_pa_dataset(self):
        """Lazily open the pyarrow dataset once per worker process."""
        if self._pa_dataset is None:
            self._pa_dataset = ds.dataset(self.parquet_path, format="parquet")
        return self._pa_dataset

    def _load_session(self, key: SessionKey) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if self.cache.key == key and self.cache.X is not None:
            return self.cache.X, self.cache.y, self.cache.t

        pa_ds = self._get_pa_dataset()
        filt = (
            (ds.field(self._dataset_col) == key.dataset)
            & (ds.field(self._subject_col) == key.subject_id)
            & (ds.field(self._session_col) == key.session_id)
        )
        table = pa_ds.to_table(columns=self._load_cols, filter=filt)
        df = table.to_pandas()

        if self._time_col:
            df = df.sort_values(self._time_col).reset_index(drop=True)
        if self._drop_na:
            df = df.dropna(subset=self._sensor_cols)

        X = df[self._sensor_cols].to_numpy(dtype=np.float32)
        y = df[self._label_col].to_numpy()
        t = df[self._time_col].to_numpy() if self._time_col else None

        self.cache = _SessionCache(key=key, X=X, y=y, t=t)
        return X, y, t

    def __getitem__(self, idx: int):
        key = self._keys[self._sess_idx[idx]]
        start = int(self._starts[idx])
        X, y, t = self._load_session(key)
        T = self._T
        Xw = X[start : start + T]
        yw = y[start : start + T]

        label = resolve_window_label(yw, self.cfg)
        if label is None:
            return None

        # gap gating at fetch time
        if self._handle_gaps and t is not None:
            dt = np.diff(t[start : start + T])
            if np.any(dt > self._max_gap_ns):
                if self._gap_method in {"drop", "split_segment"}:
                    return None

        Xproc = preprocess_window(Xw, self.cfg, self.stats)
        if Xproc is None:
            return None

        Xproc = apply_augment(Xproc, self.cfg)              # [T, C] -> [T, C]
        x_ct = torch.from_numpy(Xproc).float().T             # [T, C] -> [C, T]  (single transpose)

        if self._spec_enabled:
            out = stft_encode(x_ct, self.cfg)
            x_tensor = out[1] if isinstance(out, tuple) else out
            if x_tensor.dim() != 3:
                return None
        else:
            x_tensor = x_ct

        y_tensor = torch.tensor(label, dtype=torch.long)
        return x_tensor, y_tensor


class SessionGroupedBatchSampler:
    """Batch sampler that groups windows by session for I/O efficiency.

    Instead of random access across all windows (causing constant parquet
    cache misses on the single-entry _SessionCache), this sampler yields all
    windows from one session before moving to the next.  Sessions are shuffled
    each epoch for randomness; windows within each session are also shuffled.
    """

    def __init__(self, sess_idx: np.ndarray, batch_size: int,
                 shuffle: bool = True, drop_last: bool = False, seed: int = 0):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # sess_idx is contiguous by session from _build_window_specs
        _, self._group_starts, self._group_counts = np.unique(
            sess_idx, return_index=True, return_counts=True
        )
        self._n_sessions = len(self._group_starts)
        self._total = int(sess_idx.shape[0])

    def set_epoch(self, epoch: int):
        """For DDP compatibility."""
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)

        if self.shuffle:
            session_order = rng.permutation(self._n_sessions)
        else:
            session_order = np.arange(self._n_sessions)

        batch: list[int] = []
        for si in session_order:
            start = int(self._group_starts[si])
            count = int(self._group_counts[si])
            indices = np.arange(start, start + count)
            if self.shuffle:
                rng.shuffle(indices)
            for idx in indices:
                batch.append(int(idx))
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

        if batch and not self.drop_last:
            yield batch

        self.epoch += 1

    def __len__(self):
        if self.drop_last:
            return self._total // self.batch_size
        return (self._total + self.batch_size - 1) // self.batch_size


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0)


def _make_loader(dataset: WindowDataset, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool, session_grouped: bool = True) -> DataLoader:
    persistent = num_workers > 0
    if shuffle and session_grouped:
        batch_sampler = SessionGroupedBatchSampler(
            dataset._sess_idx, batch_size, shuffle=True, drop_last=False,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_skip_none,
            persistent_workers=persistent,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_skip_none,
        persistent_workers=persistent,
    )


def make_loaders(cfg: Any, dataset_filter=None) -> Dict[str, DataLoader]:
    """Construct DataLoaders for train/val/probe splits.

    dataset_filter: optional list of dataset names to restrict session scan.
                    When provided, builds probe_* loaders. Otherwise builds train/val/test.
    """

    parquet_path = cfg_get(cfg, ["paths", "dataset_path"])
    batch_size = int(cfg_get(cfg, ["data", "batch_size"], 256))
    eval_batch_size = int(cfg_get(cfg, ["data", "eval_batch_size"], batch_size))
    num_workers = int(cfg_get(cfg, ["data", "num_workers"], 0))
    pin_memory = bool(cfg_get(cfg, ["data", "pin_memory"], False))

    is_probe = dataset_filter is not None

    # In pretrain mode, honour data.train_on_datasets if provided.
    scan_filter = dataset_filter
    if not is_probe:
        train_on = cfg_get(cfg, ["data", "train_on_datasets"], None)
        if train_on is not None:
            scan_filter = list(train_on)

    mode = "probe" if is_probe else "pretrain"
    logger.info("[%s] build_session_index path=%s dataset_filter=%s", mode, parquet_path, scan_filter)
    session_index = build_session_index(parquet_path, cfg, dataset_filter=scan_filter)
    dataset_col = cfg_get(cfg, ["data", "dataset_column"], "dataset")
    datasets_in_run = sorted(session_index[dataset_col].unique().tolist())
    logger.info("[%s] sessions=%d datasets=%d: %s", mode, len(session_index), len(datasets_in_run), datasets_in_run)
    splits = make_splits(session_index, cfg)

    loaders: Dict[str, DataLoader] = {}
    def add(name: str, keys: List[SessionKey], bs: int, shuf: bool, session_grouped: bool = True):
        if not keys:
            return
        wds = WindowDataset(parquet_path, session_index, keys, cfg, split_name=name)
        loaders[f"{name}_loader"] = _make_loader(wds, bs, shuf, num_workers, pin_memory, session_grouped=session_grouped)
        logger.info("[%s] %s_loader: sessions=%d windows=%d batch_size=%d", mode, name, len(keys), len(wds), bs)

    if is_probe:
        add("probe_train", splits["probe_train_keys"], batch_size, True, session_grouped=False)
        add("probe_val", splits["probe_val_keys"], eval_batch_size, False)
        add("probe_test", splits["probe_test_keys"], eval_batch_size, False)
    else:
        add("train", splits["train_keys"], batch_size, True)
        add("val", splits["val_keys"], eval_batch_size, False)
        add("test", splits["test_keys"], eval_batch_size, False)

    return loaders
