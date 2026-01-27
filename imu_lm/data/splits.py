from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

try:  # required dependency for parquet IO
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
except Exception as exc:  # pragma: no cover - hard fail
    raise ImportError("pyarrow is required for dataset loading") from exc


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionKey:
    dataset: str
    subject_id: str
    session_id: str

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.dataset, self.subject_id, self.session_id)


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


def build_session_index(parquet_path: str, cfg: Any) -> pd.DataFrame:
    """Build a session-level index with streaming aggregation (no full pandas load)."""

    dataset_col = _cfg_get(cfg, ["data", "loading", "dataset_column"], "dataset")
    subject_col = _cfg_get(cfg, ["data", "loading", "subject_column"], "subject_id")
    session_col = _cfg_get(cfg, ["data", "loading", "session_column"], "session_id")
    time_col = _cfg_get(cfg, ["data", "loading", "time_column"], None)

    max_gap_ms = float(_cfg_get(cfg, ["data", "windowing", "max_gap_ms"], 200.0))
    gap_ns = int(max_gap_ms * 1e6)

    cols = [dataset_col, subject_col, session_col]
    if time_col:
        cols.append(time_col)

    dset = ds.dataset(parquet_path, format="parquet")
    scanner = dset.scanner(columns=cols, batch_size=1_000_000)

    n_rows: Dict[str, int] = {}
    t_min: Dict[str, int] = {}
    t_max: Dict[str, int] = {}
    gap_count: Dict[str, int] = {}

    prev_key0: Optional[str] = None
    prev_t0: Optional[int] = None

    for batch in scanner.to_batches():
        d = pc.cast(batch[dataset_col], pa.string())
        s = pc.cast(batch[subject_col], pa.string())
        e = pc.cast(batch[session_col], pa.string())
        key = pc.binary_join_element_wise([d, s, e], "|")

        gb = pa.table({"key": key}).group_by("key").aggregate([("key", "count")])
        keys_b = gb["key"].to_pylist()
        cnts_b = gb["key_count"].to_numpy()
        for k, c in zip(keys_b, cnts_b):
            n_rows[k] = n_rows.get(k, 0) + int(c)

        if time_col:
            t = pc.cast(batch[time_col], pa.int64()).to_numpy(zero_copy_only=False)

            tb = pa.table({"key": key, "t": pa.array(t, type=pa.int64())})
            gb2 = tb.group_by("key").aggregate([("t", "min"), ("t", "max")])
            keys2 = gb2["key"].to_pylist()
            tmins = gb2["t_min"].to_numpy()
            tmaxs = gb2["t_max"].to_numpy()
            for k, mn, mx in zip(keys2, tmins, tmaxs):
                mn = int(mn); mx = int(mx)
                t_min[k] = mn if k not in t_min else min(t_min[k], mn)
                t_max[k] = mx if k not in t_max else max(t_max[k], mx)

            key_np = np.array(key.to_pylist(), dtype=object)
            if len(t) >= 2:
                dt = t[1:] - t[:-1]
                same = (key_np[1:] == key_np[:-1])
                gaps = (dt > gap_ns) & same
                if gaps.any():
                    gap_keys = key_np[1:][gaps]
                    uniq, counts = np.unique(gap_keys, return_counts=True)
                    for k, c in zip(uniq, counts):
                        gap_count[k] = gap_count.get(k, 0) + int(c)

            if prev_key0 is not None and len(t) >= 1:
                if key_np[0] == prev_key0 and (t[0] - prev_t0) > gap_ns:
                    gap_count[prev_key0] = gap_count.get(prev_key0, 0) + 1

            prev_key0 = key_np[-1]
            prev_t0 = int(t[-1])

    rows = []
    for k, N in n_rows.items():
        ds_, subj_, sess_ = k.split("|", 2)
        rec = {
            dataset_col: ds_,
            subject_col: subj_,
            session_col: sess_,
            "n_rows": int(N),
            "gap_count": int(gap_count.get(k, 0)),
        }
        if time_col:
            rec["t_min"] = int(t_min.get(k, 0))
            rec["t_max"] = int(t_max.get(k, 0))
        rows.append(rec)

    return pd.DataFrame.from_records(rows)


def _group_key(series: pd.Series, key: str) -> pd.Series:
    vals = series.fillna("").astype(str)
    return vals if key else series


def _to_session_keys(df: pd.DataFrame, dataset_col: str, subject_col: str, session_col: str) -> List[SessionKey]:
    return [SessionKey(r[dataset_col], r[subject_col], r[session_col]) for _, r in df.iterrows()]


def _split_by_ratio(groups: List[np.ndarray], ratios: List[float]) -> List[List[np.ndarray]]:
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
    counts = len(groups)
    boundaries = np.cumsum(ratios) * counts
    idxs = [0] + [int(round(b)) for b in boundaries]
    splits: List[List[np.ndarray]] = []
    for i in range(len(ratios)):
        splits.append(groups[idxs[i]: idxs[i + 1]])
    return splits


def make_splits(session_index: pd.DataFrame, cfg: Any) -> Dict[str, List[SessionKey]]:
    """Create deterministic splits according to cfg.data.splits."""

    splits_cfg = _cfg_get(cfg, ["data", "splits"], {}) or {}
    dataset_col = _cfg_get(cfg, ["data", "loading", "dataset_column"], "dataset")
    subject_col = _cfg_get(cfg, ["data", "loading", "subject_column"], "subject_id")
    session_col = _cfg_get(cfg, ["data", "loading", "session_column"], "session_id")

    probe_dataset = splits_cfg.get("probe_dataset")
    group_key = splits_cfg.get("group_key") or session_col
    shuffle = bool(splits_cfg.get("shuffle", False))
    val_ratio = float(splits_cfg.get("val_ratio", 0.0))
    probe_ratios = splits_cfg.get("probe_ratios", [0.7, 0.1, 0.2])
    seed = int(splits_cfg.get("seed", 0))

    rng = np.random.RandomState(seed)

    def group_sessions(df: pd.DataFrame) -> List[np.ndarray]:
        groups = []
        by = _group_key(df[group_key] if group_key in df else df[session_col], group_key)
        for _, g in df.groupby(by):
            groups.append(g)
        if shuffle:
            rng.shuffle(groups)
        return groups

    # Non-probe splits (train/val)
    non_probe_df = session_index[session_index[dataset_col] != probe_dataset]
    non_probe_groups = group_sessions(non_probe_df)
    val_count = int(round(len(non_probe_groups) * val_ratio))
    val_groups = non_probe_groups[:val_count]
    train_groups = non_probe_groups[val_count:]

    train_df = pd.concat(train_groups, ignore_index=True) if train_groups else pd.DataFrame(columns=session_index.columns)
    val_df = pd.concat(val_groups, ignore_index=True) if val_groups else pd.DataFrame(columns=session_index.columns)

    # Probe splits
    probe_df = session_index[session_index[dataset_col] == probe_dataset]
    probe_groups = group_sessions(probe_df)
    pr_train, pr_val, pr_test = _split_by_ratio(probe_groups, probe_ratios)

    probe_train_df = pd.concat(pr_train, ignore_index=True) if pr_train else pd.DataFrame(columns=session_index.columns)
    probe_val_df = pd.concat(pr_val, ignore_index=True) if pr_val else pd.DataFrame(columns=session_index.columns)
    probe_test_df = pd.concat(pr_test, ignore_index=True) if pr_test else pd.DataFrame(columns=session_index.columns)

    result = {
        "train_keys": _to_session_keys(train_df, dataset_col, subject_col, session_col),
        "val_keys": _to_session_keys(val_df, dataset_col, subject_col, session_col),
        "probe_train_keys": _to_session_keys(probe_train_df, dataset_col, subject_col, session_col),
        "probe_val_keys": _to_session_keys(probe_val_df, dataset_col, subject_col, session_col),
        "probe_test_keys": _to_session_keys(probe_test_df, dataset_col, subject_col, session_col),
    }

    def log_split(name: str, df: pd.DataFrame):
        if df.empty:
            logger.info("split=%s empty", name)
            return
        rows = df["n_rows"].sum() if "n_rows" in df else len(df)
        logger.info(
            "split=%s sessions=%d rows=%s datasets=%s subjects=%s",
            name,
            len(df),
            rows,
            dict(df[dataset_col].value_counts()),
            dict(df[subject_col].value_counts()) if subject_col in df else {},
        )

    log_split("train", train_df)
    log_split("val", val_df)
    log_split("probe_train", probe_train_df)
    log_split("probe_val", probe_val_df)
    log_split("probe_test", probe_test_df)

    return result
