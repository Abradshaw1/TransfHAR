from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Optional, List

import yaml
import pandas as pd

try:
    import pyarrow.dataset as ds
except Exception as exc:
    raise ImportError("pyarrow is required for smoke") from exc

from imu_lm.data.splits import SessionKey, build_session_index
from imu_lm.data.loaders import WindowDataset

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Optionally define expected datasets for coverage checking. Set to None to use datasets seen in a quick discovery scan.
EXPECTED_DATASETS: Optional[List[str]] = ["samosa", "opportunity++", "pamap2"]


def _cfg_get(cfg: Any, path, default=None):
    cur = cfg
    for key in path:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, default)
        else:
            cur = getattr(cur, key, default)
    return cur if cur is not None else default


def load_cfg(path: str) -> Any:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _discover_datasets(parquet_path: str, dataset_col: str, limit_batches: int = 5) -> List[str]:
    dataset = ds.dataset(parquet_path, format="parquet")
    scanner = dataset.scanner(columns=[dataset_col], batch_size=100_000)
    names = []
    for i, batch in enumerate(scanner.to_batches()):
        col = batch[dataset_col].to_pandas()
        names.extend(col.dropna().unique().tolist())
        if limit_batches and (i + 1) >= limit_batches:
            break
    return sorted(list(set(names)))


def _scan_sessions_for_dataset(parquet_path: str, cfg: Any, dataset_name: str, max_batches: int = 20) -> pd.DataFrame:
    dataset_col = _cfg_get(cfg, ["data", "loading", "dataset_column"], "dataset")
    subject_col = _cfg_get(cfg, ["data", "loading", "subject_column"], "subject_id")
    session_col = _cfg_get(cfg, ["data", "loading", "session_column"], "session_id")

    columns = [dataset_col, subject_col, session_col]
    dataset = ds.dataset(parquet_path, format="parquet")
    filt = ds.field(dataset_col) == dataset_name
    scanner = dataset.scanner(columns=columns, filter=filt, batch_size=100_000)

    records: List[dict] = []
    for i, batch in enumerate(scanner.to_batches()):
        if len(batch) == 0:
            continue
        df = batch.to_pandas()
        grouped = df.groupby([dataset_col, subject_col, session_col]).size().reset_index(name="n_rows")
        records.extend(grouped.to_dict("records"))
        if max_batches and (i + 1) >= max_batches:
            break

    return pd.DataFrame.from_records(records)


def main(cfg_path: str) -> int:
    cfg = load_cfg(cfg_path)
    parquet_path = cfg["data"]["loading"].get("dataset_path")
    if not parquet_path:
        logger.error("data.loading.dataset_path not set in config")
        return 1

    dataset_col = _cfg_get(cfg, ["data", "loading", "dataset_column"], "dataset")
    subject_col = _cfg_get(cfg, ["data", "loading", "subject_column"], "subject_id")
    session_col = _cfg_get(cfg, ["data", "loading", "session_column"], "session_id")

    # Get dataset list from expectation or a quick discovery scan.
    datasets = EXPECTED_DATASETS if EXPECTED_DATASETS is not None else _discover_datasets(parquet_path, dataset_col, limit_batches=5)
    logger.info("Checking datasets: %s", ", ".join(datasets))

    failures = []

    for ds_name in datasets:
        df_ds = _scan_sessions_for_dataset(parquet_path, cfg, ds_name, max_batches=20)
        if df_ds.empty:
            logger.error("Dataset %s has no sessions", ds_name)
            failures.append(ds_name)
            continue
        sample_sessions = df_ds.sample(n=min(3, len(df_ds)), random_state=0)
        keys = [SessionKey(r[dataset_col], str(r[subject_col]), str(r[session_col])) for _, r in sample_sessions.iterrows()]

        session_fail = False
        for key in keys:
            ds_obj = WindowDataset(parquet_path, df_ds, [key], cfg, split_name=f"smoke_{ds_name}")
            if len(ds_obj) == 0:
                session_fail = True
                logger.error("Dataset %s session=%s: no windows loaded", ds_name, key.session_id)
                break
            max_checks = min(10, len(ds_obj))
            found = False
            for i in range(max_checks):
                sample = ds_obj[i]
                if sample is None:
                    continue
                x, y = sample
                logger.info("Dataset %s session=%s window %d: x.shape=%s y=%s", ds_name, key.session_id, i, tuple(x.shape), int(y))
                found = True
                break
            if not found:
                session_fail = True
                logger.error("Dataset %s session=%s: no valid windows in first %d", ds_name, key.session_id, max_checks)
                break

        if session_fail:
            failures.append(ds_name)

    if failures:
        logger.error("Smoke check FAILED for datasets: %s", ", ".join(failures))
        return 1

    logger.info("Smoke check PASSED for all datasets (%d)", len(datasets))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test parquet pipeline")
    parser.add_argument("--cfg", default="configs/base.yaml", help="Path to YAML config")
    args = parser.parse_args()
    sys.exit(main(args.cfg))
