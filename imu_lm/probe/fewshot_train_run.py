"""Stage B fewshot evaluation: frozen encoder + linear head, sweep over k shots.

For full-data probe, see train_run.py.

Supports:
- Single k: fewshot_shots_per_class = 5
- Sweep:    fewshot_shots_per_class = [1, 5, 10, 25, 50]

Output structure:
    run_dir/probe_fewshot/k1/   (checkpoints/, logs/, summary.txt, probe_meta.json)
    run_dir/probe_fewshot/k5/
    ...
    run_dir/probe_fewshot/sweep_summary.json
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List

import pyarrow.dataset as pa_ds
from torch.utils.data import DataLoader, Subset

from imu_lm.data.splits import SessionKey
from imu_lm.data.windowing import resolve_window_label
from imu_lm.probe.io import resolve_probe_dir
from imu_lm.probe.train_run import setup_probe, run_probe


def _bulk_read_session_labels(dataset) -> Dict[SessionKey, "np.ndarray"]:
    """Single-pass parquet read of label column for all sessions in the dataset.

    Returns {session_key: label_array} with labels in row order.
    One parquet query per unique dataset name (typically just 1 for probe).
    """
    import numpy as np

    pa = pa_ds.dataset(dataset.parquet_path, format="parquet")
    label_col = dataset._label_col
    subj_col = dataset._subject_col
    sess_col = dataset._session_col
    ds_col = dataset._dataset_col
    time_col = dataset._time_col

    # Unique dataset names (usually 1, e.g. "samosa")
    needed_keys = set(dataset._keys)
    ds_names = set(k.dataset for k in needed_keys)

    cols = [label_col, subj_col, sess_col]
    if time_col:
        cols.append(time_col)

    session_labels: Dict[SessionKey, np.ndarray] = {}
    for ds_name in ds_names:
        table = pa.to_table(columns=cols, filter=pa_ds.field(ds_col) == ds_name)
        df = table.to_pandas()
        if time_col and time_col in df.columns:
            df = df.sort_values([subj_col, sess_col, time_col])
        for (subj, sess), grp in df.groupby([subj_col, sess_col]):
            key = SessionKey(ds_name, str(subj), str(sess))
            if key in needed_keys:
                session_labels[key] = grp[label_col].to_numpy()

    return session_labels


def _fewshot_subset(loader: DataLoader, label_map: Dict[str, Any], shots_per_class: int, seed: int) -> DataLoader:
    """Subsample train loader to k windows per class.

    Bulk-reads labels from parquet (1 query, label column only),
    then resolves per-window labels in memory — no preprocessing/STFT.
    """
    import numpy as np

    raw_to_idx = {int(k): int(v) for k, v in label_map.get("raw_to_idx", {}).items()}
    dataset = loader.dataset
    rng = random.Random(seed)
    per_class: Dict[int, List[int]] = {idx: [] for idx in raw_to_idx.values()}

    # One bulk parquet read — label column only
    session_labels = _bulk_read_session_labels(dataset)

    for idx in range(len(dataset)):
        key = dataset._keys[dataset._sess_idx[idx]]
        y_arr = session_labels.get(key)
        if y_arr is None:
            continue
        start = int(dataset._starts[idx])
        yw = y_arr[start : start + dataset._T]
        label = resolve_window_label(yw, dataset.cfg)
        if label is None:
            continue
        raw = int(label)
        if raw not in raw_to_idx:
            continue
        mapped = raw_to_idx[raw]
        per_class[mapped].append(idx)

    keep_indices: List[int] = []
    for cls_idx, idxs in per_class.items():
        rng.shuffle(idxs)
        keep_indices.extend(idxs[:shots_per_class])

    subset = Subset(dataset, keep_indices)
    # loader.batch_size is None when using SessionGroupedBatchSampler
    bs = loader.batch_size or 128
    return DataLoader(
        subset,
        batch_size=bs,
        shuffle=True,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=False,
        collate_fn=loader.collate_fn,
    )


def main(cfg: Any, run_dir: str):
    probe_cfg = cfg.get("probe", {}) if isinstance(cfg, dict) else getattr(cfg, "probe", {})
    fewshot_seed = int(probe_cfg.get("fewshot_seed", 0))
    raw_shots = probe_cfg.get("fewshot_shots_per_class", 5)

    # Normalize to list
    if isinstance(raw_shots, list):
        shot_list = sorted([int(s) for s in raw_shots])
    else:
        shot_list = [int(raw_shots)]

    # Shared setup (encoder, loaders, label_map, etc.) — loaded once
    ctx = setup_probe(cfg, run_dir)
    logger = ctx["logger"]
    logger.info("[fewshot] shots=%s seed=%d", shot_list, fewshot_seed)

    # Sweep over each k
    all_summaries = {}
    for k in shot_list:
        logger.info("[fewshot] ========== k=%d ==========", k)
        fs_loader = _fewshot_subset(ctx["train_loader"], ctx["label_map"], k, fewshot_seed)
        logger.info("[fewshot] k=%d train_windows=%d", k, len(fs_loader.dataset))

        summary = run_probe(
            encoder=ctx["encoder"], train_loader=fs_loader,
            val_loader=ctx["val_loader"], test_loader=ctx["test_loader"],
            label_map=ctx["label_map"], label_names=ctx["label_names"],
            embed_dim=ctx["embed_dim"], num_classes=ctx["num_classes"],
            train_cfg=ctx["train_cfg"],
            paths=resolve_probe_dir(run_dir, cfg, fewshot_k=k),
            probe_dataset=ctx["probe_dataset"], device=ctx["device"],
            logger=logger, wb_prefix=f"probe_k{k}",
            extra_meta={"shots_per_class": k},
        )
        all_summaries[f"k{k}"] = summary
        logger.info("[fewshot] k=%d best_%s=%.4f test_macro_f1=%.4f",
                    k, summary["selection_metric"], summary["best_metric"],
                    summary.get("test", {}).get("macro_f1", 0.0))

    # Write combined sweep summary
    fewshot_dirname = cfg.get("probe", {}).get("fewshot_probe_dirname", "probe_fewshot")
    sweep_dir = os.path.join(run_dir, fewshot_dirname)
    os.makedirs(sweep_dir, exist_ok=True)
    sweep_path = os.path.join(sweep_dir, "sweep_summary.json")
    with open(sweep_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    logger.info("[fewshot] sweep summary written to %s", sweep_path)

    # Print final table
    print("\n=== Fewshot Sweep Results ===")
    print(f"{'k':>6}  {'best_val':>10}  {'test_f1':>10}  {'test_acc':>10}  {'epochs':>6}")
    print("-" * 50)
    for k in shot_list:
        s = all_summaries.get(f"k{k}", {})
        t = s.get("test", {})
        print(f"{k:>6}  {s.get('best_metric', 0):.4f}      {t.get('macro_f1', 0):.4f}      {t.get('acc', 0):.4f}      {s.get('best_epoch', 0):>6}")
    print("=" * 50)
