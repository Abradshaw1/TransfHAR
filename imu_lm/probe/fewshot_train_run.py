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

from torch.utils.data import DataLoader, Subset

from imu_lm.probe.io import resolve_probe_dir
from imu_lm.probe.train_run import setup_probe, run_probe


def _fewshot_subset(loader: DataLoader, label_map: Dict[str, Any], shots_per_class: int, seed: int) -> DataLoader:
    """Subsample train loader to k windows per class."""
    raw_to_idx = {int(k): int(v) for k, v in label_map.get("raw_to_idx", {}).items()}
    dataset = loader.dataset
    rng = random.Random(seed)
    per_class: Dict[int, List[int]] = {idx: [] for idx in raw_to_idx.values()}

    for idx in range(len(dataset)):
        item = dataset[idx]
        if item is None:
            continue
        _, y = item
        raw = int(y)
        if raw not in raw_to_idx:
            continue
        mapped = raw_to_idx[raw]
        per_class[mapped].append(idx)

    keep_indices: List[int] = []
    for cls_idx, idxs in per_class.items():
        rng.shuffle(idxs)
        keep_indices.extend(idxs[:shots_per_class])

    subset = Subset(dataset, keep_indices)
    return DataLoader(
        subset,
        batch_size=loader.batch_size,
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
    sweep_dir = os.path.join(run_dir, "probe_fewshot")
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
