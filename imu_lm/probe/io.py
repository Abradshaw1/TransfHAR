"""Probe I/O helpers: checkpointing, logging, path resolution."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

import torch


def resolve_probe_dir(run_dir: str, cfg: Any, fewshot_k: int | None = None) -> Dict[str, str]:
    probe_cfg = cfg.get("probe", {}) if isinstance(cfg, dict) else getattr(cfg, "probe", {})
    dirname = probe_cfg.get("probe_dirname", probe_cfg.get("output", {}).get("probe_dirname", "probe"))
    if fewshot_k is not None:
        fewshot_dirname = probe_cfg.get("fewshot_probe_dirname", "probe_fewshot")
        base = os.path.join(run_dir, fewshot_dirname, f"k{fewshot_k}")
    else:
        base = os.path.join(run_dir, dirname)
    paths = {
        "base": base,
        "logs": os.path.join(base, "logs"),
        "ckpts": os.path.join(base, "checkpoints"),
        "metrics": os.path.join(base, "logs", "metrics.txt"),
        "summary": os.path.join(base, "summary.txt"),
        "best": os.path.join(base, "checkpoints", "best.pt"),
        "latest": os.path.join(base, "checkpoints", "latest.pt"),
    }
    os.makedirs(paths["logs"], exist_ok=True)
    os.makedirs(paths["ckpts"], exist_ok=True)
    return paths


def save_checkpoint(path: str, head: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, label_map: Dict[str, Any]):
    state = {
        "epoch": epoch,
        "head": head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "label_map": label_map,
    }
    torch.save(state, path)


def load_checkpoint(path: str, device) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    state = torch.load(path, map_location=device)
    head_state = state.get("head", {})
    label_map = state.get("label_map", {})
    return head_state, label_map


def write_metrics_line(metrics_path: str, line: str):
    with open(metrics_path, "a", buffering=1) as f:
        f.write(line + "\n")


def write_summary(summary_path: str, summary: Dict[str, Any]):
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


def select_mapped_batch(
    x: torch.Tensor,
    y_raw: torch.Tensor,
    raw_to_idx: Dict[int, int],
    device,
    unknown_raw_id: int | None = None,
):
    """Map raw labels to contiguous indices.

    Behavior:
    - If raw label is in raw_to_idx, keep it.
    - Else if unknown_raw_id is configured and mapped, map to unknown class.
    - Else drop that sample.
    """

    y_raw = y_raw.to(device)
    raw_vals = y_raw.tolist()

    unknown_idx = None
    if unknown_raw_id is not None and int(unknown_raw_id) in raw_to_idx:
        unknown_idx = raw_to_idx[int(unknown_raw_id)]

    keep_rows = []
    mapped_labels = []
    for row_idx, raw in enumerate(raw_vals):
        raw_int = int(raw)
        if raw_int in raw_to_idx:
            keep_rows.append(row_idx)
            mapped_labels.append(raw_to_idx[raw_int])
            continue
        if unknown_idx is not None:
            keep_rows.append(row_idx)
            mapped_labels.append(unknown_idx)

    if len(keep_rows) == 0:
        return None, None

    keep_tensor = torch.tensor(keep_rows, device=device)
    x_sel = torch.index_select(x.to(device), 0, keep_tensor)
    y_sel = torch.tensor(mapped_labels, dtype=torch.long, device=device)
    return x_sel, y_sel
