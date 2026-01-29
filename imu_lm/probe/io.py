"""Probe I/O helpers: checkpointing, logging, path resolution."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

import torch


def resolve_probe_dir(run_dir: str, cfg: Any) -> Dict[str, str]:
    probe_cfg = cfg.get("probe", {}) if isinstance(cfg, dict) else getattr(cfg, "probe", {})
    dirname = probe_cfg.get("output", {}).get("probe_dirname", "probe")
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
