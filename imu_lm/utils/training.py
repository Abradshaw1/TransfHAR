"""Shared training utilities for model run.py files.

Common functions for optimizer, scheduler, checkpoint loading, etc.
"""

from __future__ import annotations

import math
import os
import logging
from typing import Any, Dict, Optional, List

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from imu_lm.utils.helpers import cfg_get

logger = logging.getLogger(__name__)


def resolve_resume_path(run_dir: str, resume: Optional[str]) -> Optional[str]:
    """Resolve checkpoint path from resume argument.
    
    Args:
        run_dir: Run directory path
        resume: "latest", "best", absolute path, or relative path
        
    Returns:
        Resolved absolute path or None
    """
    if not resume:
        return None
    if resume in {"latest", "best"}:
        return os.path.join(run_dir, "checkpoints", f"{resume}.pt")
    if os.path.isabs(resume):
        return resume
    return os.path.join(run_dir, resume)


def build_optimizer_from_params(params: List, cfg: Any):
    """Build optimizer from parameter list.
    
    Supports both flat config (trainer.lr) and nested config (trainer.optim.lr).
    """
    tcfg = cfg_get(cfg, ["trainer"], {}) or {}
    
    # Check for nested optim config first, then flat
    ocfg = tcfg.get("optim", {}) or {}
    
    if ocfg:
        # Nested: trainer.optim.*
        optim_name = ocfg.get("name", "adam").lower()
        lr = float(ocfg.get("lr", 0.001))
        wd = float(ocfg.get("weight_decay", 0.0))
        betas = tuple(ocfg.get("betas", [0.9, 0.999]))
        eps = float(ocfg.get("eps", 1e-8))
    else:
        # Flat: trainer.*
        optim_name = "adamw"
        lr = float(tcfg.get("lr", 1.5e-4))
        wd = float(tcfg.get("weight_decay", 0.05))
        betas = tuple(tcfg.get("betas", [0.9, 0.95]))
        eps = float(tcfg.get("eps", 1e-8))
    
    if optim_name == "adamw":
        return AdamW(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)
    return Adam(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)


def build_optimizer(model: nn.Module, cfg: Any):
    """Build optimizer from model parameters."""
    return build_optimizer_from_params(model.parameters(), cfg)


def build_scheduler(optimizer, cfg: Any):
    """Build learning rate scheduler.
    
    Supports cosine with warmup and reduce_on_plateau.
    """
    tcfg = cfg_get(cfg, ["trainer"], {}) or {}
    scfg = tcfg.get("sched", {})
    
    # Handle both nested (sched.name) and flat (sched as string) config
    if isinstance(scfg, dict):
        sched_name = scfg.get("name", "cosine").lower()
        warmup_steps = int(scfg.get("warmup_steps", 0))
    else:
        sched_name = str(scfg).lower() if scfg else "cosine"
        warmup_steps = int(tcfg.get("warmup_steps", 0))
    
    max_steps = int(tcfg.get("max_steps", 100000))
    
    if sched_name == "reduce_on_plateau":
        scfg = tcfg.get("sched", {}) if isinstance(tcfg.get("sched"), dict) else {}
        patience = int(scfg.get("patience", 5))
        factor = float(scfg.get("factor", 0.5))
        return ReduceLROnPlateau(optimizer, mode="min", patience=patience, factor=factor)
    
    if sched_name != "cosine":
        return None
    
    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return step / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    
    return LambdaLR(optimizer, lr_lambda)


def build_label_map(
    loader,
    cfg: Any,
    unknown_id: Optional[int] = None,
    drop_unknown: bool = True,
    min_count: int = 0,
) -> Dict[str, Any]:
    """Scan a DataLoader to discover unique labels and build contiguous mapping.

    Shared by supervised training and probe.

    Returns:
        Dict with keys: raw_to_idx, idx_to_raw, num_classes, unknown_id
    """
    counts: Dict[int, int] = {}
    for batch in loader:
        if batch is None:
            continue
        _, y = batch
        for v in y.tolist():
            v_int = int(v)
            if drop_unknown and unknown_id is not None and v_int == int(unknown_id):
                continue
            counts[v_int] = counts.get(v_int, 0) + 1

    kept = [k for k, c in counts.items() if c >= min_count]
    if not drop_unknown and unknown_id is not None and int(unknown_id) not in kept:
        kept.append(int(unknown_id))
    kept = sorted(kept)

    raw_to_idx = {int(r): i for i, r in enumerate(kept)}
    idx_to_raw = {i: int(r) for i, r in enumerate(kept)}
    logger.info("build_label_map: classes=%d", len(raw_to_idx))
    return {
        "raw_to_idx": raw_to_idx,
        "idx_to_raw": idx_to_raw,
        "num_classes": len(raw_to_idx),
        "unknown_id": unknown_id,
    }


def remap_labels(y: torch.Tensor, raw_to_idx: Dict[int, int]) -> torch.Tensor:
    """Remap raw label tensor to contiguous indices. Non-mapped labels get -100 (ignored by cross_entropy)."""
    mapped = torch.full_like(y, -100, dtype=torch.long)
    for raw, idx in raw_to_idx.items():
        mapped[y == raw] = idx
    return mapped


def load_checkpoint(
    resume_path: str,
    model: nn.Module,
    optimizer: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    extra_modules: Optional[Dict[str, nn.Module]] = None,
) -> int:
    """Load checkpoint and return start step.
    
    Args:
        resume_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to restore
        scheduler: Optional scheduler to restore
        extra_modules: Optional dict of named modules (e.g. head, decoder)
            whose state was saved alongside the model
        
    Returns:
        Start step from checkpoint (0 if not found)
    """
    if not resume_path:
        return 0
    if not os.path.exists(resume_path):
        logger.warning("[resume] checkpoint not found: %s — starting from scratch", resume_path)
        return 0
    
    logger.info("[resume] loading checkpoint from %s", resume_path)
    state = torch.load(resume_path, map_location="cpu", weights_only=False)
    
    if "model" in state:
        model.load_state_dict(state["model"], strict=True)
        logger.info("[resume] model loaded (%d params)", len(state["model"]))
    
    if extra_modules:
        for name, mod in extra_modules.items():
            if name in state:
                mod.load_state_dict(state[name], strict=True)
                logger.info("[resume] extra module '%s' loaded (%d params)", name, len(state[name]))
            else:
                logger.warning("[resume] extra module '%s' not found in checkpoint", name)
    
    if optimizer is not None and state.get("optimizer"):
        try:
            optimizer.load_state_dict(state["optimizer"])
            logger.info("[resume] optimizer loaded")
        except Exception as e:
            logger.warning("[resume] optimizer load failed: %s", e)
    
    if scheduler is not None and state.get("scheduler"):
        try:
            scheduler.load_state_dict(state["scheduler"])
            logger.info("[resume] scheduler loaded")
        except Exception as e:
            logger.warning("[resume] scheduler load failed: %s", e)
    
    start_step = int(state.get("step", 0))
    logger.info("[resume] resuming from step %d", start_step)
    return start_step
