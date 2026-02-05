"""Shared training utilities for model run.py files.

Common functions for optimizer, scheduler, checkpoint loading, etc.
"""

from __future__ import annotations

import math
import os
from typing import Any, Optional, List

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from imu_lm.utils.helpers import cfg_get


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


def load_checkpoint(
    resume_path: str,
    model: nn.Module,
    optimizer: Optional[Any] = None,
    scheduler: Optional[Any] = None,
) -> int:
    """Load checkpoint and return start step.
    
    Args:
        resume_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to restore
        scheduler: Optional scheduler to restore
        
    Returns:
        Start step from checkpoint (0 if not found)
    """
    if not resume_path or not os.path.exists(resume_path):
        return 0
    
    state = torch.load(resume_path, map_location="cpu")
    
    if "model" in state:
        model.load_state_dict(state["model"], strict=False)
    
    if optimizer is not None and state.get("optimizer"):
        try:
            optimizer.load_state_dict(state["optimizer"])
        except Exception:
            pass
    
    if scheduler is not None and state.get("scheduler"):
        try:
            scheduler.load_state_dict(state["scheduler"])
        except Exception:
            pass
    
    return int(state.get("step", 0))
