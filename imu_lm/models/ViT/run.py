"""ViT Stage A wiring: loaders → model → objective → trainer → artifacts."""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from imu_lm.data.loaders import make_loaders
from imu_lm.models.ViT.model import ViTEncoder
from imu_lm.objectives import mae as mae_obj
from imu_lm.runtime_consistency.artifacts import save_encoder
from imu_lm.runtime_consistency.trainer import Trainer
from imu_lm.utils.helpers import cfg_get


def _build_optimizer(model: torch.nn.Module, cfg: Any):
    ocfg = cfg_get(cfg, ["trainer", "optim"], None)
    if ocfg is None:
        ocfg = cfg_get(cfg, ["optim"], {}) or {}
    lr = float(ocfg.get("lr", 1.5e-4))
    wd = float(ocfg.get("weight_decay", 0.05))
    betas = tuple(ocfg.get("betas", [0.9, 0.95]))
    eps = float(ocfg.get("eps", 1e-8))
    return AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas, eps=eps)


def _build_scheduler(optimizer, cfg: Any):
    scfg = cfg_get(cfg, ["trainer", "sched"], None)
    if scfg is None:
        scfg = cfg_get(cfg, ["sched"], {}) or {}
    name = scfg.get("name", "cosine")
    warmup_steps = int(scfg.get("warmup_steps", 0))
    max_steps = int(cfg_get(cfg, ["trainer", "max_steps"], 100000))

    if name != "cosine":
        return None

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return step / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, lr_lambda)


def _select_objective(cfg: Any):
    name = cfg_get(cfg, ["objective", "name"], "mae")
    if name == "mae":
        return mae_obj.forward_loss
    if name == "frozen_baseline":
        def frozen_step(batch, model, cfg):
            x, _ = batch
            x = x.to(next(model.parameters()).device)
            with torch.no_grad():
                model.forward_features(x)
            zero = torch.tensor(0.0, device=x.device, requires_grad=True)
            return zero, {"loss": 0.0}
        return frozen_step
    raise ValueError(f"Unsupported objective {name}")


def _resolve_resume_path(run_dir: str, resume: Optional[str]) -> Optional[str]:
    if not resume:
        return None
    if resume in {"latest", "best"}:
        candidate = os.path.join(run_dir, "checkpoints", f"{resume}.pt")
        return candidate
    if os.path.isabs(resume):
        return resume
    return os.path.join(run_dir, resume)


def main(cfg: Any, run_dir: str, resume_ckpt: Optional[str] = None):
    loaders = make_loaders(cfg)
    train_loader = loaders.get("train_loader")
    val_loader = loaders.get("val_loader")

    model = ViTEncoder(cfg)
    if cfg_get(cfg, ["objective", "name"], "mae") == "mae":
        # Build MAE pretraining head here so optimizer sees decoder/mask token params
        mae_obj.ensure_mae_pretrain_head(model, cfg)
    objective_fn = _select_objective(cfg)
    optimizer = _build_optimizer(model, cfg)
    scheduler = _build_scheduler(optimizer, cfg)

    start_step = 0
    resume_path = _resolve_resume_path(run_dir, resume_ckpt)
    if resume_path and os.path.exists(resume_path):
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
        start_step = int(state.get("step", 0))

    trainer = Trainer(cfg, run_dir)
    trainer.fit(model, objective_fn, train_loader, val_loader, optimizer, scheduler, start_step=start_step)

    meta = {
        "embedding_dim": model.embed_dim,
        "encoding": "spectrogram_image",
        "objective": cfg_get(cfg, ["objective", "name"], "mae"),
        "backbone": model.backbone_name,
        "input_spec": {
            "channels": model.num_channels,
            "height": model.resize_hw[0],
            "width": model.resize_hw[1],
            "patch_size": model.patch_size,
        },
        "normalization": cfg_get(cfg, ["data", "preprocess", "normalize", "method"], None),
    }
    save_encoder(model, meta, run_dir)


if __name__ == "__main__":
    raise SystemExit("Use scripts/run_pretrain.py to launch training.")
