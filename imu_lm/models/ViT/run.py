"""ViT Stage A wiring: loaders → model → objective → trainer → artifacts."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch

from imu_lm.data.loaders import make_loaders
from imu_lm.models.ViT.model import ViTEncoder
from imu_lm.objectives import masked_2d as masked_2d_obj
from imu_lm.runtime_consistency.artifacts import save_encoder
from imu_lm.runtime_consistency.trainer import Trainer
from imu_lm.utils.helpers import cfg_get
from imu_lm.utils.training import (
    resolve_resume_path,
    build_optimizer,
    build_scheduler,
    load_checkpoint,
)


def _select_objective(cfg: Any):
    # Simplified: only masked_2d objective supported (frozen_baseline handled via warm_start)
    return masked_2d_obj.forward_loss


def main(cfg: Any, run_dir: str, resume_ckpt: Optional[str] = None):
    loaders = make_loaders(cfg)
    train_loader = loaders.get("train_loader")
    val_loader = loaders.get("val_loader")

    model = ViTEncoder(cfg)  # mae_model already attached with full encoder+decoder
    objective_fn = _select_objective(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # Load checkpoint if resuming
    resume_path = resolve_resume_path(run_dir, resume_ckpt)
    start_step = load_checkpoint(resume_path, model, optimizer, scheduler)

    trainer = Trainer(cfg, run_dir)
    trainer.fit(model, objective_fn, train_loader, val_loader, optimizer, scheduler, start_step=start_step)

    meta = {
        "embedding_dim": model.embed_dim,
        "encoding": "spectrogram_image",
        "objective": "masked_2d",
        "backbone": model.backbone_name,
        "input_spec": {
            "channels": model.num_channels,
            "height": model.resize_hw[0],
            "width": model.resize_hw[1],
            "patch_size": model.patch_size,
        },
        "normalization": cfg_get(cfg, ["preprocess", "normalize", "method"], None),
    }
    save_encoder(model, meta, run_dir)


if __name__ == "__main__":
    raise SystemExit("Use scripts/run_pretrain.py to launch training.")
