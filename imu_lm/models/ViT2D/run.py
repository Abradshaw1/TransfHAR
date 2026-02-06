"""ViT2D Stage A wiring: loaders → encoder + head/HF-MAE → objective → trainer → artifacts.

Supports two objectives:
- MAE (masked_2d): uses HuggingFace ViTMAE with built-in decoder
- Supervised: encoder + LinearHead, fully-supervised classification
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from imu_lm.data.loaders import make_loaders
from imu_lm.models.ViT2D.model import ViTEncoder
from imu_lm.objectives import masked_2d as masked_2d_obj
from imu_lm.objectives import supervised as supervised_obj
from imu_lm.probe.head import LinearHead
from imu_lm.runtime_consistency.artifacts import save_encoder, save_supervised_model
from imu_lm.runtime_consistency.trainer import Trainer
from imu_lm.utils.helpers import cfg_get
from imu_lm.utils.training import (
    resolve_resume_path,
    build_optimizer,
    build_optimizer_from_params,
    build_scheduler,
    load_checkpoint,
)


def _make_mae_objective_fn(encoder, cfg):
    """Wrap masked_2d.forward_loss for MAE objective (HuggingFace ViTMAE)."""
    def objective_step(batch, model, cfg):
        return masked_2d_obj.forward_loss(batch, encoder, cfg)
    return objective_step


def _make_supervised_objective_fn(encoder, head, cfg):
    """Wrap supervised.forward_loss for supervised objective."""
    def objective_step(batch, model, cfg):
        return supervised_obj.forward_loss(batch, encoder, head, cfg)
    return objective_step


def main(cfg: Any, run_dir: str, resume_ckpt: Optional[str] = None):
    # Get objective type: "mae" or "supervised"
    objective_type = cfg_get(cfg, ["objective", "type"], "mae").lower()
    
    # Create encoder (always needed)
    encoder = ViTEncoder(cfg)
    
    if objective_type == "supervised":
        # Supervised: encoder + LinearHead, needs labels
        num_classes = int(cfg_get(cfg, ["objective", "num_classes"], 10))
        head = LinearHead(encoder.embed_dim, num_classes)
        objective_fn = _make_supervised_objective_fn(encoder, head, cfg)
        all_params = list(encoder.parameters()) + list(head.parameters())
        optimizer = build_optimizer_from_params(all_params, cfg)
        loaders = make_loaders(cfg)
    else:
        # MAE: HuggingFace ViTMAE handles decoder internally
        objective_fn = _make_mae_objective_fn(encoder, cfg)
        optimizer = build_optimizer(encoder, cfg)
        loaders = make_loaders(cfg)
    
    train_loader = loaders.get("train_loader")
    val_loader = loaders.get("val_loader")
    
    scheduler = build_scheduler(optimizer, cfg)

    # Load checkpoint if resuming
    resume_path = resolve_resume_path(run_dir, resume_ckpt)
    start_step = load_checkpoint(resume_path, encoder, optimizer, scheduler)

    trainer = Trainer(cfg, run_dir)
    trainer.fit(encoder, objective_fn, train_loader, val_loader, optimizer, scheduler, start_step=start_step)

    meta = {
        "embedding_dim": encoder.embed_dim,
        "encoding": "spectrogram_image",
        "objective": objective_type,
        "backbone": encoder.backbone_name,
        "input_spec": {
            "channels": encoder.num_channels,
            "height": encoder.resize_hw[0],
            "width": encoder.resize_hw[1],
            "patch_size": encoder.patch_size,
        },
        "normalization": cfg_get(cfg, ["preprocess", "normalize", "method"], None),
    }
    
    if objective_type == "supervised":
        meta["num_classes"] = num_classes
        save_supervised_model(encoder, head, meta, run_dir)
    else:
        save_encoder(encoder, meta, run_dir)


if __name__ == "__main__":
    raise SystemExit("Use scripts/run_pretrain.py to launch training.")
