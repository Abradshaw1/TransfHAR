"""TSTransformer1D Stage A wiring: loaders → encoder + head/decoder → objective → trainer → artifacts.

Supports two objectives:
- MAE (masked): encoder + decoder, self-supervised reconstruction
- Supervised: encoder + LinearHead, fully-supervised classification
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from imu_lm.data.loaders import make_loaders
from imu_lm.data.windowing import compute_T_and_hop
from imu_lm.models.TSTransformer1D.model import TSTransformer1DEncoder, TSTransformer1DDecoder
from imu_lm.objectives import masked_1d as masked_1d_obj
from imu_lm.objectives import supervised as supervised_obj
from imu_lm.probe.head import LinearHead
from imu_lm.runtime_consistency.artifacts import save_encoder, save_supervised_model
from imu_lm.runtime_consistency.trainer import Trainer
from imu_lm.utils.helpers import cfg_get
from imu_lm.utils.training import (
    resolve_resume_path,
    build_optimizer_from_params,
    build_scheduler,
    load_checkpoint,
)


def _make_mae_objective_fn(encoder, decoder, cfg):
    """Wrap masked_1d.forward_loss for MAE objective."""
    def objective_step(batch, model, cfg):
        return masked_1d_obj.forward_loss(batch, encoder, decoder, cfg)
    return objective_step


def _make_supervised_objective_fn(encoder, head, cfg):
    """Wrap supervised.forward_loss for supervised objective."""
    def objective_step(batch, model, cfg):
        return supervised_obj.forward_loss(batch, encoder, head, cfg)
    return objective_step


def main(cfg: Any, run_dir: str, resume_ckpt: Optional[str] = None):
    # Compute input temporal dimension from windowing config
    T, _ = compute_T_and_hop(cfg)
    
    # Get objective type: "mae" or "supervised"
    objective_type = cfg_get(cfg, ["objective", "type"], "mae").lower()
    
    # Create encoder (always needed)
    encoder = TSTransformer1DEncoder(cfg)
    
    if objective_type == "supervised":
        # Supervised: encoder + LinearHead, needs labels
        num_classes = int(cfg_get(cfg, ["objective", "num_classes"], 10))
        head = LinearHead(encoder.embed_dim, num_classes)
        objective_fn = _make_supervised_objective_fn(encoder, head, cfg)
        all_params = list(encoder.parameters()) + list(head.parameters())
        loaders = make_loaders(cfg)
    else:
        # MAE: encoder + decoder, self-supervised
        decoder = TSTransformer1DDecoder(
            embed_dim=encoder.embed_dim,
            out_channels=encoder.in_channels,
            target_T=T,
            cfg=cfg,
        )
        objective_fn = _make_mae_objective_fn(encoder, decoder, cfg)
        all_params = list(encoder.parameters()) + list(decoder.parameters())
        loaders = make_loaders(cfg)
    
    train_loader = loaders.get("train_loader")
    val_loader = loaders.get("val_loader")
    
    optimizer = build_optimizer_from_params(all_params, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    
    # Load checkpoint if resuming
    resume_path = resolve_resume_path(run_dir, resume_ckpt)
    start_step = load_checkpoint(resume_path, encoder, optimizer, scheduler)
    
    trainer = Trainer(cfg, run_dir)
    trainer.fit(encoder, objective_fn, train_loader, val_loader, optimizer, scheduler, start_step=start_step)
    
    # Save artifacts
    data_cfg = cfg_get(cfg, ["data"], {}) or {}
    sensor_cols = data_cfg.get("sensor_columns", ["acc_x", "acc_y", "acc_z"])
    
    meta = {
        "embedding_dim": encoder.embed_dim,
        "encoding": "raw_1d_imu",
        "objective": objective_type,
        "backbone": "tstransformer1d",
        "input_spec": {
            "channels": len(sensor_cols),
            "time_steps": T,
            "format": "[B, C, T]",
        },
        "architecture": {
            "d_model": encoder.d_model,
            "num_layers": encoder.num_layers,
            "nhead": encoder.nhead,
            "pooling": encoder.pooling,
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
