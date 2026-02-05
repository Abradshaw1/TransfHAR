"""ViT1D Stage A wiring: loaders → encoder + decoder → objective → trainer → artifacts.

The encoder is objective-agnostic. The decoder is ViT1D-specific and lives in model.py.
After pretraining, we discard the decoder and save only the encoder.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch

from imu_lm.data.loaders import make_loaders
from imu_lm.data.windowing import compute_T_and_hop
from imu_lm.models.ViT1D.model import ViT1DEncoder, ViT1DDecoder
from imu_lm.objectives import masked_1d as masked_1d_obj
from imu_lm.runtime_consistency.artifacts import save_encoder
from imu_lm.runtime_consistency.trainer import Trainer
from imu_lm.utils.helpers import cfg_get
from imu_lm.utils.training import (
    resolve_resume_path,
    build_optimizer_from_params,
    build_scheduler,
    load_checkpoint,
)


def _make_objective_fn(encoder: ViT1DEncoder, decoder: ViT1DDecoder, cfg: Any):
    """Wrap masked_1d.forward_loss to match Trainer's expected signature.
    
    Trainer expects: objective_step(batch, model, cfg) → (loss, logs)
    masked_1d.forward_loss needs encoder and decoder separately.
    """
    def objective_step(batch, model, cfg):
        return masked_1d_obj.forward_loss(batch, encoder, decoder, cfg)
    return objective_step


def main(cfg: Any, run_dir: str, resume_ckpt: Optional[str] = None):
    loaders = make_loaders(cfg)
    train_loader = loaders.get("train_loader")
    val_loader = loaders.get("val_loader")
    
    # Compute input temporal dimension from windowing config
    T, _ = compute_T_and_hop(cfg)
    
    # Create encoder (what we keep) and decoder (throwaway after pretraining)
    encoder = ViT1DEncoder(cfg)
    decoder = ViT1DDecoder(
        embed_dim=encoder.embed_dim,
        out_channels=encoder.in_channels,
        target_T=T,
        cfg=cfg,
    )
    
    # Wrap objective to match Trainer signature
    objective_fn = _make_objective_fn(encoder, decoder, cfg)
    
    # Combine encoder + decoder params for optimizer
    all_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = build_optimizer_from_params(all_params, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    
    # Load checkpoint if resuming
    resume_path = resolve_resume_path(run_dir, resume_ckpt)
    start_step = load_checkpoint(resume_path, encoder, optimizer, scheduler)
    
    trainer = Trainer(cfg, run_dir)
    trainer.fit(encoder, objective_fn, train_loader, val_loader, optimizer, scheduler, start_step=start_step)
    
    # Save encoder only (decoder is discarded)
    data_cfg = cfg_get(cfg, ["data"], {}) or {}
    sensor_cols = data_cfg.get("sensor_columns", ["acc_x", "acc_y", "acc_z"])
    vit_cfg = cfg_get(cfg, ["vit1d"], {}) or {}
    
    meta = {
        "embedding_dim": encoder.embed_dim,
        "encoding": "raw_1d_imu",
        "objective": "masked_1d",
        "backbone": "vit1d",
        "input_spec": {
            "channels": len(sensor_cols),
            "time_steps": T,
            "patch_size": encoder.patch_size,
            "format": "[B, C, T]",
        },
        "architecture": {
            "hidden_size": encoder.embed_dim,
            "num_layers": int(vit_cfg.get("num_hidden_layers", 12)),
            "num_heads": int(vit_cfg.get("num_attention_heads", 6)),
        },
        "normalization": cfg_get(cfg, ["preprocess", "normalize", "method"], None),
    }
    save_encoder(encoder, meta, run_dir)


if __name__ == "__main__":
    raise SystemExit("Use scripts/run_pretrain.py to launch training.")
