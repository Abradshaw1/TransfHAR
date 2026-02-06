"""ViT1D Stage A wiring: loaders → encoder + head/decoder → objective → trainer → artifacts.

Supports two objectives:
- MAE (masked): encoder + decoder, self-supervised reconstruction
- Supervised: encoder + LinearHead, fully-supervised classification
"""

from __future__ import annotations

from typing import Any, Optional

from imu_lm.data.loaders import make_loaders
from imu_lm.data.windowing import compute_T_and_hop
from imu_lm.models.ViT1D.model import ViT1DEncoder, ViT1DDecoder
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
    build_label_map,
    load_checkpoint,
)


def main(cfg: Any, run_dir: str, resume_ckpt: Optional[str] = None):
    # Compute input temporal dimension from windowing config
    T, _ = compute_T_and_hop(cfg)
    
    # Get objective type: "mae" or "supervised"
    objective_type = cfg_get(cfg, ["objective", "type"], "mae").lower()
    
    # Create encoder (always needed)
    encoder = ViT1DEncoder(cfg)
    
    if objective_type == "supervised":
        # Supervised: encoder + LinearHead, auto-discover classes from data
        loaders = make_loaders(cfg)
        label_map = build_label_map(loaders["train_loader"], cfg)
        raw_to_idx = label_map["raw_to_idx"]
        num_classes = label_map["num_classes"]
        head = LinearHead(encoder.embed_dim, num_classes)
        def objective_fn(batch, model, cfg):
            return supervised_obj.forward_loss(batch, encoder, head, cfg, raw_to_idx=raw_to_idx)
        all_params = list(encoder.parameters()) + list(head.parameters())
    else:
        # MAE: encoder + decoder, self-supervised
        decoder = ViT1DDecoder(
            embed_dim=encoder.embed_dim,
            out_channels=encoder.in_channels,
            target_T=T,
            cfg=cfg,
        )
        def objective_fn(batch, model, cfg):
            return masked_1d_obj.forward_loss(batch, encoder, decoder, cfg)
        all_params = list(encoder.parameters()) + list(decoder.parameters())
        loaders = make_loaders(cfg)
    
    train_loader = loaders.get("train_loader")
    val_loader = loaders.get("val_loader")
    
    optimizer = build_optimizer_from_params(all_params, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    
    # Build extra_modules dict for checkpointing head or decoder
    if objective_type == "supervised":
        extra_modules = {"head": head}
    else:
        extra_modules = {"decoder": decoder}
    
    # Load checkpoint if resuming
    resume_path = resolve_resume_path(run_dir, resume_ckpt)
    start_step = load_checkpoint(resume_path, encoder, optimizer, scheduler, extra_modules=extra_modules)
    
    trainer = Trainer(cfg, run_dir)
    trainer.fit(encoder, objective_fn, train_loader, val_loader, optimizer, scheduler, start_step=start_step, extra_modules=extra_modules)
    
    # Save artifacts
    data_cfg = cfg_get(cfg, ["data"], {}) or {}
    sensor_cols = data_cfg.get("sensor_columns", ["acc_x", "acc_y", "acc_z"])
    vit_cfg = cfg_get(cfg, ["vit1d"], {}) or {}
    
    meta = {
        "embedding_dim": encoder.embed_dim,
        "encoding": "raw_1d_imu",
        "objective": objective_type,
        "backbone": "vit1d",
        "input_spec": {
            "channels": len(sensor_cols),
            "time_steps": T,
            "patch_size": encoder.patch_size,
            "format": "[B, C, T]",
        },
        "architecture": {
            "hidden_size": encoder.embed_dim,
            "num_layers": int(vit_cfg.get("encoder", {}).get("num_hidden_layers", 12)),
            "num_heads": int(vit_cfg.get("encoder", {}).get("num_attention_heads", 6)),
        },
        "normalization": cfg_get(cfg, ["preprocess", "normalize", "method"], None),
    }
    
    if objective_type == "supervised":
        meta["num_classes"] = num_classes
        save_supervised_model(encoder, head, meta, run_dir, label_map=label_map)
    else:
        save_encoder(encoder, meta, run_dir)


if __name__ == "__main__":
    raise SystemExit("Use scripts/run_pretrain.py to launch training.")
