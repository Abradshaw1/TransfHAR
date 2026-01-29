"""MAE objective using Hugging Face ViTMAEForPreTraining (paper-aligned masking)."""

from __future__ import annotations

from typing import Any, Dict

import torch
from transformers import ViTMAEConfig, ViTMAEForPreTraining


def ensure_mae_pretrain_head(model, cfg: Any):
    """Build and attach the MAE pretraining head once (canonical path)."""

    if hasattr(model, "mae_model"):
        return

    mae_cfg = cfg.get("objective", {}).get("mae", {}) if isinstance(cfg, dict) else getattr(cfg, "objective", {}).get("mae", {})
    mask_ratio = float(mae_cfg.get("mask_ratio", 0.75))
    norm_pix = bool(mae_cfg.get("norm_pix_loss", False))
    dec_cfg = mae_cfg.get("decoder", {}) or {}

    cfg_copy = model.backbone.config.to_dict()
    cfg_copy.update(
        dict(
            mask_ratio=mask_ratio,
            norm_pix_loss=norm_pix,
            decoder_hidden_size=int(dec_cfg.get("hidden_size", cfg_copy.get("decoder_hidden_size", 512))),
            decoder_num_hidden_layers=int(dec_cfg.get("num_hidden_layers", cfg_copy.get("decoder_num_hidden_layers", 8))),
            decoder_num_attention_heads=int(dec_cfg.get("num_attention_heads", cfg_copy.get("decoder_num_attention_heads", 16))),
            decoder_intermediate_size=int(dec_cfg.get("intermediate_size", cfg_copy.get("decoder_intermediate_size", 2048))),
            decoder_hidden_dropout_prob=float(dec_cfg.get("hidden_dropout_prob", cfg_copy.get("decoder_hidden_dropout_prob", 0.0))),
            decoder_attention_probs_dropout_prob=float(dec_cfg.get("attention_probs_dropout_prob", cfg_copy.get("decoder_attention_probs_dropout_prob", 0.0))),
            hidden_act=dec_cfg.get("hidden_act", cfg_copy.get("hidden_act", "gelu")),
        )
    )

    hf_cfg = ViTMAEConfig(**cfg_copy)
    mae_model = ViTMAEForPreTraining(hf_cfg)
    mae_model.vit.load_state_dict(model.backbone.state_dict(), strict=False)
    model.mae_model = mae_model.to(next(model.parameters()).device)


def forward_loss(batch, model, cfg):
    """
    Uses HF ViTMAEForPreTraining to handle masking + decoder per config.
    Args:
        batch: (x, y) where x is [B,3,F,TT] image-like spectrogram.
        model: ViTEncoder wrapper.
    Returns:
        loss, logs dict
    """

    x, _ = batch  # y ignored for MAE
    mae_cfg = cfg.get("objective", {}).get("mae", {}) if isinstance(cfg, dict) else getattr(cfg, "objective", {}).get("mae", {})
    mask_ratio = float(mae_cfg.get("mask_ratio", 0.75))

    ensure_mae_pretrain_head(model, cfg)

    x_img = model._prepare(x)  # [B,3,H,W]
    outputs = model.mae_model(pixel_values=x_img, mask_ratio=mask_ratio, output_hidden_states=False)
    loss = outputs.loss

    logs = {
        "loss": loss.detach().item(),
        "mask_ratio": mask_ratio,
        "recon_mse": loss.detach().item(),
    }
    return loss, logs
