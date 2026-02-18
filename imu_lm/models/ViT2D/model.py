"""ViT-MAE wrapper using HuggingFace ViTMAEForPreTraining.

Single source of truth for model loading:
- warm_start=true: load pretrained HF weights with HF defaults
- warm_start=false: random init with YAML config overrides

Exposes:
- forward(x) / forward_features(x): pooled embedding [B, D] for probes/supervised
- mae_model: full HF model for MAE pretraining (handles masking internally)
"""

from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.nn as nn
from transformers import ViTMAEConfig, ViTMAEForPreTraining

from imu_lm.utils.helpers import resize_bilinear


class ViTEncoder(nn.Module):
    def __init__(self, cfg: Any):
        super().__init__()

        vit_cfg = cfg.get("vit", {}) if isinstance(cfg, dict) else getattr(cfg, "vit", {})
        warm_start = bool(vit_cfg.get("warm_start", True))
        pretrained_id = vit_cfg.get("pretrained_id", "facebook/vit-mae-base")

        # Input params under vit
        resize_hw = vit_cfg.get("resize_hw", [224, 224])
        patch_size = int(vit_cfg.get("patch_size", 16))
        num_channels = int(vit_cfg.get("num_channels", 3))

        # Objective config (mask_ratio, norm_pix_loss directly under objective)
        obj_cfg = cfg.get("objective", {}) if isinstance(cfg, dict) else getattr(cfg, "objective", {})
        mask_ratio = float(obj_cfg.get("mask_ratio", 0.75))
        norm_pix = bool(obj_cfg.get("norm_pix_loss", False))

        # Encoder/decoder arch configs (top-level sections)
        enc_cfg = cfg.get("encoder", {}) if isinstance(cfg, dict) else getattr(cfg, "encoder", {})
        dec_cfg = cfg.get("decoder", {}) if isinstance(cfg, dict) else getattr(cfg, "decoder", {})

        if warm_start:
            # Load pretrained with HF defaults, only override mask_ratio/norm_pix
            self.mae_model = ViTMAEForPreTraining.from_pretrained(pretrained_id)
            self.mae_model.config.mask_ratio = mask_ratio
            self.mae_model.config.norm_pix_loss = norm_pix
        else:
            # Build from scratch with YAML overrides
            if resize_hw[0] != resize_hw[1]:
                raise ValueError("ViTMAE scratch init expects square resize_hw")
            hf_cfg = ViTMAEConfig(
                image_size=int(resize_hw[0]),
                patch_size=patch_size,
                num_channels=num_channels,
                # Encoder (from top-level encoder section)
                hidden_size=int(enc_cfg.get("hidden_size", 768)),
                num_hidden_layers=int(enc_cfg.get("num_hidden_layers", 12)),
                num_attention_heads=int(enc_cfg.get("num_attention_heads", 12)),
                intermediate_size=int(enc_cfg.get("intermediate_size", 3072)),
                hidden_act=enc_cfg.get("hidden_act", "gelu"),
                hidden_dropout_prob=float(enc_cfg.get("hidden_dropout_prob", 0.0)),
                attention_probs_dropout_prob=float(enc_cfg.get("attention_probs_dropout_prob", 0.0)),
                qkv_bias=bool(enc_cfg.get("qkv_bias", True)),
                layer_norm_eps=float(enc_cfg.get("layer_norm_eps", 1e-12)),
                initializer_range=float(enc_cfg.get("initializer_range", 0.02)),
                # Decoder (from top-level decoder section)
                decoder_hidden_size=int(dec_cfg.get("hidden_size", 512)),
                decoder_num_hidden_layers=int(dec_cfg.get("num_hidden_layers", 8)),
                decoder_num_attention_heads=int(dec_cfg.get("num_attention_heads", 16)),
                decoder_intermediate_size=int(dec_cfg.get("intermediate_size", 2048)),
                # MAE
                mask_ratio=mask_ratio,
                norm_pix_loss=norm_pix,
            )
            self.mae_model = ViTMAEForPreTraining(hf_cfg)

        self.resize_hw: Tuple[int, int] = (int(resize_hw[0]), int(resize_hw[1]))
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = int(self.mae_model.config.hidden_size)
        self.pooling = enc_cfg.get("pooling", "mean")  # pooling is under encoder
        self.backbone_name = pretrained_id if warm_start else "vit_mae_scratch"

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected x with shape [B,3,H,W], got {x.shape}")
        return resize_bilinear(x.float(), self.resize_hw)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled embedding [B, D] for probes."""
        x_img = self._prepare(x)
        out = self.mae_model.vit(pixel_values=x_img)
        tokens = out.last_hidden_state  # [B, 1+N, D]
        if self.pooling == "cls":
            return tokens[:, 0]
        return tokens[:, 1:, :].mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)
