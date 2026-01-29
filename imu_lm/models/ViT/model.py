"""ViT-MAE encoder wrapper (facebook/vit-mae-base family).

Exposes two entry points:
- forward_tokens(x_img): patch tokens (no CLS) for MAE-style objectives.
- forward_features(x_img): pooled embedding [B, D] for probes.

Assumes input from loader is already image-like: [B, 3, F, TT] with return_image=True.
Resizes internally to vit.input.resize_hw.
"""

from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.nn as nn
from transformers import ViTMAEConfig, ViTMAEModel

from imu_lm.utils.helpers import resize_bilinear


class ViTEncoder(nn.Module):
    def __init__(self, cfg: Any):
        super().__init__()

        vit_cfg = getattr(cfg, "vit", cfg.get("vit")) if cfg is not None else {}
        warm_start = bool(vit_cfg.get("warm_start", True))
        pretrained_id = vit_cfg.get("pretrained_id", "facebook/vit-mae-base")

        input_cfg = vit_cfg.get("input", {}) or {}
        resize_hw = input_cfg.get("resize_hw", [224, 224])
        patch_size = int(input_cfg.get("patch_size", 16))
        num_channels = int(input_cfg.get("num_channels", 3))

        if warm_start:
            self.backbone = ViTMAEModel.from_pretrained(pretrained_id)
        else:
            arch = vit_cfg.get("arch", {}) or {}
            if resize_hw[0] != resize_hw[1]:
                raise ValueError("ViTMAE scratch init expects square resize_hw")
            hf_cfg = ViTMAEConfig(
                image_size=int(resize_hw[0]),
                patch_size=patch_size,
                num_channels=num_channels,
                hidden_size=int(arch.get("hidden_size", 768)),
                num_hidden_layers=int(arch.get("num_hidden_layers", 12)),
                num_attention_heads=int(arch.get("num_attention_heads", 12)),
                intermediate_size=int(arch.get("intermediate_size", 3072)),
                hidden_act=arch.get("hidden_act", "gelu"),
                hidden_dropout_prob=float(arch.get("hidden_dropout_prob", 0.0)),
                attention_probs_dropout_prob=float(arch.get("attention_probs_dropout_prob", 0.0)),
                qkv_bias=bool(arch.get("qkv_bias", True)),
                layer_norm_eps=float(arch.get("layer_norm_eps", 1e-12)),
                initializer_range=float(arch.get("initializer_range", 0.02)),
            )
            self.backbone = ViTMAEModel(hf_cfg)

        self.resize_hw: Tuple[int, int] = (int(resize_hw[0]), int(resize_hw[1]))
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = int(self.backbone.config.hidden_size)
        self.pooling = vit_cfg.get("pooling", "mean")
        self.backbone_name = pretrained_id if warm_start else "vit_mae_scratch"

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, F, TT] -> resize to [B, 3, H, W]
        if x.dim() != 4:
            raise ValueError(f"Expected x with shape [B,3,H,W], got {x.shape}")
        return resize_bilinear(x.float(), self.resize_hw)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x_img = self._prepare(x)
        out = self.backbone(pixel_values=x_img)
        tokens = out.last_hidden_state  # [B, 1+N, D]
        return tokens[:, 1:, :]  # drop CLS

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x_img = self._prepare(x)
        out = self.backbone(pixel_values=x_img)
        tokens = out.last_hidden_state  # [B, 1+N, D]
        if self.pooling == "cls":
            return tokens[:, 0]
        # mean pool patch tokens
        return tokens[:, 1:, :].mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # optional; defaults to features
        return self.forward_features(x)
