"""ViT-1D encoder and decoder for raw 1D IMU/sensor data (LSM-2 style architecture).

Encoder: [B, C, T] → [B, embed_dim]
- 1D patch embedding with shared kernel across channels
- 2D positional embedding (time + channel)
- Transformer encoder layers
- Mean pooling over tokens

Decoder: [B, embed_dim] → [B, C, T]
- Lightweight transformer decoder
- Projects back to input space

Architecture-specific components live here; objective is architecture-agnostic.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from imu_lm.utils.helpers import cfg_get


class PatchEmbed1D(nn.Module):
    """1D Patch Embedding: [B, C, T] → [B, N, D] where N = C * (T // patch_size)."""
    
    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        embed_dim: int,
        shared_kernel: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.shared_kernel = shared_kernel
        
        if shared_kernel:
            # Single Conv1d applied to each channel separately (LSM-2 style)
            self.proj = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            # Separate projection per channel
            self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, T] → [B, N, D] where N = C * num_patches_per_channel."""
        B, C, T = x.shape
        num_patches = T // self.patch_size
        
        if self.shared_kernel:
            # Apply same conv to each channel: [B, C, T] → [B, C, num_patches, D]
            patches = []
            for c in range(C):
                # [B, 1, T] → [B, D, num_patches]
                patch_c = self.proj(x[:, c:c+1, :])  # [B, D, num_patches]
                patches.append(patch_c.transpose(1, 2))  # [B, num_patches, D]
            # Stack channels: [B, C * num_patches, D]
            out = torch.cat(patches, dim=1)
        else:
            # [B, C, T] → [B, D, num_patches] → [B, num_patches, D]
            out = self.proj(x).transpose(1, 2)
        
        return out


class PositionalEmbedding2D(nn.Module):
    """2D positional embedding: separate embeddings for time and channel positions."""
    
    def __init__(
        self,
        max_patches_per_channel: int,
        max_channels: int,
        embed_dim: int,
    ):
        super().__init__()
        self.time_embed = nn.Embedding(max_patches_per_channel, embed_dim)
        self.channel_embed = nn.Embedding(max_channels, embed_dim)
        self.max_patches_per_channel = max_patches_per_channel
        self.max_channels = max_channels
    
    def forward(self, num_patches_per_channel: int, num_channels: int, device: torch.device) -> torch.Tensor:
        """Generate positional embeddings for [N] tokens where N = num_channels * num_patches_per_channel."""
        # Time positions: [0, 1, 2, ..., num_patches-1] repeated for each channel
        time_pos = torch.arange(num_patches_per_channel, device=device)
        time_pos = time_pos.repeat(num_channels)  # [N]
        
        # Channel positions: [0, 0, ..., 1, 1, ..., C-1, C-1, ...]
        channel_pos = torch.arange(num_channels, device=device)
        channel_pos = channel_pos.repeat_interleave(num_patches_per_channel)  # [N]
        
        # Sum embeddings
        pos_embed = self.time_embed(time_pos) + self.channel_embed(channel_pos)  # [N, D]
        return pos_embed


class ViT1DEncoder(nn.Module):
    """ViT-1D Encoder: [B, C, T] → [B, embed_dim].
    
    Architecture (LSM-2 style):
    - 1D patch embedding with shared kernel
    - 2D positional embedding (time + channel)
    - Transformer encoder
    - Mean pooling
    """
    
    def __init__(self, cfg: Any):
        super().__init__()
        
        # Get config
        vit_cfg = cfg_get(cfg, ["vit1d"], {}) or {}
        enc_cfg = vit_cfg.get("encoder", {}) or {}
        data_cfg = cfg_get(cfg, ["data"], {}) or {}
        
        # Input dimensions
        sensor_cols = data_cfg.get("sensor_columns", ["acc_x", "acc_y", "acc_z"])
        self.in_channels = len(sensor_cols)
        
        # Patch embedding config (top-level vit1d)
        self.patch_size = int(vit_cfg.get("patch_size", 16))
        shared_kernel = bool(vit_cfg.get("shared_kernel", True))
        
        # Encoder config (nested under vit1d.encoder)
        self.embed_dim = int(enc_cfg.get("hidden_size", 384))
        num_layers = int(enc_cfg.get("num_hidden_layers", 12))
        num_heads = int(enc_cfg.get("num_attention_heads", 6))
        mlp_ratio = float(enc_cfg.get("intermediate_size", 1536)) / self.embed_dim
        dropout = float(enc_cfg.get("hidden_dropout_prob", 0.0))
        attn_dropout = float(enc_cfg.get("attention_probs_dropout_prob", 0.0))
        layer_norm_eps = float(enc_cfg.get("layer_norm_eps", 1e-6))
        
        # Positional embedding config (top-level vit1d)
        max_patches = int(vit_cfg.get("max_patches_per_channel", 256))
        max_channels = int(vit_cfg.get("max_channels", 32))
        
        # Pooling
        self.pooling = vit_cfg.get("pooling", "mean")
        
        # Layers
        self.patch_embed = PatchEmbed1D(
            in_channels=self.in_channels,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            shared_kernel=shared_kernel,
        )
        
        self.pos_embed = PositionalEmbedding2D(
            max_patches_per_channel=max_patches,
            max_channels=max_channels,
            embed_dim=self.embed_dim,
        )
        
        # Use PyTorch's built-in transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=int(self.embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,  # pre-norm like ViT
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(self.embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, T] → [B, embed_dim]."""
        B, C, T = x.shape
        num_patches_per_channel = T // self.patch_size
        
        # Patch embedding: [B, C, T] → [B, N, D]
        tokens = self.patch_embed(x)
        
        # Add positional embedding
        pos_embed = self.pos_embed(num_patches_per_channel, C, x.device)
        tokens = tokens + pos_embed.unsqueeze(0)
        
        # Transformer encoder
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        
        # Pooling: [B, N, D] → [B, D]
        if self.pooling == "mean":
            out = tokens.mean(dim=1)
        elif self.pooling == "cls":
            out = tokens[:, 0]
        else:
            out = tokens.mean(dim=1)
        
        return out


class ViT1DDecoder(nn.Module):
    """ViT-1D Decoder: [B, embed_dim] → [B, C, T].
    
    Lightweight transformer decoder for reconstruction.
    This is throwaway after pretraining - only encoder is kept.
    """
    
    def __init__(self, embed_dim: int, out_channels: int, target_T: int, cfg: Any):
        super().__init__()
        
        # Get decoder config (nested under vit1d.decoder)
        vit_cfg = cfg_get(cfg, ["vit1d"], {}) or {}
        dec_cfg = vit_cfg.get("decoder", {}) or {}
        
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.target_T = target_T
        self.patch_size = int(vit_cfg.get("patch_size", 16))
        
        # Decoder config
        hidden_size = int(dec_cfg.get("hidden_size", 192))
        num_layers = int(dec_cfg.get("num_hidden_layers", 4))
        num_heads = int(dec_cfg.get("num_attention_heads", 4))
        mlp_ratio = float(dec_cfg.get("intermediate_size", 768)) / hidden_size
        
        # Number of output patches
        self.num_patches_per_channel = target_T // self.patch_size
        self.num_patches = out_channels * self.num_patches_per_channel
        
        # Project from encoder embed_dim to decoder hidden_size
        self.embed_to_decoder = nn.Linear(embed_dim, hidden_size)
        
        # Learnable query tokens for reconstruction
        self.query_tokens = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        
        # Positional embedding for decoder
        max_patches = int(vit_cfg.get("max_patches_per_channel", 256))
        max_channels = int(vit_cfg.get("max_channels", 32))
        self.pos_embed = PositionalEmbedding2D(max_patches, max_channels, hidden_size)
        
        # Use PyTorch's built-in transformer encoder (used as decoder here)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=int(hidden_size * mlp_ratio),
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(hidden_size)
        
        # Project to patch values: each token → patch_size values
        self.head = nn.Linear(hidden_size, self.patch_size)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """[B, embed_dim] → [B, C, T]."""
        B = z.shape[0]
        
        # Project encoder output and expand to sequence
        z_proj = self.embed_to_decoder(z)  # [B, hidden_size]
        
        # Use learnable queries + encoder context
        queries = self.query_tokens.expand(B, -1, -1)  # [B, N, hidden_size]
        
        # Add positional embedding
        pos_embed = self.pos_embed(self.num_patches_per_channel, self.out_channels, z.device)
        queries = queries + pos_embed.unsqueeze(0)
        
        # Add encoder context to all queries
        queries = queries + z_proj.unsqueeze(1)
        
        # Transformer decoder
        queries = self.decoder(queries)
        queries = self.norm(queries)
        
        # Project to patch values: [B, N, hidden_size] → [B, N, patch_size]
        patch_values = self.head(queries)  # [B, N, patch_size]
        
        # Reshape to [B, C, T]
        # Tokens are ordered: [C0_P0, C0_P1, ..., C1_P0, C1_P1, ...]
        patch_values = patch_values.view(B, self.out_channels, self.num_patches_per_channel, self.patch_size)
        out = patch_values.view(B, self.out_channels, self.num_patches_per_channel * self.patch_size)
        
        # Interpolate to exact target length if needed
        if out.shape[-1] != self.target_T:
            out = torch.nn.functional.interpolate(out, size=self.target_T, mode='linear', align_corners=False)
        
        return out
