#!/usr/bin/env python3
"""Self-contained ViT1D encoder export & inference script.

Send this file + best.pt + encoder_meta.json to your collaborator.
No dependency on the TransfHAR repo — just PyTorch (and optionally coremltools).

Usage
-----
# 1) Inference with the raw checkpoint:
python export_and_inference.py --checkpoint best.pt --mode inference

# 2) Export to TorchScript (for CoreML conversion):
python export_and_inference.py --checkpoint best.pt --mode trace --out encoder_traced.pt

# 3) Export directly to CoreML (requires `pip install coremltools`):
python export_and_inference.py --checkpoint best.pt --mode coreml --out encoder.mlpackage

# 4) Inference with an already-traced TorchScript model:
python export_and_inference.py --traced encoder_traced.pt --mode inference

Input spec
----------
- Shape:  [B, 6, 128]   float32
- Channels: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
- Sample rate: 50 Hz, window: 2.56 s → 128 samples
- Normalization: z-score (zero mean, unit variance per channel)
- The encoder outputs [B, 384] embedding vectors.

Architecture (ViT-1D, LSM-2 style)
-----------------------------------
- 1D patch embedding (shared Conv1d kernel, patch_size=4) → 6 × 32 = 192 tokens
- 2D positional embedding (time + channel)
- 12-layer Transformer encoder (d=384, 6 heads, FFN=1536, GELU, pre-norm)
- Mean pooling over all tokens → [B, 384]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn

class PatchEmbed1D(nn.Module):
    """1D Patch Embedding: [B, C, T] → [B, N, D] where N = C * (T // patch_size)."""

    def __init__(self, in_channels: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        # Shared Conv1d applied to each channel separately (LSM-2 style)
        self.proj = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        patches = []
        for c in range(C):
            patch_c = self.proj(x[:, c : c + 1, :])  # [B, D, num_patches]
            patches.append(patch_c.transpose(1, 2))   # [B, num_patches, D]
        return torch.cat(patches, dim=1)  # [B, C * num_patches, D]


class PositionalEmbedding2D(nn.Module):
    """Additive 2-D positional embedding (time + channel axes)."""

    def __init__(self, max_patches_per_channel: int, max_channels: int, embed_dim: int):
        super().__init__()
        self.time_embed = nn.Embedding(max_patches_per_channel, embed_dim)
        self.channel_embed = nn.Embedding(max_channels, embed_dim)
        self.max_patches_per_channel = max_patches_per_channel
        self.max_channels = max_channels

    def forward(self, num_patches_per_channel: int, num_channels: int, device: torch.device) -> torch.Tensor:
        time_pos = torch.arange(num_patches_per_channel, device=device).repeat(num_channels)
        channel_pos = torch.arange(num_channels, device=device).repeat_interleave(num_patches_per_channel)
        return self.time_embed(time_pos) + self.channel_embed(channel_pos)


class ViT1DEncoder(nn.Module):
    """ViT-1D encoder: [B, 6, 128] to [B, 384]."""

    def __init__(
        self,
        in_channels: int = 6,
        patch_size: int = 4,
        embed_dim: int = 384,
        num_layers: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_patches_per_channel: int = 256,
        max_channels: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed1D(in_channels, patch_size, embed_dim)
        self.pos_embed = PositionalEmbedding2D(max_patches_per_channel, max_channels, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, T] → [B, embed_dim]."""
        B, C, T = x.shape
        num_patches_per_channel = T // self.patch_size

        tokens = self.patch_embed(x)
        pos = self.pos_embed(num_patches_per_channel, C, x.device)
        tokens = tokens + pos.unsqueeze(0)

        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)

        return tokens.mean(dim=1)  # mean-pool → [B, D]


def load_encoder_from_checkpoint(ckpt_path: str, device: str = "cpu") -> ViT1DEncoder:
    """Load ViT1DEncoder weights from a TransfHAR best.pt / latest.pt checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    vit_cfg = cfg.get("vit1d", {})
    enc_cfg = vit_cfg.get("encoder", {})
    data_cfg = cfg.get("data", {})

    encoder = ViT1DEncoder(
        in_channels=len(data_cfg.get("sensor_columns", ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"])),
        patch_size=int(vit_cfg.get("patch_size", 4)),
        embed_dim=int(enc_cfg.get("hidden_size", 384)),
        num_layers=int(enc_cfg.get("num_hidden_layers", 12)),
        num_heads=int(enc_cfg.get("num_attention_heads", 6)),
        mlp_ratio=float(enc_cfg.get("intermediate_size", 1536)) / float(enc_cfg.get("hidden_size", 384)),
        dropout=float(enc_cfg.get("hidden_dropout_prob", 0.0)),
        max_patches_per_channel=int(vit_cfg.get("max_patches_per_channel", 256)),
        max_channels=32,
    )

    encoder.load_state_dict(ckpt["model"], strict=True)
    encoder.eval()
    print(f" Loaded encoder from {ckpt_path}")
    print(f"  arch: ViT-1D  embed_dim={encoder.embed_dim}  layers=12  heads=6")
    print(f"  input: [B, {encoder.in_channels}, 128]  patch_size={encoder.patch_size}")
    print(f"  output: [B, {encoder.embed_dim}]")
    return encoder


def trace_encoder(encoder: ViT1DEncoder, out_path: str):
    """Trace the encoder and save as TorchScript .pt (input for coremltools)."""
    encoder.eval()
    dummy = torch.randn(1, encoder.in_channels, 128)
    with torch.no_grad():
        traced = torch.jit.trace(encoder, dummy)
    traced.save(out_path)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Saved TorchScript model to {out_path}  ({size_mb:.1f} MB)")
    return traced


def export_coreml(encoder: ViT1DEncoder, out_path: str):
    """Convert to CoreML .mlpackage via coremltools (must be installed)."""
    try:
        import coremltools as ct
    except ImportError:
        print("ERROR: coremltools not installed. Run: pip install coremltools")
        return

    encoder.eval()
    dummy = torch.randn(1, encoder.in_channels, 128)
    with torch.no_grad():
        traced = torch.jit.trace(encoder, dummy)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="imu_window", shape=(1, 6, 128))],
        outputs=[ct.TensorType(name="embedding")],
        minimum_deployment_target=ct.target.watchOS7,
    )
    mlmodel.save(out_path)
    print(f"Saved CoreML model to {out_path}")



def run_sample_inference(model, device: str = "cpu"):
    """Run inference on synthetic 6-axis IMU data and print results + latency."""
    model.eval()
    model.to(device)

    # Simulate one 2.56-second window of 6-axis IMU at 50 Hz
    # In practice: z-score normalize each channel (mean=0, std=1) before passing in
    x = torch.randn(1, 6, 128, device=device)

    # Warm-up
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)

    # Timed inference
    n_runs = 100
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            out = model(x)
    elapsed = time.perf_counter() - start

    print(f"\n── Sample Inference ──")
    print(f"Input shape:   {list(x.shape)}  (batch=1, channels=6, time=128)")
    print(f"Output shape:  {list(out.shape)}  (batch=1, embed_dim={out.shape[-1]})")
    print(f"Output (first 8 dims): {out[0, :8].tolist()}")
    print(f"Latency:       {elapsed / n_runs * 1000:.2f} ms  (avg over {n_runs} runs, {device})")
    print(f"Parameters:    {sum(p.numel() for p in model.parameters()):,}")



def main():
    ap = argparse.ArgumentParser(description="ViT-1D IMU encoder: inference & export")
    ap.add_argument("--checkpoint", type=str, default=None, help="Path to best.pt or latest.pt")
    ap.add_argument("--traced", type=str, default=None, help="Path to already-traced TorchScript .pt")
    ap.add_argument("--mode", choices=["inference", "trace", "coreml"], default="inference",
                    help="inference: run sample; trace: export TorchScript; coreml: export .mlpackage")
    ap.add_argument("--out", type=str, default=None, help="Output path for trace/coreml export")
    ap.add_argument("--device", type=str, default="cpu", help="Device (cpu / cuda / mps)")
    args = ap.parse_args()

    if args.traced:
        print(f"Loading traced model from {args.traced}")
        model = torch.jit.load(args.traced, map_location=args.device)
    elif args.checkpoint:
        model = load_encoder_from_checkpoint(args.checkpoint, device=args.device)
    else:
        ap.error("Provide either --checkpoint (best.pt) or --traced (traced .pt)")

    if args.mode == "inference":
        run_sample_inference(model, device=args.device)

    elif args.mode == "trace":
        out = args.out or "encoder_traced.pt"
        trace_encoder(model, out)
        # Verify round-trip
        reloaded = torch.jit.load(out, map_location=args.device)
        run_sample_inference(reloaded, device=args.device)

    elif args.mode == "coreml":
        out = args.out or "encoder.mlpackage"
        export_coreml(model, out)


if __name__ == "__main__":
    main()
