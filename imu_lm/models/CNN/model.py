"""CNN1D encoder for raw IMU signals (SAMoSA-style architecture).

Architecture from SAMoSA paper:
- 4 Conv1D layers: [128, 128, 256, 256], kernel=10, stride=1, ReLU, batch norm, max pool=2
- 3 FC layers: [1000, 500, 250] with ReLU, dropout=0.5
- Output: embedding [B, embed_dim]

This encoder is objective-agnostic. Different objectives attach different heads:
- Masked SSL: decoder reconstructs [B, C, T] → MSE loss
- Supervised: classifier → [B, num_classes] → CE loss
- Linear probe: frozen encoder + linear layer

Input: raw 1D IMU [B, C, T] (channels × time)
"""

from __future__ import annotations

from typing import Any, List

import torch
import torch.nn as nn


class CNN1DEncoder(nn.Module):
    """1D CNN encoder for raw IMU signals [B, C, T].
    
    Produces embeddings that can be used for any downstream task.
    """
    
    def __init__(self, cfg: Any):
        super().__init__()
        
        cnn_cfg = cfg.get("cnn1d", {}) if isinstance(cfg, dict) else getattr(cfg, "cnn1d", {})
        enc_cfg = cnn_cfg.get("encoder", {}) or {}
        
        # Conv stack config (SAMoSA defaults) - nested under cnn1d.encoder
        conv_filters: List[int] = enc_cfg.get("conv_filters", [128, 128, 256, 256])
        kernel_size: int = int(enc_cfg.get("kernel_size", 10))
        stride: int = int(enc_cfg.get("stride", 1))
        use_batch_norm: bool = bool(enc_cfg.get("use_batch_norm", True))
        pool_size: int = int(enc_cfg.get("pool_size", 2))
        dropout: float = float(enc_cfg.get("dropout", 0.5))
        activation: str = enc_cfg.get("activation", "relu")
        
        # FC layers config (SAMoSA defaults) - nested under cnn1d.encoder
        fc_dims: List[int] = enc_cfg.get("fc_dims", [1000, 500, 250])
        
        # Input channels from data config
        data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else getattr(cfg, "data", {})
        sensor_cols = data_cfg.get("sensor_columns", ["acc_x", "acc_y", "acc_z"])
        in_channels = len(sensor_cols)
        
        # Build conv layers
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        prev_channels = in_channels
        for out_channels in conv_filters:
            layers = [
                nn.Conv1d(prev_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
            ]
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_channels))
            if activation == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif activation == "gelu":
                layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            
            self.conv_layers.append(nn.Sequential(*layers))
            self.pool_layers.append(nn.MaxPool1d(pool_size))
            prev_channels = out_channels
        
        self.conv_out_channels = conv_filters[-1] if conv_filters else in_channels
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # FC layers
        self.fc_layers = nn.ModuleList()
        prev_dim = self.conv_out_channels
        for fc_dim in fc_dims:
            self.fc_layers.append(nn.Sequential(
                nn.Linear(prev_dim, fc_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ))
            prev_dim = fc_dim
        
        # Final embedding dimension
        self.embed_dim = fc_dims[-1] if fc_dims else self.conv_out_channels
        
        # Store arch info for external use (e.g., decoder construction)
        self.in_channels = in_channels
        self.conv_filters = conv_filters
        self.pool_size = pool_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full encoder: [B, C, T] → [B, embed_dim]."""
        # Conv stack
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            x = conv(x)
            x = pool(x)
        
        # Global pooling
        x = self.adaptive_pool(x).squeeze(-1)  # [B, conv_out_channels]
        
        # FC layers
        for fc in self.fc_layers:
            x = fc(x)
        
        return x


class CNN1DDecoder(nn.Module):
    """1D CNN decoder for reconstruction [B, embed_dim] → [B, C, T].
    
    This is a throwaway component - only used during masked pretraining.
    After pretraining, we discard it and keep only the encoder.
    """
    
    def __init__(self, embed_dim: int, out_channels: int, target_T: int, cfg: Any):
        super().__init__()
        
        # Get decoder config (nested under cnn1d.decoder)
        cnn_cfg = cfg.get("cnn1d", {}) if isinstance(cfg, dict) else getattr(cfg, "cnn1d", {})
        dec_cfg = cnn_cfg.get("decoder", {}) or {}
        hidden_size = int(dec_cfg.get("decoder_hidden_size", 256))
        num_layers = int(dec_cfg.get("decoder_num_layers", 4))
        
        # Start with small T, upsample through deconvs
        T_start = max(1, target_T // (2 ** (num_layers - 1)))
        
        self.T_start = T_start
        self.hidden_size = hidden_size
        self.target_T = target_T
        
        # FC to expand from embedding
        self.fc_expand = nn.Sequential(
            nn.Linear(embed_dim, hidden_size * T_start),
            nn.ReLU(inplace=True),
        )
        
        # Transpose conv layers to upsample
        self.deconv_layers = nn.ModuleList()
        prev_ch = hidden_size
        for i in range(num_layers - 1):
            out_ch = hidden_size // (2 ** (i + 1))
            out_ch = max(out_ch, out_channels)
            self.deconv_layers.append(nn.Sequential(
                nn.ConvTranspose1d(prev_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ))
            prev_ch = out_ch
        
        # Final layer to target channels
        self.final_conv = nn.Conv1d(prev_ch, out_channels, kernel_size=3, padding=1)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """[B, embed_dim] → [B, C, T]."""
        B = z.shape[0]
        
        # Expand to [B, hidden_size, T_start]
        x = self.fc_expand(z)
        x = x.view(B, self.hidden_size, self.T_start)
        
        # Upsample with transpose convs
        for deconv in self.deconv_layers:
            x = deconv(x)
        
        # Final conv
        x = self.final_conv(x)
        
        # Interpolate to exact target length
        if x.shape[-1] != self.target_T:
            x = torch.nn.functional.interpolate(x, size=self.target_T, mode='linear', align_corners=False)
        
        return x
