"""TSTransformer1D: Time-Series Transformer for raw 1D IMU signals [B, C, T].

Encoder: Projects input to d_model, applies positional encoding, transformer layers, mean pooling.
Decoder: Lightweight transformer decoder for masked reconstruction.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from imu_lm.utils.helpers import cfg_get


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence data."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TSTransformer1DEncoder(nn.Module):
    """Time-Series Transformer encoder for raw 1D signals [B, C, T] → [B, embed_dim].
    
    Architecture:
    1. Transpose to [B, T, C]
    2. Linear projection to d_model
    3. Positional encoding
    4. Transformer encoder layers
    5. Mean pooling over time → [B, d_model]
    """
    
    def __init__(self, cfg: Any):
        super().__init__()
        
        # Get nested config: tstransformer1d.encoder.*
        ts_cfg = cfg_get(cfg, ["tstransformer1d"], {}) or {}
        enc_cfg = ts_cfg.get("encoder", {}) or {}
        
        # Input config
        data_cfg = cfg_get(cfg, ["data"], {}) or {}
        sensor_cols = data_cfg.get("sensor_columns", ["acc_x", "acc_y", "acc_z"])
        self.in_channels = len(sensor_cols)
        
        # Encoder hyperparameters
        self.d_model = int(enc_cfg.get("d_model", 128))
        self.nhead = int(enc_cfg.get("nhead", 4))
        self.num_layers = int(enc_cfg.get("num_layers", 2))
        self.dim_feedforward = int(enc_cfg.get("dim_feedforward", 256))
        self.dropout = float(enc_cfg.get("dropout", 0.1))
        self.pooling = str(enc_cfg.get("pooling", "mean"))
        
        # For compatibility with other encoders
        self.embed_dim = self.d_model
        
        # Input projection: C → d_model
        self.input_proj = nn.Linear(self.in_channels, self.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, dropout=self.dropout)
        
        # CLS token for cls pooling
        if self.pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        else:
            self.cls_token = None
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Output normalization
        self.norm = nn.LayerNorm(self.d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, T] → [B, embed_dim]"""
        # Transpose to [B, T, C]
        x = x.transpose(1, 2)
        B, T, C = x.shape
        
        # Project to d_model
        x = self.input_proj(x)  # [B, T, d_model]
        
        # Add CLS token if using cls pooling
        if self.pooling == "cls" and self.cls_token is not None:
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)  # [B, T+1, d_model]
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.encoder(x)
        
        # Pooling
        if self.pooling == "cls":
            emb = x[:, 0]  # Take CLS token
        else:
            emb = x.mean(dim=1)  # Mean pooling over time
        
        # Normalize
        emb = self.norm(emb)
        
        return emb  # [B, embed_dim]


class TSTransformer1DDecoder(nn.Module):
    """Lightweight transformer decoder for masked reconstruction [B, embed_dim] → [B, C, T].
    
    Architecture:
    1. Project embedding to sequence of queries
    2. Positional encoding
    3. Transformer decoder layers  
    4. Project back to [B, C, T]
    """
    
    def __init__(self, embed_dim: int, out_channels: int, target_T: int, cfg: Any):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.target_T = target_T
        
        # Get decoder config (nested under tstransformer1d.decoder)
        ts_cfg = cfg_get(cfg, ["tstransformer1d"], {}) or {}
        dec_cfg = ts_cfg.get("decoder", {}) or {}
        
        self.d_model = int(dec_cfg.get("d_model", 128))
        self.nhead = int(dec_cfg.get("nhead", 4))
        self.num_layers = int(dec_cfg.get("num_layers", 2))
        self.dim_feedforward = int(dec_cfg.get("dim_feedforward", 256))
        self.dropout = float(dec_cfg.get("dropout", 0.1))
        
        # Project from encoder embedding to decoder d_model
        self.embed_proj = nn.Linear(embed_dim, self.d_model)
        
        # Learnable query tokens for each time step
        self.query_tokens = nn.Parameter(torch.zeros(1, target_T, self.d_model))
        
        # Positional encoding for queries
        self.pos_encoder = PositionalEncoding(self.d_model, dropout=self.dropout)
        
        # Transformer decoder layers (self-attention + cross-attention to encoder embedding)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)
        
        # Output projection: d_model → C
        self.output_proj = nn.Linear(self.d_model, out_channels)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """[B, embed_dim] → [B, C, T]"""
        B = z.shape[0]
        
        # Project encoder embedding: [B, embed_dim] → [B, 1, d_model]
        memory = self.embed_proj(z).unsqueeze(1)  # [B, 1, d_model]
        
        # Expand query tokens for batch
        queries = self.query_tokens.expand(B, -1, -1)  # [B, T, d_model]
        queries = self.pos_encoder(queries)
        
        # Transformer decoder
        decoded = self.decoder(queries, memory)  # [B, T, d_model]
        
        # Project to output channels
        out = self.output_proj(decoded)  # [B, T, C]
        
        # Transpose to [B, C, T]
        out = out.transpose(1, 2)
        
        return out
