"""Probe heads (start with a simple linear classifier)."""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearHead(nn.Module):
    """Minimal linear probe head."""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
