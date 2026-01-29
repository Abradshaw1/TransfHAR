"""Save/load encoder artifacts and metadata."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import torch


def artifact_paths(run_dir: str) -> Dict[str, str]:
    art_dir = os.path.join(run_dir, "artifacts")
    os.makedirs(art_dir, exist_ok=True)
    return {
        "dir": art_dir,
        "encoder": os.path.join(art_dir, "encoder.pt"),
        "meta": os.path.join(art_dir, "encoder_meta.json"),
    }


def save_encoder(encoder: torch.nn.Module, meta: Dict[str, Any], run_dir: str):
    paths = artifact_paths(run_dir)
    torch.save(encoder.state_dict(), paths["encoder"])
    with open(paths["meta"], "w") as f:
        json.dump(meta, f, indent=2)


def load_encoder(encoder: torch.nn.Module, run_dir: str, map_location=None):
    paths = artifact_paths(run_dir)
    state = torch.load(paths["encoder"], map_location=map_location)
    encoder.load_state_dict(state)
    return encoder
