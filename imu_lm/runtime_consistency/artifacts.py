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
    # Save the full module for robust Stage B loading (no reconstruction needed)
    torch.save(encoder, paths["encoder"])
    with open(paths["meta"], "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[artifact] saved encoder to {paths['encoder']}")
    print(f"[artifact] saved encoder_meta to {paths['meta']}")


def load_encoder(run_dir: str, map_location=None):
    paths = artifact_paths(run_dir)
    encoder = torch.load(paths["encoder"], map_location=map_location, weights_only=False)
    return encoder


def save_supervised_model(encoder: torch.nn.Module, head: torch.nn.Module, meta: Dict[str, Any], run_dir: str):
    """Save encoder + classification head for fully supervised training.
    
    Unlike MAE which discards the decoder, supervised training saves both
    encoder and head since this is the complete trained classifier.
    """
    paths = artifact_paths(run_dir)
    # Save encoder
    torch.save(encoder, paths["encoder"])
    # Save head separately
    head_path = os.path.join(paths["dir"], "head.pt")
    torch.save(head, head_path)
    # Save combined model dict for easy loading
    model_path = os.path.join(paths["dir"], "model.pt")
    torch.save({"encoder": encoder, "head": head}, model_path)
    with open(paths["meta"], "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[artifact] saved encoder to {paths['encoder']}")
    print(f"[artifact] saved head to {head_path}")
    print(f"[artifact] saved combined model to {model_path}")
    print(f"[artifact] saved meta to {paths['meta']}")
