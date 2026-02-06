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


def save_supervised_model(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    meta: Dict[str, Any],
    run_dir: str,
    label_map: Dict[str, Any] | None = None,
):
    """Save encoder + classification head + label_map for fully supervised training."""
    paths = artifact_paths(run_dir)
    torch.save(encoder, paths["encoder"])
    head_path = os.path.join(paths["dir"], "head.pt")
    torch.save(head, head_path)
    model_path = os.path.join(paths["dir"], "model.pt")
    torch.save({"encoder": encoder, "head": head}, model_path)
    if label_map is not None:
        label_map_path = os.path.join(paths["dir"], "label_map.json")
        with open(label_map_path, "w") as f:
            json.dump({k: v for k, v in label_map.items() if k != "unknown_id" or v is not None}, f, indent=2)
        meta["label_map_path"] = "label_map.json"
    with open(paths["meta"], "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[artifact] saved encoder to {paths['encoder']}")
    print(f"[artifact] saved head to {head_path}")
    print(f"[artifact] saved combined model to {model_path}")
    print(f"[artifact] saved meta to {paths['meta']}")
