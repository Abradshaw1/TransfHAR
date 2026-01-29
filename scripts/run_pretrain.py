"""
run_pretrain.py
----------------
Stage A entrypoint: load/merge configs, create run dir, route to model run.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict

import yaml

from imu_lm.models.ViT import run as vit_run


def _deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _resolve_run_dir(cfg: Dict[str, Any], run_name: str | None) -> str:
    runs_root = cfg.get("paths", {}).get("runs_root", "runs")
    if run_name is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        run_name = f"run-{ts}"
    run_dir = os.path.join(runs_root, run_name)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "artifacts"), exist_ok=True)
    return run_dir


def _log_resolved(cfg: Dict[str, Any], run_dir: str):
    log_path = os.path.join(run_dir, "logs", "stdout.log")
    with open(log_path, "a") as f:
        f.write("# Resolved config\n")
        f.write(json.dumps(cfg, indent=2))
        f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Base config YAML")
    ap.add_argument("--model-config", required=True, help="Model config YAML")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    base_cfg = _load_yaml(args.config)
    model_cfg = _load_yaml(args.model_config)
    cfg = _deep_update(base_cfg, model_cfg)

    run_dir = _resolve_run_dir(cfg, args.run_name)
    _log_resolved(cfg, run_dir)

    model_name = cfg.get("model", {}).get("name", "")
    if model_name == "vit":
        vit_run.main(cfg, run_dir)
    else:
        raise ValueError(f"Unsupported model.name={model_name}")


if __name__ == "__main__":
    main()
