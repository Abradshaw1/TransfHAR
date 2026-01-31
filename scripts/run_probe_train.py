"""Stage B probe training entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import logging
from typing import Any, Dict

import yaml

from imu_lm.probe import train_run


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


def _log_resolved(cfg: Dict[str, Any], run_dir: str):
    log_path = os.path.join(run_dir, "logs", "stdout.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write("# Resolved config (probe)\n")
        f.write(json.dumps(cfg, indent=2))
        f.write("\n")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Base config YAML")
    ap.add_argument("--probe-config", required=True, help="Probe config YAML")
    ap.add_argument("--run", required=True, help="Run name under runs/<run>")
    args = ap.parse_args()

    base_cfg = _load_yaml(args.config)
    probe_cfg = _load_yaml(args.probe_config)
    cfg = _deep_update(base_cfg, probe_cfg)

    runs_root = cfg.get("paths", {}).get("runs_root", "runs")
    run_dir = os.path.join(runs_root, args.run)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"Run dir not found: {run_dir}")

    _log_resolved(cfg, run_dir)

    train_run.main(cfg, run_dir)


if __name__ == "__main__":
    main()
