"""Stage B probe training entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import logging
from typing import Any, Dict

from imu_lm.probe import train_run
from imu_lm.utils.helpers import deep_update, load_yaml

try:
    import wandb
except ImportError:
    wandb = None


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

    base_cfg = load_yaml(args.config)
    probe_cfg = load_yaml(args.probe_config)
    cfg = deep_update(base_cfg, probe_cfg)

    runs_root = cfg.get("paths", {}).get("runs_root", "runs")
    run_dir = os.path.join(runs_root, args.run)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"Run dir not found: {run_dir}")

    _log_resolved(cfg, run_dir)

    # Minimal wandb init for probe
    if wandb is not None:
        wb_cfg = cfg.get("wandb", {}) or {}
        try:
            wandb.init(
                project=wb_cfg.get("project", "imu-lm"),
                entity=wb_cfg.get("entity", None),
                name=f"{args.run}-probe",
                config=cfg,
                dir=run_dir,
            )
        except Exception as e:
            logging.getLogger(__name__).warning("wandb init failed: %s", e)

    train_run.main(cfg, run_dir)


if __name__ == "__main__":
    main()
