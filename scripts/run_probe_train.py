"""Stage B probe training entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import logging
from typing import Any, Dict

from imu_lm.probe import train_run
from imu_lm.probe.io import resolve_probe_dir
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
                name=f"{args.run}-{resolve_probe_dir(os.path.join(runs_root, args.run), cfg)['base'].split(os.sep)[-1]}",
                config=cfg,
                dir=run_dir,
            )
        except Exception as e:
            logging.getLogger(__name__).warning("wandb init failed: %s", e)

    if cfg.get("probe", {}).get("user_study_mode", False):
        import copy, numpy as np
        from imu_lm.data.loaders import build_session_index
        subj_col = cfg.get("data", {}).get("subject_column", "subject_id")
        probe_ds = cfg.get("splits", {}).get("probe_dataset", None)
        si = build_session_index(cfg["paths"]["dataset_path"], cfg, dataset_filter=[probe_ds] if probe_ds else None)
        participants = sorted(si[subj_col].astype(str).unique().tolist())
        log = logging.getLogger(__name__)
        log.info("[user_study] %d participants: %s", len(participants), participants)
        probe_local = cfg.get("probe", {})
        if probe_local.get("fewshot_enabled", False):
            _fs_dn = probe_local.get("fewshot_probe_dirname", "fewshot_probe")
            _fs_k = int(probe_local.get("fewshot_shots_per_class", 5))
            base_dirname = f"{_fs_dn}_k{_fs_k}"
        else:
            base_dirname = probe_local.get("probe_dirname", "probe")
        all_summaries = []
        for pid in participants:
            log.info("[user_study] ===== participant=%s =====", pid)
            if wandb is not None and wandb.run is not None:
                wandb.finish()
            if wandb is not None:
                wb_cfg = cfg.get("wandb", {}) or {}
                try:
                    wandb.init(project=wb_cfg.get("project", "imu-lm"), entity=wb_cfg.get("entity", None),
                               name=f"{args.run}-{base_dirname}-{pid}", config=cfg, dir=run_dir, reinit=True)
                except Exception as e:
                    log.warning("wandb init for %s failed: %s", pid, e)
            pcfg = copy.deepcopy(cfg)
            pcfg["_participant_id"] = str(pid)
            pcfg["probe"]["probe_dirname"] = f"{base_dirname}/{pid}"
            try:
                train_run.main(pcfg, run_dir)
                summary_path = os.path.join(run_dir, base_dirname, str(pid), "summary.txt")
                if os.path.exists(summary_path):
                    with open(summary_path) as sf:
                        all_summaries.append({"participant": pid, **json.load(sf)})
            except Exception as e:
                log.error("[user_study %s] FAILED: %s", pid, e, exc_info=True)
        # Aggregate
        agg_path = os.path.join(run_dir, base_dirname, "aggregate_summary.json")
        os.makedirs(os.path.dirname(agg_path), exist_ok=True)
        agg = {"num_participants": len(all_summaries), "participants": all_summaries, "aggregate": {}}
        for k in ["bal_acc", "macro_f1", "macro_precision", "macro_recall"]:
            vals = [s.get("test", {}).get(k) for s in all_summaries if isinstance(s.get("test", {}).get(k), (int, float))]
            if vals:
                a = np.array(vals)
                agg["aggregate"][k] = {"mean": float(a.mean()), "std": float(a.std()), "n": len(vals)}
        with open(agg_path, "w") as af:
            json.dump(agg, af, indent=2)
        log.info("[user_study] aggregate written to %s", agg_path)
        print("\n===== AGGREGATE PROBE RESULTS =====")
        for k, v in agg["aggregate"].items():
            print(f"  {k}: mean={v['mean']:.4f} +/- std={v['std']:.4f} (n={v['n']})")
        print("====================================\n")
    else:
        train_run.main(cfg, run_dir)


if __name__ == "__main__":
    main()
