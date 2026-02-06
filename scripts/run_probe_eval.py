"""Stage B: evaluate frozen head/encoder on probe test split."""

from __future__ import annotations

import argparse
import json
import logging
import os

import torch

from imu_lm.data.loaders import make_loaders
from imu_lm.probe.eval import eval_head
from imu_lm.probe.head import LinearHead
from imu_lm.probe.io import load_checkpoint, resolve_probe_dir, write_metrics_line, write_summary
from imu_lm.runtime_consistency import artifacts
from imu_lm.utils.helpers import deep_update, load_yaml
from imu_lm.utils.metrics import format_metrics_txt


def _infer_embed_dim(encoder, loader, label_map, device):
    for batch in loader:
        if batch is None:
            continue
        x, y_raw = batch
        y_raw = y_raw.to(device)
        mapped = [label_map["raw_to_idx"].get(int(v), None) for v in y_raw.tolist()]
        if all(m is None for m in mapped):
            continue
        x = x.to(device)
        with torch.no_grad():
            feats = encoder.forward_features(x)
        return feats.shape[-1]
    raise RuntimeError("Could not infer embedding dim from encoder")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Base config YAML")
    ap.add_argument("--probe-config", required=True, help="Probe config YAML")
    ap.add_argument("--run", required=True, help="Run name under runs/<run>")
    ap.add_argument("--ckpt", default=None, help="Optional checkpoint path (defaults to best.pt)")
    args = ap.parse_args()

    base_cfg = load_yaml(args.config)
    probe_cfg = load_yaml(args.probe_config)
    cfg = deep_update(base_cfg, probe_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runs_root = cfg.get("paths", {}).get("runs_root", "runs")
    run_dir = os.path.join(runs_root, args.run)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"Run dir not found: {run_dir}")

    paths = resolve_probe_dir(run_dir, cfg)

    logger.info("[probe eval] loading encoder from %s", run_dir)
    encoder = artifacts.load_encoder(run_dir, map_location=device).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    probe_dataset = cfg.get("splits", {}).get("probe_dataset", None) if isinstance(cfg, dict) else None
    if probe_dataset is None and isinstance(cfg, dict):
        probe_dataset = cfg.get("data", {}).get("splits", {}).get("probe_dataset", None)
    logger.info("[probe eval] building data loaders (probe_dataset=%s)...", probe_dataset)
    loaders = make_loaders(cfg, dataset_filter=[probe_dataset] if probe_dataset else None)
    logger.info("[probe eval] data loaders ready")
    test_loader = loaders.get("probe_test_loader") if loaders else None
    if test_loader is None:
        raise SystemExit("probe_test_loader missing; cannot eval")

    # Load checkpoint
    ckpt_path = args.ckpt or paths["best"]
    if not os.path.exists(ckpt_path):
        ckpt_path = paths.get("latest", ckpt_path)
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    state, label_map = load_checkpoint(ckpt_path, device)
    raw_to_idx = label_map.get("raw_to_idx", {})
    num_classes = len(raw_to_idx)
    if num_classes == 0:
        raise SystemExit("label_map in checkpoint is empty; cannot eval")

    embed_dim = label_map.get("embedding_dim")
    if embed_dim is None:
        meta_path = artifacts.artifact_paths(run_dir)["meta"]
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
                embed_dim = meta.get("embedding_dim")
    if embed_dim is None:
        embed_dim = _infer_embed_dim(encoder, test_loader, label_map, device)

    head = LinearHead(embed_dim, num_classes).to(device)
    head.load_state_dict(state)
    head.eval()

    logger.info("[probe eval] running evaluation on test split...")
    metrics = eval_head(encoder, head, test_loader, label_map, device)

    line = "split=test " + format_metrics_txt(metrics)
    write_metrics_line(paths["metrics"], line)

    summary = {
        "checkpoint": os.path.relpath(ckpt_path, run_dir),
        "metrics": metrics,
        "num_classes": num_classes,
    }
    write_summary(paths["summary"], summary)

    # Print summary metrics
    labels = metrics.get("labels", [])
    per_class = metrics.get("per_class", {})
    
    print(f"\n[probe eval] ===== TEST RESULTS =====")
    print(f"[probe eval] classes evaluated: {len(labels)} (expected: {num_classes})")
    print(f"[probe eval] bal_acc={metrics.get('bal_acc', 0):.4f}  macro_f1={metrics.get('macro_f1', 0):.4f}")
    print(f"\n[probe eval] Per-class F1:")
    for lbl, class_metrics in per_class.items():
        f1_val = class_metrics.get("f1", 0.0) if isinstance(class_metrics, dict) else 0.0
        print(f"  class {lbl}: F1={f1_val:.4f}")
    print(f"\n[probe eval] summary written to {paths['summary']}")


if __name__ == "__main__":
    main()
