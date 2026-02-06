"""Evaluate a fully-supervised model (encoder + head) on the held-out test split.

Usage:
    python -m scripts.run_supervised_eval \
        --config configs/base.yaml \
        --model-config configs/vit1d.yaml \
        --run my_supervised_run
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import torch

from imu_lm.data.loaders import make_loaders
from imu_lm.runtime_consistency import artifacts
from imu_lm.utils.helpers import deep_update, load_yaml
from imu_lm.utils.metrics import compute_metrics, format_metrics_txt
from imu_lm.utils.training import remap_labels


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser(description="Evaluate supervised model on test split")
    ap.add_argument("--config", required=True, help="Base config YAML")
    ap.add_argument("--model-config", required=True, help="Model config YAML")
    ap.add_argument("--run", required=True, help="Run name under runs/<run>")
    args = ap.parse_args()

    base_cfg = load_yaml(args.config)
    model_cfg = load_yaml(args.model_config)
    cfg = deep_update(base_cfg, model_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runs_root = cfg.get("paths", {}).get("runs_root", "runs")
    run_dir = os.path.join(runs_root, args.run)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"Run dir not found: {run_dir}")

    # Load encoder
    logger.info("[sup eval] loading encoder from %s", run_dir)
    encoder = artifacts.load_encoder(run_dir, map_location=device).to(device)
    encoder.eval()

    # Load head
    head_path = os.path.join(artifacts.artifact_paths(run_dir)["dir"], "head.pt")
    if not os.path.exists(head_path):
        raise SystemExit(f"Head not found: {head_path} — was this a supervised run?")
    head = torch.load(head_path, map_location=device, weights_only=False).to(device)
    head.eval()

    # Load meta
    meta_path = artifacts.artifact_paths(run_dir)["meta"]
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
    logger.info("[sup eval] meta: %s", meta)

    # Load label_map for remapping raw labels → contiguous indices
    label_map_path = os.path.join(artifacts.artifact_paths(run_dir)["dir"], "label_map.json")
    raw_to_idx = None
    if os.path.exists(label_map_path):
        with open(label_map_path, "r") as f:
            label_map = json.load(f)
        raw_to_idx = {int(k): int(v) for k, v in label_map.get("raw_to_idx", {}).items()}
        logger.info("[sup eval] loaded label_map: %d classes", len(raw_to_idx))
    else:
        logger.warning("[sup eval] no label_map.json found — assuming labels are already contiguous")

    # Build test loader
    loaders = make_loaders(cfg)
    test_loader = loaders.get("test_loader")
    if test_loader is None:
        raise SystemExit("test_loader missing — set splits.test_ratio > 0 in config")

    logger.info("[sup eval] evaluating on test split (%d windows)...", len(test_loader.dataset))

    # Aggregated eval
    all_y_true = []
    all_y_pred = []
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue
            x, y = batch
            x, y = x.to(device), y.to(device)

            # Remap raw labels to contiguous indices
            if raw_to_idx is not None:
                y = remap_labels(y, raw_to_idx)
            valid = y >= 0
            if not valid.any():
                continue
            x, y = x[valid], y[valid]

            z = encoder(x)
            logits = head(z)
            loss = torch.nn.functional.cross_entropy(logits, y)
            preds = logits.argmax(dim=1)

            total_loss += float(loss.item()) * y.shape[0]
            count += y.shape[0]
            all_y_true.extend(y.cpu().tolist())
            all_y_pred.extend(preds.cpu().tolist())

    if count == 0:
        raise SystemExit("No test samples found")

    avg_loss = total_loss / count
    metrics = compute_metrics(all_y_true, all_y_pred)
    metrics["test_loss"] = avg_loss
    metrics["test_samples"] = count

    # Print results
    print(f"\n[sup eval] ===== TEST RESULTS =====")
    print(f"[sup eval] samples={count} loss={avg_loss:.6f}")
    print(f"[sup eval] bal_acc={metrics['bal_acc']:.4f}  macro_f1={metrics['macro_f1']:.4f}  macro_prec={metrics['macro_precision']:.4f}  macro_rec={metrics['macro_recall']:.4f}")

    per_class = metrics.get("per_class", {})
    if per_class:
        print(f"\n[sup eval] Per-class metrics:")
        print(f"  {'class':<20s} {'F1':>8s} {'Prec':>8s} {'Rec':>8s} {'Acc':>8s} {'Support':>8s}")
        for lbl, cm in per_class.items():
            if isinstance(cm, dict):
                print(f"  {str(lbl):<20s} {cm.get('f1',0):.4f}   {cm.get('precision',0):.4f}   {cm.get('recall',0):.4f}   {cm.get('accuracy',0):.4f}   {cm.get('support',0):>6d}")

    # Save summary
    summary_path = os.path.join(run_dir, "test_summary.json")
    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[sup eval] summary written to {summary_path}")


if __name__ == "__main__":
    main()
