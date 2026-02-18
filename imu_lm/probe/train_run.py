"""Stage B training for frozen encoder + linear head."""

from __future__ import annotations

import json
import os
import warnings
from typing import Any, Dict

import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from imu_lm.data.loaders import make_loaders
from imu_lm.probe.eval import eval_head
from imu_lm.probe.head import LinearHead
from imu_lm.probe.io import (
    load_checkpoint,
    resolve_probe_dir,
    save_checkpoint,
    select_mapped_batch,
    write_metrics_line,
    write_summary,
)
from imu_lm.runtime_consistency import artifacts
from imu_lm.utils.helpers import cfg_get
from imu_lm.utils.metrics import compute_metrics, format_metrics_summary, format_metrics_txt
from imu_lm.utils.training import build_label_map

try:
    import wandb
except ImportError:
    wandb = None


def _load_encoder_meta(run_dir: str) -> Dict[str, Any]:
    paths = artifacts.artifact_paths(run_dir)
    if not os.path.exists(paths["meta"]):
        return {}
    with open(paths["meta"], "r") as f:
        return json.load(f)


def _build_label_names(cfg: Any, logger: logging.Logger) -> Dict[int, str]:
    """Build mapping from dataset_activity_id → string activity name."""
    parquet_path = cfg_get(cfg, ["paths", "dataset_path"], None)
    label_col = cfg_get(cfg, ["data", "label_column"], "dataset_activity_id")
    name_col = cfg_get(cfg, ["data", "label_name_column"], None)
    
    if not parquet_path or not name_col:
        logger.info("[probe] no label_name_column configured, using numeric IDs")
        return {}
    
    if not os.path.exists(parquet_path):
        logger.warning("[probe] parquet file not found: %s", parquet_path)
        return {}
    
    try:
        import pyarrow.dataset as pa_ds
        probe_dataset = cfg_get(cfg, ["splits", "probe_dataset"], None)
        dset = pa_ds.dataset(parquet_path, format="parquet")
        dataset_col = cfg_get(cfg, ["data", "dataset_column"], "dataset")
        filt = pa_ds.field(dataset_col) == probe_dataset if probe_dataset else None
        table = dset.to_table(columns=[label_col, name_col], filter=filt)
        df = table.to_pandas().drop_duplicates(subset=[label_col])
        label_names = {int(row[label_col]): str(row[name_col]) for _, row in df.iterrows()}
        logger.info("[probe] built label_names mapping: %d activities", len(label_names))
        return label_names
    except Exception as e:
        logger.warning("[probe] failed to build label_names: %s", e)
        return {}


def _train_epoch(train_loader, encoder, head, optimizer, device, label_map, use_amp: bool, grad_clip_norm: float, scaler: GradScaler, logger: logging.Logger, log_every_steps: int, epoch: int = 0, global_step: int = 0, wb_prefix: str = "probe"):
    encoder.eval()
    head.train()
    total_loss = 0.0
    n_samples = 0
    y_true = []
    y_pred = []

    raw_to_idx = {int(k): int(v) for k, v in label_map.get("raw_to_idx", {}).items()}
    unknown_id = label_map.get("unknown_id")

    for batch_idx, batch in enumerate(train_loader, 1):
        if batch is None:
            continue
        x, y_raw = batch
        x, y = select_mapped_batch(x, y_raw, raw_to_idx, device, unknown_raw_id=unknown_id)
        if x is None:
            continue
        global_step += 1
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=use_amp):
            with torch.no_grad():
                feats = encoder(x)
            logits = head(feats)
            loss = F.cross_entropy(logits, y)

        scaler.scale(loss).backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        preds = logits.detach().argmax(dim=1)
        total_loss += float(loss.detach().item() * y.shape[0])
        n_samples += y.shape[0]
        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

        if log_every_steps and batch_idx % log_every_steps == 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                step_metrics = compute_metrics(y.cpu().tolist(), preds.cpu().tolist()) if y.numel() > 0 else {}
            step_metrics["loss"] = float(loss.detach().item())
            step_metrics["acc"] = float((preds == y).float().mean().item())
            print(f"epoch={epoch} step={global_step} split=train {format_metrics_summary(step_metrics)}")
            if wandb is not None and wandb.run is not None:
                wb_train = {f"{wb_prefix}_train/{k}": v for k, v in step_metrics.items() if isinstance(v, (int, float))}
                wandb.log(wb_train, step=global_step)

    if n_samples == 0:
        return {"loss": 0.0, "acc": 0.0, "bal_acc": 0.0, "macro_f1": 0.0}, global_step

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / n_samples
    metrics["acc"] = float(np.mean(np.array(y_true) == np.array(y_pred)))
    return metrics, global_step


def _apply_probe_batch_size(cfg: Any, probe_cfg: Dict[str, Any]):
    bs = probe_cfg.get("batch_size", None)
    if bs is None:
        return
    if isinstance(cfg, dict):
        cfg.setdefault("data", {})
        cfg["data"]["batch_size"] = bs
        cfg["data"]["eval_batch_size"] = bs
    else:
        if not hasattr(cfg, "data"):
            return
        cfg.data.batch_size = bs
        cfg.data.eval_batch_size = bs


def setup_probe(cfg: Any, run_dir: str) -> Dict[str, Any]:
    """Shared setup: load encoder, build loaders, label map, infer embed dim.

    Returns a dict with all objects needed by run_probe().
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe_cfg = cfg.get("probe", {}) if isinstance(cfg, dict) else getattr(cfg, "probe", {})
    unknown_id = probe_cfg.get("unknown_id", cfg_get(cfg, ["data", "unknown_label_id"], None))
    train_cfg = {
        "num_epochs": probe_cfg.get("num_epochs", 1000000),
        "batch_size": probe_cfg.get("batch_size", 256),
        "lr": probe_cfg.get("lr", 0.0001),
        "weight_decay": probe_cfg.get("weight_decay", 0.0),
        "grad_clip_norm": probe_cfg.get("grad_clip_norm", 1.0),
        "early_stop_patience": probe_cfg.get("early_stop_patience", 20),
        "amp": probe_cfg.get("amp", True),
        "selection_metric": probe_cfg.get("selection_metric", "macro_f1"),
        "log_every_steps": probe_cfg.get("log_every_steps", 5),
    }
    _apply_probe_batch_size(cfg, probe_cfg)
    logger = logging.getLogger(__name__)

    logger.info("[probe] loading encoder from %s", run_dir)
    encoder = artifacts.load_encoder(run_dir, map_location=device)
    encoder = encoder.to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    meta = _load_encoder_meta(run_dir)

    probe_dataset = cfg.get("splits", {}).get("probe_dataset", None) if isinstance(cfg, dict) else None
    logger.info("[probe] building probe loaders (probe_dataset=%s)", probe_dataset)
    loaders = make_loaders(cfg, dataset_filter=[probe_dataset] if probe_dataset else None)
    train_loader = loaders.get("probe_train_loader")
    val_loader = loaders.get("probe_val_loader") if loaders else None
    test_loader = loaders.get("probe_test_loader") if loaders else None

    if train_loader is None:
        raise RuntimeError("probe_train_loader missing; ensure make_loaders returns probe splits")

    logger.info(
        "[probe] splits (by loaders): train=%d val=%d test=%d",
        len(train_loader.dataset),
        len(val_loader.dataset) if val_loader else 0,
        len(test_loader.dataset) if test_loader else 0,
    )

    drop_unknown = bool(probe_cfg.get("drop_unknown", True))
    min_count = int(probe_cfg.get("min_count_per_class", 0))

    # Fast label discovery via pyarrow (avoids iterating full DataLoader)
    import pyarrow.dataset as pa_ds
    import pyarrow.compute as pc
    parquet_path = cfg_get(cfg, ["paths", "dataset_path"])
    label_col = cfg_get(cfg, ["data", "label_column"], "dataset_activity_id")
    dataset_col = cfg_get(cfg, ["data", "dataset_column"], "dataset")
    pa_dset = pa_ds.dataset(parquet_path, format="parquet")
    filt = pa_ds.field(dataset_col) == probe_dataset if probe_dataset else None
    lbl_arr = pa_dset.to_table(columns=[label_col], filter=filt)[label_col]
    vc = lbl_arr.value_counts().to_pylist()
    counts = {int(entry["values"]): int(entry["counts"]) for entry in vc}
    if drop_unknown and unknown_id is not None:
        counts.pop(int(unknown_id), None)
    kept = sorted([k for k, c in counts.items() if c >= min_count])
    raw_to_idx = {r: i for i, r in enumerate(kept)}
    idx_to_raw = {i: r for i, r in enumerate(kept)}
    logger.info("build_label_map (fast): classes=%d", len(raw_to_idx))
    label_map = {
        "raw_to_idx": raw_to_idx,
        "idx_to_raw": idx_to_raw,
        "num_classes": len(raw_to_idx),
        "unknown_id": unknown_id,
    }

    # Build activity name mapping (dataset_activity_id → string name)
    raw_label_names = _build_label_names(cfg, logger)
    raw_keys = sorted(list(label_map.get("raw_to_idx", {}).keys()))
    logger.info("[probe] label_map classes=%d", len(raw_keys))

    # Build idx → name mapping for metrics display
    idx_to_raw = label_map.get("idx_to_raw", {})
    label_names = {}
    for idx, raw in idx_to_raw.items():
        label_names[idx] = raw_label_names.get(raw, str(raw))
    label_map["label_names"] = label_names

    # Determine embedding dim
    num_classes = len(label_map.get("raw_to_idx", {}))
    if num_classes == 0:
        raise RuntimeError("No probe classes found after filtering; check labels/unknown handling")
    embed_dim = meta.get("embedding_dim", None)
    if embed_dim is None:
        for batch in train_loader:
            if batch is None:
                continue
            x, y_raw = batch
            y_raw = y_raw.to(device)
            mapped = [label_map["raw_to_idx"].get(int(v), None) for v in y_raw.tolist()]
            if all(m is None for m in mapped):
                continue
            x = x.to(device)
            with torch.no_grad():
                feats = encoder(x)
            embed_dim = feats.shape[-1]
            break
    if embed_dim is None:
        raise RuntimeError("Could not infer embedding dim from encoder/meta")
    label_map["embedding_dim"] = int(embed_dim)

    logger.info("[probe] ready: embed_dim=%d num_classes=%d", embed_dim, num_classes)
    return {
        "device": device, "encoder": encoder, "train_cfg": train_cfg,
        "train_loader": train_loader, "val_loader": val_loader,
        "test_loader": test_loader, "label_map": label_map,
        "label_names": label_names, "embed_dim": embed_dim,
        "num_classes": num_classes, "probe_dataset": probe_dataset,
        "logger": logger,
    }


def run_probe(
    encoder, train_loader, val_loader, test_loader, label_map, label_names,
    embed_dim, num_classes, train_cfg, paths, probe_dataset, device, logger,
    wb_prefix: str = "probe", extra_meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Train a linear probe to convergence → test eval → summary.

    Used by both train_run.main() and fewshot_train_run.main().
    """
    # Save probe metadata
    probe_meta = {
        "embed_dim": int(embed_dim),
        "num_classes": num_classes,
        "head_type": "linear",
        "selection_metric": train_cfg.get("selection_metric", "macro_f1"),
        "label_map": {
            "raw_to_idx": label_map.get("raw_to_idx", {}),
            "idx_to_raw": label_map.get("idx_to_raw", {}),
        },
        "label_names": label_names,
        "probe_dataset": probe_dataset,
        "train_config": train_cfg,
    }
    if extra_meta:
        probe_meta.update(extra_meta)
    probe_meta_path = os.path.join(paths["base"], "probe_meta.json")
    with open(probe_meta_path, "w") as f:
        json.dump(probe_meta, f, indent=2)
    logger.info("[%s] saved probe_meta.json to %s", wb_prefix, probe_meta_path)

    head = LinearHead(embed_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    num_epochs = int(train_cfg.get("num_epochs", 1000000))
    grad_clip = float(train_cfg.get("grad_clip_norm", 0.0))
    use_amp = bool(train_cfg.get("amp", True))
    selection_metric = train_cfg.get("selection_metric", "macro_f1")
    patience = int(train_cfg.get("early_stop_patience", 20))
    log_every_steps = int(train_cfg.get("log_every_steps", 5))
    scaler = GradScaler("cuda", enabled=use_amp)

    best_metric = -1e9
    best_epoch = -1
    epochs_no_improve = 0

    global_step = 0
    for epoch in range(1, num_epochs + 1):
        train_metrics, global_step = _train_epoch(train_loader, encoder, head, optimizer, device, label_map, use_amp, grad_clip, scaler, logger, log_every_steps, epoch=epoch, global_step=global_step, wb_prefix=wb_prefix)
        val_metrics = eval_head(encoder, head, val_loader, label_map, device) if val_loader is not None else {}

        # Full metrics to file (includes per-class)
        line_train_full = f"epoch={epoch} split=train " + format_metrics_txt(train_metrics)
        write_metrics_line(paths["metrics"], line_train_full)
        if val_loader is not None:
            line_val_full = f"epoch={epoch} split=val " + format_metrics_txt(val_metrics)
            write_metrics_line(paths["metrics"], line_val_full)

        # Summary to terminal (clean, one line per split)
        print(f"epoch={epoch} split=train {format_metrics_summary(train_metrics)}")
        if val_loader is not None:
            print(f"epoch={epoch} split=val {format_metrics_summary(val_metrics)}")

        # wandb: val metrics logged per epoch at current global_step
        if wandb is not None and wandb.run is not None and val_metrics:
            wb_val = {f"{wb_prefix}_val/{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))}
            wandb.log(wb_val, step=global_step)

        # checkpoints
        save_checkpoint(paths["latest"], head, optimizer, epoch, label_map)

        if val_loader is not None:
            metric_val = val_metrics.get(selection_metric, None)
        else:
            metric_val = train_metrics.get(selection_metric, None)

        if metric_val is not None and metric_val > best_metric:
            best_metric = metric_val
            best_epoch = epoch
            epochs_no_improve = 0
            save_checkpoint(paths["best"], head, optimizer, epoch, label_map)
            logger.info("[%s] new best %s=%.6f at epoch %d; saved %s", wb_prefix, selection_metric, best_metric, epoch, paths["best"])
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    # Test eval using best checkpoint if available
    if os.path.exists(paths["best"]):
        head_state, saved_label_map = load_checkpoint(paths["best"], device)
        head.load_state_dict(head_state)
        label_map = saved_label_map or label_map

    test_metrics = eval_head(encoder, head, test_loader, label_map, device) if test_loader is not None else {}

    summary = {
        "best_epoch": best_epoch,
        "selection_metric": selection_metric,
        "best_metric": best_metric,
        "test": test_metrics,
        "num_classes": num_classes,
    }
    if extra_meta:
        summary.update(extra_meta)
    write_summary(paths["summary"], summary)
    logger.info("[%s] summary written to %s", wb_prefix, paths["summary"])
    return summary


def main(cfg: Any, run_dir: str):
    ctx = setup_probe(cfg, run_dir)
    paths = resolve_probe_dir(run_dir, cfg)
    run_probe(
        encoder=ctx["encoder"], train_loader=ctx["train_loader"],
        val_loader=ctx["val_loader"], test_loader=ctx["test_loader"],
        label_map=ctx["label_map"], label_names=ctx["label_names"],
        embed_dim=ctx["embed_dim"], num_classes=ctx["num_classes"],
        train_cfg=ctx["train_cfg"], paths=paths,
        probe_dataset=ctx["probe_dataset"], device=ctx["device"],
        logger=ctx["logger"],
    )
