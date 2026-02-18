"""Stage B training for frozen encoder + linear head."""

from __future__ import annotations

import json
import os
import random
import warnings
from typing import Any, Dict, List, Tuple

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

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


def _fewshot_subset(loader: DataLoader, label_map: Dict[str, Any], shots_per_class: int, seed: int) -> DataLoader:
    """Subsample train loader to k windows per class.

    Uses a lightweight label-only parquet scan (no preprocessing/STFT).
    """
    import pyarrow.dataset as pa_ds
    from imu_lm.data.windowing import resolve_window_label

    raw_to_idx = {int(k): int(v) for k, v in label_map.get("raw_to_idx", {}).items()}
    dataset = loader.dataset
    rng = random.Random(seed)
    per_class: Dict[int, List[int]] = {idx: [] for idx in raw_to_idx.values()}

    # Bulk read label column from parquet — one query per unique dataset name
    pa = pa_ds.dataset(dataset.parquet_path, format="parquet")
    needed_keys = set(dataset._keys)
    ds_names = set(k.dataset for k in needed_keys)

    label_col = dataset._label_col
    subj_col = dataset._subject_col
    sess_col = dataset._session_col
    ds_col = dataset._dataset_col
    time_col = dataset._time_col
    cols = [label_col, subj_col, sess_col]
    if time_col:
        cols.append(time_col)

    session_labels: Dict = {}
    for ds_name in ds_names:
        table = pa.to_table(columns=cols, filter=pa_ds.field(ds_col) == ds_name)
        df = table.to_pandas()
        if time_col and time_col in df.columns:
            df = df.sort_values([subj_col, sess_col, time_col])
        for (subj, sess), grp in df.groupby([subj_col, sess_col]):
            from imu_lm.data.splits import SessionKey
            key = SessionKey(ds_name, str(subj), str(sess))
            if key in needed_keys:
                session_labels[key] = grp[label_col].to_numpy()

    # Resolve per-window labels in memory (no sensor data loaded)
    T = dataset._T
    for idx in range(len(dataset)):
        key = dataset._keys[dataset._sess_idx[idx]]
        y_arr = session_labels.get(key)
        if y_arr is None:
            continue
        start = int(dataset._starts[idx])
        yw = y_arr[start : start + T]
        label = resolve_window_label(yw, dataset.cfg)
        if label is None:
            continue
        raw = int(label)
        if raw not in raw_to_idx:
            continue
        mapped = raw_to_idx[raw]
        per_class[mapped].append(idx)

    keep_indices: List[int] = []
    for cls_idx, idxs in per_class.items():
        rng.shuffle(idxs)
        keep_indices.extend(idxs[:shots_per_class])

    subset = Subset(dataset, keep_indices)
    bs = loader.batch_size or 256
    return DataLoader(
        subset,
        batch_size=bs,
        shuffle=True,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=False,
        collate_fn=loader.collate_fn,
    )


def _train_epoch(train_loader, encoder, head, optimizer, device, label_map, use_amp: bool, grad_clip_norm: float, scaler: GradScaler, logger: logging.Logger, log_every_steps: int, epoch: int = 0):
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
            print(f"epoch={epoch} step={batch_idx} split=train {format_metrics_summary(step_metrics)}")

    if n_samples == 0:
        return {"loss": 0.0, "acc": 0.0, "bal_acc": 0.0, "macro_f1": 0.0}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / n_samples
    metrics["acc"] = float(np.mean(np.array(y_true) == np.array(y_pred)))
    return metrics


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


def main(cfg: Any, run_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    probe_cfg = cfg.get("probe", {}) if isinstance(cfg, dict) else getattr(cfg, "probe", {})
    unknown_id = probe_cfg.get("unknown_id", cfg_get(cfg, ["data", "unknown_label_id"], None))
    # Flattened probe config: labels/fewshot/train fields are now directly under probe
    labels_cfg = {
        "unknown_id": unknown_id,
        "drop_unknown": probe_cfg.get("drop_unknown", True),
        "min_count_per_class": probe_cfg.get("min_count_per_class", 0),
    }
    fewshot_cfg = {
        "enabled": probe_cfg.get("fewshot_enabled", False),
        "shots_per_class": probe_cfg.get("fewshot_shots_per_class", 5),
        "seed": probe_cfg.get("fewshot_seed", 0),
    }
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

    paths = resolve_probe_dir(run_dir, cfg)
    # ensure probe batch size overrides loaders
    _apply_probe_batch_size(cfg, probe_cfg)

    logger = logging.getLogger(__name__)

    logger.info("[probe] loading encoder from %s", run_dir)
    encoder = artifacts.load_encoder(run_dir, map_location=device)
    encoder = encoder.to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    meta = _load_encoder_meta(run_dir)

    # Auto-detect spectrogram settings from pretrain config if encoder needs images
    if meta.get("encoding") == "spectrogram_image":
        ckpt_path = os.path.join(run_dir, "checkpoints", "best.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(run_dir, "checkpoints", "latest.pt")
        saved_cfg = torch.load(ckpt_path, map_location="cpu").get("cfg", {})
        saved_spec = saved_cfg.get("spectrogram", {})
        if saved_spec:
            cfg["spectrogram"] = saved_spec
            logger.info("[probe] auto-loaded spectrogram config from pretrain checkpoint: %s", saved_spec)

    probe_dataset = cfg.get("splits", {}).get("probe_dataset", None) if isinstance(cfg, dict) else None
    logger.info("[probe] building probe loaders (probe_dataset=%s)", probe_dataset)
    loaders = make_loaders(cfg, dataset_filter=[probe_dataset] if probe_dataset else None)
    train_loader = loaders.get("probe_train_loader")
    val_loader = loaders.get("probe_val_loader") if loaders else None
    test_loader = loaders.get("probe_test_loader") if loaders else None

    if train_loader is None:
        raise RuntimeError("probe_train_loader missing; ensure make_loaders returns probe splits")

    logger.info(
        "[probe] splits sessions (by loaders): train=%d val=%d test=%d",
        len(train_loader.dataset),
        len(val_loader.dataset) if val_loader else 0,
        len(test_loader.dataset) if test_loader else 0,
    )

    unknown_id = labels_cfg.get("unknown_id", None)
    drop_unknown = bool(labels_cfg.get("drop_unknown", True))
    min_count = int(labels_cfg.get("min_count_per_class", 0))
    label_map = build_label_map(
        train_loader, cfg,
        unknown_id=unknown_id,
        drop_unknown=drop_unknown,
        min_count=min_count,
    )
    
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

    if fewshot_cfg.get("enabled", False):
        shots = int(fewshot_cfg.get("shots_per_class", 5))
        seed = int(fewshot_cfg.get("seed", 0))
        train_loader = _fewshot_subset(train_loader, label_map, shots, seed)
        logger.info("[probe] fewshot enabled: shots_per_class=%d seed=%d", shots, seed)

    # Determine embedding dim and num_classes
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
    logger.info(
        "[probe] ready: embed_dim=%d num_classes=%d train_batches=%d",
        embed_dim,
        num_classes,
        len(train_loader),
    )

    # Save probe metadata (architecture, classes, config)
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
        "fewshot": fewshot_cfg,
        "train_config": train_cfg,
    }
    probe_meta_path = os.path.join(paths["base"], "probe_meta.json")
    with open(probe_meta_path, "w") as f:
        json.dump(probe_meta, f, indent=2)
    logger.info("[probe] saved probe_meta.json to %s", probe_meta_path)

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

    for epoch in range(1, num_epochs + 1):
        train_metrics = _train_epoch(train_loader, encoder, head, optimizer, device, label_map, use_amp, grad_clip, scaler, logger, log_every_steps, epoch=epoch)
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

        # wandb logging
        if wandb is not None and wandb.run is not None:
            wb = {f"probe_train/{k}": v for k, v in train_metrics.items() if isinstance(v, (int, float))}
            if val_metrics:
                wb.update({f"probe_val/{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))})
            wandb.log(wb, step=epoch)

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
            logger.info("[probe] new best %s=%.6f at epoch %d; saved %s", selection_metric, best_metric, epoch, paths["best"])
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
        "shots_per_class": fewshot_cfg.get("shots_per_class") if fewshot_cfg.get("enabled", False) else None,
    }
    write_summary(paths["summary"], summary)
    logger.info("[probe] summary written to %s", paths["summary"])
