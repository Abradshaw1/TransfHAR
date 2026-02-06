"""Stage B training for frozen encoder + linear head."""

from __future__ import annotations

import json
import os
import random
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
from imu_lm.probe.io import load_checkpoint, resolve_probe_dir, save_checkpoint, write_metrics_line, write_summary
from imu_lm.runtime_consistency import artifacts
from imu_lm.utils.helpers import cfg_get
from imu_lm.utils.metrics import compute_metrics, format_metrics_txt


def _load_encoder_meta(run_dir: str) -> Dict[str, Any]:
    paths = artifacts.artifact_paths(run_dir)
    if not os.path.exists(paths["meta"]):
        return {}
    with open(paths["meta"], "r") as f:
        return json.load(f)


def _build_label_map(loader: DataLoader, labels_cfg: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    unknown_id = labels_cfg.get("unknown_id", None)
    drop_unknown = bool(labels_cfg.get("drop_unknown", True))
    min_count = int(labels_cfg.get("min_count_per_class", 0))

    counts: Dict[int, int] = {}
    seen_batches = 0
    for batch in loader:
        seen_batches += 1
        if batch is None:
            continue
        _, y = batch
        for v in y.tolist():
            v_int = int(v)
            if drop_unknown and unknown_id is not None and v_int == int(unknown_id):
                continue
            counts[v_int] = counts.get(v_int, 0) + 1
        if seen_batches % 100 == 0:
            logger.info("[probe] label_map progress: batches=%d classes=%d", seen_batches, len(counts))

    kept = [k for k, c in counts.items() if c >= min_count]
    kept = sorted(kept)

    raw_to_idx = {int(r): i for i, r in enumerate(kept)}
    idx_to_raw = {i: int(r) for i, r in enumerate(kept)}
    logger.info("[probe] label_map built: classes=%d from %d batches", len(raw_to_idx), seen_batches)
    return {"raw_to_idx": raw_to_idx, "idx_to_raw": idx_to_raw}


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
        df = pd.read_parquet(parquet_path, columns=[label_col, name_col])
        unique_pairs = df.drop_duplicates(subset=[label_col])
        label_names = {int(row[label_col]): str(row[name_col]) for _, row in unique_pairs.iterrows()}
        logger.info("[probe] built label_names mapping: %d activities", len(label_names))
        return label_names
    except Exception as e:
        logger.warning("[probe] failed to build label_names: %s", e)
        return {}


def _fewshot_subset(loader: DataLoader, label_map: Dict[str, Any], shots_per_class: int, seed: int) -> DataLoader:
    raw_to_idx = {int(k): int(v) for k, v in label_map.get("raw_to_idx", {}).items()}
    dataset = loader.dataset
    rng = random.Random(seed)
    per_class: Dict[int, List[int]] = {idx: [] for idx in raw_to_idx.values()}

    for idx in range(len(dataset)):
        item = dataset[idx]
        if item is None:
            continue
        _, y = item
        raw = int(y)
        if raw not in raw_to_idx:
            continue
        mapped = raw_to_idx[raw]
        per_class[mapped].append(idx)

    keep_indices: List[int] = []
    for cls_idx, idxs in per_class.items():
        rng.shuffle(idxs)
        keep_indices.extend(idxs[: shots_per_class])

    subset = Subset(dataset, keep_indices)
    return DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=True,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=False,
        collate_fn=loader.collate_fn,
    )


def _train_epoch(train_loader, encoder, head, optimizer, device, label_map, use_amp: bool, grad_clip_norm: float, scaler: GradScaler, logger: logging.Logger, log_every_steps: int):
    encoder.eval()
    head.train()
    total_loss = 0.0
    n_samples = 0
    y_true = []
    y_pred = []

    raw_to_idx = {int(k): int(v) for k, v in label_map.get("raw_to_idx", {}).items()}

    for batch_idx, batch in enumerate(train_loader, 1):
        if batch is None:
            continue
        x, y_raw = batch
        y_raw = y_raw.to(device)
        idxs = [i for i, v in enumerate(y_raw.tolist()) if int(v) in raw_to_idx]
        if len(idxs) == 0:
            continue
        y = torch.tensor([raw_to_idx[int(y_raw[i])] for i in idxs], dtype=torch.long, device=device)

        x = torch.index_select(x.to(device), 0, torch.tensor(idxs, device=device))
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
            step_metrics = compute_metrics(y.cpu().tolist(), preds.cpu().tolist()) if y.numel() > 0 else {}
            batch_bal_acc = step_metrics.get("bal_acc", 0.0)
            batch_macro_f1 = step_metrics.get("macro_f1", 0.0)
            logger.info(
                "[probe] step=%d loss=%.6f bal_acc=%.4f macro_f1=%.4f",
                batch_idx,
                loss.detach().item(),
                batch_bal_acc,
                batch_macro_f1,
            )

    if n_samples == 0:
        return {"loss": 0.0, "acc": 0.0, "bal_acc": 0.0, "macro_f1": 0.0}

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
    # Flattened probe config: labels/fewshot/train fields are now directly under probe
    labels_cfg = {
        "unknown_id": probe_cfg.get("unknown_id"),
        "drop_unknown": probe_cfg.get("drop_unknown", True),
        "min_count_per_class": probe_cfg.get("min_count_per_class", 0),
    }
    fewshot_cfg = {
        "enabled": probe_cfg.get("fewshot_enabled", False),
        "shots_per_class": probe_cfg.get("fewshot_shots_per_class", 5),
        "seed": probe_cfg.get("fewshot_seed", 0),
    }
    train_cfg = {
        "num_epochs": probe_cfg.get("num_epochs", 100),
        "batch_size": probe_cfg.get("batch_size", 256),
        "lr": probe_cfg.get("lr", 0.0001),
        "weight_decay": probe_cfg.get("weight_decay", 0.0),
        "grad_clip_norm": probe_cfg.get("grad_clip_norm", 1.0),
        "early_stop_patience": probe_cfg.get("early_stop_patience", 20),
        "amp": probe_cfg.get("amp", True),
        "selection_metric": probe_cfg.get("selection_metric", "macro_f1"),
        "log_every_batches": probe_cfg.get("log_every_batches", 5),
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

    label_map = _build_label_map(train_loader, labels_cfg, logger)
    
    # Build activity name mapping (dataset_activity_id → string name)
    raw_label_names = _build_label_names(cfg, logger)
    
    raw_keys = sorted(list(label_map.get("raw_to_idx", {}).keys()))
    preview = raw_keys[:10]
    logger.info(
        "[probe] label_map classes=%d raw_labels_preview=%s", len(raw_keys), preview
    )
    
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

    head = LinearHead(embed_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    num_epochs = int(train_cfg.get("num_epochs", 50))
    grad_clip = float(train_cfg.get("grad_clip_norm", 0.0))
    use_amp = bool(train_cfg.get("amp", True))
    selection_metric = train_cfg.get("selection_metric", "macro_f1")
    patience = int(train_cfg.get("early_stop_patience", 5))
    log_every_steps = int(train_cfg.get("log_every_steps", train_cfg.get("log_every_batches", 1))) or 1
    scaler = GradScaler("cuda", enabled=use_amp)

    best_metric = -1e9
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        logger.info("[probe] epoch %d/%d train...", epoch, num_epochs)
        train_metrics = _train_epoch(train_loader, encoder, head, optimizer, device, label_map, use_amp, grad_clip, scaler, logger, log_every_steps)
        val_metrics = eval_head(encoder, head, val_loader, label_map, device) if val_loader is not None else {}

        # log
        line_train = f"epoch={epoch} split=train " + format_metrics_txt(train_metrics)
        write_metrics_line(paths["metrics"], line_train)
        if val_loader is not None:
            line_val = f"epoch={epoch} split=val " + format_metrics_txt(val_metrics)
            write_metrics_line(paths["metrics"], line_val)
        logger.info(line_train)
        if val_loader is not None:
            logger.info(line_val)

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

