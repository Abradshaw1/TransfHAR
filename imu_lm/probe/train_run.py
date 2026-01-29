"""Stage B training for frozen encoder + linear head."""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

from imu_lm.data.loaders import make_loaders
from imu_lm.probe.eval import eval_head
from imu_lm.probe.head import LinearHead
from imu_lm.probe.io import load_checkpoint, resolve_probe_dir, save_checkpoint, write_metrics_line, write_summary
from imu_lm.runtime_consistency import artifacts
from imu_lm.utils.metrics import compute_metrics, format_metrics_txt


def _cfg_get(cfg: Any, path, default=None):
    cur = cfg
    for key in path:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, default)
        else:
            cur = getattr(cur, key, default)
    return cur if cur is not None else default


def _load_encoder_meta(run_dir: str) -> Dict[str, Any]:
    paths = artifacts.artifact_paths(run_dir)
    if not os.path.exists(paths["meta"]):
        return {}
    with open(paths["meta"], "r") as f:
        return json.load(f)


def _build_label_map(loader: DataLoader, labels_cfg: Dict[str, Any]) -> Dict[str, Any]:
    unknown_id = labels_cfg.get("unknown_id", None)
    drop_unknown = bool(labels_cfg.get("drop_unknown", True))
    min_count = int(labels_cfg.get("min_count_per_class", 0))

    counts: Dict[int, int] = {}
    for batch in loader:
        if batch is None:
            continue
        _, y = batch
        for v in y.tolist():
            v_int = int(v)
            if drop_unknown and unknown_id is not None and v_int == int(unknown_id):
                continue
            counts[v_int] = counts.get(v_int, 0) + 1

    kept = [k for k, c in counts.items() if c >= min_count]
    kept = sorted(kept)

    raw_to_idx = {int(r): i for i, r in enumerate(kept)}
    idx_to_raw = {i: int(r) for i, r in enumerate(kept)}
    return {"raw_to_idx": raw_to_idx, "idx_to_raw": idx_to_raw}


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


def _train_epoch(train_loader, encoder, head, optimizer, device, label_map, use_amp: bool, grad_clip_norm: float, scaler: GradScaler):
    encoder.eval()
    head.train()
    total_loss = 0.0
    n_samples = 0
    y_true = []
    y_pred = []

    raw_to_idx = {int(k): int(v) for k, v in label_map.get("raw_to_idx", {}).items()}

    for batch in train_loader:
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
        with autocast(enabled=use_amp):
            with torch.no_grad():
                feats = encoder.forward_features(x)
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

    if n_samples == 0:
        return {"loss": 0.0, "acc": 0.0, "bal_acc": 0.0, "macro_f1": 0.0}

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / n_samples
    metrics["acc"] = float(np.mean(np.array(y_true) == np.array(y_pred)))
    return metrics


def _apply_probe_batch_size(cfg: Any, train_cfg: Dict[str, Any]):
    bs = train_cfg.get("batch_size", None)
    if bs is None:
        return
    if isinstance(cfg, dict):
        cfg.setdefault("data", {}).setdefault("loading", {})
        cfg["data"]["loading"]["batch_size"] = bs
        cfg["data"]["loading"]["eval_batch_size"] = bs
    else:
        if not hasattr(cfg, "data"):
            return
        if not hasattr(cfg.data, "loading"):
            return
        cfg.data.loading.batch_size = bs
        cfg.data.loading.eval_batch_size = bs


def main(cfg: Any, run_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    probe_cfg = cfg.get("probe", {}) if isinstance(cfg, dict) else getattr(cfg, "probe", {})
    labels_cfg = probe_cfg.get("labels", {})
    fewshot_cfg = probe_cfg.get("fewshot", {})
    train_cfg = probe_cfg.get("train", {})

    paths = resolve_probe_dir(run_dir, cfg)
    # ensure probe batch size overrides loaders
    _apply_probe_batch_size(cfg, train_cfg)

    encoder = artifacts.load_encoder(run_dir, map_location=device)
    encoder = encoder.to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    meta = _load_encoder_meta(run_dir)

    loaders = make_loaders(cfg)
    train_loader = loaders.get("probe_train_loader")
    val_loader = loaders.get("probe_val_loader") if loaders else None
    test_loader = loaders.get("probe_test_loader") if loaders else None

    if train_loader is None:
        raise RuntimeError("probe_train_loader missing; ensure make_loaders returns probe splits")

    label_map = _build_label_map(train_loader, labels_cfg)

    if fewshot_cfg.get("enabled", False):
        shots = int(fewshot_cfg.get("shots_per_class", 5))
        seed = int(fewshot_cfg.get("seed", 0))
        train_loader = _fewshot_subset(train_loader, label_map, shots, seed)

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
                feats = encoder.forward_features(x)
            embed_dim = feats.shape[-1]
            break
    if embed_dim is None:
        raise RuntimeError("Could not infer embedding dim from encoder/meta")

    label_map["embedding_dim"] = int(embed_dim)

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

    scaler = GradScaler(enabled=use_amp)

    best_metric = -1e9
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        train_metrics = _train_epoch(train_loader, encoder, head, optimizer, device, label_map, use_amp, grad_clip, scaler)
        val_metrics = eval_head(encoder, head, val_loader, label_map, device) if val_loader is not None else {}

        # log
        line_train = f"epoch={epoch} split=train " + format_metrics_txt(train_metrics)
        write_metrics_line(paths["metrics"], line_train)
        if val_loader is not None:
            line_val = f"epoch={epoch} split=val " + format_metrics_txt(val_metrics)
            write_metrics_line(paths["metrics"], line_val)

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

