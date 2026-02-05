"""
Single-step smoke test for Stage A (MAE) pretrain.
- Verifies config essentials, split leakage, input shape, forward/backward, logging, checkpoint, encoder artifact.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Tuple

import torch
import yaml
from torch.cuda.amp import GradScaler, autocast

from imu_lm.data.loaders import make_loaders
from imu_lm.data.splits import build_session_index, make_splits
from imu_lm.models.ViT2D import run as vit_run
from imu_lm.models.ViT2D.model import ViTEncoder
from imu_lm.objectives import mae as mae_obj
from imu_lm.runtime_consistency.artifacts import save_encoder
from imu_lm.utils.helpers import cfg_get


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
        run_name = f"smoke-{ts}"
    run_dir = os.path.join(runs_root, run_name)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "artifacts"), exist_ok=True)
    return run_dir


def _log_resolved(cfg: Dict[str, Any], run_dir: str):
    log_path = os.path.join(run_dir, "logs", "resolved_config.json")
    essentials = {
        "vit.warm_start": cfg_get(cfg, ["vit", "warm_start"], None),
        "vit.pretrained_id": cfg_get(cfg, ["vit", "pretrained_id"], None),
        "splits.probe_dataset": cfg_get(cfg, ["splits", "probe_dataset"], None),
        "splits.probe_ratios": cfg_get(cfg, ["splits", "probe_ratios"], None),
        "data.batch_size": cfg_get(cfg, ["data", "batch_size"], None),
        "data.eval_batch_size": cfg_get(cfg, ["data", "eval_batch_size"], None),
        "paths.runs_root": cfg_get(cfg, ["paths", "runs_root"], None),
        "run_dir": run_dir,
    }
    with open(log_path, "w") as f:
        json.dump({"essentials": essentials, "cfg": cfg}, f, indent=2)
    print("# essentials", json.dumps(essentials))


def _assert_no_probe_leakage(splits: Dict[str, Iterable], probe_dataset: str | None):
    if probe_dataset is None:
        return
    for name in ["train_keys", "val_keys"]:
        keys = splits.get(name, []) or []
        bad = [k for k in keys if getattr(k, "dataset", None) == probe_dataset]
        if bad:
            raise AssertionError(f"probe leakage: split={name} contains probe_dataset={probe_dataset}")


def _first_non_none_batch(loader):
    if loader is None:
        return None
    for batch in loader:
        if batch is not None:
            return batch
    return None


def _inspect_batch(x: torch.Tensor):
    if not torch.is_floating_point(x):
        raise AssertionError(f"batch dtype must be float; got {x.dtype}")
    if x.dim() != 4:
        raise AssertionError(f"batch must be image-like [B,C,H,W]; got shape {tuple(x.shape)}")
    stats = {
        "shape": list(x.shape),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
    }
    print("# batch", json.dumps(stats))


def _format_metrics(step: int, split: str, loss: float, logs: Dict[str, Any]) -> str:
    toks = [f"step={step}", f"split={split}", f"loss={loss:.6f}"]
    for k, v in logs.items():
        if k == "loss":
            continue
        if isinstance(v, float):
            toks.append(f"{k}={v:.6f}")
        else:
            toks.append(f"{k}={v}")
    return " ".join(toks)


def _save_latest(model, optimizer, cfg, run_dir: str, step: int):
    state = {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "cfg": cfg}
    path = os.path.join(run_dir, "checkpoints", "latest.pt")
    torch.save(state, path)
    return path


def _build_meta(model: ViTEncoder, cfg: Any) -> Dict[str, Any]:
    return {
        "embedding_dim": model.embed_dim,
        "encoding": "spectrogram_image",
        "objective": "mae",
        "backbone": model.backbone_name,
        "input_spec": {
            "channels": model.num_channels,
            "height": model.resize_hw[0],
            "width": model.resize_hw[1],
            "patch_size": model.patch_size,
        },
        "normalization": cfg_get(cfg, ["preprocess", "normalize", "method"], None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--steps", type=int, default=1, help="Number of train steps to run (0 to skip)")
    args = ap.parse_args()

    base_cfg = _load_yaml(args.config)
    model_cfg = _load_yaml(args.model_config)
    cfg = _deep_update(base_cfg, model_cfg)

    run_dir = _resolve_run_dir(cfg, args.run_name)
    _log_resolved(cfg, run_dir)

    # Build session index/splits and assert no probe leakage in train/val
    session_index = build_session_index(cfg_get(cfg, ["paths", "dataset_path"]), cfg)
    splits = make_splits(session_index, cfg)
    probe_dataset = cfg_get(cfg, ["splits", "probe_dataset"], None)
    _assert_no_probe_leakage(splits, probe_dataset)

    # Build loaders
    loaders = make_loaders(cfg)
    train_loader = loaders.get("train_loader")
    if train_loader is None:
        raise SystemExit("train_loader missing; cannot run smoke")

    # Grab one batch for inspection
    batch = _first_non_none_batch(train_loader)
    if batch is None:
        raise SystemExit("No non-empty batch found in train_loader")
    x, y = batch
    _inspect_batch(x)

    # Build model/objective/optim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTEncoder(cfg).to(device)  # mae_model already attached
    objective_fn = mae_obj.forward_loss
    optimizer = vit_run._build_optimizer(model, cfg)
    scheduler = vit_run._build_scheduler(optimizer, cfg)
    use_amp = bool(cfg_get(cfg, ["trainer", "amp"], False))
    scaler = GradScaler(enabled=use_amp)

    # Single or zero step
    metrics_path = os.path.join(run_dir, "logs", "metrics.txt")
    with open(metrics_path, "a", buffering=1) as mf:
        if args.steps > 0:
            x = x.to(device)
            y = y.to(device)
            step = 1
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                loss, logs = objective_fn((x, y), model, cfg)
            if not torch.isfinite(loss):
                raise AssertionError("Loss is not finite")
            scaler.scale(loss).backward()
            grads = [p for p in model.parameters() if p.grad is not None]
            if len(grads) == 0:
                raise AssertionError("No gradients found after backward")
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            line = _format_metrics(step, "train", float(loss.detach().item()), logs)
            mf.write(line + "\n")
            print(line)
        else:
            print("steps=0: skipping train step; only saving artifacts")

    # Save checkpoint + encoder artifact
    ckpt_path = _save_latest(model, optimizer, cfg, run_dir, step=1)
    meta = _build_meta(model, cfg)
    save_encoder(model, meta, run_dir)

    print(json.dumps({"checkpoint": ckpt_path, "artifacts": run_dir + "/artifacts"}))


if __name__ == "__main__":
    main()
