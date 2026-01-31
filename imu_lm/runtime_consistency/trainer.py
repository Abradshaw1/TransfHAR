"""Shared training loop (model-agnostic, objective_step contract)."""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, Optional

import numpy as np
import random
import torch
from torch.cuda.amp import GradScaler, autocast

from imu_lm.utils.helpers import cfg_get


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, cfg: Any, run_dir: str):
        self.cfg = cfg
        self.run_dir = run_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        seed = int(cfg_get(cfg, ["trainer", "seed"], 42))
        _set_seed(seed)

        os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

        self.metrics_path = os.path.join(run_dir, "logs", "metrics.txt")
        self.ckpt_latest = os.path.join(run_dir, "checkpoints", "latest.pt")

        self.log_every = int(cfg_get(cfg, ["logging", "log_every_steps"], 100))
        self.ckpt_every = int(cfg_get(cfg, ["logging", "ckpt_every_steps"], 1000))
        self.max_steps = int(cfg_get(cfg, ["trainer", "max_steps"], 100000))
        self.use_amp = bool(cfg_get(cfg, ["trainer", "amp"], False))

    def fit(
        self,
        model: torch.nn.Module,
        objective_step: Callable[[Any, torch.nn.Module, Any], tuple],
        train_loader,
        val_loader=None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
        start_step: int = 0,
    ):
        if optimizer is None:
            raise ValueError("Trainer.fit requires an optimizer; got None")
        model.to(self.device)
        scaler = GradScaler(enabled=self.use_amp)

        step = int(start_step)
        metrics_f = open(self.metrics_path, "a", buffering=1)

        while step < self.max_steps:
            for batch in train_loader:
                if batch is None:
                    continue
                step += 1
                model.train()

                batch = self._to_device(batch)
                with autocast(enabled=self.use_amp):
                    loss, logs = objective_step(batch, model, self.cfg)

                optimizer.zero_grad(set_to_none=True)
                if self.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                if step % self.log_every == 0:
                    lr = self._get_lr(optimizer)
                    line = self._format_log(step, "train", lr, logs)
                    metrics_f.write(line + "\n")
                    print(line)

                if step % self.ckpt_every == 0:
                    self._save_ckpt(model, optimizer, step)

                if step >= self.max_steps:
                    break

            # Simple val pass per epoch
            if val_loader is not None:
                print(f"[train] eval at step {step}")
                self._eval(val_loader, model, objective_step, step, metrics_f)

        metrics_f.close()
        # final checkpoint
        self._save_ckpt(model, optimizer, step)

    def _eval(self, val_loader, model, objective_step, step, metrics_f):
        model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                batch = self._to_device(batch)
                loss, logs = objective_step(batch, model, self.cfg)
                total_loss += float(loss.detach().item())
                count += 1
        if count == 0:
            return
        avg_loss = total_loss / count
        line = f"step={step} split=val loss={avg_loss:.6f}"
        metrics_f.write(line + "\n")
        print(line)

    def _get_lr(self, optimizer):
        if optimizer is None or len(optimizer.param_groups) == 0:
            return 0.0
        return optimizer.param_groups[0].get("lr", 0.0)

    def _format_log(self, step: int, split: str, lr: float, logs: Dict[str, float]):
        tokens = [f"step={step}", f"split={split}", f"loss={logs.get('loss', 0.0):.6f}", f"lr={lr:.6f}"]
        for k, v in logs.items():
            if k == "loss":
                continue
            if isinstance(v, float):
                tokens.append(f"{k}={v:.6f}")
            else:
                tokens.append(f"{k}={v}")
        return " ".join(tokens)

    def _save_ckpt(self, model, optimizer, step: int):
        state = {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "cfg": self.cfg,
        }
        torch.save(state, self.ckpt_latest)
        print(f"[train] saved checkpoint {self.ckpt_latest} (step={step})")

    def _to_device(self, batch):
        if isinstance(batch, (list, tuple)):
            return tuple(self._to_device(b) for b in batch)
        if isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        if torch.is_tensor(batch):
            return batch.to(self.device)
        return batch
