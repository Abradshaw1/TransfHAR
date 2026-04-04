#!/usr/bin/env python3
"""TransfHAR Real-Time Demo Server
===================================
Apple Watch (Watch-Data-Streamer) streams live IMU over UDP.
Server windows, preprocesses (same repo pipeline: impute → lowpass 24Hz → z-score),
encodes with frozen 6-axis ViT1D, trains linear probe on demand, classifies in real-time.

Data flow:
  Watch CoreMotion → UDP → parse (gravity+userAcc → total acc, gyro as-is)
  → ring buffer → 128-sample windows (64 hop) → preprocess_window (repo)
  → ViT1D encoder → 384-dim embedding → record / classify

Usage:
  python demo/server.py                        # test mode (random embeddings + simulator)
  python demo/server.py --simulate             # test with simulated IMU, real encoder
  python demo/server.py --udp-port 8001        # real watch, real encoder
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Project root on sys.path so we can import imu_lm.* for preprocessing
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO

from imu_lm.data.augmentations.preprocess import preprocess_window as _repo_preprocess

logging.basicConfig(level=logging.INFO, format="[%(levelname)s %(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("demo")

# ═══════════════════════════════════════════════════════════════════════════
# Hard-coded config (matches base.yaml preprocessing exactly)
# ═══════════════════════════════════════════════════════════════════════════
CHANNELS = 6                   # acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
SAMPLE_RATE_HZ = 50.0
WINDOW_T = 128                 # 2.56 s @ 50 Hz
WINDOW_HOP = 64                # 50% overlap
EMBED_DIM = 384                # ViT1D output
G_MS2 = 9.80665               # only used if we want m/s²; z-score normalizes anyway

# This dict is passed to preprocess_window() — must match training config
PREPROCESS_CFG = {
    "preprocess": {
        "impute":    {"enabled": True,  "method": "linear", "max_missing_frac": 0.2},
        "filter":    {"enabled": True,  "type": "lowpass", "order": 4, "low_hz": 0.25, "high_hz": 24},
        "normalize": {"enabled": True,  "method": "zscore", "eps": 1e-6},
    }
}

# Default encoder paths (relative to repo root)
DEFAULT_TRACED = PROJECT_ROOT / "export_for_collaborator" / "encoder_traced.pt"
DEFAULT_CKPT   = PROJECT_ROOT / "export_for_collaborator" / "best.pt"


# ═══════════════════════════════════════════════════════════════════════════
# ViT1D Encoder (self-contained, from export_and_inference.py)
# ═══════════════════════════════════════════════════════════════════════════

class PatchEmbed1D(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.proj = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        patches = []
        for c in range(C):
            patch_c = self.proj(x[:, c : c + 1, :])
            patches.append(patch_c.transpose(1, 2))
        return torch.cat(patches, dim=1)


class PositionalEmbedding2D(nn.Module):
    def __init__(self, max_patches_per_channel: int, max_channels: int, embed_dim: int):
        super().__init__()
        self.time_embed = nn.Embedding(max_patches_per_channel, embed_dim)
        self.channel_embed = nn.Embedding(max_channels, embed_dim)
        self.max_patches_per_channel = max_patches_per_channel
        self.max_channels = max_channels

    def forward(self, num_patches_per_channel: int, num_channels: int, device: torch.device) -> torch.Tensor:
        time_pos = torch.arange(num_patches_per_channel, device=device).repeat(num_channels)
        channel_pos = torch.arange(num_channels, device=device).repeat_interleave(num_patches_per_channel)
        return self.time_embed(time_pos) + self.channel_embed(channel_pos)


class ViT1DEncoder(nn.Module):
    """ViT-1D encoder: [B, 6, 128] → [B, 384]."""
    def __init__(self, in_channels=6, patch_size=4, embed_dim=384, num_layers=12,
                 num_heads=6, mlp_ratio=4.0, dropout=0.0,
                 max_patches_per_channel=256, max_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed1D(in_channels, patch_size, embed_dim)
        self.pos_embed = PositionalEmbedding2D(max_patches_per_channel, max_channels, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        num_patches_per_channel = T // self.patch_size
        tokens = self.patch_embed(x)
        pos = self.pos_embed(num_patches_per_channel, C, x.device)
        tokens = tokens + pos.unsqueeze(0)
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        return tokens.mean(dim=1)


def load_encoder_from_checkpoint(ckpt_path: str, device: torch.device) -> ViT1DEncoder:
    """Load ViT1DEncoder from a TransfHAR best.pt checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    vit_cfg = cfg.get("vit1d", {})
    enc_cfg = vit_cfg.get("encoder", {})

    encoder = ViT1DEncoder(
        in_channels=len(cfg.get("data", {}).get("sensor_columns",
                        ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"])),
        patch_size=int(vit_cfg.get("patch_size", 4)),
        embed_dim=int(enc_cfg.get("hidden_size", 384)),
        num_layers=int(enc_cfg.get("num_hidden_layers", 12)),
        num_heads=int(enc_cfg.get("num_attention_heads", 6)),
        mlp_ratio=float(enc_cfg.get("intermediate_size", 1536)) / float(enc_cfg.get("hidden_size", 384)),
        dropout=float(enc_cfg.get("hidden_dropout_prob", 0.0)),
        max_patches_per_channel=int(vit_cfg.get("max_patches_per_channel", 256)),
    )
    encoder.load_state_dict(ckpt["model"], strict=True)
    encoder.to(device).eval()
    logger.info("Loaded ViT1D from checkpoint: %s  (embed=%d, layers=%d)",
                ckpt_path, encoder.embed_dim, 12)
    return encoder


def load_encoder(device: torch.device) -> nn.Module:
    """Load encoder: prefer checkpoint (device-safe) over traced model."""
    if DEFAULT_CKPT.exists():
        return load_encoder_from_checkpoint(str(DEFAULT_CKPT), device)
    if DEFAULT_TRACED.exists():
        # Traced model may have hardcoded CPU tensors; force CPU if needed
        model = torch.jit.load(str(DEFAULT_TRACED), map_location="cpu")
        model.eval()
        logger.info("Loaded traced encoder (CPU only): %s", DEFAULT_TRACED)
        return model
    logger.warning("No encoder found — using random embeddings (test mode)")
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Flask + SocketIO
# ═══════════════════════════════════════════════════════════════════════════
app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
app.config["SECRET_KEY"] = "transfhar-demo"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


# ═══════════════════════════════════════════════════════════════════════════
# Window buffer (thread-safe ring buffer producing overlapping windows)
# ═══════════════════════════════════════════════════════════════════════════

class WindowBuffer:
    def __init__(self):
        self.buf: collections.deque = collections.deque(maxlen=WINDOW_T * 4)
        self.lock = threading.Lock()
        self._new = 0              # samples since last window was consumed

    def add(self, sample: list):
        with self.lock:
            self.buf.append(sample)
            self._new += 1

    def try_window(self) -> Optional[np.ndarray]:
        """Return [T, C] numpy if a full window is ready, else None."""
        with self.lock:
            if len(self.buf) < WINDOW_T or self._new < WINDOW_HOP:
                return None
            w = list(self.buf)[-WINDOW_T:]
            self._new = 0
            return np.array(w, dtype=np.float32)

    def recent(self, n: int = 150) -> np.ndarray:
        with self.lock:
            items = list(self.buf)[-n:]
            if not items:
                return np.zeros((1, CHANNELS))
            return np.array(items, dtype=np.float32)

    def clear(self):
        with self.lock:
            self.buf.clear()
            self._new = 0


# ═══════════════════════════════════════════════════════════════════════════
# App state
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AppState:
    mode: str = "idle"                  # idle | record | train | infer
    current_label: str = ""
    encoder: Any = None
    probe: Optional[nn.Module] = None
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    labels: List[str] = field(default_factory=list)
    recorded: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    pred_label: str = "—"
    pred_conf: float = 0.0
    infer_ms: float = 0.0


STATE = AppState()
BUFFER = WindowBuffer()


# ═══════════════════════════════════════════════════════════════════════════
# Encode a window (preprocess via repo pipeline → ViT1D → embedding)
# ═══════════════════════════════════════════════════════════════════════════

def encode_window(window_np: np.ndarray) -> Optional[np.ndarray]:
    """[T, C] numpy → [D] embedding numpy.

    Preprocessing: impute → lowpass filter (24 Hz, order 4) → z-score normalize.
    Same pipeline as training (imu_lm.data.augmentations.preprocess.preprocess_window).
    """
    processed = _repo_preprocess(window_np.copy(), PREPROCESS_CFG)
    if processed is None:
        return None

    # processed is [C, T] float32
    x = torch.from_numpy(processed).unsqueeze(0).to(STATE.device)  # [1, C, T]

    if STATE.encoder is not None:
        t0 = time.perf_counter()
        with torch.no_grad():
            emb = STATE.encoder(x)  # [1, D]
        STATE.infer_ms = (time.perf_counter() - t0) * 1000
    else:
        emb = torch.randn(1, EMBED_DIM)
        STATE.infer_ms = 0.0

    return emb.cpu().numpy()[0]


# ═══════════════════════════════════════════════════════════════════════════
# UDP listener — receives Watch-Data-Streamer packets
# ═══════════════════════════════════════════════════════════════════════════

def parse_sample(fields: list) -> Optional[list]:
    """Parse one CoreMotion sample → [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z].

    Fields: timestamp userAcc_x/y/z gravity_x/y/z gyro_x/y/z quat_x/y/z/w
    Conversion: total_acc = gravity + userAcceleration (in g-units, z-score normalizes)
    """
    if len(fields) < 10:
        return None
    try:
        ua = [float(fields[1]), float(fields[2]), float(fields[3])]
        grav = [float(fields[4]), float(fields[5]), float(fields[6])]
        gyro = [float(fields[7]), float(fields[8]), float(fields[9])]
        acc = [grav[i] + ua[i] for i in range(3)]
        return acc + gyro  # [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    except (ValueError, IndexError):
        return None


def udp_listener(host: str, port: int):
    """Background thread: receive UDP packets from Watch-Data-Streamer."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    sock.settimeout(1.0)
    logger.info("UDP listening on %s:%d", host, port)

    while True:
        try:
            data, addr = sock.recvfrom(65535)
        except socket.timeout:
            continue
        except Exception as e:
            logger.error("UDP error: %s", e)
            continue

        text = data.decode("utf-8", errors="ignore").strip()

        # Control messages from Watch-Data-Streamer
        if text in {"client initialized", "stop"}:
            logger.info("Watch control: %s from %s", text, addr)
            continue

        # Watch-Data-Streamer batch format: "id;deviceId;motion:s1&s2&..."
        if ";motion:" in text:
            payload = text.split(";motion:", 1)[-1]
            for chunk in payload.split("&"):
                chunk = chunk.strip()
                if not chunk:
                    continue
                sample = parse_sample(chunk.split())
                if sample is not None:
                    BUFFER.add(sample)
            continue

        # Single-line format (one sample per packet)
        fields = text.split()
        if len(fields) >= 10:
            sample = parse_sample(fields)
            if sample is not None:
                BUFFER.add(sample)
            continue

        # Multi-line batch
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            sample = parse_sample(line.split())
            if sample is not None:
                BUFFER.add(sample)


# ═══════════════════════════════════════════════════════════════════════════
# Inference loop — continuously processes windows from the buffer
# ═══════════════════════════════════════════════════════════════════════════

def inference_loop():
    """Background thread: pull windows → preprocess → encode → record/classify → push dashboard."""
    logger.info("Inference loop running")

    while True:
        window_np = BUFFER.try_window()
        if window_np is None:
            time.sleep(0.02)
            continue

        # Raw signal for chart
        recent = BUFFER.recent(150)
        sig = {
            "acc_x": recent[:, 0].tolist()[-100:],
            "acc_y": recent[:, 1].tolist()[-100:],
            "acc_z": recent[:, 2].tolist()[-100:],
            "gyro_x": recent[:, 3].tolist()[-100:],
            "gyro_y": recent[:, 4].tolist()[-100:],
            "gyro_z": recent[:, 5].tolist()[-100:],
        }

        # Encode
        emb = encode_window(window_np)

        if emb is not None:
            if STATE.mode == "record" and STATE.current_label:
                STATE.recorded.setdefault(STATE.current_label, []).append(emb)

            elif STATE.mode == "infer" and STATE.probe is not None:
                with torch.no_grad():
                    logits = STATE.probe(torch.from_numpy(emb).unsqueeze(0).to(STATE.device))
                    probs = torch.softmax(logits, dim=-1)[0]
                    conf, idx = probs.max(0)
                    STATE.pred_label = STATE.labels[idx.item()]
                    STATE.pred_conf = conf.item() * 100

        # Build all-class predictions list (sorted by confidence)
        all_preds = []
        if STATE.mode == "infer" and STATE.probe is not None and emb is not None:
            with torch.no_grad():
                logits = STATE.probe(torch.from_numpy(emb).unsqueeze(0).to(STATE.device))
                probs = torch.softmax(logits, dim=-1)[0]
                for i, lbl in enumerate(STATE.labels):
                    all_preds.append({"label": lbl, "confidence": round(probs[i].item(), 4)})
                all_preds.sort(key=lambda x: x["confidence"], reverse=True)

        socketio.emit("update", {
            "mode": STATE.mode,
            "signals": sig,
            "predictions": all_preds,
            "top_label": STATE.pred_label if STATE.mode == "infer" else "",
            "top_confidence": round(STATE.pred_conf, 1) if STATE.mode == "infer" else 0.0,
            "inference_ms": round(STATE.infer_ms, 1),
            "current_label": STATE.current_label,
            "recorded_counts": {k: len(v) for k, v in STATE.recorded.items()},
        })

        time.sleep(0.01)


# ═══════════════════════════════════════════════════════════════════════════
# Probe training (LinearHead — same as repo's imu_lm.probe.head)
# ═══════════════════════════════════════════════════════════════════════════

class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def train_probe() -> dict:
    """Fit linear probe on recorded embeddings. Returns result dict."""
    if len(STATE.recorded) < 2:
        return {"ok": False, "error": "Need at least 2 activity classes to train"}

    X_all, y_all = [], []
    labels = sorted(STATE.recorded.keys())
    for idx, lbl in enumerate(labels):
        embs = STATE.recorded[lbl]
        # Trim first/last window (transition edges, ~1.28s each)
        if len(embs) > 2:
            embs = embs[1:-1]
        for emb in embs:
            X_all.append(emb)
            y_all.append(idx)

    if len(X_all) < 4:
        return {"ok": False, "error": "Need more data (at least 4 windows total)"}

    X = torch.tensor(np.array(X_all), dtype=torch.float32).to(STATE.device)
    y = torch.tensor(y_all, dtype=torch.long).to(STATE.device)

    probe = LinearHead(EMBED_DIM, len(labels)).to(STATE.device)
    opt = torch.optim.AdamW(probe.parameters(), lr=5e-4, weight_decay=1e-4)

    probe.train()
    for _ in range(300):
        logits = probe(X)
        loss = nn.functional.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        acc = (probe(X).argmax(1) == y).float().mean().item()

    STATE.probe = probe
    STATE.labels = labels

    msg = f"Trained: {len(labels)} classes, {len(X)} windows, acc {acc*100:.1f}%"
    logger.info(msg)
    return {"ok": True, "message": msg, "classes": len(labels), "accuracy": round(acc * 100, 1)}


# ═══════════════════════════════════════════════════════════════════════════
# HTTP API
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/start_record", methods=["POST"])
def api_start_record():
    data = request.get_json(force=True) or {}
    label = data.get("label", "").strip()
    if not label:
        return jsonify({"ok": False, "error": "Label required"}), 400
    STATE.mode = "record"
    STATE.current_label = label
    BUFFER.clear()
    logger.info("Recording: %s", label)
    socketio.emit("status", {"msg": f"Recording '{label}'…", "ok": True})
    return jsonify({"ok": True, "label": label})


@app.route("/api/stop_record", methods=["POST"])
def api_stop_record():
    lbl = STATE.current_label
    count = len(STATE.recorded.get(lbl, []))
    STATE.mode = "idle"
    STATE.current_label = ""
    logger.info("Stopped recording: %s (%d windows)", lbl, count)
    socketio.emit("status", {"msg": f"Recorded '{lbl}': {count} windows", "ok": True})
    return jsonify({"ok": True, "label": lbl, "windows": count})


@app.route("/api/train", methods=["POST"])
def api_train():
    STATE.mode = "train"
    socketio.emit("status", {"msg": "Training probe…", "ok": True})
    result = train_probe()
    STATE.mode = "idle"
    socketio.emit("status", {"msg": result.get("message", result.get("error", "")), "ok": result["ok"]})
    return jsonify(result)


@app.route("/api/start_infer", methods=["POST"])
def api_start_infer():
    if STATE.probe is None:
        return jsonify({"ok": False, "error": "Train a probe first"}), 400
    STATE.mode = "infer"
    STATE.pred_label = "—"
    STATE.pred_conf = 0.0
    BUFFER.clear()
    logger.info("Inference started (%d classes)", len(STATE.labels))
    socketio.emit("status", {"msg": "Live inference started", "ok": True})
    return jsonify({"ok": True, "classes": len(STATE.labels)})


@app.route("/api/stop_infer", methods=["POST"])
def api_stop_infer():
    STATE.mode = "idle"
    STATE.pred_label = "—"
    STATE.pred_conf = 0.0
    socketio.emit("status", {"msg": "Inference stopped", "ok": True})
    return jsonify({"ok": True})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    STATE.recorded.clear()
    STATE.probe = None
    STATE.labels = []
    STATE.mode = "idle"
    STATE.current_label = ""
    STATE.pred_label = "—"
    STATE.pred_conf = 0.0
    BUFFER.clear()
    logger.info("Reset all data")
    socketio.emit("status", {"msg": "Reset — all data cleared", "ok": True})
    return jsonify({"ok": True})


@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify({
        "mode": STATE.mode,
        "labels": STATE.labels,
        "recorded_counts": {k: len(v) for k, v in STATE.recorded.items()},
        "has_probe": STATE.probe is not None,
        "encoder_loaded": STATE.encoder is not None,
        "current_label": STATE.current_label,
    })


# ═══════════════════════════════════════════════════════════════════════════
# SocketIO handlers (for web dashboard buttons)
# ═══════════════════════════════════════════════════════════════════════════

@socketio.on("start_record")
def ws_start_record(data):
    label = (data or {}).get("label", "").strip()
    if not label:
        socketio.emit("status", {"msg": "Enter an activity name first", "ok": False})
        return
    STATE.mode = "record"
    STATE.current_label = label
    BUFFER.clear()
    logger.info("Recording: %s", label)
    socketio.emit("status", {"msg": f"Recording '{label}'…", "ok": True})


@socketio.on("stop_record")
def ws_stop_record(_data=None):
    lbl = STATE.current_label
    count = len(STATE.recorded.get(lbl, []))
    STATE.mode = "idle"
    STATE.current_label = ""
    socketio.emit("status", {"msg": f"Recorded '{lbl}': {count} windows", "ok": True})


@socketio.on("train")
def ws_train(_data=None):
    STATE.mode = "train"
    socketio.emit("status", {"msg": "Training probe…", "ok": True})
    result = train_probe()
    STATE.mode = "idle"
    socketio.emit("status", {"msg": result.get("message", result.get("error", "")), "ok": result["ok"]})


@socketio.on("start_infer")
def ws_start_infer(_data=None):
    if STATE.probe is None:
        socketio.emit("status", {"msg": "Train a probe first", "ok": False})
        return
    STATE.mode = "infer"
    STATE.pred_label = "—"
    STATE.pred_conf = 0.0
    BUFFER.clear()
    socketio.emit("status", {"msg": "Live inference started", "ok": True})


@socketio.on("stop_infer")
def ws_stop_infer(_data=None):
    STATE.mode = "idle"
    STATE.pred_label = "—"
    STATE.pred_conf = 0.0
    socketio.emit("status", {"msg": "Inference stopped", "ok": True})


@socketio.on("reset")
def ws_reset(_data=None):
    api_reset()


# ═══════════════════════════════════════════════════════════════════════════
# Simulator (for testing without a real watch)
# ═══════════════════════════════════════════════════════════════════════════

def simulator():
    """Generate synthetic 6-axis IMU at 50 Hz → feed into buffer."""
    logger.info("Simulator active: synthetic IMU @ 50 Hz")
    dt = 1.0 / SAMPLE_RATE_HZ
    t = 0.0

    def _walk(t):
        ua = [0.02 * math.sin(2*math.pi*2*t), 0.05 * math.sin(2*math.pi*2*t+0.5),
              0.03 * math.sin(2*math.pi*4*t)]
        grav = [0.01, 0.01, -1.0]
        acc = [grav[i] + ua[i] + np.random.normal(0, 0.005) for i in range(3)]
        gyro = [np.random.normal(0, 0.1) for _ in range(3)]
        return acc + gyro

    def _type(t):
        ua = [np.random.normal(0, 0.03), np.random.normal(0, 0.02), np.random.normal(0, 0.015)]
        grav = [0.01, 0.01, -1.0]
        acc = [grav[i] + ua[i] for i in range(3)]
        gyro = [np.random.normal(0, 0.5), np.random.normal(0, 0.3), np.random.normal(0, 0.2)]
        return acc + gyro

    def _stir(t):
        ua = [0.06*math.sin(2*math.pi*1.5*t), 0.06*math.cos(2*math.pi*1.5*t),
              np.random.normal(0, 0.002)]
        grav = [0.01, 0.01, -1.0]
        acc = [grav[i] + ua[i] + np.random.normal(0, 0.01) for i in range(3)]
        gyro = [0.3*math.cos(2*math.pi*1.5*t), 0.3*math.sin(2*math.pi*1.5*t),
                math.sin(2*math.pi*1.5*t)]
        return acc + gyro

    def _idle(t):
        grav = [0.01, 0.01, -1.0]
        acc = [g + np.random.normal(0, 0.001) for g in grav]
        gyro = [np.random.normal(0, 0.01) for _ in range(3)]
        return acc + gyro

    generators = [_walk, _type, _stir, _idle]

    while True:
        for gen in generators:
            for _ in range(int(15.0 * SAMPLE_RATE_HZ)):  # 15s per pattern
                BUFFER.add(gen(t))
                t += dt
                time.sleep(dt)


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="TransfHAR real-time demo server")
    p.add_argument("--simulate", action="store_true", help="Generate synthetic IMU (no watch)")
    p.add_argument("--no-encoder", action="store_true", help="Skip encoder loading (random embeddings)")
    p.add_argument("--udp-host", default="0.0.0.0")
    p.add_argument("--udp-port", type=int, default=8001)
    p.add_argument("--web-port", type=int, default=5000)
    p.add_argument("--device", default="auto", help="cpu | cuda | mps | auto")
    return p.parse_args()


def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(args.device)
    STATE.device = dev
    logger.info("Device: %s", dev)

    # Load encoder
    if not args.no_encoder:
        STATE.encoder = load_encoder(dev)
    else:
        logger.warning("Encoder disabled — random embeddings")

    # Background threads
    if args.simulate:
        threading.Thread(target=simulator, daemon=True).start()
    else:
        threading.Thread(target=udp_listener, args=(args.udp_host, args.udp_port), daemon=True).start()

    threading.Thread(target=inference_loop, daemon=True).start()

    logger.info("═" * 50)
    logger.info("Dashboard  → http://localhost:%d", args.web_port)
    logger.info("UDP port   → %d", args.udp_port)
    logger.info("Watch API  → http://<laptop-ip>:%d/api/...", args.web_port)
    logger.info("═" * 50)
    socketio.run(app, host="0.0.0.0", port=args.web_port, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
