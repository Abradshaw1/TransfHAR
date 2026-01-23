# IMU-LM SSL Benchmark (TransfHAR)

Self-supervised pretraining and frozen linear probing for IMU-based activity recognition. The goal is to measure **representation quality under dataset shift**—encoders are pretrained once (Stage A) and **frozen** during evaluation (Stage B).

## Goals
- Pretrain IMU encoders (FastViT, ViT, TS-Transformer, CNN) with SSL or supervised objectives while holding out the probe dataset.
- Freeze the encoder and train a linear classifier on the probe dataset (e.g., SAMoSA or any target split).
- Report standardized metrics (balanced accuracy, macro-F1, per-class F1, confusion) and keep everything reproducible in `runs/`.

## Repository layout
- `configs/` — `base.yaml` (paths/logging/windowing/loader/optim), `probe.yaml`, and backbone configs (`cnn.yaml`, `tstransformer.yaml`, `vit.yaml`, `fastvit.yaml`).
- `scripts/` — orchestration entrypoints:
  - `print_config.py` (inspect merged config)
  - `run_pretrain.py` (Stage A)
  - `run_probe_train.py` (Stage B training)
  - `run_probe_eval.py` (Stage B eval)
- `imu_lm/` — reusable ML logic:
  - `data/` (windowing, splits, loaders, augmentations for raw/spectrogram)
  - `objectives/` (`mae`, `consistency`, `supervised` stubs)
  - `runtime_consistency/` (shared trainer + artifacts I/O)
  - `probe/` (head/trainer/eval/io for frozen encoders)
  - `utils/` (config merge/paths/seeding, metrics)
  - `models/` (CNN, TSTransformer, ViTd, FastViT wiring to shared trainer)
- `runs/` — all outputs (logs, checkpoints, artifacts, probe results). Never commit contents.
- `smoke/` — quick integrity and loop checks.

## Quick Start

### 1) Environment
```bash
python -m venv TransfHAR_env
source TransfHAR_env/bin/activate
pip install -r requirements.txt
```
> If using GPU, install the matching CUDA-enabled PyTorch wheels before the rest.

### 2) Data
Place the unified IMU parquet outside version control, e.g.:
```
/data/imu_ssl/unified_dataset.parquet
```
Set paths in `configs/base.yaml` (e.g., `paths.data_root`, `paths.runs_root`) and specify probe dataset in `configs/probe.yaml`.

### 3) Configure an experiment
Pick backbone/objective encoding via backbone config (`cnn.yaml`, `tstransformer.yaml`, `vit.yaml`, `fastvit.yaml`) plus `base.yaml`. Use `configs/probe.yaml` for probe settings (shots, metric, split policy). Run `scripts/print_config.py` to inspect the merged config.

### 4) Pretrain (Stage A)
```bash
python scripts/run_pretrain.py --config configs/base.yaml --model-config configs/vit.yaml
```
Writes logs/checkpoints/artifacts under `runs/<run_name>/`.

### 5) Linear probe train (Stage B)
```bash
python scripts/run_probe_train.py --config configs/base.yaml --probe-config configs/probe.yaml --run <run_name>
```
Loads the frozen encoder from `runs/<run_name>/artifacts/encoder.pt`, trains a linear head on the probe dataset, and writes to `runs/<run_name>/probe/`.

### 6) Probe eval
```bash
python scripts/run_probe_eval.py --config configs/base.yaml --probe-config configs/probe.yaml --run <run_name>
```
Evaluates the best head on the frozen encoder, writes `metrics.txt` and `summary.txt` in `runs/<run_name>/probe/`.

## Runs directory contract
- `logs/stdout.log` — tee of stdout/stderr with resolved config header.
- `logs/metrics.txt` — key=value lines; every line starts with `step=` or `epoch=`.
- `checkpoints/` — `latest.pt`, optional periodic `step_*.pt`.
- `artifacts/encoder.pt`, `artifacts/encoder_meta.json` — frozen encoder + metadata.
- `probe/` — probe logs, checkpoints (`latest.pt`, `best.pt`), `metrics.txt`, `summary.txt`.

## Mantra
“IMU-LM is an evaluation system for representations, not a training framework for task accuracy.”
# TransfHAR