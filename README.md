# IMU-LM SSL Benchmark (TransfHAR)

Self-supervised pretraining and frozen linear probing for IMU-based activity recognition. Everything is built to measure **representation quality under dataset shift**—encoders are pretrained once (Stage A) and **frozen** during evaluation (Stage B).

## What this repo is for
- Pretrain IMU encoders (ViT2D on spectrograms; ViT1D/TS-Transformer1D/CNN1D on raw) with SSL or supervised objectives while holding out the probe dataset.
- Freeze the encoder and train a linear classifier on the probe dataset (e.g., SAMoSA or other target splits).
- Report standardized metrics (balanced accuracy, macro-F1, per-class F1, confusion) and keep everything reproducible in `runs/`.

## Repository layout
- `configs/` — `base.yaml` (paths/logging/windowing/loader/optim), `probe.yaml`, backbone configs (`cnn1D.yaml`, `tstransformer1d.yaml`, `vit1d.yaml`, `vit2d.yaml`).
- `scripts/` — orchestration entrypoints:
  - `print_config.py` (placeholder/WIP)
  - `run_pretrain.py` (Stage A)
  - `run_probe_train.py` (Stage B training)
  - `run_probe_eval.py` (Stage B eval)
- `imu_lm/` — reusable ML logic:
  - `data/` (windowing, splits, loaders, augmentations for raw/spectrogram; schema matches continuous_stream v3)
  - `objectives/` (`mae`, `consistency`, `supervised` stubs)
  - `runtime_consistency/` (shared trainer + artifacts I/O)
  - `probe/` (head/trainer/eval/io for frozen encoders)
  - `utils/` (config merge/paths/seeding, metrics)
  - `models/` (CNN, TSTransformer, ViTd, FastViT wiring to shared trainer)

## Quick Start

### 1) Environment
```bash
python -m venv TransfHAR_env
source TransfHAR_env/bin/activate
pip install -r requirements.txt
```
> If using GPU, install the matching CUDA-enabled PyTorch wheels before the rest.

### 2) Data
1. Download `IMULM_master_dataset.parquet` from the shared Google Drive folder:
   **https://drive.google.com/drive/u/0/folders/1Tfsgh4eUo_ZMPHbfZG2i6Yo2VbddWezf**
2. Place it in the data drive directory:
   ```
   imu_lm/data/data_drive/IMULM_master_dataset.parquet
   ```
3. Set `paths.dataset_path` in `configs/base.yaml` to point to that file (and update `paths.runs_root` if needed). Specify the probe dataset in `configs/probe.yaml`.

Schema expectation: continuous_stream v3 (50 Hz, FLU axes, acc m/s², gyro rad/s, required keys for dataset/subject/session/timestamp/labels).

### 3) Configure an experiment
Pick backbone/objective/encoding via backbone config (`cnn1D.yaml`, `tstransformer1d.yaml`, `vit1d.yaml`, or `vit2d.yaml`) plus `base.yaml`. Use `configs/probe.yaml` for probe settings (shots, metric, split policy).

### 4) Pretrain (Stage A)
```bash
python scripts/run_pretrain.py --config configs/base.yaml --model-config configs/vit2d.yaml --run-name <run_name>
```
Writes logs/checkpoints/artifacts under `runs/<run_name>/`.

### 5) Linear probe train (Stage B)
```bash
python scripts/run_probe_train.py --config configs/base.yaml --probe-config configs/probe.yaml --run <encoder_run_name>
```
Loads the frozen encoder from `runs/<encoder_run_name>/`, trains a linear head on the probe dataset (SAMoSA by default), and writes probe outputs to `runs/<encoder_run_name>/probe/`.

> **Note:** The probe lives *inside* the encoder's run directory. Use the **encoder run name** (e.g. `test_new_mae`) for both probe train and eval. There is no separate probe run name.

### 6) Probe eval
```bash
python scripts/run_probe_eval.py --config configs/base.yaml --probe-config configs/probe.yaml --run <encoder_run_name>
```
Loads the best probe head from `runs/<encoder_run_name>/probe/checkpoints/best.pt` and evaluates on the test split. Writes `metrics.txt` and `summary.txt` in `runs/<encoder_run_name>/probe/`.

## Runs directory contract
```
runs/<encoder_run_name>/
├── logs/
│   ├── stdout.log              # tee of stdout/stderr with resolved config header
│   └── metrics.txt             # key=value lines; every line starts with step= or epoch=
├── checkpoints/
│   ├── latest.pt               # encoder + decoder checkpoint
│   └── best.pt                 # best by val_loss (if validation enabled)
├── artifacts/
│   └── encoder_meta.json       # encoder architecture metadata
└── probe/                      # created by run_probe_train.py
    ├── checkpoints/
    │   ├── best.pt             # head weights at best val macro_f1
    │   └── latest.pt           # head weights after last epoch
    ├── logs/
    │   └── metrics.txt         # per-epoch train/val/test metrics (includes per-class)
    └── summary.txt             # JSON: best_epoch, test results, num_classes
```

The probe checkpoint (`best.pt`) contains the head state_dict, optimizer state, and full `label_map` (class mapping, label names, embedding dim). This is self-contained — `run_probe_eval.py` reconstructs the head architecture from it.
