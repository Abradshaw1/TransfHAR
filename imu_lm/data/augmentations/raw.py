"""
raw.py
------
Raw windows → [B, C, T] float32 with normalization.

Pseudocode:
- expects dict with keys: {"acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"}
- stack channels → tensor [C, T] (C=3 or 6 depending on gyro availability)
- apply normalization using train-split stats (mean/std) provided in cfg/meta
- return tensor, metadata (e.g., mask for missing gyro)
"""
