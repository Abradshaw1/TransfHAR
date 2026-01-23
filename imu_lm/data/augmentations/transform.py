"""
transform.py
-------------
Representation-agnostic augmentations (jitter/mask/crop/drop/etc.).

Pseudocode:
- define augmentation pipeline configurable via cfg.augment.*
- ops may include:
    - time crop / resize
    - jitter (Gaussian noise)
    - time masking / channel dropout
    - permutation / time-warp (if allowed)
- ensure augmentations are label-agnostic and dataset-agnostic
- expose `apply_transforms(window, cfg, is_train=True)` returning augmented window
"""
