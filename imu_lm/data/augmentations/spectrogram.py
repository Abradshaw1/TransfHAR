"""
spectrogram.py
--------------
STFT-based encoding → [B, C, F, TT] with normalization.

Pseudocode:
- input: raw window tensor [C, T] (acc +/- gyro)
- compute spectrogram (e.g., torch.stft) with cfg params (n_fft, hop, win_length)
- normalize magnitude (train-split stats) channel-wise
- optionally drop/keep phase; return magnitude-only tensor
- output: spectrogram tensor, metadata (freq bins, time bins)
"""
