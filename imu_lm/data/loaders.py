"""
loaders.py
----------
Parquet → datasets/dataloaders. Chooses encoding (raw/spectrogram) and augmentations.

Schema contract (continuous_stream v3):
- primary_index: dataset, subject_id, session_id, timestamp_ns
- columns: acc_x/y/z (float32, non-null), gyro_x/y/z (float32, nullable),
  global_activity_id/label (non-null), dataset_activity_id/label (non-null)
- rate_hz: 50; axis_frame: FLU; acc units m/s^2; gyro rad/s.
- expectations: monotonic_timestamp_per_session, approx_rate_hz_tolerance=0.5,
  required_not_null on key columns.

Pseudocode:
- load parquet via pyarrow dataset or pandas with schema validation
- filter rows per split descriptor (dataset/subject/session)
- apply windowing (imu_lm.data.windowing)
- choose encoding: raw (imu_lm.data.augmentations.raw) or spectrogram (...spectrogram)
- apply transform augmentations (imu_lm.data.augmentations.transform) when in pretrain
- build torch Dataset that yields (window_tensor, labels/meta)
- build DataLoader with cfg.loader.batch_size/num_workers/pin_memory
"""
