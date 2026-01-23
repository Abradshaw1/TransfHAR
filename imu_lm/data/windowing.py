"""
windowing.py
------------
Gap-aware window extraction and fixed-length slicing at uniform sample_rate_hz.

Pseudocode:
- enforce sample_rate_hz (resample if needed)
- for each (dataset, subject_id, session_id):
    - group by session, sort by timestamp_ns
    - apply gap policy (break on gaps > threshold)
    - slide fixed windows (window_secs, stride_secs)
    - attach labels (global + dataset) with mixed-label policy for overlaps
- return iterable of windows with metadata for downstream datasets/dataloaders
"""
