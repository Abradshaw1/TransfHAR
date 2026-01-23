"""
splits.py
---------
Defines pretrain vs probe selection and probe train/val/test splits.

Pseudocode:
- load metadata (dataset, subject_id, session_id) from parquet manifest
- implement split policies:
    - within-dataset (subject-disjoint where possible)
    - cross-dataset (train on source, eval on target)
    - LODO (hold out one dataset for eval)
- produce split descriptors with subject/session lists and label distributions
- expose helpers to get split filters for loaders/windowing
"""
