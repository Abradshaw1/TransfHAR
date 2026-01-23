"""
data_integrity_checker.py
-------------------------
Purpose: quick audit of parquet/schema/timestamps/NaNs/run-purity before training.

Pseudocode:
- define expected schema (see continuous_stream v3 contract)
- walk datadrive/ for parquet files
- for each file:
    - load schema with pyarrow
    - verify columns/dtypes/nullability match expectation
    - sample rows to check monotonic timestamp per session
    - check approx_rate_hz_tolerance
    - flag required_not_null violations
- summarize findings; nonzero violations -> exit nonzero
"""
