"""
config.py
----------
YAML load/merge, validation, run naming, seed, and path helpers.

Pseudocode:
- load base.yaml + model/probe configs; apply CLI overrides
- validate required fields (paths, model, objective, encoding)
- set seeds for torch/random/numpy
- resolve run_name and create runs/<run_name> with subdirs (logs, checkpoints, artifacts, probe)
- emit resolved config for logging
"""
