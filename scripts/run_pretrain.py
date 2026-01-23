"""
run_pretrain.py
----------------
Stage A: merge config, set up run_dir/logging, route to models/<name>/run.py.
"""

# Pseudocode:
# - parse CLI args: --config base.yaml, --model-config cnn.yaml, overrides
# - cfg = imu_lm.utils.config.load_and_merge(...)
# - run_dir = imu_lm.utils.config.init_run_dir(cfg)
# - log resolved cfg to run_dir/logs/stdout.log header
# - select model package by cfg.model (e.g., imu_lm.models.CNN.run)
# - call model_run.main(cfg, run_dir)
# - exit with status
