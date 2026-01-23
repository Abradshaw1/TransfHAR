"""
run_probe_eval.py
-----------------
Stage B: evaluate best frozen head on frozen encoder; write metrics/summary.
"""

# Pseudocode:
# - parse CLI args: --config base.yaml, --probe-config probe.yaml, overrides
# - cfg = imu_lm.utils.config.load_and_merge(...)
# - encoder = imu_lm.runtime_consistency.artifacts.load_encoder(cfg.paths.artifact_path, device)
# - head = imu_lm.probe.io.load_probe_head(cfg.paths.probe_checkpoint, device)
# - dataloaders = imu_lm.data.loaders.make_probe_eval_loaders(cfg)
# - metrics = imu_lm.probe.eval.evaluate(encoder, head, dataloaders, cfg)
# - write metrics.txt and summary.txt into runs/<run>/probe/
# - exit
