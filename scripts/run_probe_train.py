"""
run_probe_train.py
------------------
Stage B: load encoder artifact, train frozen linear head.
"""

# Pseudocode:
# - parse CLI args: --config base.yaml, --probe-config probe.yaml, overrides
# - cfg = imu_lm.utils.config.load_and_merge(...)
# - encoder = imu_lm.runtime_consistency.artifacts.load_encoder(cfg.paths.artifact_path)
# - dataloaders = imu_lm.data.loaders.make_probe_loaders(cfg)
# - head = imu_lm.probe.head.LinearProbeHead(...)
# - trainer = imu_lm.probe.trainer.ProbeTrainer(cfg)
# - best_ckpt = trainer.fit(encoder, head, dataloaders)
# - save metrics/logs/checkpoints into runs/<run>/probe/
