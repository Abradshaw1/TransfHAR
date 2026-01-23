"""
run.py (CNN)
------------
Build/wire encoder, dataloaders, objective, optim/sched, then delegate to shared trainer.

Pseudocode:
- def main(cfg, run_dir):
    - set seed/device via imu_lm.utils.config helpers
    - dataloaders = imu_lm.data.loaders.make_pretrain_loaders(cfg)
    - encoder = CNNEncoder(cfg)
    - objective_fn = imu_lm.objectives.<cfg.objective>.forward_loss
    - optim, sched = build from cfg.optim
    - trainer = imu_lm.runtime_consistency.trainer.Trainer(cfg, run_dir, optim, sched)
    - trainer.fit(encoder, objective_fn, dataloaders)
    - save encoder artifact via imu_lm.runtime_consistency.artifacts.save_encoder(...)
"""
