"""
single_loop_pretrain.py
-----------------------
Minimal end-to-end smoke test: loader → model → objective → backward for N steps.

Pseudocode:
- cfg = small inline/default config (tiny batch/steps)
- build dummy dataloader from synthetic tensors respecting input spec
- choose a minimal model (e.g., small CNN) from imu_lm.models.CNN.model
- choose objective_fn = imu_lm.objectives.mae.forward_loss (or stub)
- trainer = lightweight loop: forward_loss -> backward -> step for few steps
- assert loss is finite and decreases slightly; exit 0 on success
"""
