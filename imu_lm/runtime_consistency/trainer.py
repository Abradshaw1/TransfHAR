"""
trainer.py
----------
ONE shared training loop for all backbones to ensure identical optimization semantics.

Pseudocode:
- class Trainer:
    - __init__(cfg): set seed, resolve device, prepare logger/metrics writer, load optim/sched
    - fit(model, objective_fn, dataloaders):
        - loop over steps/epochs:
            - for batch in train_loader:
                - forward_loss = objective_fn(batch, model, cfg)
                - loss.backward()
                - optim.step(); sched.step(); zero_grad()
                - log key=value to metrics.txt (step=..., split=train, loss=..., lr=...)
                - checkpoint periodically to runs/<run>/checkpoints/latest.pt
            - eval on val_loader per eval cadence; early stop by metric if configured
        - return best checkpoint path / state
    - _log(line): write to stdout and logs/metrics.txt
- Uses AMP if cfg.trainer.amp
- No model-specific hacks here; purely generic loop
"""
