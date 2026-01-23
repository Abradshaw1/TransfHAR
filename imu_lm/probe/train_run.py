"""
trainer.py
-----------
Trainer for frozen encoder + linear head.

Pseudocode:
- class ProbeTrainer:
    - __init__(cfg): setup device, optimizer, metric selection (e.g., macro_f1)
    - fit(encoder, head, loaders):
        - encoder.eval(); head.train()
        - loop epochs:
            - for batch in train_loader:
                - forward encoder (no grad), forward head, compute loss
                - backward/update head only
                - log metrics to probe/logs/metrics.txt (key=value, epoch/step prefix)
            - eval on val_loader; track best by cfg.probe.metric
            - save checkpoints (latest.pt, best.pt) under runs/<run>/probe/checkpoints/
        - return best checkpoint path/metrics
"""
