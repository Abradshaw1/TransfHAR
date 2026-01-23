"""
eval.py
-------
Evaluation for frozen encoder + trained head.

Pseudocode:
- def evaluate(encoder, head, loaders, cfg):
    - encoder.eval(); head.eval()
    - iterate over eval loader(s):
        - forward encoder (no grad), forward head -> logits
        - collect preds/labels
    - metrics = imu_lm.utils.metrics.compute_metrics(...)
    - write metrics.txt + summary.txt under runs/<run>/probe/
    - return metrics
"""
