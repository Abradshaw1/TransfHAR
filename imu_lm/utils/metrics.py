"""
metrics.py
-----------
Metrics for probes/eval: balanced accuracy, macro F1, per-class F1, confusion matrix.

Pseudocode:
- def compute_metrics(y_true, y_pred, labels):
    - use sklearn metrics (balanced_accuracy_score, f1_score)
    - compute per-class F1 and confusion matrix
    - return dict of metrics
- def format_metrics_txt(metrics, prefix=""):
    - serialize as key=value tokens for metrics.txt lines
"""
