"""
consistency.py
--------------
Consistency-style SSL objective (e.g., student-teacher / perturbation invariance).

Contract:
def forward_loss(batch, encoder, cfg):
    Returns: loss (torch.Tensor), logs (Dict[str, float])

Pseudocode:
- generate two augmented views via imu_lm.data.augmentations.transform
- encode both views
- compute consistency loss (e.g., mse/kl) between embeddings
- log view similarity stats
- return loss, logs
"""
