"""
supervised.py
--------------
Supervised objective for baselines.

Contract:
def forward_loss(batch, encoder, cfg):
    Returns: loss (torch.Tensor), logs (Dict[str, float])

Pseudocode:
- unpack batch: inputs, labels
- embed = encoder(inputs)
- logits = head_on_top(???) -> but for encoder-only pretrain, may attach simple classifier
- compute cross-entropy loss
- log accuracy metrics
- return loss, logs
"""
