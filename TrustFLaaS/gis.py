"""
Gradient Interaction Score (GIS), HealSplit Eq. 7.

Design decision -- what "gradient of a label" means:
Eq. 7 writes G_p[i,j] = cos(grad(y_tx^(i)), grad(y_ty^(j))) -- cosine
similarity between "the gradient of label i in task tx" and "the gradient
of label j in task ty." A gradient of a label by itself isn't a standard
object; the paper cites multi-task gradient-alignment literature (Yu et al.
2020, "gradient surgery for multi-task learning") for this idea, where the
standard construction is: the gradient of a task's loss, restricted to
samples carrying a specific label, taken with respect to parameters SHARED
across tasks. We implement it that way: for label value i in task tx,
restrict to samples with y_tx==i, take the mean loss over just those
samples, and differentiate with respect to `MultiHeadModel.shared_trunk`'s
parameters (flattened into one vector). Two labels' gradients being
cosine-similar means: pushing the shared trunk to reduce one task's loss
for that label also tends to reduce the other task's loss for its label --
"cooperative," per the paper's own description of what large G_p values mean.
"""
from typing import List

import torch
import torch.nn as nn

from .multihead_model import MultiHeadModel


def _per_label_gradients(model: MultiHeadModel, Z: torch.Tensor, labels: torch.Tensor,
                          head: nn.Module, num_label_values: int,
                          criterion: nn.Module) -> List[torch.Tensor]:
    """One flattened shared-trunk gradient vector per label value 0..num_label_values-1.
    Label values with no samples in this batch get a zero vector (cosine
    similarity against a zero vector is defined as 0 below, not NaN)."""
    trunk_params = list(model.shared_trunk.parameters())
    numel = sum(p.numel() for p in trunk_params)
    grads = []
    for v in range(num_label_values):
        mask = labels == v
        if mask.sum() == 0:
            grads.append(torch.zeros(numel, device=Z.device))
            continue
        feats = model.shared_features(Z[mask])
        logits = head(feats)
        loss = criterion(logits, labels[mask])
        g = torch.autograd.grad(loss, trunk_params, retain_graph=False, allow_unused=True)
        flat = torch.cat([
            (gi if gi is not None else torch.zeros_like(pi)).flatten()
            for gi, pi in zip(g, trunk_params)
        ])
        grads.append(flat.detach())
    return grads


def compute_gis(model: MultiHeadModel, Z: torch.Tensor,
                 labels_tx: torch.Tensor, head_tx: nn.Module, num_tx: int,
                 labels_ty: torch.Tensor, head_ty: nn.Module, num_ty: int) -> torch.Tensor:
    """Returns G_p, shape [num_tx, num_ty]."""
    criterion = nn.CrossEntropyLoss()
    grads_tx = _per_label_gradients(model, Z, labels_tx, head_tx, num_tx, criterion)
    grads_ty = _per_label_gradients(model, Z, labels_ty, head_ty, num_ty, criterion)

    G = torch.zeros(num_tx, num_ty)
    for i in range(num_tx):
        for j in range(num_ty):
            gi, gj = grads_tx[i], grads_ty[j]
            denom = gi.norm() * gj.norm()
            G[i, j] = torch.dot(gi, gj) / denom if denom > 1e-12 else 0.0
    return G
