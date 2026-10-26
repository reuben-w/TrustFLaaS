"""
Client-level shift detection for TrustFLaaS (replaces TAS as the detector).

Why this exists: an offline comparison on real smashed data (MNIST, Fashion-MNIST, CIFAR-10, PAMAP2; IID and
non-IID; see eval_detectors.py) found that TAS's score barely separates smashed-data poisoning from clean data
(AUROC 0.54-0.70), while a per-client shift score separated them perfectly (AUROC 1.000 everywhere).

How it works (all on the server, from what it already holds -- smashed data, labels, client ids):
  1. For each client and each class it holds (>= 3 samples), take the mean smashed vector.
  2. Compare that with the coordinate-wise MEDIAN of the other clients' means for the same class, divided by the
     class's typical spread. The client's score is the median of these over classes.
  3. A client is flagged if its score is more than `tau` robust spreads (MAD) above the median client score.
     If the scores form one cluster, nothing is flagged -- there is no fixed percentage target.

Assumption: honest majority. The median and MAD only describe "normal" if most clients are honest, and at most
`max_fraction` of the clients can be flagged in a round (a safety ceiling, not a target).

This module is our own design, not taken from a paper.
"""
from typing import Dict, Set

import numpy as np
import torch

from .detectors import det_client_shift


def client_shift_scores(Z: torch.Tensor, y: torch.Tensor, cid: torch.Tensor) -> Dict[int, float]:
    """One suspicion score per client present in the pool (higher = more shifted from the other clients)."""
    per_sample = det_client_shift(Z.cpu(), y.cpu(), cid.cpu())
    cid_np = cid.cpu().numpy()
    return {int(c): float(per_sample[cid_np == c][0]) for c in np.unique(cid_np)}


def flag_clients(scores: Dict[int, float], tau: float = 6.0, max_fraction: float = 0.4) -> Set[int]:
    """MAD rule at client level. Returns the set of flagged client ids (possibly empty)."""
    if len(scores) < 3:
        return set()            # not enough clients to define 'normal'
    ids = sorted(scores)
    s = np.array([scores[i] for i in ids])
    med = np.median(s)
    mad = 1.4826 * np.median(np.abs(s - med))
    spread = max(mad, 0.05 * abs(med), 1e-9)
    flagged = [i for i, v in zip(ids, s) if v > med + tau * spread]
    max_n = int(np.floor(max_fraction * len(ids)))
    if len(flagged) > max_n:
        flagged = sorted(flagged, key=lambda i: -scores[i])[:max_n]
    return set(flagged)
