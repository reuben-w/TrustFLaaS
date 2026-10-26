"""
Dirichlet-distribution-based non-IID client partitioning -- kept separate
from src/data.py's own partition_indices (the IID path) rather than
modifying it, since non-IID partitioning is a concern specific to this
track's testing needs, not the shared FL+DP baseline.

Standard approach, matching the convention used across the non-IID FL
literature this project has been reading (the EWC/MConFL paper, CLIMB,
MOON, Astraea all partition this way): for each class independently, draw
per-client proportions from a Dirichlet(alpha) distribution, then split
that class's sample indices across clients according to those proportions.
Lower alpha means more skew -- clients end up with very different class
mixes; alpha -> infinity converges to the IID case (every client gets an
equal share of every class).

Every sample is assigned to EXACTLY ONE client -- this partitions the
dataset, it doesn't duplicate or drop anything. Note this can legitimately
produce a client with ZERO samples of a given class at low alpha /
low class-sample-count combinations (e.g. an already-scarce minority class
under severe skew) -- this is a real, expected property of Dirichlet
partitioning at low alpha, not a bug, though it's worth knowing about if a
downstream computation ever assumes every client has at least one sample
of every class.
"""
import numpy as np
import torch
from typing import List


def dirichlet_partition_indices(
    labels: torch.Tensor, num_clients: int, alpha: float, seed: int
) -> List[np.ndarray]:
    """labels: [N] tensor of class indices for the full (already-restricted/
    subsampled, if applicable) training set, in dataset order. Returns a
    list of num_clients index arrays, each a subset of range(N) -- directly
    usable the same way partition_indices' own output is (e.g. wrapped in
    torch.utils.data.Subset)."""
    rng = np.random.default_rng(seed)
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    num_classes = int(labels_np.max()) + 1

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        class_idx = np.nonzero(labels_np == c)[0]
        if len(class_idx) == 0:
            continue  # this class has no samples at all in the dataset -- nothing to split
        rng.shuffle(class_idx)

        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        split_points = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        splits = np.split(class_idx, split_points)

        for cid, split in enumerate(splits):
            client_indices[cid].extend(split.tolist())

    for cid in range(num_clients):
        rng.shuffle(client_indices[cid])

    return [np.array(idx, dtype=np.int64) for idx in client_indices]


def summarize_partition(labels: torch.Tensor, shards: List[np.ndarray], num_classes: int) -> str:
    """Human-readable per-client class-count table, for --non-iid setup logging
    -- lets a user SEE the actual skew a run used, rather than trusting the
    alpha value alone (two runs at the same alpha can look quite different,
    since the Dirichlet draw itself is random)."""
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    lines = []
    for cid, shard in enumerate(shards):
        counts = np.bincount(labels_np[shard], minlength=num_classes)
        lines.append(f"  client {cid}: " + ", ".join(f"c{c}={counts[c]}" for c in range(num_classes)))
    return "\n".join(lines)
