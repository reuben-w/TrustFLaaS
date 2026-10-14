"""
Label Poisoning (LP) attack, matching HealSplit's threat model exactly
(paper, "Threat Model" section, A1):

    y'_j = (y_j + delta_y) mod C

where C is the number of classes and delta_y is a fixed label shift. This is
an "all-to-all" cyclic label shift -- every malicious client relabels every
one of its samples by the same shift -- as opposed to a targeted attack
(e.g. only flipping one specific source class to one specific target class).
We implement the paper's definition as written rather than substituting a
different (e.g. targeted) label-flip variant.
"""
from typing import Dict, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class LabelFlipDataset(Dataset):
    """Wraps a Dataset/Subset and relabels every sample via y' = (y + delta) mod C.
    Inputs (x) are left untouched -- this is a pure label-poisoning attack,
    not a data-poisoning (feature perturbation) attack (that's HealSplit's
    separate DP/A2 threat, a natural follow-up attack to implement)."""

    def __init__(self, base_dataset: Dataset, delta_y: int, num_classes: int):
        self.base_dataset = base_dataset
        self.delta_y = delta_y
        self.num_classes = num_classes

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        y_int = int(y)
        y_flipped = (y_int + self.delta_y) % self.num_classes
        return x, torch.tensor(y_flipped, dtype=torch.long)


def select_malicious_clients(num_clients: int, malicious_ratio: float, seed: int) -> Set[int]:
    """Deterministically picks which client IDs are malicious, given the
    experiment seed -- so "client 2 is malicious" is reproducible and doesn't
    depend on run order the way naive random.sample() calls scattered through
    the main loop would.

    Design decision -- selection independent of the training-loop RNG:
    Uses its own seeded generator (numpy default_rng), matching the pattern
    already used for IID data partitioning in src/data.py::partition_indices
    -- attacker-identity assignment shouldn't be entangled with (and
    perturbed by) unrelated randomness elsewhere in the pipeline.
    """
    rng = np.random.default_rng(seed + 1)  # +1 so this doesn't reuse the exact
                                            # same stream as data partitioning,
                                            # which is also seeded from `seed`
    num_malicious = round(num_clients * malicious_ratio)
    malicious_ids = rng.choice(num_clients, size=num_malicious, replace=False)
    return set(int(i) for i in malicious_ids)


def make_smashed_poison_deltas(malicious_client_ids: Set[int], z_shape: Tuple[int, ...],
                                sigma: float, seed: int) -> Dict[int, torch.Tensor]:
    """Generates ONE fixed perturbation vector delta_z per malicious client,
    deterministically from `seed`. z_shape is the shape of a single sample's
    smashed data (no batch dim), e.g. (16, 14, 14).

    Design decision -- fixed per-client, not resampled per sample/batch:
    Caught during testing (see sfl_healsplit/README.md "Verified before
    handoff"): an earlier version generated fresh `torch.randn_like(z)`
    noise on every call, which resamples independently per element of every
    sample. But the paper's own equation, z'_j = g(x_j) + delta_z, has NO
    subscript j on delta_z -- it's meant to be the SAME perturbation for
    every sample from that attacker. Fresh-per-sample noise scatters
    poisoned points randomly through feature space instead of clustering
    them together, which contradicts the paper's own stated premise that
    poisoned samples are "locally dense" -- and empirically, scattered
    points end up maximally isolated even from EACH OTHER, which triggers
    the same teleport-term-dominance pathology documented in detection.py
    (a near-zero-degree node's v_k inflates rather than suppresses its
    score). A single fixed delta_z per client keeps poisoned samples
    clustered together, matching what TAS is actually built to detect.
    """
    deltas = {}
    for cid in malicious_client_ids:
        g = torch.Generator().manual_seed(seed * 1000 + cid)
        deltas[cid] = torch.randn(z_shape, generator=g) * sigma
    return deltas


def add_smashed_poison(z: torch.Tensor, delta_z: torch.Tensor) -> torch.Tensor:
    """Smashed Poisoning (SP, A3) attack, paper Threat Model: z'_j = g(x_j) + delta_z.
    `delta_z` is a fixed per-client perturbation from make_smashed_poison_deltas
    (shape matches a single sample; broadcasts over the batch dimension)."""
    return z + delta_z.to(z.device).unsqueeze(0)
