"""
Multi-head model shared by the AD Teacher and the real consistency
validation student. Both need three task outputs (paper's Methodology
section, "Gradient Interaction Score" and Eq. 13): a = poisoning
identification, b = client identification, c = category classification.

Design decision -- shared trunk, three heads:
The paper describes GIS as measuring gradient alignment "through shared
parameters" (citing multi-task learning literature, Yu et al. 2020). That
only makes sense if there IS a shared parameter set multiple task losses
route gradients through -- a shared trunk with per-task heads is the
standard architecture this describes, and it's also what GIS (gis.py)
computes gradients with respect to.
"""
from typing import Tuple

import torch
import torch.nn as nn


class MultiHeadModel(nn.Module):
    def __init__(self, in_channels: int, num_clients: int, num_classes: int = 10,
                 trunk_channels: int = 32):
        super().__init__()
        self.shared_trunk = nn.Sequential(
            nn.Conv2d(in_channels, trunk_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, trunk_channels),
            nn.ReLU(),
            nn.Conv2d(trunk_channels, trunk_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, trunk_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        feat_dim = trunk_channels * 4 * 4

        self.head_a = nn.Sequential(nn.Linear(feat_dim, 32), nn.ReLU(), nn.Linear(32, 2))
        self.head_b = nn.Sequential(nn.Linear(feat_dim, 32), nn.ReLU(), nn.Linear(32, num_clients))
        self.head_c = nn.Sequential(nn.Linear(feat_dim, 32), nn.ReLU(), nn.Linear(32, num_classes))

    def shared_features(self, z: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.shared_trunk(z), 1)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (logits_a, logits_b, logits_c)."""
        feat = self.shared_features(z)
        return self.head_a(feat), self.head_b(feat), self.head_c(feat)
