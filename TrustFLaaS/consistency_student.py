"""
Stand-in "consistency validation student" for stage 3.

The paper's real consistency validation student is trained via adversarial
multi-teacher distillation (Vanilla Teacher + AD Teacher, Eqs. 10-13) --
that's stage 4, not built yet. This module provides a SIMPLER stand-in: a
plain classifier trained only on TAS-identified clean smashed data with
ordinary cross-entropy loss. That's functionally equivalent to just the
paper's "Vanilla Teacher" (Eq. 10) on its own, used here directly as the
GAN-output gatekeeper so stage 3 has a complete, working, testable
detect->generate->validate->substitute pipeline without waiting on stage 4.

When stage 4 builds the real student (Vanilla + AD teacher distillation),
it should REPLACE the model this module trains, not sit alongside it --
`recovery.py`'s gating function takes any classifier with a
`(logits) forward(z)` signature, so swapping in the stage-4 student later
is a one-line change at the call site, not a rewrite of the gating logic.

Added for the PAMAP2 extension track (sensor time-series, per supervisor
request): PAMAP2's smashed data z has shape (channels, window_len) -- e.g.
(16, 50) -- with NO spatial H/W dimensions, unlike the image datasets'
(channels, H, W) -- e.g. (16, 14, 14). The original SmashedDataClassifier is
Conv2d-based and assumes a 4D (N, C, H, W) input; fed a 3D PAMAP2 z batch
(N, C, L), PyTorch's Conv2d silently reinterprets it as UNBATCHED (C, H, W)
-- i.e. it reads the batch dimension itself as the channel count -- which is
exactly the "expected input to have 16 channels but got 32" error this
produces in practice (32 being the batch size, not a real channel count).

Fix: Pamap2SmashedDataClassifier below is a Conv1d-based structural mirror
of SmashedDataClassifier (same two-conv-block + AdaptiveAvgPool + two-FC
shape, same GroupNorm/ReLU pattern), used only when z has no spatial
dimensions. build_smashed_data_classifier() is the dispatch point: it picks
Conv1d vs Conv2d based on len(z_shape) (2 -> 1D/PAMAP2, 3 -> 2D/image),
which is already available at every call site (both callers already compute
z_shape via estimate_smashed_data_scale). SmashedDataClassifier's own code
is UNCHANGED -- image-dataset runs (MNIST/Fashion-MNIST/CIFAR-10) go through
the exact same class as before, so none of that prior work's results are
affected by this addition.
"""
import torch
import torch.nn as nn


class SmashedDataClassifier(nn.Module):
    """Small CNN operating directly on smashed-data-shaped input (e.g. the
    16x14x14 feature maps ClientModel produces for MNIST). Architecturally
    similar in spirit to ServerModel's first stage, but standalone -- this
    is NOT the server-side model, it's a separate classifier used only for
    gating/validation, matching Table 1's "4-layer CNN" for the student."""

    def __init__(self, in_channels: int, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.fc1 = nn.Linear(32 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.net(z)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class Pamap2SmashedDataClassifier(nn.Module):
    """PAMAP2 analogue of SmashedDataClassifier: Conv1d instead of Conv2d,
    AdaptiveAvgPool1d instead of AdaptiveAvgPool2d, otherwise the identical
    two-conv-block + two-FC structure. Operates on PAMAP2's smashed data z,
    shape (batch, 16, window_len // 2) -- e.g. (batch, 16, 50) -- with no
    spatial H/W dimensions."""

    def __init__(self, in_channels: int, num_classes: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.fc1 = nn.Linear(32 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.net(z)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def build_smashed_data_classifier(z_shape, num_classes: int = 10) -> nn.Module:
    """Dispatch point: builds the right smashed-data classifier for a given
    per-sample z shape (NOT including the batch dimension), e.g. (16, 14, 14)
    for MNIST/Fashion-MNIST/CIFAR-10, or (16, 50) for PAMAP2.

    len(z_shape) is the discriminator: 3 -> image-shaped (channels, H, W) ->
    the original Conv2d-based SmashedDataClassifier, unchanged; 2 ->
    sensor-window-shaped (channels, length), no spatial dims -> the new
    Conv1d-based Pamap2SmashedDataClassifier. Both call sites in defense.py
    already compute z_shape via estimate_smashed_data_scale, so this needs
    no dataset name threaded through -- the shape itself is enough."""
    in_channels = z_shape[0]
    if len(z_shape) == 2:
        return Pamap2SmashedDataClassifier(in_channels=in_channels, num_classes=num_classes)
    return SmashedDataClassifier(in_channels=in_channels, num_classes=num_classes)


def train_classifier(model: nn.Module, Z: torch.Tensor, y: torch.Tensor,
                      device: torch.device, num_epochs: int = 20,
                      batch_size: int = 32, lr: float = 0.01) -> nn.Module:
    """Plain supervised training -- Eq. 10's L_Van, applied here to whatever
    classifier is passed in (the stand-in student for stage 3; the stage-4
    student would be trained differently, via distillation, not this
    function)."""
    model.to(device)
    Z, y = Z.to(device), y.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    n = Z.size(0)

    model.train()
    for epoch in range(num_epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            logits = model(Z[idx])
            loss = criterion(logits, y[idx])
            loss.backward()
            optimizer.step()
    return model
