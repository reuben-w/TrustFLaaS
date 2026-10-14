"""
Client-side / server-side model split for Split Federated Learning.

Same overall architecture as src/model.py::SmallCNN (kept identical on
purpose, so any future comparison between the plain-FL baseline and this SFL
track isn't confounded by architecture differences) but cut into two pieces
at a fixed split point.

Design decision -- where to split:
We split after the first conv block (conv1 -> GroupNorm -> ReLU -> pool).
This is a standard shallow split point in the SFL literature: the client
keeps only enough of the network to extract low-level features, and the
"smashed data" it transmits is a 16-channel feature map rather than raw
pixels -- some representational privacy benefit, without needing much
client-side compute (the stated appeal of split learning for
resource-constrained clients). The split point is not currently
configurable; that's a natural extension point (see README) once we care
about characterizing HealSplit's detection performance vs. split depth.

Design decision -- GroupNorm, not BatchNorm:
Kept for consistency with the FL+DP baseline (src/model.py) and because it
composes cleanly if this track is ever combined with DP-SGD on the
client-side model. Not required for correctness here the way it was for
Opacus, but there's no reason to diverge.

Added for the PAMAP2 extension track (sensor time-series, per supervisor
request) -- Pamap2ClientModel/Pamap2ServerModel below are a parallel 1D-CNN
pair for PAMAP2's [channels, timesteps] windows, mirroring the image
models' same split-after-first-conv-block structure AND, critically, the
same non-negative final activation (ReLU). That invariant is what lets
gan.py's Generator (whose own final ReLU exists specifically to match
"real smashed data is always non-negative" -- see gan.py's docstring) and
DALSE's centroid tracking keep working completely unmodified on PAMAP2's
z -- verified generic in both files, no changes needed there.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClientModel(nn.Module):
    """Everything the client computes locally, up to the smashed-data cut
    point. Output shape: (batch, 16, img_size/2, img_size/2)."""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=16)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.relu(self.gn1(self.conv1(x))))


class ServerModel(nn.Module):
    """Everything from the smashed data onward. This is the single, shared
    model instance in our SFLV1-style engine (see engine.py) -- there is
    exactly one ServerModel, updated directly and sequentially by every
    client's batch, not N per-client copies that get aggregated."""

    def __init__(self, num_classes: int = 10, img_size: int = 28):
        super().__init__()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=32)
        self.pool = nn.MaxPool2d(2, 2)

        feat_size = img_size // 4  # two 2x2 pools total (one client-side, one here)
        self.fc1 = nn.Linear(32 * feat_size * feat_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, smashed: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.gn2(self.conv2(smashed))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class Pamap2ClientModel(nn.Module):
    """PAMAP2 analogue of ClientModel: Conv1d instead of Conv2d, same
    split-after-first-conv-block structure, same GroupNorm/ReLU/pool
    pattern, same non-negative-output invariant (see module docstring).
    Input: (batch, 18, window_size). Output ("smashed data" z):
    (batch, 16, window_size // 2)."""

    def __init__(self, in_channels: int = 18):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=5, padding=2)
        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=16)
        self.pool = nn.MaxPool1d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.relu(self.gn1(self.conv1(x))))


class Pamap2ServerModel(nn.Module):
    """PAMAP2 analogue of ServerModel: Conv1d instead of Conv2d, otherwise
    structurally identical. num_classes=12 for the 12-activity protocol set
    (see pamap2_data.py::PROTOCOL_ACTIVITY_IDS)."""

    def __init__(self, num_classes: int = 12, window_size: int = 100):
        super().__init__()
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=32)
        self.pool = nn.MaxPool1d(2, 2)

        feat_len = window_size // 4  # two 2x pools total (one client-side, one here)
        self.fc1 = nn.Linear(32 * feat_len, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, smashed: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.gn2(self.conv2(smashed))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def build_split_model(dataset_name: str, num_classes: int = 10, window_size: int = 100):
    """Returns (ClientModel, ServerModel) for the given dataset.
    num_classes defaults to 10 (every dataset this project has used natively
    has 10 classes) but is a real parameter -- needed for class-restricted
    experiments (e.g. CIFAR-10 cut to 3 classes), where the natural class
    count no longer matches the dataset's default.

    PAMAP2 gets its own 1D-CNN pair (Pamap2ClientModel/Pamap2ServerModel)
    instead of going through DATASET_STATS -- it isn't an image dataset, so
    there's no channels/img_size spec to look up there; window_size (default
    100 = 1.0s @ 100Hz, matching pamap2_data.py's default) plays the role
    img_size plays for the image models."""
    if dataset_name == "pamap2":
        client_model = Pamap2ClientModel(in_channels=18)
        server_model = Pamap2ServerModel(num_classes=num_classes, window_size=window_size)
        return client_model, server_model

    from src.data import DATASET_STATS
    spec = DATASET_STATS[dataset_name]
    client_model = ClientModel(in_channels=spec["channels"])
    server_model = ServerModel(num_classes=num_classes, img_size=spec["img_size"])
    return client_model, server_model
