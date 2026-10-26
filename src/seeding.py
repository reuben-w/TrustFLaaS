"""
Central seeding utility.

Design decision: we seed Python's `random`, NumPy, and Torch (CPU + CUDA) once
at process start from a single top-level seed. Per-client determinism (e.g.
which client gets which data indices) is derived from this same seed via the
partitioning code, NOT from independent per-client seeds -- this keeps the
IID partition identical across repeated runs regardless of client count.

Note on DP-SGD: Opacus's Poisson (uniform-with-replacement) batch sampler is
itself stochastic every epoch, as required for the privacy accounting to be
valid. Seeding makes this stochasticity reproducible across runs, but it does
NOT make the sampler deterministic in the sense of "same minibatches every
round" -- that would defeat the point of Poisson sampling.
"""
import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Deterministic cuDNN kernels (slower, but reproducible on GPU).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
