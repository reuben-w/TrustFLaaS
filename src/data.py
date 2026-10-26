"""
Dataset loading and client partitioning.

Only IID partitioning is implemented for this baseline (per the task spec:
"simple IID split first, non-IID splitting to be added later"). The
partitioning function is written so that a non-IID strategy (e.g. Dirichlet
label-skew, shard-based pathological non-IID) can be dropped in later behind
the same interface -- see `partition_indices`.
"""
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from typing import List, Optional, Tuple

DATASET_STATS = {
    "mnist": {
        "cls": datasets.MNIST,
        "mean": (0.1307,),
        "std": (0.3081,),
        "channels": 1,
        "img_size": 28,
    },
    "fashion-mnist": {
        "cls": datasets.FashionMNIST,
        "mean": (0.2860,),
        "std": (0.3530,),
        "channels": 1,
        "img_size": 28,
    },
    "cifar10": {
        "cls": datasets.CIFAR10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "channels": 3,
        "img_size": 32,
    },
}


def get_datasets(dataset_name: str, data_root: str) -> Tuple[Dataset, Dataset]:
    """Download (if needed) and return (train_dataset, test_dataset)."""
    spec = DATASET_STATS[dataset_name]
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(spec["mean"], spec["std"]),
    ])
    train_ds = spec["cls"](root=data_root, train=True, download=True, transform=tfm)
    test_ds = spec["cls"](root=data_root, train=False, download=True, transform=tfm)
    return train_ds, test_ds


def partition_indices(num_samples: int, num_clients: int, iid: bool, seed: int) -> List[np.ndarray]:
    """Split `num_samples` indices into `num_clients` roughly-equal shards.

    IID design decision: indices are shuffled once (with the global seed) and
    then cut into `num_clients` contiguous, near-equal blocks. Because the
    shuffle is uniform-random and unconditioned on label, each client's shard
    is an unbiased sample of the label distribution -- i.e. IID by
    construction, not by post-hoc checking.
    """
    if not iid:
        raise NotImplementedError(
            "Non-IID partitioning is not implemented in this baseline. "
            "Reserved for future work (see README)."
        )
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)
    shards = np.array_split(indices, num_clients)
    return shards


def build_client_dataloaders(
    dataset_name: str,
    data_root: str,
    num_clients: int,
    iid: bool,
    seed: int,
    batch_size: int,
) -> Tuple[List[DataLoader], DataLoader, Dataset]:
    """Returns (list of per-client train DataLoaders, centralized test DataLoader, train_dataset).

    The train_dataset is also returned so client-side code can build its own
    Opacus-compatible DataLoader (Opacus's PrivacyEngine replaces the
    DataLoader's sampler internally, so client.py constructs its own loader
    from a Subset rather than reusing the loaders returned here directly).
    """
    train_ds, test_ds = get_datasets(dataset_name, data_root)
    shards = partition_indices(len(train_ds), num_clients, iid, seed)

    client_loaders = []
    for shard in shards:
        subset = Subset(train_ds, shard.tolist())
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=False)
        client_loaders.append(loader)

    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    return client_loaders, test_loader, train_ds


def get_client_subset(train_ds: Dataset, num_clients: int, iid: bool, seed: int, client_id: int) -> Subset:
    """Convenience accessor used by client processes that only need their own shard."""
    shards = partition_indices(len(train_ds), num_clients, iid, seed)
    return Subset(train_ds, shards[client_id].tolist())


# --- Class restriction + minority subsampling (outlier/scarcity experiments) ---
#
# Design decision -- why this exists:
# HealSplit's TAS detects samples that look topologically isolated. MNIST
# digits all sit reasonably close to at least one other digit, so testing
# "does TAS mistreat a genuinely rare-and-different class" needs a dataset
# where classes are actually far apart. This restricts a dataset to a small
# set of named classes, remaps their labels to a compact 0..N-1 range, and
# optionally subsamples ONE class (the "minority") down to a fraction of its
# original count -- e.g. CIFAR-10 restricted to {automobile, truck, frog},
# with frog cut to 20% of its original count, giving a class that is both
# numerically rare AND structurally distinct from the majority (no wheels,
# no boxy shape, organic rather than mechanical) -- unlike an MNIST digit,
# which is always visually close to at least one other digit regardless of
# how rare you make it.
CIFAR10_CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
                        "dog", "frog", "horse", "ship", "truck"]


def get_class_index(dataset_name: str, class_name: str) -> int:
    if dataset_name == "cifar10":
        return CIFAR10_CLASS_NAMES.index(class_name)
    if dataset_name in ("mnist", "fashion-mnist"):
        # Neither dataset gets a class-NAME list (e.g. Fashion-MNIST's own
        # category names) -- --minority-class has always been documented
        # and used as the raw digit index (0-9) for these two datasets (see
        # config.py's --minority-class help text), so --restrict-classes
        # uses that SAME convention rather than introducing a second,
        # parallel naming scheme. This also means the "list every class to
        # unlock minority subsampling" trick used for CIFAR-10
        # (--restrict-classes airplane,automobile,...,truck) has a direct
        # MNIST/Fashion-MNIST equivalent: --restrict-classes 0,1,...,9.
        try:
            idx = int(class_name)
        except ValueError:
            raise ValueError(
                f"For dataset='{dataset_name}', class identifiers passed to "
                f"--restrict-classes/--minority-class must be a raw digit index "
                f"as a string ('0'-'9'), got {class_name!r}."
            )
        if not (0 <= idx <= 9):
            raise ValueError(
                f"For dataset='{dataset_name}', class index must be in 0-9, got {idx}."
            )
        return idx
    raise NotImplementedError(
        f"Class-name restriction isn't defined for '{dataset_name}' yet -- "
        f"only cifar10 (CIFAR10_CLASS_NAMES), mnist, and fashion-mnist "
        f"(raw digit index) are supported right now."
    )


class RestrictedDataset(Dataset):
    """Wraps a Dataset, keeping only a chosen subset of sample indices and
    remapping their original labels to a compact 0..N-1 range. Used for
    class-restriction + minority-subsampling experiments."""

    def __init__(self, base_dataset: Dataset, sample_indices: List[int], label_remap: dict):
        self.base_dataset = base_dataset
        self.sample_indices = sample_indices
        self.label_remap = label_remap

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        x, y = self.base_dataset[self.sample_indices[idx]]
        return x, self.label_remap[int(y)]


def build_restricted_dataset(
    dataset_name: str,
    data_root: str,
    class_names: List[str],
    minority_class_names: Optional[List[str]] = None,   # CHANGED: was minority_class_name: Optional[str]
    minority_keep_fraction: float = 1.0,
    seed: int = 0,
) -> Tuple[Dataset, Dataset, int]:
    """Returns (train_dataset, test_dataset, num_classes), restricted to
    `class_names` (labels remapped to 0..len(class_names)-1 in that order).

    minority_class_names may now list MORE THAN ONE class -- each is
    independently subsampled to minority_keep_fraction of its own
    original count. A single-element list behaves identically to the
    original single-minority design.

    Minority subsampling applies to the TRAIN set only, never the test
    set: the test set stays full/balanced across all restricted classes,
    so evaluation measures true per-class performance rather than being
    handicapped by an already-scarce test sample.
    """
    train_ds, test_ds = get_datasets(dataset_name, data_root)
    class_indices = [get_class_index(dataset_name, c) for c in class_names]
    label_remap = {orig: new for new, orig in enumerate(class_indices)}
    num_classes = len(class_names)
    minority_class_names = minority_class_names or []

    def filter_and_subsample(ds: Dataset, apply_minority_subsample: bool) -> List[int]:
        targets = ds.targets if hasattr(ds, "targets") else [ds[i][1] for i in range(len(ds))]
        targets = torch.as_tensor(targets)
        keep_mask = torch.zeros(len(ds), dtype=torch.bool)
        for ci in class_indices:
            keep_mask |= (targets == ci)
        keep_idx = torch.nonzero(keep_mask, as_tuple=True)[0]

        if apply_minority_subsample and minority_class_names and minority_keep_fraction < 1.0:
            minority_idx_vals = {get_class_index(dataset_name, m) for m in minority_class_names}
            is_minority = torch.tensor([targets[i].item() in minority_idx_vals for i in keep_idx])
            minority_positions = keep_idx[is_minority]
            other_positions = keep_idx[~is_minority]

            # Subsample EACH minority class independently, not the pooled
            # minority set as a whole -- otherwise a keep_fraction applied
            # to the combined pool would let one minority class crowd out
            # another rather than each being scarce on its own terms.
            rng = np.random.default_rng(seed)
            kept_minority_chunks = []
            for m_name in minority_class_names:
                m_idx_val = get_class_index(dataset_name, m_name)
                this_class_mask = targets[minority_positions] == m_idx_val
                this_class_positions = minority_positions[this_class_mask]
                n_keep = max(1, int(len(this_class_positions) * minority_keep_fraction))
                chosen = rng.choice(this_class_positions.numpy(), size=n_keep, replace=False)
                kept_minority_chunks.append(torch.as_tensor(chosen, dtype=torch.long))

            keep_idx = torch.cat([other_positions] + kept_minority_chunks)

        return keep_idx.tolist()

    train_keep = filter_and_subsample(train_ds, apply_minority_subsample=True)
    test_keep = filter_and_subsample(test_ds, apply_minority_subsample=False)

    train_restricted = RestrictedDataset(train_ds, train_keep, label_remap)
    test_restricted = RestrictedDataset(test_ds, test_keep, label_remap)
    return train_restricted, test_restricted, num_classes
