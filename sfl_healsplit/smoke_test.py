"""
Optional environment sanity check for the Split FL track -- NOT a substitute
for a real experiment. See src/../smoke_test.py for the equivalent on the
FL+DP baseline; same rationale (no MNIST download in this sandbox, synthetic
data of the same shape instead).

Run:
    python -m sfl_healsplit.smoke_test
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from src.seeding import set_global_seed

from sfl_healsplit.attacks import LabelFlipDataset, select_malicious_clients
from sfl_healsplit.engine import run_split_fl
from sfl_healsplit.model import ClientModel, ServerModel


def make_synthetic_mnist(n_train=600, n_test=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    Xtr = torch.randn(n_train, 1, 28, 28, generator=g)
    ytr = torch.randint(0, 10, (n_train,), generator=g)
    Xte = torch.randn(n_test, 1, 28, 28, generator=g)
    yte = torch.randint(0, 10, (n_test,), generator=g)
    return TensorDataset(Xtr, ytr), TensorDataset(Xte, yte)


def partition(n, num_clients, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return np.array_split(idx, num_clients)


def main():
    seed = 42
    num_clients = 3
    malicious_ratio = 0.34  # -> 1 of 3 clients malicious
    label_flip_delta = 1
    set_global_seed(seed)
    device = torch.device("cpu")

    train_ds, test_ds = make_synthetic_mnist(seed=seed)
    shards = partition(len(train_ds), num_clients, seed)
    malicious_ids = select_malicious_clients(num_clients, malicious_ratio, seed)
    print(f"malicious_client_ids={sorted(malicious_ids)}")

    client_loaders = []
    for cid, shard in enumerate(shards):
        subset = Subset(train_ds, shard.tolist())
        if cid in malicious_ids:
            subset = LabelFlipDataset(subset, label_flip_delta, num_classes=10)
        client_loaders.append(DataLoader(subset, batch_size=32, shuffle=True))

    test_loader = DataLoader(test_ds, batch_size=64)

    client_model = ClientModel(in_channels=1)
    server_model = ServerModel(num_classes=10, img_size=28)

    os.makedirs("./sfl_smoke_results", exist_ok=True)
    run_split_fl(
        client_model=client_model,
        server_model=server_model,
        client_loaders=client_loaders,
        test_loader=test_loader,
        device=device,
        num_rounds=2,
        local_epochs=1,
        lr_client=0.05,
        lr_server=0.05,
        results_path="./sfl_smoke_results/metrics.csv",
        label_flip_delta=label_flip_delta,
    )
    print("SMOKE TEST OK")


if __name__ == "__main__":
    main()
