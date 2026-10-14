"""
Dump real smashed data from a short training run, for offline detector comparison.

It sets up the run exactly as main.py does from a SAVED CONFIG of one of your definitive runs
(dataset, class restriction, minority class, IID or Dirichlet split, batch size, learning rates, seed),
trains with the smashed-poisoning attack active (undefended, so the poison is in the training as it
would be), and at chosen checkpoints collects several detection pools of CLEAN smashed data (4 batches
per client, as the defence does). The attack is NOT baked into the dump: the offset is stored as a unit
vector per malicious client, so eval_detectors.py can add it at any strength afterwards. The saved
data is therefore exactly what the server would receive, with or without the attack.

Run from the folder that contains sfl_healsplit/ and src/ (same place as python -m sfl_healsplit.main):

    python -m sfl_healsplit.dump_smashed --from-config path/to/mnist_digit8_sp_deterministic_seed42_config.json \
        --out lab_dumps --split iid

Options: --split config|iid|noniid (override), --seed N (override), --checkpoints 0,1,3,5,9 (rounds of
training completed before each collection), --pools 2 (independent pools per checkpoint),
--device auto|cpu|cuda, --data-root (override).
"""
import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import build_restricted_dataset, get_datasets, partition_indices
from src.seeding import set_global_seed

from .attacks import make_smashed_poison_deltas, select_malicious_clients
from .config import SFLConfig
from .dirichlet_partition import dirichlet_partition_indices
from .engine import collect_round_smashed_data, estimate_smashed_data_scale, train_one_round
from .model import build_split_model
from .pamap2_data import build_pamap2_datasets


def labels_of(ds) -> torch.Tensor:
    """Labels of a dataset without assuming a .targets attribute (restricted and PAMAP2 wrappers)."""
    if hasattr(ds, "sample_indices") and hasattr(ds, "label_remap") and hasattr(ds.base_dataset, "targets"):
        base = torch.as_tensor(ds.base_dataset.targets)
        return torch.tensor([ds.label_remap[int(base[i])] for i in ds.sample_indices])
    return torch.tensor([int(ds[i][1]) for i in range(len(ds))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-config", required=True)
    ap.add_argument("--out", default="lab_dumps")
    ap.add_argument("--split", choices=["config", "iid", "noniid"], default="config")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--checkpoints", default="0,1,3,5,9")
    ap.add_argument("--pools", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()

    with open(args.from_config) as f:
        raw = json.load(f)
    fields = {f.name for f in dataclasses.fields(SFLConfig)}
    cfg = SFLConfig(**{k: v for k, v in raw.items() if k in fields})
    if args.seed is not None:
        cfg.seed = args.seed
    if args.split != "config":
        cfg.iid = args.split == "iid"
    if args.data_root:
        cfg.data_root = args.data_root
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")
    set_global_seed(cfg.seed)

    # ---- data, exactly as main.py ----
    if cfg.dataset == "pamap2":
        train_ds, test_ds, num_classes = build_pamap2_datasets(
            cfg.data_root, window_size=cfg.pamap2_window_size, stride=cfg.pamap2_stride)
    else:
        train_ds, test_ds = get_datasets(cfg.dataset, cfg.data_root)
        num_classes = 10
        if cfg.restrict_classes:
            names = [c.strip() for c in cfg.restrict_classes.split(",") if c.strip()]
            minority = [m.strip() for m in cfg.minority_class.split(",") if m.strip()] if cfg.minority_class else None
            train_ds, test_ds, num_classes = build_restricted_dataset(
                cfg.dataset, cfg.data_root, names, minority_class_names=minority,
                minority_keep_fraction=cfg.minority_keep_fraction, seed=cfg.seed)
    train_labels = labels_of(train_ds)
    counts = torch.bincount(train_labels, minlength=num_classes)
    minority_idx = int(counts.argmin())

    if cfg.iid:
        shards = partition_indices(len(train_ds), cfg.num_clients, cfg.iid, cfg.seed)
    else:
        shards = dirichlet_partition_indices(train_labels, cfg.num_clients, cfg.dirichlet_alpha, cfg.seed)
    malicious_ids = sorted(select_malicious_clients(cfg.num_clients, cfg.malicious_ratio, cfg.seed))

    loaders = [DataLoader(torch.utils.data.Subset(train_ds, s.tolist()), batch_size=cfg.batch_size, shuffle=True)
               for s in shards]
    client_model, server_model = build_split_model(cfg.dataset, num_classes=num_classes,
                                                   window_size=cfg.pamap2_window_size)
    client_model.to(device)
    server_model.to(device)
    z_std0, z_shape = estimate_smashed_data_scale(client_model, loaders[0], device)
    train_deltas = make_smashed_poison_deltas(set(malicious_ids), z_shape, cfg.smashed_poison_sigma * z_std0, cfg.seed)
    unit = make_smashed_poison_deltas(set(malicious_ids), z_shape, 1.0, cfg.seed)  # same generator, sigma 1
    print(f"[dump] {cfg.dataset} iid={cfg.iid} seed={cfg.seed} malicious={malicious_ids} z_shape={z_shape} "
          f"z_std0={z_std0:.4f} minority_class={minority_idx} (count {int(counts[minority_idx])})")

    server_opt = torch.optim.SGD(server_model.parameters(), lr=cfg.lr_server)
    checkpoints = sorted({int(c) for c in args.checkpoints.split(",")})
    Zs, ys, cids, rounds = [], [], [], []
    trained = 0
    for cp in checkpoints:
        while trained < cp:
            client_model = train_one_round(client_model, server_model, loaders, server_opt, device,
                                           cfg.local_epochs, cfg.lr_client, smashed_poison_deltas=train_deltas)
            trained += 1
            print(f"[dump] trained round {trained}")
        for _ in range(args.pools):
            Z, y, cid, _ = collect_round_smashed_data(client_model, loaders, device, smashed_poison_deltas=None,
                                                      max_batches_per_client=4)
            Zs.append(Z.reshape(Z.size(0), -1).numpy().astype(np.float32))
            ys.append(y.numpy())
            cids.append(cid.numpy())
            rounds.append(trained)
        print(f"[dump] collected {args.pools} pool(s) after {trained} trained round(s)")

    os.makedirs(args.out, exist_ok=True)
    tag = f"{cfg.dataset}_{'iid' if cfg.iid else 'noniid'}_seed{cfg.seed}"
    meta = dict(dataset=cfg.dataset, iid=cfg.iid, seed=cfg.seed, num_clients=cfg.num_clients,
                num_classes=num_classes, minority_class=minority_idx, malicious_ids=malicious_ids,
                z_std0=z_std0, z_shape=list(z_shape), train_sigma=cfg.smashed_poison_sigma,
                batch_size=cfg.batch_size, class_counts=counts.tolist())
    np.savez_compressed(
        os.path.join(args.out, tag + ".npz"),
        Z=np.stack(Zs), y=np.stack(ys), cid=np.stack(cids), round=np.array(rounds),
        unit_ids=np.array(malicious_ids),
        unit_deltas=np.stack([unit[m].reshape(-1).numpy() for m in malicious_ids]),
        meta=json.dumps(meta))
    print(f"[dump] wrote {os.path.join(args.out, tag + '.npz')}")


if __name__ == "__main__":
    main()
