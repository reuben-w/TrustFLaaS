"""
Evaluates TAS detection on real smashed data produced by the Split FL
engine -- trains a few rounds under an attack, then collects one round's
smashed data (with ground truth) and checks whether TAS's flags line up.

This is also the script demonstrating the key finding from this stage:
TAS can detect Smashed Poisoning (SP) but CANNOT detect Label Poisoning
(LP), because LP never touches the smashed data z -- only the label. TAS
has no way to see a wrong label; it only sees z's position in feature
space, which is untouched by LP. Run this script with both --attack-type
values to see the asymmetry directly.

Usage:
    python -m sfl_healsplit.evaluate_detection --attack-type smashed_poison
    python -m sfl_healsplit.evaluate_detection --attack-type label_flip
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Subset

from src.data import get_datasets, partition_indices
from src.seeding import set_global_seed

from .attacks import LabelFlipDataset, make_smashed_poison_deltas, select_malicious_clients
from .detection import detect_poisoned
from .engine import collect_round_smashed_data, estimate_smashed_data_scale, train_one_round
from .model import build_split_model


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate TAS detection on real Split FL smashed data.")
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion-mnist", "cifar10"])
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--num-clients", type=int, default=5)
    p.add_argument("--malicious-ratio", type=float, default=0.2)
    p.add_argument("--attack-type", type=str, default="smashed_poison",
                    choices=["label_flip", "smashed_poison"])
    p.add_argument("--label-flip-delta", type=int, default=1)
    p.add_argument("--smashed-poison-sigma", type=float, default=3.0)
    p.add_argument("--warmup-rounds", type=int, default=3,
                    help="Rounds of real training before collecting smashed data for "
                         "detection -- an untrained client model's smashed data is close "
                         "to random for any input, which isn't representative of what TAS "
                         "would actually see partway through a real run.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr-client", type=float, default=0.05)
    p.add_argument("--lr-server", type=float, default=0.05)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--percentile", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, _ = get_datasets(args.dataset, args.data_root)
    shards = partition_indices(len(train_ds), args.num_clients, True, args.seed)
    malicious_ids = select_malicious_clients(args.num_clients, args.malicious_ratio, args.seed)
    print(f"attack_type={args.attack_type}  malicious_client_ids={sorted(malicious_ids)}")

    client_loaders = []
    for cid, shard in enumerate(shards):
        subset = Subset(train_ds, shard.tolist())
        if args.attack_type == "label_flip" and cid in malicious_ids:
            subset = LabelFlipDataset(subset, args.label_flip_delta, num_classes=10)
        client_loaders.append(DataLoader(subset, batch_size=args.batch_size, shuffle=True))

    client_model, server_model = build_split_model(args.dataset)
    client_model.to(device)
    server_model.to(device)
    server_optimizer = torch.optim.SGD(server_model.parameters(), lr=args.lr_server)

    smashed_poison_deltas = None
    if args.attack_type == "smashed_poison":
        z_std, z_shape = estimate_smashed_data_scale(client_model, client_loaders[0], device)
        smashed_poison_deltas = make_smashed_poison_deltas(
            malicious_ids, z_shape, args.smashed_poison_sigma * z_std, args.seed
        )
        print(f"smashed-data scale estimate: std={z_std:.4f}, shape={z_shape} "
              f"-> delta_z magnitude ~{args.smashed_poison_sigma * z_std:.4f} per element")

    print(f"Training {args.warmup_rounds} warm-up round(s) so smashed data reflects a "
          f"partially-trained model, not random init...")
    for _ in range(args.warmup_rounds):
        client_model = train_one_round(
            client_model, server_model, client_loaders, server_optimizer, device,
            local_epochs=1, lr_client=args.lr_client,
            smashed_poison_deltas=smashed_poison_deltas,
        )

    print("Collecting smashed data for detection...")
    Z, y, client_ids, is_poisoned_z = collect_round_smashed_data(
        client_model, client_loaders, device,
        smashed_poison_deltas=smashed_poison_deltas,
        max_batches_per_client=4,
    )
    # For label_flip, z is never touched -- ground truth "attacker's samples"
    # (for reporting purposes only, NOT what TAS is being asked to find,
    # since TAS operates on z and z is clean under LP) is which client a
    # sample came from, not a z-level corruption flag.
    if args.attack_type == "label_flip":
        is_poisoned_ground_truth = torch.tensor(
            [cid.item() in malicious_ids for cid in client_ids], dtype=torch.bool
        )
    else:
        is_poisoned_ground_truth = is_poisoned_z

    print(f"Collected {Z.shape[0]} smashed-data vectors, "
          f"{is_poisoned_ground_truth.sum().item()} from malicious clients.")

    result = detect_poisoned(Z, k=args.k, alpha=args.alpha, percentile=args.percentile)

    pred = result.predicted_poisoned
    gt = is_poisoned_ground_truth
    tp = (pred & gt).sum().item()
    fp = (pred & ~gt).sum().item()
    fn = (~pred & gt).sum().item()
    tn = (~pred & ~gt).sum().item()
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    print(f"\n--- TAS detection results (attack_type={args.attack_type}) ---")
    print(f"threshold={result.threshold:.5f}")
    print(f"mean TAS, malicious-client samples: {result.tas_scores[gt].mean().item():.5f}")
    print(f"mean TAS, clean-client samples:     {result.tas_scores[~gt].mean().item():.5f}")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")

    if args.attack_type == "label_flip":
        print("\nNote: TAS operates on smashed data (z), which label-flipping never "
              "touches -- these numbers are expected to be near chance level. This "
              "is the mismatch documented in sfl_healsplit/README.md, not a bug. "
              "Re-run with --attack-type smashed_poison to see TAS actually work.")


if __name__ == "__main__":
    main()
