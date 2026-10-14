"""
Evaluates the full recovery pipeline (detect -> generate -> validate ->
substitute) by comparing a downstream classifier trained on three versions
of the same round's smashed data:

  (a) attacked   -- untouched, includes the SP-poisoned samples as-is
  (b) healed     -- TAS-flagged samples replaced via semantically_consistent_substitution
  (c) clean oracle -- the same samples with NO attack applied at all (upper bound)

If HealSplit's recovery mechanism works, (b) should land closer to (c) than
(a) does. This isn't the paper's own evaluation protocol (they measure
global-model accuracy across full training runs under attack, Table 2) --
it's a faster, more direct way to isolate whether *this specific pipeline
stage* (substitution) is doing what it's supposed to, before it's wired
into a full multi-round training loop.

Usage:
    python -m sfl_healsplit.evaluate_recovery
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Subset

from src.data import get_datasets, partition_indices
from src.seeding import set_global_seed

from .attacks import make_smashed_poison_deltas, select_malicious_clients
from .consistency_student import SmashedDataClassifier, train_classifier
from .detection import detect_poisoned
from .engine import collect_round_smashed_data, estimate_smashed_data_scale, train_one_round
from .gan import train_gan
from .model import build_split_model
from .recovery import semantically_consistent_substitution


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the GAN + validator recovery pipeline.")
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion-mnist", "cifar10"])
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--num-clients", type=int, default=5)
    p.add_argument("--malicious-ratio", type=float, default=0.2)
    p.add_argument("--smashed-poison-sigma", type=float, default=3.0)
    p.add_argument("--warmup-rounds", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr-client", type=float, default=0.05)
    p.add_argument("--lr-server", type=float, default=0.05)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--percentile", type=float, default=20.0)
    p.add_argument("--gan-epochs", type=int, default=100)
    p.add_argument("--confidence-threshold", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def eval_downstream_accuracy(Z_train, y_train, Z_test, y_test, in_channels, device, seed):
    """Trains a fresh SmashedDataClassifier on (Z_train, y_train), reports
    accuracy on (Z_test, y_test). A fresh model each time -- comparing three
    conditions fairly means none of them get a head start from a
    previously-trained model."""
    torch.manual_seed(seed)
    model = SmashedDataClassifier(in_channels=in_channels)
    model = train_classifier(model, Z_train, y_train, device, num_epochs=15, lr=0.005)
    model.eval()
    with torch.no_grad():
        preds = model(Z_test.to(device)).argmax(dim=1).cpu()
    return (preds == y_test).float().mean().item()


def main():
    args = parse_args()
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, _ = get_datasets(args.dataset, args.data_root)
    shards = partition_indices(len(train_ds), args.num_clients, True, args.seed)
    malicious_ids = select_malicious_clients(args.num_clients, args.malicious_ratio, args.seed)
    print(f"malicious_client_ids={sorted(malicious_ids)}")

    client_loaders = [
        DataLoader(Subset(train_ds, shard.tolist()), batch_size=args.batch_size, shuffle=True)
        for shard in shards
    ]
    # Separate, UNSHUFFLED loaders for the two collect_round_smashed_data
    # calls below (attacked vs. oracle) -- shuffle=True loaders reshuffle
    # independently on every iteration, so two calls against `client_loaders`
    # would NOT return samples in the same order, breaking the assumption
    # that Z_attacked[i] and Z_oracle[i] are the same underlying sample.
    # Caught during testing (see README): comparing misaligned "attacked"
    # and "oracle" tensors gave near-chance accuracy in ALL THREE
    # conditions, including the oracle upper bound, which should have been
    # trivial on this task -- that was the tell that something was
    # misaligned rather than the classifier failing to learn.
    collection_loaders = [
        DataLoader(Subset(train_ds, shard.tolist()), batch_size=args.batch_size, shuffle=False)
        for shard in shards
    ]

    client_model, server_model = build_split_model(args.dataset)
    client_model.to(device)
    server_model.to(device)
    server_optimizer = torch.optim.SGD(server_model.parameters(), lr=args.lr_server)

    z_std, z_shape = estimate_smashed_data_scale(client_model, client_loaders[0], device)
    attacked_deltas = make_smashed_poison_deltas(malicious_ids, z_shape, args.smashed_poison_sigma * z_std, args.seed)
    print(f"smashed-data scale: std={z_std:.4f} shape={z_shape}")

    print(f"Training {args.warmup_rounds} warm-up round(s) under attack...")
    for _ in range(args.warmup_rounds):
        client_model = train_one_round(
            client_model, server_model, client_loaders, server_optimizer, device,
            local_epochs=1, lr_client=args.lr_client, smashed_poison_deltas=attacked_deltas,
        )

    # Collect the SAME round's smashed data under two conditions: attacked (for
    # detection+healing to work on) and clean oracle (upper-bound comparison,
    # using the SAME warmed-up client_model so it isn't a confound -- only the
    # attack is toggled, not the model doing the forward pass).
    print("Collecting attacked and clean-oracle smashed data...")
    Z_attacked, y, client_ids, is_poisoned_gt = collect_round_smashed_data(
        client_model, collection_loaders, device, smashed_poison_deltas=attacked_deltas, max_batches_per_client=6,
    )
    Z_oracle, _, _, _ = collect_round_smashed_data(
        client_model, collection_loaders, device, smashed_poison_deltas=None, max_batches_per_client=6,
    )

    print(f"Collected {Z_attacked.shape[0]} samples, {is_poisoned_gt.sum().item()} actually poisoned.")

    # --- Detect (detect_poisoned flattens internally, so Z's natural shape is fine here) ---
    result = detect_poisoned(Z_attacked, k=args.k, alpha=args.alpha, percentile=args.percentile)
    flagged = result.predicted_poisoned
    tp = (flagged & is_poisoned_gt).sum().item()
    fp = (flagged & ~is_poisoned_gt).sum().item()
    fn = (~flagged & is_poisoned_gt).sum().item()
    print(f"TAS: flagged {flagged.sum().item()} of {len(flagged)} "
          f"(TP={tp} FP={fp} FN={fn} -- {is_poisoned_gt.sum().item()} truly poisoned)")

    # --- Generate: train GAN on the TAS-clean remainder ---
    clean_z_for_gan = Z_attacked[~flagged]
    print(f"Training GAN on {clean_z_for_gan.shape[0]} TAS-clean samples...")
    generator, discriminator, gan_log = train_gan(
        clean_z_for_gan, z_shape, device, num_epochs=args.gan_epochs, seed=args.seed,
    )
    print(f"GAN final epoch: d_real_acc={gan_log.d_real_acc[-1]:.3f} d_fake_acc={gan_log.d_fake_acc[-1]:.3f} "
          f"(both near 0.5 = discriminator can no longer tell real from fake -- good convergence sign)")

    # --- Validate: train the stand-in student on the TAS-clean remainder ---
    validator = SmashedDataClassifier(in_channels=z_shape[0])
    validator = train_classifier(validator, clean_z_for_gan, y[~flagged], device, num_epochs=20)

    # --- Substitute ---
    sub_result = semantically_consistent_substitution(
        Z_attacked, y, flagged, generator, validator, device,
        confidence_threshold=args.confidence_threshold,
    )
    print(f"Substitution: replaced={sub_result.num_replaced} dropped={sub_result.num_dropped} "
          f"unflagged={sub_result.num_unflagged}")

    # --- Compare downstream accuracy: attacked vs healed vs clean oracle ---
    # Held-out test split: last 20% of the collected samples (not part of any
    # training condition below), evaluated on the ORACLE (uncorrupted) version
    # of those samples -- we're testing whether each condition teaches a
    # classifier the TRUE underlying task, not whether it fits corrupted data.
    n = Z_attacked.size(0)
    n_test = max(1, n // 5)
    test_idx = torch.arange(n - n_test, n)
    train_idx = torch.arange(0, n - n_test)

    Z_test, y_test = Z_oracle[test_idx], y[test_idx]

    acc_attacked = eval_downstream_accuracy(
        Z_attacked[train_idx], y[train_idx], Z_test, y_test, z_shape[0], device, args.seed)
    acc_healed = eval_downstream_accuracy(
        sub_result.healed_z[train_idx][sub_result.keep_mask[train_idx]],
        y[train_idx][sub_result.keep_mask[train_idx]],
        Z_test, y_test, z_shape[0], device, args.seed)
    acc_oracle = eval_downstream_accuracy(
        Z_oracle[train_idx], y[train_idx], Z_test, y_test, z_shape[0], device, args.seed)

    print(f"\n--- Downstream classifier accuracy (held-out test set) ---")
    print(f"attacked (no defense):  {acc_attacked:.4f}")
    print(f"healed (HealSplit):     {acc_healed:.4f}")
    print(f"clean oracle (upper bound, no attack ever applied): {acc_oracle:.4f}")


if __name__ == "__main__":
    main()
