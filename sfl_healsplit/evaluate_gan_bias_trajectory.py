"""
Extends evaluate_gan_bias.py's single-snapshot measurement into a full
trajectory across one continuous, real experiment.

evaluate_gan_bias.py compares two SEPARATELY-initialized runs at different
warmup depths (1 round vs. 10 rounds) -- useful, but it can't distinguish
"a more-trained model represents frog worse" from "this specific run
happened to land differently." This script settles that: it runs ONE
continuous N-round experiment (matching cifar_frog_gan's setup exactly) and
logs the nearest-centroid bias measurement at EVERY round, so the bias can
be plotted directly against round number from a single consistent source.

Design decision -- a "shadow" measurement GAN, not modifying defense.py:
Each round, this script does two things: (1) calls the REAL, validated
train_one_round_with_healing (unchanged) to advance the actual experiment,
exactly as a normal `main.py --defense` run would; (2) independently
collects that same round's data and trains a SEPARATE GAN purely for
measurement -- sampling from it and checking nearest-centroid, without
using its output for any real substitution. This roughly doubles per-round
GAN training cost (two GANs instead of one), but keeps the validated
pipeline completely untouched and guarantees the logged training
trajectory (accuracy) matches what a real experiment would actually do.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Subset

from src.data import build_restricted_dataset, partition_indices
from src.seeding import set_global_seed

from .consistency_student import SmashedDataClassifier, train_classifier
from .defense import train_one_round_with_healing
from .detection import detect_poisoned
from .engine import collect_round_smashed_data, estimate_smashed_data_scale, evaluate
from .gan import train_gan
from .model import build_split_model
from .recovery import semantically_consistent_substitution


def parse_args():
    p = argparse.ArgumentParser(description="Log GAN nearest-centroid bias at every round of "
                                              "one continuous real experiment.")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--num-clients", type=int, default=5)
    p.add_argument("--minority-keep-fraction", type=float, default=0.2)
    p.add_argument("--num-rounds", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr-client", type=float, default=0.05)
    p.add_argument("--lr-server", type=float, default=0.05)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--percentile", type=float, default=20.0)
    p.add_argument("--gan-epochs", type=int, default=40)
    p.add_argument("--validator-epochs", type=int, default=15)
    p.add_argument("--confidence-threshold", type=float, default=0.7)
    p.add_argument("--num-gan-samples", type=int, default=300)
    p.add_argument("--max-batches-per-client-for-detection", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", type=str, default="./cifar_results")
    p.add_argument("--run-name", type=str, default="gan_bias_trajectory")
    return p.parse_args()


def nearest_centroid(candidate: torch.Tensor, centroids: dict) -> int:
    best_class, best_dist = None, float("inf")
    for cls, centroid in centroids.items():
        dist = torch.norm(candidate.flatten() - centroid.flatten()).item()
        if dist < best_dist:
            best_dist = dist
            best_class = cls
    return best_class


def measure_bias_this_round(client_model, collection_loaders, device, args, num_classes,
                             class_names, frog_idx, round_seed):
    """Independent 'shadow' measurement -- collects this round's data fresh
    (using the CURRENT client_model), trains a separate measurement-only
    GAN, and returns the nearest-centroid percentages plus frog fidelity.
    Does not affect the real training happening this round at all."""
    Z, y, _, _ = collect_round_smashed_data(
        client_model, collection_loaders, device,
        max_batches_per_client=args.max_batches_per_client_for_detection,
    )
    result = detect_poisoned(Z, k=args.k, alpha=args.alpha, percentile=args.percentile)
    flagged = result.predicted_poisoned
    clean_z, clean_y = Z[~flagged], y[~flagged]

    centroids = {}
    for c in range(num_classes):
        mask = clean_y == c
        if mask.sum() > 0:
            centroids[c] = clean_z[mask].mean(dim=0)

    if clean_z.size(0) < 10:  # not enough to train a meaningful measurement GAN this round
        return None

    _, z_shape = estimate_smashed_data_scale(client_model, collection_loaders[0], device)
    generator, _, _ = train_gan(clean_z, z_shape, device, num_epochs=args.gan_epochs, seed=round_seed)

    with torch.no_grad():
        samples = generator.sample(args.num_gan_samples, device).cpu()
    nearest_counts = {c: 0 for c in centroids}
    for i in range(samples.size(0)):
        cls = nearest_centroid(samples[i], centroids)
        nearest_counts[cls] += 1
    pct = {class_names[c]: 100 * nearest_counts[c] / args.num_gan_samples for c in centroids}

    # Frog fidelity: independent validator + real substitution logic, measurement-only
    validator = SmashedDataClassifier(in_channels=z_shape[0], num_classes=num_classes)
    validator = train_classifier(validator, clean_z, clean_y, device, num_epochs=args.validator_epochs)
    sub_result = semantically_consistent_substitution(
        Z, y, flagged, generator, validator, device, confidence_threshold=args.confidence_threshold,
    )
    frog_flagged_idx = torch.nonzero(flagged & (y == frog_idx), as_tuple=True)[0]
    frog_accepted = sum(1 for i in frog_flagged_idx if sub_result.keep_mask[i])

    return {
        "num_flagged": int(flagged.sum().item()),
        "num_clean_for_gan": int(clean_z.size(0)),
        "pct_automobile": pct.get("automobile", 0.0),
        "pct_truck": pct.get("truck", 0.0),
        "pct_frog": pct.get("frog", 0.0),
        "frog_flagged": len(frog_flagged_idx),
        "frog_accepted": frog_accepted,
    }


def main():
    args = parse_args()
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = ["automobile", "truck", "frog"]
    frog_idx = class_names.index("frog")

    train_ds, test_ds, num_classes = build_restricted_dataset(
        "cifar10", args.data_root, class_names,
        minority_class_name="frog", minority_keep_fraction=args.minority_keep_fraction,
        seed=args.seed,
    )
    shards = partition_indices(len(train_ds), args.num_clients, True, args.seed)
    client_loaders = [
        DataLoader(Subset(train_ds, s.tolist()), batch_size=args.batch_size, shuffle=True)
        for s in shards
    ]
    collection_loaders = [
        DataLoader(Subset(train_ds, s.tolist()), batch_size=args.batch_size, shuffle=False)
        for s in shards
    ]
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    client_model, server_model = build_split_model("cifar10", num_classes=num_classes)
    client_model.to(device)
    server_model.to(device)
    server_optimizer = torch.optim.SGD(server_model.parameters(), lr=args.lr_server)
    _, z_shape = estimate_smashed_data_scale(client_model, client_loaders[0], device)

    run_dir = os.path.join(args.results_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    results_path = os.path.join(run_dir, f"{args.run_name}_bias_trajectory.csv")

    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "test_accuracy", "num_flagged", "num_clean_for_gan",
                          "pct_automobile", "pct_truck", "pct_frog",
                          "frog_flagged", "frog_accepted"])

        loss0, acc0 = evaluate(client_model, server_model, test_loader, device)[:2]
        writer.writerow([0, acc0, 0, 0, 0.0, 0.0, 0.0, 0, 0])
        f.flush()
        print(f"[Round 0] test_accuracy={acc0:.4f}")

        for round_num in range(1, args.num_rounds + 1):
            # --- Measurement (shadow GAN, this round's CURRENT client_model state) ---
            bias = measure_bias_this_round(
                client_model, collection_loaders, device, args, num_classes,
                class_names, frog_idx, round_seed=args.seed * 1000 + round_num,
            )

            # --- Real training (the actual, validated pipeline, advancing the experiment) ---
            client_model, _stats = train_one_round_with_healing(
                client_model, server_model, client_loaders, server_optimizer, device,
                local_epochs=1, lr_client=args.lr_client, z_shape=z_shape,
                num_clients=args.num_clients, num_classes=num_classes,
                detection_k=args.k, detection_alpha=args.alpha, detection_percentile=args.percentile,
                gan_epochs=args.gan_epochs, validator_epochs=args.validator_epochs,
                confidence_threshold=args.confidence_threshold, validator_mode="standin",
                healing_mode="gan",
                max_batches_per_client_for_detection=args.max_batches_per_client_for_detection,
                seed=args.seed + round_num,
            )

            loss, acc = evaluate(client_model, server_model, test_loader, device)[:2]

            if bias is None:
                row = [round_num, acc, 0, 0, 0.0, 0.0, 0.0, 0, 0]
                print(f"[Round {round_num}] test_accuracy={acc:.4f} "
                      f"(not enough clean data this round to run the bias measurement)")
            else:
                row = [round_num, acc, bias["num_flagged"], bias["num_clean_for_gan"],
                       bias["pct_automobile"], bias["pct_truck"], bias["pct_frog"],
                       bias["frog_flagged"], bias["frog_accepted"]]
                print(f"[Round {round_num}] test_accuracy={acc:.4f} "
                      f"frog_share={bias['pct_frog']:.1f}% "
                      f"(auto={bias['pct_automobile']:.1f}% truck={bias['pct_truck']:.1f}%) "
                      f"frog_flagged={bias['frog_flagged']} frog_accepted={bias['frog_accepted']}")
            writer.writerow(row)
            f.flush()

    print(f"\nResults written to {results_path}")


if __name__ == "__main__":
    main()
