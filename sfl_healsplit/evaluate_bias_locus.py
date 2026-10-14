"""
RQ1 diagnostic: does minority-class healing degradation originate in the
shared client-side representation, or in the healing mechanism itself?

Design (see project notes for the full reasoning):
  Phase 1 -- run ONE continuous training trajectory on the CIFAR
  automobile/truck/frog scarcity setup, and save the client model's
  weights at several checkpoint rounds along the way. Two modes:
    --phase1-mode plain (default) -- ordinary, undefended training. This
      isolates pure representation drift from imbalanced gradient descent,
      with no healing-mechanism effects feeding back into the representation.
    --phase1-mode defended -- the real HealSplit pipeline (detect + heal +
      train) runs every round, exactly as it would in a genuine deployment.
      This tests whether a representation that has itself been shaped by
      ongoing healing decisions (flagged samples dropped/substituted each
      round) shows the same locus-of-bias pattern as the plain trajectory,
      or a different one -- run both and compare.

  Phase 2 -- IDENTICAL regardless of phase1-mode, so the two runs stay
  directly comparable. At EACH checkpoint, reload the frozen weights fresh
  and repeat N times: collect a new batch of smashed data, run detection,
  train a brand-new GAN from scratch (fresh seed), and measure frog's share
  of the GAN's output plus its real accept/reject healing success rate. The
  representation never changes within a checkpoint's repeats -- only the
  freshly-collected batch and the GAN's own random training draw do.

Interpretation:
  - Tight spread WITHIN a checkpoint, but a big shift ACROSS checkpoints
    -> the representation is doing most of the work; retraining the
    healing mechanism can't recover what a degraded representation already
    lost.
  - Wide spread WITHIN a single checkpoint (representation held fixed)
    -> the healing mechanism's own training instability is a real,
    independent contributor, not just inheriting a bad representation.

Usage:
    python -m sfl_healsplit.evaluate_bias_locus --num-repeats 20 --phase1-mode plain
    python -m sfl_healsplit.evaluate_bias_locus --num-repeats 20 --phase1-mode defended --run-name rq1_bias_locus_defended
"""
import argparse
import copy
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
from .engine import collect_round_smashed_data, estimate_smashed_data_scale, train_one_round
from .gan import train_gan
from .model import ClientModel, build_split_model
from .recovery import semantically_consistent_substitution


def parse_args():
    p = argparse.ArgumentParser(description="RQ1: isolate representation-level vs "
                                              "mechanism-level minority-class bias.")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--num-clients", type=int, default=5)
    p.add_argument("--minority-keep-fraction", type=float, default=0.2)
    p.add_argument("--checkpoint-rounds", type=str, default="1,3,6,10,15,20",
                    help="Comma-separated rounds (within the phase-1 trajectory) "
                         "at which to freeze and save the client model's weights.")
    p.add_argument("--phase1-mode", type=str, default="plain", choices=["plain", "defended"],
                    help="plain: ordinary undefended training generates the checkpoints. "
                         "defended: the real detect+heal+train pipeline runs every round, "
                         "so the checkpointed representation has itself been shaped by "
                         "ongoing healing decisions. Phase 2's measurement is identical "
                         "either way, so the two modes stay directly comparable.")
    p.add_argument("--num-repeats", type=int, default=20,
                    help="How many independent healing-mechanism repeats to run "
                         "against EACH frozen checkpoint.")
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
    p.add_argument("--run-name", type=str, default="rq1_bias_locus")
    return p.parse_args()


def nearest_centroid(candidate: torch.Tensor, centroids: dict) -> int:
    best_class, best_dist = None, float("inf")
    for cls, centroid in centroids.items():
        dist = torch.norm(candidate.flatten() - centroid.flatten()).item()
        if dist < best_dist:
            best_dist = dist
            best_class = cls
    return best_class


def main():
    args = parse_args()
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = ["automobile", "truck", "frog"]
    frog_idx = class_names.index("frog")
    checkpoint_rounds = sorted(int(r) for r in args.checkpoint_rounds.split(","))
    max_round = max(checkpoint_rounds)

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
    # shuffle=True deliberately -- Phase 2 wants a genuinely fresh random batch
    # collected on each repeat, not a fixed aligned sample.
    collection_loaders = [
        DataLoader(Subset(train_ds, s.tolist()), batch_size=args.batch_size, shuffle=True)
        for s in shards
    ]

    client_model, server_model = build_split_model("cifar10", num_classes=num_classes)
    client_model.to(device)
    server_model.to(device)
    server_optimizer = torch.optim.SGD(server_model.parameters(), lr=args.lr_server)
    input_channels = client_model.conv1.in_channels
    _, z_shape = estimate_smashed_data_scale(client_model, client_loaders[0], device)

    # --- Phase 1: trajectory (plain or defended), saving frozen checkpoints along the way ---
    print(f"Phase 1 ({args.phase1_mode}): training to round {max_round}, "
          f"freezing checkpoints at {checkpoint_rounds}...")
    checkpoints = {}
    for round_num in range(1, max_round + 1):
        if args.phase1_mode == "defended":
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
        else:
            client_model = train_one_round(
                client_model, server_model, client_loaders, server_optimizer,
                device, local_epochs=1, lr_client=args.lr_client,
            )
        if round_num in checkpoint_rounds:
            checkpoints[round_num] = copy.deepcopy(client_model.state_dict())
            print(f"  checkpoint saved at round {round_num}")

    # --- Phase 2: repeated healing-mechanism trials against each frozen checkpoint ---
    run_dir = os.path.join(args.results_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    results_path = os.path.join(run_dir, f"{args.run_name}_metrics.csv")

    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["checkpoint_round", "repeat", "num_flagged", "num_clean_for_gan",
                          "pct_automobile", "pct_truck", "pct_frog",
                          "frog_flagged", "frog_accepted", "frog_success_rate"])

        for round_num in checkpoint_rounds:
            print(f"\nPhase 2: checkpoint round {round_num}, {args.num_repeats} repeats...")
            for repeat in range(1, args.num_repeats + 1):
                repeat_seed = args.seed * 10000 + round_num * 100 + repeat

                frozen_model = ClientModel(in_channels=input_channels).to(device)
                frozen_model.load_state_dict(checkpoints[round_num])
                frozen_model.eval()

                Z, y, _, _ = collect_round_smashed_data(
                    frozen_model, collection_loaders, device,
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

                if clean_z.size(0) < 10:
                    print(f"  repeat {repeat}: not enough clean data, skipping")
                    writer.writerow([round_num, repeat, int(flagged.sum().item()),
                                      int(clean_z.size(0)), None, None, None, 0, 0, None])
                    continue

                generator, _, _ = train_gan(clean_z, z_shape, device,
                                             num_epochs=args.gan_epochs, seed=repeat_seed)

                with torch.no_grad():
                    samples = generator.sample(args.num_gan_samples, device).cpu()
                nearest_counts = {c: 0 for c in centroids}
                for i in range(samples.size(0)):
                    cls = nearest_centroid(samples[i], centroids)
                    nearest_counts[cls] += 1
                pct = {c: 100 * nearest_counts[c] / args.num_gan_samples for c in centroids}

                validator = SmashedDataClassifier(in_channels=z_shape[0], num_classes=num_classes)
                validator = train_classifier(validator, clean_z, clean_y, device,
                                              num_epochs=args.validator_epochs)
                sub_result = semantically_consistent_substitution(
                    Z, y, flagged, generator, validator, device,
                    confidence_threshold=args.confidence_threshold,
                )
                frog_flagged_idx = torch.nonzero(flagged & (y == frog_idx), as_tuple=True)[0]
                frog_accepted = sum(1 for i in frog_flagged_idx if sub_result.keep_mask[i])
                frog_success_rate = (frog_accepted / len(frog_flagged_idx)
                                      if len(frog_flagged_idx) > 0 else None)

                writer.writerow([
                    round_num, repeat, int(flagged.sum().item()), int(clean_z.size(0)),
                    pct.get(0, 0.0), pct.get(1, 0.0), pct.get(frog_idx, 0.0),
                    len(frog_flagged_idx), frog_accepted, frog_success_rate,
                ])
                f.flush()
                print(f"  repeat {repeat}: frog_share={pct.get(frog_idx, 0.0):.1f}% "
                      f"frog_flagged={len(frog_flagged_idx)} frog_accepted={frog_accepted}")

    print(f"\nResults written to {results_path}")


if __name__ == "__main__":
    main()
