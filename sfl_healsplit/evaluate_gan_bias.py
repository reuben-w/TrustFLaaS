"""
Diagnostic for the supervisor's question: is GAN reconstruction of the
minority class (frog) being influenced/contaminated by the majority
classes (automobile, truck)?

We're operating in smashed-data space (intermediate activations), not
pixels -- there's no image to visually inspect. The direct quantitative
analogue: for each GAN-generated candidate, which class's TRUE centroid
(the mean of that round's actual clean smashed data for that class) is it
closest to? If frog-targeted candidates cluster nearer to automobile/truck
than to frog's own centroid, that's measurable evidence of contamination --
arguably more precise than an eyeballed image comparison would be anyway.

Two measurements:
1. Overall generative bias -- sample many candidates from the (unconditioned)
   GAN, check nearest-centroid for each. Tests whether the GAN's output
   DISTRIBUTION skews toward the majority classes' representation, independent
   of what the validator later accepts or rejects.
2. Accepted-candidate fidelity -- for candidates that actually passed the
   real accept/reject pipeline as frog replacements, how close are THEY to
   frog's true centroid specifically? Tests whether a "successful" healing
   is a genuinely good frog, or one that barely scraped past the validator
   while structurally resembling something else.

Usage:
    python -m sfl_healsplit.evaluate_gan_bias
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Subset

from src.data import build_restricted_dataset, partition_indices
from src.seeding import set_global_seed

from .consistency_student import SmashedDataClassifier, train_classifier
from .detection import detect_poisoned
from .engine import collect_round_smashed_data, estimate_smashed_data_scale, train_one_round
from .gan import train_gan
from .model import build_split_model
from .recovery import semantically_consistent_substitution


def parse_args():
    p = argparse.ArgumentParser(description="Check whether GAN reconstruction of a minority "
                                              "class is biased toward the majority classes.")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--num-clients", type=int, default=5)
    p.add_argument("--minority-keep-fraction", type=float, default=0.2)
    p.add_argument("--warmup-rounds", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr-client", type=float, default=0.05)
    p.add_argument("--lr-server", type=float, default=0.05)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--percentile", type=float, default=20.0)
    p.add_argument("--gan-epochs", type=int, default=40)
    p.add_argument("--validator-epochs", type=int, default=15)
    p.add_argument("--confidence-threshold", type=float, default=0.7)
    p.add_argument("--num-gan-samples", type=int, default=300,
                    help="How many candidates to sample from the GAN for the bias measurement.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", type=str, default="./cifar_results",
                    help="Matches main.py's convention -- results are written to "
                         "<results-dir>/<run-name>/<run-name>_gan_bias.json")
    p.add_argument("--run-name", type=str, default="gan_bias_check",
                    help="Used to build a self-identifying output filename, same convention "
                         "as main.py -- avoids the generic-filename collisions that have bitten "
                         "this project before.")
    return p.parse_args()


def nearest_centroid(candidate: torch.Tensor, centroids: dict) -> int:
    """Returns the class index of centroids whose centroid is closest (L2) to candidate."""
    best_class, best_dist = None, float("inf")
    for cls, centroid in centroids.items():
        dist = torch.norm(candidate.flatten() - centroid.flatten()).item()
        if dist < best_dist:
            best_dist = dist
            best_class = cls
    return best_class, best_dist


def main():
    args = parse_args()
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {"config": vars(args)}

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

    client_model, server_model = build_split_model("cifar10", num_classes=num_classes)
    client_model.to(device)
    server_model.to(device)
    server_optimizer = torch.optim.SGD(server_model.parameters(), lr=args.lr_server)

    print(f"Training {args.warmup_rounds} warm-up round(s), clean (no attack) -- matching "
          f"the cifar_frog_* experiments...")
    for _ in range(args.warmup_rounds):
        client_model = train_one_round(
            client_model, server_model, client_loaders, server_optimizer, device,
            local_epochs=1, lr_client=args.lr_client,
        )

    print("Collecting a round's smashed data...")
    Z, y, client_ids, _ = collect_round_smashed_data(
        client_model, collection_loaders, device, max_batches_per_client=4,
    )
    _, z_shape = estimate_smashed_data_scale(client_model, client_loaders[0], device)

    result = detect_poisoned(Z, k=args.k, alpha=args.alpha, percentile=args.percentile)
    flagged = result.predicted_poisoned
    clean_z, clean_y = Z[~flagged], y[~flagged]
    print(f"TAS flagged {flagged.sum().item()}/{len(flagged)}; "
          f"{(~flagged).sum().item()} clean samples available to train the GAN on.")

    # --- Ground truth: this round's TRUE per-class centroids, from clean data ---
    centroids = {}
    for c in range(num_classes):
        mask = clean_y == c
        if mask.sum() > 0:
            centroids[c] = clean_z[mask].mean(dim=0)
    print(f"True clean centroids computed for classes: "
          f"{[class_names[c] for c in centroids]}")

    print(f"\nTraining GAN on {clean_z.size(0)} TAS-clean samples...")
    generator, _, gan_log = train_gan(clean_z, z_shape, device, num_epochs=args.gan_epochs, seed=args.seed)
    print(f"GAN final epoch: d_real_acc={gan_log.d_real_acc[-1]:.3f} d_fake_acc={gan_log.d_fake_acc[-1]:.3f}")
    results["num_flagged"] = int(flagged.sum().item())
    results["num_clean_for_gan"] = int(clean_z.size(0))
    results["gan_final_d_real_acc"] = gan_log.d_real_acc[-1]
    results["gan_final_d_fake_acc"] = gan_log.d_fake_acc[-1]

    # --- Measurement 1: overall generative bias ---
    print(f"\n--- Measurement 1: nearest-centroid distribution over {args.num_gan_samples} "
          f"unconditioned GAN samples ---")
    with torch.no_grad():
        samples = generator.sample(args.num_gan_samples, device).cpu()
    nearest_counts = {c: 0 for c in centroids}
    for i in range(samples.size(0)):
        cls, _ = nearest_centroid(samples[i], centroids)
        nearest_counts[cls] += 1
    results["measurement1_nearest_centroid_pct"] = {}
    for c, name in enumerate(class_names):
        if c in nearest_counts:
            pct = 100 * nearest_counts[c] / args.num_gan_samples
            print(f"  {name}: {nearest_counts[c]}/{args.num_gan_samples} ({pct:.1f}%) of GAN "
                  f"samples land nearest to this class's true centroid")
            results["measurement1_nearest_centroid_pct"][name] = pct

    # --- Measurement 2: fidelity of ACCEPTED frog substitutions specifically ---
    print(f"\n--- Measurement 2: how close are ACCEPTED frog substitutions to frog's true centroid, "
          f"vs. automobile/truck's? ---")
    validator = SmashedDataClassifier(in_channels=z_shape[0], num_classes=num_classes)
    validator = train_classifier(validator, clean_z, clean_y, device, num_epochs=args.validator_epochs)

    sub_result = semantically_consistent_substitution(
        Z, y, flagged, generator, validator, device, confidence_threshold=args.confidence_threshold,
    )
    frog_flagged_idx = torch.nonzero(flagged & (y == frog_idx), as_tuple=True)[0]
    frog_accepted_idx = [i.item() for i in frog_flagged_idx if sub_result.keep_mask[i]]
    print(f"Frog samples flagged: {len(frog_flagged_idx)}, accepted (successfully healed): "
          f"{len(frog_accepted_idx)}")
    results["frog_flagged"] = len(frog_flagged_idx)
    results["frog_accepted"] = len(frog_accepted_idx)

    if frog_accepted_idx and frog_idx in centroids:
        dists_to_frog, dists_to_others = [], []
        for i in frog_accepted_idx:
            candidate = sub_result.healed_z[i]
            for c, centroid in centroids.items():
                d = torch.norm(candidate.flatten() - centroid.flatten()).item()
                if c == frog_idx:
                    dists_to_frog.append(d)
                else:
                    dists_to_others.append(d)
        avg_frog_dist = sum(dists_to_frog) / len(dists_to_frog)
        avg_other_dist = sum(dists_to_others) / len(dists_to_others) if dists_to_others else float("nan")
        print(f"  Average distance to frog's own centroid:        {avg_frog_dist:.4f}")
        print(f"  Average distance to automobile/truck centroids: {avg_other_dist:.4f}")
        results["measurement2_avg_dist_to_frog_centroid"] = avg_frog_dist
        results["measurement2_avg_dist_to_other_centroids"] = avg_other_dist
        if avg_frog_dist < avg_other_dist:
            print("  -> Accepted frog candidates are, on average, closer to frog's own centroid "
                  "than to the majority classes' -- no strong evidence of contamination in the "
                  "ACCEPTED subset specifically.")
            results["measurement2_verdict"] = "no_strong_contamination_in_accepted_subset"
        else:
            print("  -> Accepted frog candidates are, on average, CLOSER to automobile/truck's "
                  "centroid than to frog's own -- direct evidence the validator is accepting "
                  "candidates that structurally resemble the majority classes.")
            results["measurement2_verdict"] = "contamination_detected_in_accepted_subset"
    else:
        print("  No frog samples were successfully healed this round -- cannot measure fidelity "
              "of an empty set. (This itself may be informative, depending on how often it happens.)")
        results["measurement2_verdict"] = "no_accepted_frog_samples_this_round"

    run_dir = os.path.join(args.results_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, f"{args.run_name}_gan_bias.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
