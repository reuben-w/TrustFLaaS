"""
Evaluates the real consistency validation student (AD Teacher + Vanilla
Teacher + adversarial multi-teacher distillation) against stage 3's plain
stand-in, and confirms it plugs into recovery.py's gating function as a
drop-in replacement.

Usage:
    python -m sfl_healsplit.evaluate_distillation
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Subset

from src.data import get_datasets, partition_indices
from src.seeding import set_global_seed

from .ad_teacher import train_ad_teacher
from .attacks import make_smashed_poison_deltas, select_malicious_clients
from .consistency_student import SmashedDataClassifier, train_classifier
from .detection import build_mutual_knn_graph, detect_poisoned
from .distillation import CategoryHeadWrapper, train_consistency_student
from .engine import collect_round_smashed_data, estimate_smashed_data_scale, train_one_round
from .model import build_split_model
from .recovery import semantically_consistent_substitution
from .gan import train_gan


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the real (distilled) consistency validation student.")
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
    p.add_argument("--ad-beta", type=float, default=0.3)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def eval_category_accuracy(model_fn, Z_test, y_test, device):
    with torch.no_grad():
        logits = model_fn(Z_test.to(device))
        preds = logits.argmax(dim=1).cpu()
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
    # Unshuffled, for aligned collection -- same fix as evaluate_recovery.py's
    # stage-3 bug (see README "Verified before handoff, stage 3").
    collection_loaders = [
        DataLoader(Subset(train_ds, shard.tolist()), batch_size=args.batch_size, shuffle=False)
        for shard in shards
    ]

    client_model, server_model = build_split_model(args.dataset)
    client_model.to(device)
    server_model.to(device)
    server_optimizer = torch.optim.SGD(server_model.parameters(), lr=args.lr_server)

    z_std, z_shape = estimate_smashed_data_scale(client_model, client_loaders[0], device)
    deltas = make_smashed_poison_deltas(malicious_ids, z_shape, args.smashed_poison_sigma * z_std, args.seed)

    print(f"Training {args.warmup_rounds} warm-up round(s) under attack...")
    for _ in range(args.warmup_rounds):
        client_model = train_one_round(
            client_model, server_model, client_loaders, server_optimizer, device,
            local_epochs=1, lr_client=args.lr_client, smashed_poison_deltas=deltas,
        )

    print("Collecting smashed data...")
    Z, y_c, client_ids, is_poisoned_gt = collect_round_smashed_data(
        client_model, collection_loaders, device, smashed_poison_deltas=deltas, max_batches_per_client=8,
    )
    print(f"Collected {Z.shape[0]} samples, {is_poisoned_gt.sum().item()} truly poisoned.")

    # Train/test split -- fixed order (collection_loaders are unshuffled), so
    # this split is consistent across every tensor derived from Z below.
    n = Z.size(0)
    n_test = max(1, n // 5)
    train_idx = torch.arange(0, n - n_test)
    test_idx = torch.arange(n - n_test, n)

    Z_train, y_c_train, cid_train = Z[train_idx], y_c[train_idx], client_ids[train_idx]
    Z_test, y_c_test = Z[test_idx], y_c[test_idx]

    # --- Detect ---
    result = detect_poisoned(Z_train, k=args.k, alpha=args.alpha, percentile=args.percentile)
    flagged = result.predicted_poisoned
    y_a_train = flagged.long()  # task a's training target: TAS's OWN prediction,
                                 # not ground truth -- matches what's actually
                                 # available in a real deployment.
    print(f"TAS flagged {flagged.sum().item()}/{len(flagged)} in the training split.")

    # --- Stage 3 stand-in (Vanilla Teacher only, plain training) ---
    print("\nTraining stage-3 stand-in (Vanilla-only)...")
    standin = SmashedDataClassifier(in_channels=z_shape[0], num_classes=10)
    standin = train_classifier(standin, Z_train[~flagged], y_c_train[~flagged], device, num_epochs=20)
    acc_standin = eval_category_accuracy(standin, Z_test, y_c_test, device)

    # --- Stage 4: AD Teacher + Vanilla Teacher + real student ---
    print("Training AD Teacher...")
    R = build_mutual_knn_graph(Z_train, k=args.k)
    ad_teacher = train_ad_teacher(
        Z_train, y_a_train, cid_train, y_c_train, R,
        num_clients=args.num_clients, num_classes=10, device=device, beta=args.ad_beta,
    )
    acc_ad_c = eval_category_accuracy(lambda z: ad_teacher(z)[2], Z_test, y_c_test, device)

    print("Training Vanilla Teacher (identical procedure to stage 3's stand-in)...")
    vanilla_teacher = SmashedDataClassifier(in_channels=z_shape[0], num_classes=10)
    vanilla_teacher = train_classifier(vanilla_teacher, Z_train[~flagged], y_c_train[~flagged], device, num_epochs=20)

    print("Training the real student via adversarial multi-teacher distillation...")
    student = train_consistency_student(
        Z_train, y_a_train, cid_train, y_c_train, vanilla_teacher, ad_teacher,
        num_clients=args.num_clients, num_classes=10, device=device, temperature=args.temperature,
    )
    student_wrapped = CategoryHeadWrapper(student)
    acc_student = eval_category_accuracy(student_wrapped, Z_test, y_c_test, device)

    print(f"\n--- Category accuracy on held-out test split ---")
    print(f"stage-3 stand-in (Vanilla-only, plain training): {acc_standin:.4f}")
    print(f"AD Teacher alone:                                {acc_ad_c:.4f}")
    print(f"real student (distilled from both teachers):     {acc_student:.4f}")

    # --- Confirm plug-and-play with recovery.py's gate ---
    print("\nConfirming the real student plugs into recovery.py's gate...")
    clean_z_for_gan = Z_train[~flagged]
    generator, _, gan_log = train_gan(clean_z_for_gan, z_shape, device, num_epochs=80, seed=args.seed)
    sub_result_standin = semantically_consistent_substitution(
        Z_train, y_c_train, flagged, generator, standin, device)
    sub_result_student = semantically_consistent_substitution(
        Z_train, y_c_train, flagged, generator, student_wrapped, device)
    print(f"substitution with stand-in gate: replaced={sub_result_standin.num_replaced} "
          f"dropped={sub_result_standin.num_dropped}")
    print(f"substitution with real student gate: replaced={sub_result_student.num_replaced} "
          f"dropped={sub_result_student.num_dropped}")


if __name__ == "__main__":
    main()
