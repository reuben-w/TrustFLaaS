"""
Entry point for the Split FL track. Covers stages 1-5: label-flip/smashed-
poison attacks (stages 1-2), and optionally HealSplit's live detect+heal
defense (stages 3-5, via --defense).

Run with, e.g.:
    python -m sfl_healsplit.main --num-clients 5 --num-rounds 20 \
        --malicious-ratio 0.2 --label-flip-delta 1 --seed 42 \
        --run-name mnist_lp_attack

Run the clean (no-attack) counterpart for comparison with:
    python -m sfl_healsplit.main --no-attack --run-name mnist_clean

Run with the live defense active (stage 5) -- SP attack is the one TAS can
actually detect (see README's "LP-vs-SP finding"):
    python -m sfl_healsplit.main --attack-type smashed_poison --defense \
        --validator-mode standin --run-name mnist_sp_defended

    # Full stage-4 pipeline as the validator (slower, more faithful):
    python -m sfl_healsplit.main --attack-type smashed_poison --defense \
        --validator-mode distilled --run-name mnist_sp_defended_distilled

PAMAP2 (sensor time-series, added per supervisor request) skips the image
dataset path entirely -- see pamap2_data.py for the data pipeline and
model.py for the 1D-CNN client/server pair. --restrict-classes,
--minority-class and hybrid_dalse_bagan escalation are all no-ops for
--dataset pamap2 (a warning is printed if any are set), since PAMAP2's
minority class (rope jumping) is left naturally rare rather than
artificially restricted/subsampled. Example:
    python -m sfl_healsplit.main --dataset pamap2 --defense \
        --healing-mode dalse --run-name pamap2_dalse_deterministic
"""
import dataclasses
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# torch's CUDA initialization probes the deprecated `pynvml` package -- this
# is a heads-up for torch's own maintainers, not something wrong with this
# project. Same fix already applied to the FL+DP baseline project
# (src/dp_client.py); this track never got it since it doesn't use
# Flower/Ray, so the warning wasn't visible until now.
warnings.filterwarnings("ignore", message=".*pynvml package is deprecated.*")

import torch
from torch.utils.data import DataLoader

from src.data import build_restricted_dataset, get_datasets, partition_indices, DATASET_STATS
from src.seeding import set_global_seed

from .attacks import LabelFlipDataset, make_smashed_poison_deltas, select_malicious_clients
from .dirichlet_partition import dirichlet_partition_indices, summarize_partition
from .config import parse_config
from .defense import run_split_fl_with_defense
from .engine import estimate_smashed_data_scale, run_split_fl
from .model import build_split_model
from .pamap2_data import build_pamap2_datasets


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    cfg = parse_config()
    set_global_seed(cfg.seed)
    device = resolve_device(cfg.device)

    run_dir = os.path.join(cfg.results_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)

    hybrid_escalate_class_indices = None

    if cfg.dataset == "pamap2":
        # Separate pipeline entirely -- not a torchvision dataset, so none of
        # get_datasets/build_restricted_dataset/DATASET_STATS apply. Minority
        # class (rope jumping) is naturally rare, not restricted/subsampled,
        # so --restrict-classes/--minority-class/hybrid escalation are no-ops
        # here; warn rather than silently ignore if the user set any of them.
        if cfg.restrict_classes:
            print("[setup] WARNING: --dataset pamap2 ignores --restrict-classes -- "
                  "PAMAP2 always uses its full fixed 12-activity protocol set "
                  "(see pamap2_data.py::PROTOCOL_ACTIVITY_IDS).")
        if cfg.minority_class:
            print("[setup] WARNING: --dataset pamap2 ignores --minority-class -- "
                  "its minority class (rope jumping) is naturally rare, not "
                  "artificially subsampled (see pamap2_data.py).")
        if cfg.healing_mode == "hybrid_dalse_bagan":
            print("[setup] WARNING: --dataset pamap2 ignores hybrid_dalse_bagan escalation "
                  "(no --restrict-classes class names to resolve it against) -- "
                  "every class routes to DALSE, identical to healing_mode=dalse.")

        train_ds, test_ds, num_classes = build_pamap2_datasets(
            cfg.data_root, window_size=cfg.pamap2_window_size, stride=cfg.pamap2_stride,
        )
        print(f"[setup] pamap2: window_size={cfg.pamap2_window_size} stride={cfg.pamap2_stride} "
              f"(num_classes={num_classes}), train_size={len(train_ds)} test_size={len(test_ds)}")
    else:
        train_ds, test_ds = get_datasets(cfg.dataset, cfg.data_root)
        num_classes = 10
        if cfg.restrict_classes:
            class_names = [c.strip() for c in cfg.restrict_classes.split(",") if c.strip()]
            minority_names = (
                [m.strip() for m in cfg.minority_class.split(",") if m.strip()]
                if cfg.minority_class else None
            )
            train_ds, test_ds, num_classes = build_restricted_dataset(
                cfg.dataset, cfg.data_root, class_names,
                minority_class_names=minority_names,
                minority_keep_fraction=cfg.minority_keep_fraction,
                seed=cfg.seed,
            )
            print(f"[setup] restricted to classes={class_names} (num_classes={num_classes}), "
                  f"minority_class={cfg.minority_class or 'none'} "
                  f"keep_fraction={cfg.minority_keep_fraction}, "
                  f"train_size={len(train_ds)} test_size={len(test_ds)}")

            if cfg.healing_mode == "hybrid_dalse_bagan" and cfg.hybrid_escalate_classes:
                escalate_names = [c.strip() for c in cfg.hybrid_escalate_classes.split(",") if c.strip()]
                # Position within class_names IS the remapped index (0..num_classes-1) --
                # build_restricted_dataset remaps labels in exactly this order.
                hybrid_escalate_class_indices = {class_names.index(c) for c in escalate_names}
                print(f"[setup] hybrid_dalse_bagan: escalating classes={escalate_names} "
                      f"(indices={sorted(hybrid_escalate_class_indices)}) to BAGAN-GP, "
                      f"all others route to DALSE")
        elif cfg.healing_mode == "hybrid_dalse_bagan":
            print("[setup] WARNING: healing_mode=hybrid_dalse_bagan but --restrict-classes is empty -- "
                  "no class names to resolve escalation against, so no classes will be escalated "
                  "(every class routes to DALSE, identical to healing_mode=dalse).")

    if cfg.iid:
        shards = partition_indices(len(train_ds), cfg.num_clients, cfg.iid, cfg.seed)
    else:
        # Extracted via __getitem__ rather than assuming a .targets attribute --
        # train_ds may be a plain torchvision dataset, a RestrictedDataset
        # (build_restricted_dataset's remapped/subsampled wrapper), OR a
        # Pamap2WindowDataset, and this approach is correct for any of them
        # without needing to know which.
        train_labels = torch.tensor([train_ds[i][1] for i in range(len(train_ds))])
        shards = dirichlet_partition_indices(train_labels, cfg.num_clients, cfg.dirichlet_alpha, cfg.seed)
        print(f"[setup] non-IID partitioning: Dirichlet alpha={cfg.dirichlet_alpha}, "
              f"seed={cfg.seed}")
        print(summarize_partition(train_labels, shards, num_classes))

    malicious_ids = select_malicious_clients(cfg.num_clients, cfg.malicious_ratio, cfg.seed) \
        if cfg.attack else set()

    print(f"[setup] device={device} run_dir={run_dir}")
    print(f"[setup] malicious_client_ids={sorted(malicious_ids)} "
          f"({len(malicious_ids)}/{cfg.num_clients} clients, attack={cfg.attack}, "
          f"attack_type={cfg.attack_type if cfg.attack else 'n/a'})")

    client_loaders = []
    for cid, shard in enumerate(shards):
        subset = torch.utils.data.Subset(train_ds, shard.tolist())
        if cfg.attack and cfg.attack_type == "label_flip" and cid in malicious_ids:
            subset = LabelFlipDataset(subset, cfg.label_flip_delta, num_classes)
        client_loaders.append(DataLoader(subset, batch_size=cfg.batch_size, shuffle=True))

    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    client_model, server_model = build_split_model(
        cfg.dataset, num_classes=num_classes, window_size=cfg.pamap2_window_size
    )

    smashed_poison_deltas = None
    if cfg.attack and cfg.attack_type == "smashed_poison":
        z_std, z_shape = estimate_smashed_data_scale(client_model, client_loaders[0], device)
        smashed_poison_deltas = make_smashed_poison_deltas(
            malicious_ids, z_shape, cfg.smashed_poison_sigma * z_std, cfg.seed
        )
        print(f"[setup] smashed-data scale estimate: std={z_std:.4f}, shape={z_shape}")

    config_dict = dataclasses.asdict(cfg)
    config_dict["malicious_client_ids"] = sorted(malicious_ids)
    # Filenames include the run_name, not just the directory -- if these
    # files get downloaded/uploaded flat (out of their sfl_results/<run_name>/
    # subfolder), they still self-identify instead of colliding as generic
    # "config.json"/"metrics.csv" (this happened repeatedly -- see project history).
    config_path = os.path.join(run_dir, f"{cfg.run_name}_config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    results_path = os.path.join(run_dir, f"{cfg.run_name}_metrics.csv")
    if cfg.defense:
        run_split_fl_with_defense(
            client_model=client_model,
            server_model=server_model,
            client_loaders=client_loaders,
            test_loader=test_loader,
            device=device,
            num_rounds=cfg.num_rounds,
            local_epochs=cfg.local_epochs,
            lr_client=cfg.lr_client,
            lr_server=cfg.lr_server,
            results_path=results_path,
            num_clients=cfg.num_clients,
            num_classes=num_classes,
            smashed_poison_deltas=smashed_poison_deltas,
            heal_every_n_rounds=cfg.heal_every_n_rounds,
            validator_mode=cfg.validator_mode,
            detection_k=cfg.detection_k,
            detection_alpha=cfg.detection_alpha,
            detection_percentile=cfg.detection_percentile,
            detector=cfg.detector,
            client_mad_tau=cfg.detection_client_mad_tau,
            max_flagged_client_fraction=cfg.detection_max_flagged_client_fraction,
            flagged_client_action=cfg.flagged_client_action,
            gan_epochs=cfg.gan_epochs,
            validator_epochs=cfg.validator_epochs,
            confidence_threshold=cfg.confidence_threshold,
            ad_beta=cfg.ad_beta,
            ad_epochs=cfg.ad_epochs,
            distill_epochs=cfg.distill_epochs,
            distill_temperature=cfg.distill_temperature,
            healing_mode=cfg.healing_mode,
            dalse_beta=cfg.dalse_beta,
            dalse_decay_lambda=cfg.dalse_decay_lambda,
            dalse_eta=cfg.dalse_eta,
            dalse_min_exemplars_for_active=cfg.dalse_min_exemplars_for_active,
            dalse_noise_sigma_fraction=cfg.dalse_noise_sigma_fraction,
            dalse_feedback_confidence_threshold=cfg.dalse_feedback_confidence_threshold,
            dalse_feedback_weight=cfg.dalse_feedback_weight,
            dalse_feedback_max_share=cfg.dalse_feedback_max_share,
            dalse_feedback_buffer_size=cfg.dalse_feedback_buffer_size,
            dalse_feedback_success_rate_threshold=cfg.dalse_feedback_success_rate_threshold,
            hybrid_escalate_class_indices=hybrid_escalate_class_indices,
            cb_loss_beta=cfg.cb_loss_beta,
            la_loss_psi=cfg.la_loss_psi,
            la_counts=cfg.la_counts,
            ewc_lambda=cfg.ewc_lambda,
            ewc_start_round=cfg.ewc_start_round,
            ewc_fisher_num_batches=cfg.ewc_fisher_num_batches,
            contrastive_lambda=cfg.contrastive_lambda,
            contrastive_temperature=cfg.contrastive_temperature,
            contrastive_num_batches=cfg.contrastive_num_batches,
            bagan_ae_epochs=cfg.bagan_ae_epochs,
            max_batches_per_client_for_detection=(
                None if cfg.max_batches_per_client_for_detection == 0
                else cfg.max_batches_per_client_for_detection
            ),
            seed=cfg.seed,
        )
    else:
        run_split_fl(
            client_model=client_model,
            server_model=server_model,
            client_loaders=client_loaders,
            test_loader=test_loader,
            device=device,
            num_rounds=cfg.num_rounds,
            local_epochs=cfg.local_epochs,
            lr_client=cfg.lr_client,
            lr_server=cfg.lr_server,
            results_path=results_path,
            label_flip_delta=cfg.label_flip_delta if (cfg.attack and cfg.attack_type == "label_flip") else None,
            smashed_poison_deltas=smashed_poison_deltas,
            num_classes=num_classes,
        )

    print(f"[done] results written to {run_dir}")


if __name__ == "__main__":
    main()
