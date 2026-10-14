"""
Stage 5: wires detection (TAS) + recovery (GAN + validator + substitution)
into the live, per-round training loop, so a full multi-round run produces
an accuracy-under-attack-WITH-defense trajectory -- comparable to the
paper's own Table 2 protocol, rather than the single-snapshot evaluation
scripts (evaluate_detection.py, evaluate_recovery.py, evaluate_distillation.py)
used to validate each piece individually in stages 2-4.

This is a SEPARATE module from engine.py, not a modification of it --
engine.py's train_one_round/run_split_fl are unchanged and still used
directly for the "no defense" and "between healing rounds" cases (see
`heal_every_n_rounds` below), so stages 1-4's verified behavior carries
over unmodified.

Design decision -- three-phase round structure:
engine.py's SFLV1 loop forwards and immediately backward+updates, batch by
batch, client by client -- there's no point where "the whole round's
smashed data" exists at once for detection to run on. This module
restructures a healing round into three phases:

  Phase A (forward-only, no_grad): every client's every batch is forwarded
    through the CURRENT global client model to produce z, with the same SP
    injection rule as engine.py. x and y are kept (not just z) so Phase C
    can re-forward WITH gradients later -- Phase A's z is only for
    detection, it can't be backpropped through (it was computed under
    no_grad).

  Phase B (detect + heal): TAS runs on the pooled round's smashed data.
    Flagged samples get GAN-generated replacements, gated by a validator
    (either stage 3's stand-in or stage 4's real distilled student --
    `validator_mode`), exactly as in evaluate_recovery.py /
    evaluate_distillation.py, just now happening automatically every
    healing round instead of as a one-off manual evaluation.

  Phase C (train): batches are split into three groups per Phase B's
    result -- normal (unflagged), healed (flagged + substitute accepted),
    dropped (flagged + substitute rejected after retries). Normal samples
    get the standard client-forward -> server-forward -> backward-both
    treatment (identical to engine.py). Healed samples get a
    server-only forward+backward using the GAN-generated z as an
    independent leaf tensor -- deliberately NOT connected to the client's
    computation graph, since that z did not come from the client's actual
    forward pass; backpropagating "the server's gradient for this
    synthetic sample" into the client's real weights doesn't correspond to
    anything physically meaningful. Dropped samples get no forward pass at
    all, contributing nothing to either model this round.

Consequence: a client whose entire batch gets flagged-and-healed this round
receives NO client_optimizer step, so its FedAvg contribution is just its
(unchanged) copy of the incoming global weights. This is intentional, not
an oversight -- a client with no admitted-normal samples this round has
contributed nothing directly trustworthy, and FedAvg-ing in its unchanged
weights is a mild protective behavior (it can't pull the aggregate in an
arbitrary direction, only dilute toward "no change") rather than a bug to
route around.

Design decision -- healing cadence and validator cost:
Running the FULL detect->GAN->validate pipeline (and especially the
distilled-student path's AD Teacher + Vanilla Teacher + distillation) every
single round of a multi-round experiment is expensive -- training four
separate models' worth of epochs, every round, for e.g. 20-50 rounds. Two
knobs manage this: `heal_every_n_rounds` (skip Phase A-C entirely on
non-healing rounds, falling back to engine.py's plain train_one_round) and
`validator_mode` ("standin" for stage 3's cheap plain classifier, or
"distilled" for the full stage-4 pipeline). The paper itself supports
periodic-not-continuous healing: "HealSplit synchronizes with SFL at
appropriate intervals, using only the smashed data from the current update
round to train the GAN" -- consistent with retraining fresh each healing
round rather than persisting/incrementally fine-tuning a generator or
validator across rounds, which is what this module does.
"""
import csv
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .ad_teacher import train_ad_teacher
from .attacks import add_smashed_poison
from .consistency_student import SmashedDataClassifier, train_classifier
from .detection import build_mutual_knn_graph, detect_poisoned
from .distillation import CategoryHeadWrapper, train_consistency_student
from .engine import estimate_smashed_data_scale, evaluate, fedavg_state_dicts, train_one_round
from .gan import train_gan
from .model import ClientModel, ServerModel
from .recovery import semantically_consistent_substitution

MIN_CLEAN_FOR_HEALING = 20  # need at least this many TAS-clean samples to
                            # train a meaningful GAN/validator this round;
                            # below this, healing is skipped for the round
                            # (see train_one_round_with_healing's fallback).

DEFAULT_MAX_BATCHES_PER_CLIENT_FOR_DETECTION = 4  # see module docstring,
                            # "Design decision -- capping the detection
                            # subsample, not the training data" below.


def train_one_round_with_healing(
    global_client_model: ClientModel,
    server_model: ServerModel,
    client_loaders: List[DataLoader],
    server_optimizer: torch.optim.Optimizer,
    device: torch.device,
    local_epochs: int,
    lr_client: float,
    z_shape: Tuple[int, ...],
    num_clients: int,
    num_classes: int,
    smashed_poison_deltas: Optional[Dict[int, torch.Tensor]] = None,
    detection_k: int = 10,
    detection_alpha: float = 0.5,
    detection_percentile: float = 20.0,
    gan_epochs: int = 40,
    validator_epochs: int = 15,
    confidence_threshold: float = 0.7,
    validator_mode: str = "standin",   # "standin" (stage 3) or "distilled" (stage 4)
    ad_beta: float = 0.3,
    ad_epochs: int = 15,
    distill_epochs: int = 15,
    distill_temperature: float = 2.0,
    max_batches_per_client_for_detection: Optional[int] = DEFAULT_MAX_BATCHES_PER_CLIENT_FOR_DETECTION,
    seed: int = 0,
) -> Tuple[ClientModel, dict]:
    """One healing round. Returns (updated global client model, stats dict
    for logging -- num_flagged/num_replaced/num_dropped/num_total).

    Design decision -- capping the detection subsample, not the training
    data, caught during testing on real (not sandbox-scale) MNIST:
    With no cap, Phase A pools an ENTIRE round's data -- on real MNIST with
    5 clients and full local batches, that's all 60,000 training samples in
    one round. TAS's KNN graph (detection.py) needs an O(K^2) pairwise
    distance matrix; the AD Teacher path's M_p (ad_teacher.py, Eq. 8) needs
    an O(K^3) matrix solve. At K=60,000 this is not just slow -- it's
    computationally infeasible (a 60,000x60,000 float32 matrix alone is
    ~14GB) and was observed to fail outright: a `torch.linalg.solve`
    "singular matrix" error in practice, and a reproducible out-of-memory
    kill in testing at K=20,000, a third of that scale. This was never
    exercised at this scale before -- every stage 2-4 validation script
    (evaluate_detection.py etc.) caps its own sample collection the same
    way this parameter does now; only the live per-round loop in this file
    was missing an equivalent cap.

    The fix caps ONLY the subsample that TAS/GAN/AD-Teacher/validator
    operate on -- every client still trains on the FULL local batch data
    it always did, so this doesn't change the fairness of an
    attacked-vs-defended comparison by shrinking how much data a defended
    run actually learns from. Any batch beyond the cap simply isn't
    inspected for poisoning this round -- the same treatment an unflagged
    (TAS-cleared) sample already gets. This is a real, deliberate
    trade-off: a bounded fraction of each round's data goes uninspected in
    exchange for the pipeline being computationally tractable at all. Set
    to None to restore the original (uncapped, unsafe-at-scale) behavior.
    """
    smashed_poison_deltas = smashed_poison_deltas or {}
    global_client_model.to(device).eval()

    # --- Phase A: forward-only collection, preserving (x, y) for Phase C's redo ---
    # round_batches: every batch, for full-data training in Phase C.
    # detect_idx_by_batch[i] is (start, end) into Z_round/flagged/etc. if
    # batch i was included in the (capped) detection subsample, else None.
    round_batches: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
    detect_idx_by_batch: List[Optional[Tuple[int, int]]] = []
    all_z, all_y, all_cid = [], [], []
    running_idx = 0
    with torch.no_grad():
        for cid, loader in enumerate(client_loaders):
            delta_z = smashed_poison_deltas.get(cid)
            for batch_idx, (x, y) in enumerate(loader):
                x, y = x.to(device), y.to(device)
                round_batches.append((cid, x, y))

                in_detection_pool = (
                    max_batches_per_client_for_detection is None
                    or batch_idx < max_batches_per_client_for_detection
                )
                if in_detection_pool:
                    z = global_client_model(x)
                    if delta_z is not None:
                        z = add_smashed_poison(z, delta_z)
                    all_z.append(z.cpu())
                    all_y.append(y.cpu())
                    all_cid.append(torch.full((y.size(0),), cid, dtype=torch.long))
                    n = y.size(0)
                    detect_idx_by_batch.append((running_idx, running_idx + n))
                    running_idx += n
                else:
                    detect_idx_by_batch.append(None)

    Z_round = torch.cat(all_z)
    y_round = torch.cat(all_y)
    cid_round = torch.cat(all_cid)

    # --- Phase B: detect + heal (operates only on the capped subsample) ---
    result = detect_poisoned(Z_round, k=detection_k, alpha=detection_alpha,
                              percentile=detection_percentile)
    flagged = result.predicted_poisoned
    stats = {"num_total": len(flagged), "num_flagged": int(flagged.sum().item()),
             "num_replaced": 0, "num_dropped": 0, "healed": False}

    keep_mask = torch.ones(len(Z_round), dtype=torch.bool)
    healed_z = Z_round.clone()

    can_heal = flagged.any().item() and (~flagged).sum().item() >= MIN_CLEAN_FOR_HEALING
    if can_heal:
        clean_z = Z_round[~flagged]
        clean_y = y_round[~flagged]
        generator, _, _ = train_gan(clean_z, z_shape, device, num_epochs=gan_epochs, seed=seed)

        if validator_mode == "distilled":
            R = build_mutual_knn_graph(Z_round, k=detection_k)
            y_a = flagged.long()
            ad_teacher = train_ad_teacher(
                Z_round, y_a, cid_round, y_round, R,
                num_clients=num_clients, num_classes=num_classes, device=device,
                num_epochs=ad_epochs, beta=ad_beta, seed=seed,
            )
            vanilla_teacher = SmashedDataClassifier(in_channels=z_shape[0], num_classes=num_classes)
            vanilla_teacher = train_classifier(vanilla_teacher, clean_z, clean_y, device,
                                                num_epochs=validator_epochs)
            student = train_consistency_student(
                Z_round, y_a, cid_round, y_round, vanilla_teacher, ad_teacher,
                num_clients=num_clients, num_classes=num_classes, device=device,
                num_epochs=distill_epochs, temperature=distill_temperature, seed=seed,
            )
            validator = CategoryHeadWrapper(student)
        else:
            validator = SmashedDataClassifier(in_channels=z_shape[0], num_classes=num_classes)
            validator = train_classifier(validator, clean_z, clean_y, device, num_epochs=validator_epochs)

        sub_result = semantically_consistent_substitution(
            Z_round, y_round, flagged, generator, validator, device,
            confidence_threshold=confidence_threshold,
        )
        healed_z = sub_result.healed_z
        keep_mask = sub_result.keep_mask
        stats["num_replaced"] = sub_result.num_replaced
        stats["num_dropped"] = sub_result.num_dropped
        stats["healed"] = True
    # else: flagged stays as detected but healed_z==Z_round (untouched) and
    # keep_mask==all-True -- Phase C's "healed" branch below still fires for
    # flagged samples in this fallback case, but with the ORIGINAL
    # (unhealed) z. This denies the malicious client a client-side gradient
    # update for those samples (a mild protective effect) even when there
    # wasn't enough clean data this round to actually synthesize a
    # replacement -- graceful degradation, not a special-cased branch.

    # --- Phase C: train using the (possibly healed) round data ---
    criterion = nn.CrossEntropyLoss()
    client_results: List[Tuple[Dict[str, torch.Tensor], int]] = []

    per_client_batches: Dict[int, List[Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[int, int]]]]] = {}
    for (cid, x, y), idx_range in zip(round_batches, detect_idx_by_batch):
        per_client_batches.setdefault(cid, []).append((x, y, idx_range))

    for cid in range(len(client_loaders)):
        if cid not in per_client_batches:
            continue
        client_model = type(global_client_model)(
            in_channels=global_client_model.conv1.in_channels
        ).to(device)
        client_model.load_state_dict(global_client_model.state_dict())
        client_optimizer = torch.optim.SGD(client_model.parameters(), lr=lr_client)
        num_examples = 0

        for epoch_idx in range(local_epochs):
            for x, y, idx_range in per_client_batches[cid]:
                if idx_range is None:
                    # Not in the detection subsample this round -- treat as
                    # entirely normal (same as an unflagged sample), full batch.
                    normal_mask = torch.ones(y.size(0), dtype=torch.bool)
                    healed_mask = torch.zeros(y.size(0), dtype=torch.bool)
                    batch_healed_z = None
                else:
                    start, end = idx_range
                    batch_keep = keep_mask[start:end]
                    batch_flagged = flagged[start:end]
                    batch_healed_z = healed_z[start:end]
                    normal_mask = batch_keep & (~batch_flagged)
                    healed_mask = batch_keep & batch_flagged
                    # dropped: ~batch_keep -- excluded entirely, no forward pass

                if normal_mask.any():
                    x_n, y_n = x[normal_mask], y[normal_mask]
                    z_n = client_model(x_n)
                    z_for_server = z_n.detach().requires_grad_(True)
                    logits = server_model(z_for_server)
                    loss = criterion(logits, y_n)
                    server_optimizer.zero_grad()
                    loss.backward()
                    server_optimizer.step()
                    client_optimizer.zero_grad()
                    z_n.backward(z_for_server.grad)
                    client_optimizer.step()
                    num_examples += x_n.size(0)

                if healed_mask.any():
                    z_h = batch_healed_z[healed_mask].to(device).clone().detach().requires_grad_(True)
                    y_h = y[healed_mask]
                    logits = server_model(z_h)
                    loss = criterion(logits, y_h)
                    server_optimizer.zero_grad()
                    loss.backward()
                    server_optimizer.step()
                    # No client_optimizer step -- see module docstring.
                    num_examples += y_h.size(0)

        client_results.append((client_model.state_dict(), max(num_examples, 1)))

    if client_results:
        new_client_state = fedavg_state_dicts(client_results)
        global_client_model.load_state_dict(new_client_state)

    return global_client_model, stats


def run_split_fl_with_defense(
    client_model: ClientModel,
    server_model: ServerModel,
    client_loaders: List[DataLoader],
    test_loader: DataLoader,
    device: torch.device,
    num_rounds: int,
    local_epochs: int,
    lr_client: float,
    lr_server: float,
    results_path: str,
    num_clients: int,
    num_classes: int = 10,
    smashed_poison_deltas: Optional[Dict[int, torch.Tensor]] = None,
    heal_every_n_rounds: int = 1,
    validator_mode: str = "standin",
    detection_k: int = 10,
    detection_alpha: float = 0.5,
    detection_percentile: float = 20.0,
    gan_epochs: int = 40,
    validator_epochs: int = 15,
    confidence_threshold: float = 0.7,
    ad_beta: float = 0.3,
    ad_epochs: int = 15,
    distill_epochs: int = 15,
    distill_temperature: float = 2.0,
    max_batches_per_client_for_detection: Optional[int] = DEFAULT_MAX_BATCHES_PER_CLIENT_FOR_DETECTION,
    seed: int = 0,
) -> None:
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    server_optimizer = torch.optim.SGD(server_model.parameters(), lr=lr_server)
    client_model.to(device)
    server_model.to(device)

    _, z_shape = estimate_smashed_data_scale(client_model, client_loaders[0], device)

    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "test_loss", "test_accuracy", "healed_this_round",
                          "num_flagged", "num_replaced", "num_dropped", "num_total"])

        loss, acc, _ = evaluate(client_model, server_model, test_loader, device)
        writer.writerow([0, loss, acc, False, 0, 0, 0, 0])
        f.flush()
        print(f"[Round 0] test_loss={loss:.4f} test_accuracy={acc:.4f}")

        for round_num in range(1, num_rounds + 1):
            do_heal = (round_num % heal_every_n_rounds == 0)

            if do_heal:
                client_model, stats = train_one_round_with_healing(
                    client_model, server_model, client_loaders, server_optimizer, device,
                    local_epochs, lr_client, z_shape, num_clients, num_classes,
                    smashed_poison_deltas=smashed_poison_deltas,
                    detection_k=detection_k, detection_alpha=detection_alpha,
                    detection_percentile=detection_percentile,
                    gan_epochs=gan_epochs, validator_epochs=validator_epochs,
                    confidence_threshold=confidence_threshold, validator_mode=validator_mode,
                    ad_beta=ad_beta, ad_epochs=ad_epochs, distill_epochs=distill_epochs,
                    distill_temperature=distill_temperature,
                    max_batches_per_client_for_detection=max_batches_per_client_for_detection,
                    seed=seed + round_num,
                )
            else:
                client_model = train_one_round(
                    client_model, server_model, client_loaders, server_optimizer,
                    device, local_epochs, lr_client, smashed_poison_deltas=smashed_poison_deltas,
                )
                stats = {"num_flagged": 0, "num_replaced": 0, "num_dropped": 0,
                          "num_total": 0, "healed": False}

            loss, acc, _ = evaluate(client_model, server_model, test_loader, device)
            writer.writerow([round_num, loss, acc, stats["healed"],
                              stats["num_flagged"], stats["num_replaced"],
                              stats["num_dropped"], stats["num_total"]])
            f.flush()
            print(f"[Round {round_num}] test_loss={loss:.4f} test_accuracy={acc:.4f} "
                  f"healed={stats['healed']} flagged={stats['num_flagged']} "
                  f"replaced={stats['num_replaced']} dropped={stats['num_dropped']}")
