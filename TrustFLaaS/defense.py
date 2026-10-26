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
import time

from .attacks import add_smashed_poison
from .consistency_student import build_smashed_data_classifier, train_classifier
from .dalse_healing import DALSEHealer
from .detection import build_mutual_knn_graph, detect_poisoned
from .client_detection import client_shift_scores, flag_clients
from .distillation import CategoryHeadWrapper, train_consistency_student
from .engine import estimate_smashed_data_scale, evaluate, fedavg_state_dicts, train_one_round
from .gan import train_gan
from .model import ClientModel, ServerModel
from .recovery import bagan_gp_consistent_substitution, dalse_consistent_substitution, hybrid_consistent_substitution, semantically_consistent_substitution
from .ewc import EWCState
from .bagan_gp import train_bagan_gp
from .reporting import TransparencyLog
from .per_class_eval import PerClassLogger

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
    bagan_ae_epochs: int = 20,
    validator_epochs: int = 15,
    confidence_threshold: float = 0.7,
    validator_mode: str = "standin",   # "standin" (stage 3) or "distilled" (stage 4)
    ad_beta: float = 0.3,
    ad_epochs: int = 15,
    distill_epochs: int = 15,
    distill_temperature: float = 2.0,
    max_batches_per_client_for_detection: Optional[int] = DEFAULT_MAX_BATCHES_PER_CLIENT_FOR_DETECTION,
    healing_mode: str = "gan",         # "gan" (HealSplit's own mechanism) or "dalse" (see dalse_healing.py)
    dalse_healer: Optional[DALSEHealer] = None,  # REQUIRED if healing_mode=="dalse" OR
                                                   # "hybrid_dalse_bagan" -- must be created
                                                   # ONCE outside the round loop and passed in
                                                   # every round; see dalse_healing.py's module
                                                   # docstring for why.
    hybrid_escalate_class_indices=None,  # set/list of class indices routed to BAGAN-GP when
                                          # healing_mode=="hybrid_dalse_bagan"; every other
                                          # class routes to DALSE. Ignored for other healing_modes.
    cb_loss_beta: float = 0.0,  # Class-Balanced Loss (Cui et al., CVPR 2019) reweighting
                                 # strength for Phase C's training loss. 0.0 = disabled
                                 # (uniform weights, identical to plain CrossEntropyLoss);
                                 # typical enabled values are 0.9--0.9999 per the original
                                 # paper. Independent of healing_mode -- applies regardless
                                 # of which generator/healer produced this round's substitutes.
    la_counts: str = "full",
    la_loss_psi: float = 0.0,  # Logit-Adjusted Loss (MConFL Stage 1, Yuan et al. 2026)
                                # strength. 0.0 = disabled (zero logit offset, identical to
                                # plain loss). Independent of cb_loss_beta -- a DIFFERENT
                                # mechanism (shifts logits, not loss weight); see Phase C's
                                # own comment block for the full reasoning.
    ewc_state: "EWCState" = None,  # MConFL Stage 2 (Yuan et al. 2026): protects the CLIENT
                                # model's parameters across rounds. REQUIRED (non-None) if
                                # ewc_lambda>0.0 -- created ONCE outside the round loop and
                                # passed in every round, same persistence pattern as
                                # dalse_healer; see ewc.py's module docstring for why.
    ewc_start_round: int = 5,  # EWC's penalty only applies from this round onward -- per
                                # MConFL's own ablation, applying it from round 1 caused a
                                # catastrophic collapse in their experiments, not just
                                # underperformance. Ignored if ewc_state is None.
    ewc_fisher_num_batches: int = 5,  # how many of this round's raw (x, y) batches to use
                                # when estimating the Fisher information matrix each round
                                # EWC is active -- an estimate, not an exact computation,
                                # so kept small and bounded.
    detector: str = "tas",             # "tas" (HealSplit's detector, unchanged) or "client_shift"
                                        # (TrustFLaaS's detector, see client_detection.py).
    client_mad_tau: float = 6.0,       # client_shift only: flag a client scoring > median + tau*MAD.
    max_flagged_client_fraction: float = 0.4,  # client_shift only: safety ceiling on flagged clients.
    flagged_client_action: str = "heal",  # client_shift only: "heal" (DALSE/GAN replacement, server-only
                                        # training) or "drop" (flagged client's data is skipped).
    transparency_log=None,
    round_num: int = 0,  # the REAL round number, for transparency_log only -- deliberately
                         # separate from `seed` below: seed is called as seed+round_num from
                         # run_split_fl_with_defense, so reusing it here would log the wrong
                         # round number (this was a real bug, caught via write_round_report's
                         # own assertion failing when the two got conflated).
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
    if detector not in ("tas", "client_shift"):
        raise ValueError(f"unknown detector {detector!r}")
    client_mode = (detector == "client_shift")
    flagged_clients = set()
    client_scores = {}
    if client_mode:
        # Client-level detection: the verdict is computed on the (small) inspection pool, then applies to
        # EVERY batch of a flagged client -- this is what removes the 'uninspected batches' coverage gap.
        # `flagged` stays pool-level (it selects the clean pool for DALSE / the validator / the GAN);
        # everything downstream of it (stats, log, Phase C) uses the full-coverage arrays below.
        client_scores = client_shift_scores(Z_round, y_round, cid_round)
        flagged_clients = flag_clients(client_scores, tau=client_mad_tau,
                                       max_fraction=max_flagged_client_fraction)
        fc = torch.tensor(sorted(flagged_clients), dtype=torch.long)
        flagged = torch.isin(cid_round, fc)
        y_stat = torch.cat([yb.cpu() for (_, _, yb) in round_batches])
        cid_stat = torch.cat([torch.full((yb.size(0),), c, dtype=torch.long)
                              for (c, _, yb) in round_batches])
        flagged_stat = torch.isin(cid_stat, fc)
        # Phase C indexes healed_z / keep_mask via these ranges: one range per batch of a flagged client
        # into the "flagged clients' samples" arrays; None (= train as normal) for unflagged clients.
        detect_idx_by_batch = []
        ext_y_parts, pos = [], 0
        for (c, _, yb) in round_batches:
            if c in flagged_clients:
                n = yb.size(0)
                detect_idx_by_batch.append((pos, pos + n))
                ext_y_parts.append(yb.cpu())
                pos += n
            else:
                detect_idx_by_batch.append(None)
        y_ext = torch.cat(ext_y_parts) if ext_y_parts else torch.zeros(0, dtype=torch.long)
        n_ext = int(y_ext.numel())
    else:
        result = detect_poisoned(Z_round, k=detection_k, alpha=detection_alpha,
                                  percentile=detection_percentile)
        flagged = result.predicted_poisoned
        y_stat, cid_stat, flagged_stat = y_round, cid_round, flagged
    if transparency_log is not None:
        transparency_log.record_detection(round_num=round_num, y=y_stat, flagged=flagged_stat, client_ids=cid_stat)
    stats = {"num_total": len(flagged_stat), "num_flagged": int(flagged_stat.sum().item()),
             "num_replaced": 0, "num_dropped": 0, "healed": False}
    if client_mode:
        stats["client_scores"] = client_scores
        stats["flagged_clients"] = sorted(flagged_clients)

    if client_mode:
        keep_mask = torch.ones(n_ext, dtype=torch.bool)
        healed_z = torch.zeros(1, *z_shape).expand(n_ext, *z_shape)   # placeholder, no memory
    else:
        keep_mask = torch.ones(len(Z_round), dtype=torch.bool)
        healed_z = Z_round.clone()

    can_heal = flagged.any().item() and (~flagged).sum().item() >= MIN_CLEAN_FOR_HEALING
    if client_mode and flagged_client_action == "drop":
        can_heal = False
    stats["healing_mode"] = healing_mode
    stats["gen_train_time"] = 0.0
    stats["validator_train_time"] = 0.0
    stats["substitution_time"] = 0.0
    stats["total_attempts"] = 0

    if can_heal:
        clean_z = Z_round[~flagged]
        clean_y = y_round[~flagged]

        t0 = time.time()
        if healing_mode == "dalse":
            if dalse_healer is None:
                raise ValueError(
                    "healing_mode='dalse' requires a DALSEHealer instance passed as "
                    "dalse_healer=..., created ONCE outside the round loop -- see "
                    "dalse_healing.py's module docstring for why persistence matters here."
                )
            class_success_rates = None
            if transparency_log is not None:
                # Cumulative as of the END OF THE PREVIOUS round -- this
                # round's own outcome isn't known yet at this point (Phase B
                # hasn't run substitution yet), matching the same "historical,
                # not this-round" framing record_detection's own report uses.
                class_success_rates = {
                    c: transparency_log.cumulative_by_class[c].success_rate
                    for c in range(num_classes)
                }
            dalse_healer.update_and_project(clean_z, clean_y, class_success_rates=class_success_rates)
            generator = None  # no generator trained at all -- this IS the point of the comparison
        elif healing_mode == "hybrid_dalse_bagan":
            if dalse_healer is None:
                raise ValueError(
                    "healing_mode='hybrid_dalse_bagan' requires a DALSEHealer instance passed "
                    "as dalse_healer=..., same as healing_mode='dalse' -- see "
                    "dalse_healing.py's module docstring for why."
                )
            # BOTH mechanisms run every round: BAGAN-GP still trains on ALL
            # clean data (not just the escalated class) -- its whole benefit
            # comes from sharing structure across classes, so narrowing its
            # training data to just the escalated class would defeat that.
            # DALSE's update is identical to healing_mode="dalse" -- the
            # ROUTING happens only at substitution time, not here.
            class_success_rates = None
            if transparency_log is not None:
                class_success_rates = {
                    c: transparency_log.cumulative_by_class[c].success_rate
                    for c in range(num_classes)
                }
            dalse_healer.update_and_project(clean_z, clean_y, class_success_rates=class_success_rates)
            generator, _ = train_bagan_gp(clean_z, clean_y, z_shape, num_classes, device,
                               ae_epochs=bagan_ae_epochs, gan_epochs=gan_epochs, seed=seed)
        elif healing_mode == "bagan_gp":
            generator, _ = train_bagan_gp(clean_z, clean_y, z_shape, num_classes, device,
                               ae_epochs=bagan_ae_epochs, gan_epochs=gan_epochs, seed=seed)
        else:
            generator, _, _ = train_gan(clean_z, z_shape, device, num_epochs=gan_epochs, seed=seed)
        stats["gen_train_time"] = time.time() - t0

        t1 = time.time()
        if validator_mode == "distilled":
            R = build_mutual_knn_graph(Z_round, k=detection_k)
            y_a = flagged.long()
            ad_teacher = train_ad_teacher(
                Z_round, y_a, cid_round, y_round, R,
                num_clients=num_clients, num_classes=num_classes, device=device,
                num_epochs=ad_epochs, beta=ad_beta, seed=seed,
            )
            vanilla_teacher = build_smashed_data_classifier(z_shape, num_classes=num_classes)
            vanilla_teacher = train_classifier(vanilla_teacher, clean_z, clean_y, device,
                                                num_epochs=validator_epochs)
            student = train_consistency_student(
                Z_round, y_a, cid_round, y_round, vanilla_teacher, ad_teacher,
                num_clients=num_clients, num_classes=num_classes, device=device,
                num_epochs=distill_epochs, temperature=distill_temperature, seed=seed,
            )
            validator = CategoryHeadWrapper(student)
        else:
            validator = build_smashed_data_classifier(z_shape, num_classes=num_classes)
            validator = train_classifier(validator, clean_z, clean_y, device, num_epochs=validator_epochs)
        stats["validator_train_time"] = time.time() - t1

        t2 = time.time()
        if client_mode:
            # Candidates depend only on the label (and fresh noise), so a zero tensor stands in for the
            # flagged clients' original (untrusted) smashed data, which is never used.
            sub_Z = torch.zeros(n_ext, *z_shape)
            sub_y = y_ext
            sub_flag = torch.ones(n_ext, dtype=torch.bool)
        else:
            sub_Z, sub_y, sub_flag = Z_round, y_round, flagged
        if healing_mode == "dalse":
            sub_result = dalse_consistent_substitution(
                sub_Z, sub_y, sub_flag, dalse_healer, validator, device,
                confidence_threshold=confidence_threshold,
            )
        elif healing_mode == "hybrid_dalse_bagan":
            sub_result = hybrid_consistent_substitution(
                sub_Z, sub_y, sub_flag, generator, dalse_healer, validator, device,
                escalate_class_indices=hybrid_escalate_class_indices or set(),
                confidence_threshold=confidence_threshold,
            )
        elif healing_mode == "bagan_gp":
            sub_result = bagan_gp_consistent_substitution(sub_Z, sub_y, sub_flag, generator, validator, device,
                confidence_threshold=confidence_threshold,)
        else:
            sub_result = semantically_consistent_substitution(
                sub_Z, sub_y, sub_flag, generator, validator, device,
                confidence_threshold=confidence_threshold,
            )
        stats["substitution_time"] = time.time() - t2

        healed_z = sub_result.healed_z
        keep_mask = sub_result.keep_mask
        stats["num_replaced"] = sub_result.num_replaced
        stats["num_dropped"] = sub_result.num_dropped
        stats["total_attempts"] = sub_result.total_attempts
        stats["healed"] = True
    elif client_mode and n_ext > 0:
        # Flagged client(s) but healing was not possible (action="drop", or too little clean data to
        # build a generator/validator this round): drop their data rather than train on it unhealed.
        keep_mask = torch.zeros(n_ext, dtype=torch.bool)
    # else: flagged stays as detected but healed_z==Z_round (untouched) and
    # keep_mask==all-True -- Phase C's "healed" branch below still fires for
    # flagged samples in this fallback case, but with the ORIGINAL
    # (unhealed) z. This denies the malicious client a client-side gradient
    # update for those samples (a mild protective effect) even when there
    # wasn't enough clean data this round to actually synthesize a
    # replacement -- graceful degradation, not a special-cased branch.

    # Per-class breakdown -- computed from flagged/keep_mask/y_round, which
    # are valid regardless of whether healing actually ran above (in the
    # can_heal==False fallback, keep_mask stays all-True, so every flagged
    # sample counts as "replaced" there -- correct, since Phase C's "healed"
    # branch does treat those samples that way, just with unhealed z). This
    # is what a per-class investigation (e.g. "is a scarce class flagged
    # disproportionately to its share of the round") actually needs -- the
    # aggregate num_flagged/num_replaced/num_dropped counts alone can't
    # answer that, since they don't say WHICH classes those samples were.
    if client_mode:
        keep_stat = torch.ones(len(y_stat), dtype=torch.bool)
        if n_ext > 0:
            keep_stat[flagged_stat] = keep_mask      # same batch order as the flagged-clients arrays
        if not can_heal:
            stats["num_replaced"] = 0
            stats["num_dropped"] = n_ext
    else:
        keep_stat = keep_mask
    replaced_mask = flagged_stat & keep_stat
    dropped_mask = flagged_stat & (~keep_stat)
    stats["total_per_class"] = torch.bincount(y_stat, minlength=num_classes).tolist()
    stats["flagged_per_class"] = torch.bincount(y_stat[flagged_stat], minlength=num_classes).tolist()
    stats["replaced_per_class"] = torch.bincount(y_stat[replaced_mask], minlength=num_classes).tolist()
    stats["dropped_per_class"] = torch.bincount(y_stat[dropped_mask], minlength=num_classes).tolist()

    if transparency_log is not None:
        transparency_log.record_outcome(y=y_stat, flagged=flagged_stat, client_ids=cid_stat, keep_mask=keep_stat)

    if client_mode:
        # Phase C indexes `flagged` with the per-batch ranges built above, which address the flagged
        # clients' samples (all of which are flagged by definition).
        flagged = torch.ones(n_ext, dtype=torch.bool)

    # EWC's anchor/Fisher refresh happens HERE, using global_client_model in
    # its state as of the START of this round -- Phase A/B never modify
    # global_client_model itself, only read from it, so this is still valid.
    # Uses a PLAIN CrossEntropyLoss for Fisher estimation regardless of
    # whatever cb_loss_beta/la_loss_psi have set the real training criterion
    # to -- Fisher just needs a reasonable loss signal to estimate parameter
    # importance from, it doesn't need to exactly match the actual training
    # objective; a deliberate simplification, not an oversight.
    stats["ewc_fisher_mean"] = 0.0
    stats["ewc_anchor_distance"] = 0.0
    if ewc_state is not None and ewc_state.ewc_lambda > 0.0 and round_num >= ewc_start_round:
        fisher_samples = [(x, y) for (_, x, y) in round_batches[:ewc_fisher_num_batches]]
        ewc_log = ewc_state.refresh_anchor_and_fisher(
            global_client_model, server_model, fisher_samples, nn.CrossEntropyLoss(), device
        )
        stats["ewc_fisher_mean"] = ewc_log.fisher_mean
        stats["ewc_anchor_distance"] = ewc_log.anchor_distance

    # --- Phase C: train using the (possibly healed) round data ---
    # Two independently toggleable training-time interventions, both
    # deliberately scoped to THIS round's own total_per_class (already
    # computed above), not a fixed global class-count assumption -- matches
    # how everything else in this pipeline treats "this round's data" as the
    # operating unit, and avoids baking in an imbalance ratio that may not
    # match what a given round actually has to train on.
    #
    # Class-Balanced Loss (Cui et al., CVPR 2019): REWEIGHTS how much each
    # sample's loss counts, via each class's "effective number of samples."
    # cb_loss_beta=0.0 (default) -> uniform weights, no effect.
    #
    # Logit-Adjusted Loss (per MConFL's Stage 1, Yuan et al. 2026, itself
    # building on the logit-adjustment literature): a DIFFERENT mechanism --
    # shifts the LOGITS THEMSELVES before softmax, by psi*log(class_share),
    # rather than reweighting the loss magnitude. Forces the model to output
    # a larger raw signal for a rare class to compensate for the artificial
    # suppression, which the paper's own Theorem 2 shows expands that
    # class's gradient magnitude. la_loss_psi=0.0 (default) -> zero offset,
    # no effect. Deliberately independent of cb_loss_beta -- either can be
    # tested alone, matching how every other intervention in this project
    # has been isolated one variable at a time; the two CAN be combined
    # (mathematically compatible) but that's a second-order experiment, not
    # the default assumption.
    counts = torch.tensor(stats["total_per_class"], dtype=torch.float32, device=device)
    if la_counts == "kept":
        # Count only samples that actually train (round minus those the defence dropped), so the
        # prior matches the data the loss sees instead of the flagged client's discarded samples.
        counts = (counts - torch.tensor(stats["dropped_per_class"], dtype=torch.float32, device=device)).clamp(min=0)

    class_weights = None
    if cb_loss_beta > 0.0:
        beta_t = torch.tensor(cb_loss_beta, device=device)
        effective_num = torch.where(counts > 0, (1.0 - beta_t.pow(counts)) / (1.0 - beta_t),
                                     torch.ones_like(counts))
        class_weights = (1.0 - cb_loss_beta) / effective_num.clamp(min=1e-8)
        # Normalized so weights sum to num_classes (Cui et al.'s own convention).
        class_weights = class_weights * (num_classes / class_weights.sum())

    log_pi_offset = None
    if la_loss_psi > 0.0:
        pi = counts / counts.sum().clamp(min=1e-8)
        # counts[c]=0 -> pi=0 -> log_pi_offset=-inf for that class -- correct,
        # not a bug: exp(-inf + anything)=0 in the softmax denominator, and
        # since no sample this round has y==c there, nothing ever backprops
        # through that -inf entry. clamp still applied as a defensive floor.
        log_pi_offset = la_loss_psi * torch.log(pi.clamp(min=1e-8))

    def criterion(logits, y):
        adjusted = logits + log_pi_offset.unsqueeze(0) if log_pi_offset is not None else logits
        return nn.functional.cross_entropy(adjusted, y, weight=class_weights)

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
        delta_z = smashed_poison_deltas.get(cid)   # SP attack: malicious client's fixed perturbation

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
                    # BUGFIX: the attacker's smashed-data perturbation must also be applied on
                    # the TRAINING path (as engine.train_one_round does). Previously it was only
                    # applied in Phase A (detection snapshot), so SP never reached the server's
                    # training in defended runs. Healed z (below) is server-side only: unaffected.
                    if delta_z is not None:
                        z_n = add_smashed_poison(z_n, delta_z)
                    z_for_server = z_n.detach().requires_grad_(True)
                    logits = server_model(z_for_server)
                    loss = criterion(logits, y_n)
                    server_optimizer.zero_grad()
                    loss.backward()
                    server_optimizer.step()
                    client_optimizer.zero_grad()
                    z_n.backward(z_for_server.grad)
                    # EWC penalty, if active: a SECOND, separate backward() call,
                    # deliberately with NO zero_grad() in between -- PyTorch
                    # accumulates gradients across backward() calls by default,
                    # so this correctly SUMS with the task-loss gradient just
                    # computed above, matching MConFL's own additive loss
                    # formulation h_e(theta) = f_e(theta) + EWC_penalty(theta).
                    # Only ever applies here (the normal-samples branch) --
                    # the healed-samples branch below never takes a
                    # client_optimizer step at all, so an EWC penalty there
                    # would have nothing to attach to.
                    if ewc_state is not None and ewc_state.ewc_lambda > 0.0 and round_num >= ewc_start_round:
                        ewc_penalty = ewc_state.compute_penalty(client_model)
                        ewc_penalty.backward()
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
    bagan_ae_epochs: int = 20,
    validator_epochs: int = 15,
    confidence_threshold: float = 0.7,
    ad_beta: float = 0.3,
    ad_epochs: int = 15,
    distill_epochs: int = 15,
    distill_temperature: float = 2.0,
    max_batches_per_client_for_detection: Optional[int] = DEFAULT_MAX_BATCHES_PER_CLIENT_FOR_DETECTION,
    healing_mode: str = "gan",       # "gan" (HealSplit's own) or "dalse" (see dalse_healing.py)
    dalse_beta: float = 0.9,
    dalse_decay_lambda: float = 0.95,
    dalse_eta: float = 0.3,
    dalse_min_exemplars_for_active: int = 5,
    dalse_noise_sigma_fraction: float = 0.15,
    dalse_feedback_confidence_threshold: float = 0.85,
    dalse_feedback_weight: float = 0.5,
    dalse_feedback_max_share: float = 0.25,
    dalse_feedback_buffer_size: int = 20,
    dalse_feedback_success_rate_threshold: float = 0.5,
    hybrid_escalate_class_indices=None,  # set/list of class indices routed to BAGAN-GP when
                                          # healing_mode=="hybrid_dalse_bagan"
    cb_loss_beta: float = 0.0,  # Class-Balanced Loss strength; 0.0 = disabled. See
                                 # train_one_round_with_healing's docstring for details.
    la_counts: str = "full",
    la_loss_psi: float = 0.0,  # Logit-Adjusted Loss strength; 0.0 = disabled. See
                                # train_one_round_with_healing's docstring for details.
    ewc_lambda: float = 0.0,   # EWC penalty strength; 0.0 = disabled. See ewc.py's
                                # module docstring for the full mechanism.
    ewc_start_round: int = 5,  # round at which EWC's penalty begins applying.
    ewc_fisher_num_batches: int = 5,  # batches used to estimate Fisher, each active round.
    detector: str = "tas",
    client_mad_tau: float = 6.0,
    max_flagged_client_fraction: float = 0.4,
    flagged_client_action: str = "heal",
    seed: int = 0,
) -> None:
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    server_optimizer = torch.optim.SGD(server_model.parameters(), lr=lr_server)
    client_model.to(device)
    server_model.to(device)

    _, z_shape = estimate_smashed_data_scale(client_model, client_loaders[0], device)

    # Created ONCE, here, outside the round loop -- persistence across
    # healing rounds is the entire point of DALSE's drift-tracking design
    # (see dalse_healing.py's module docstring). Needed for "hybrid_dalse_bagan"
    # too, since non-escalated classes still route through it every round.
    # Only instantiated when actually needed, so the "gan"/"bagan_gp" paths'
    # memory/behavior is untouched.
    dalse_healer = (
        DALSEHealer(num_classes, z_shape, beta=dalse_beta, decay_lambda=dalse_decay_lambda,
                    eta=dalse_eta, min_exemplars_for_active=dalse_min_exemplars_for_active,
                    noise_sigma_fraction=dalse_noise_sigma_fraction,
                    feedback_confidence_threshold=dalse_feedback_confidence_threshold,
                    feedback_weight=dalse_feedback_weight,
                    feedback_max_share=dalse_feedback_max_share,
                    feedback_buffer_size=dalse_feedback_buffer_size,
                    feedback_success_rate_threshold=dalse_feedback_success_rate_threshold)
        if healing_mode in ("dalse", "hybrid_dalse_bagan") else None
    )

    transparency_log = TransparencyLog(num_classes, num_clients)
    ewc_state = EWCState(ewc_lambda=ewc_lambda, fisher_num_batches=ewc_fisher_num_batches) \
        if ewc_lambda > 0.0 else None

    per_class_cols = (
        [f"total_class{i}" for i in range(num_classes)]
        + [f"flagged_class{i}" for i in range(num_classes)]
        + [f"replaced_class{i}" for i in range(num_classes)]
        + [f"dropped_class{i}" for i in range(num_classes)]
    )
    zero_per_class = [0] * (num_classes * 4)
    per_class_logger = PerClassLogger(results_path, num_classes)
    client_log_path = os.path.splitext(results_path)[0] + "_client_detection.csv"
    if detector == "client_shift":
        with open(client_log_path, "w", newline="") as cf:
            csv.writer(cf).writerow(["round", "flagged_clients"] + [f"score_client{i}" for i in range(num_clients)])

    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["round", "test_loss", "test_accuracy", "healed_this_round",
             "num_flagged", "num_replaced", "num_dropped", "num_total",
             "healing_mode", "gen_train_time", "validator_train_time",
             "substitution_time", "total_attempts", "ewc_fisher_mean", "ewc_anchor_distance"]
            + per_class_cols
        )

        loss, acc, _ = evaluate(client_model, server_model, test_loader, device)
        writer.writerow([0, loss, acc, False, 0, 0, 0, 0, healing_mode, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0] + zero_per_class)
        f.flush()
        print(f"[Round 0] test_loss={loss:.4f} test_accuracy={acc:.4f}")
        per_class_logger.log(0, client_model, server_model, test_loader, device)

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
                    healing_mode=healing_mode, dalse_healer=dalse_healer,
                    bagan_ae_epochs=bagan_ae_epochs, transparency_log=transparency_log,
                    hybrid_escalate_class_indices=hybrid_escalate_class_indices,
                    cb_loss_beta=cb_loss_beta,
                    la_loss_psi=la_loss_psi, la_counts=la_counts,
                    ewc_state=ewc_state, ewc_start_round=ewc_start_round,
                    ewc_fisher_num_batches=ewc_fisher_num_batches,
                    detector=detector, client_mad_tau=client_mad_tau,
                    max_flagged_client_fraction=max_flagged_client_fraction,
                    flagged_client_action=flagged_client_action,
                    round_num=round_num, seed=seed + round_num,
                )
            else:
                client_model = train_one_round(
                    client_model, server_model, client_loaders, server_optimizer,
                    device, local_epochs, lr_client, smashed_poison_deltas=smashed_poison_deltas,
                )
                stats = {"num_flagged": 0, "num_replaced": 0, "num_dropped": 0,
                          "num_total": 0, "healed": False, "healing_mode": healing_mode,
                          "gen_train_time": 0.0, "validator_train_time": 0.0,
                          "substitution_time": 0.0, "total_attempts": 0,
                          "ewc_fisher_mean": 0.0, "ewc_anchor_distance": 0.0,
                          "total_per_class": [0] * num_classes,
                          "flagged_per_class": [0] * num_classes,
                          "replaced_per_class": [0] * num_classes,
                          "dropped_per_class": [0] * num_classes}

            loss, acc, _ = evaluate(client_model, server_model, test_loader, device)
            per_class_logger.log(round_num, client_model, server_model, test_loader, device,
                     final=(round_num == num_rounds))
            per_class_row = (
                stats["total_per_class"] + stats["flagged_per_class"]
                + stats["replaced_per_class"] + stats["dropped_per_class"]
            )
            writer.writerow([round_num, loss, acc, stats["healed"],
                              stats["num_flagged"], stats["num_replaced"],
                              stats["num_dropped"], stats["num_total"],
                              stats["healing_mode"], stats["gen_train_time"],
                              stats["validator_train_time"], stats["substitution_time"],
                              stats["total_attempts"], stats["ewc_fisher_mean"],
                              stats["ewc_anchor_distance"]]
                             + per_class_row)
            f.flush()
            if "client_scores" in stats:
                with open(client_log_path, "a", newline="") as cf:
                    csv.writer(cf).writerow(
                        [round_num, " ".join(str(c) for c in stats["flagged_clients"])]
                        + [round(stats["client_scores"].get(i, float("nan")), 4) for i in range(num_clients)])
                print(f"           detector=client_shift flagged_clients={stats['flagged_clients']} "
                      f"scores={ {k: round(v, 2) for k, v in stats['client_scores'].items()} }")
            print(f"[Round {round_num}] test_loss={loss:.4f} test_accuracy={acc:.4f} "
                  f"healed={stats['healed']} flagged={stats['num_flagged']} "
                  f"replaced={stats['num_replaced']} dropped={stats['num_dropped']} "
                  f"gen_time={stats['gen_train_time']:.2f}s "
                  f"sub_time={stats['substitution_time']:.2f}s "
                  f"attempts={stats['total_attempts']}")

            if transparency_log is not None and do_heal:
                transparency_log.write_round_report(os.path.dirname(results_path), round_num)
