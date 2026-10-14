"""
Config for the Split FL / HealSplit extension track. Mirrors the pattern in
src/config.py (dataclass + argparse, every hyperparameter a CLI flag, full
config snapshot written to config.json per run) for consistency with the
FL+DP baseline, but is deliberately a separate config -- this track has a
different, non-overlapping set of concerns (split point, attack params, no
DP fields) and keeping it separate avoids a single bloated config class
trying to serve two different experiments.

PAMAP2 (sensor time-series) fields added per supervisor request to test
generalization beyond image datasets -- see sfl_healsplit/pamap2_data.py for
the data pipeline and module docstring for the design decisions (naturally
rare "rope jumping" minority class, subject-holdout split, channel choice).
--restrict-classes / --minority-class / --minority-keep-fraction are all
no-ops for --dataset pamap2 (documented on each flag below) -- PAMAP2's
minority class is left naturally rare rather than artificially subsampled.
"""
import argparse
import dataclasses


@dataclasses.dataclass
class SFLConfig:
    # --- Split FL setup ---
    num_clients: int = 5
    num_rounds: int = 10
    local_epochs: int = 1
    batch_size: int = 32
    lr_client: float = 0.05
    lr_server: float = 0.05

    # --- Data ---
    dataset: str = "mnist"
    iid: bool = True
    dirichlet_alpha: float = 0.5      # non-IID skew, used only when --non-iid is set. Lower
                                       # = more skewed client class distributions; higher
                                       # converges toward the IID case. 0.5 chosen as a
                                       # moderate first test -- real, meaningful skew without
                                       # the more extreme edge cases (e.g. a client landing on
                                       # exactly zero samples of a class) that lower values
                                       # like 0.1 can produce.
    data_root: str = "./data"
    restrict_classes: str = ""       # comma-separated class names, e.g. "automobile,truck,frog" (empty = no restriction)
    minority_class: str = ""         # which of restrict_classes gets subsampled
    minority_keep_fraction: float = 1.0  # e.g. 0.2 = keep 20% of the minority class's original count

    # --- Label-flipping attack (HealSplit threat model A1) ---
    attack: bool = True
    attack_type: str = "label_flip"   # "label_flip" (A1) or "smashed_poison" (A3)
    malicious_ratio: float = 0.2      # matches HealSplit's Table 1 default (20%)
    label_flip_delta: int = 1         # y' = (y + delta) mod C
    smashed_poison_sigma: float = 3.0  # z' = z + sigma * std(z) * noise, for attack_type="smashed_poison"

    # --- HealSplit defense (stage 5: live detection + recovery) ---
    defense: bool = False
    heal_every_n_rounds: int = 1
    validator_mode: str = "standin"      # "standin" (stage 3, cheap) or "distilled" (stage 4, full pipeline)
    detection_k: int = 10
    detection_alpha: float = 0.5
    detection_percentile: float = 20.0
    detector: str = "tas"              # "tas" (HealSplit's detector, unchanged) or "client_shift"
                                        # (TrustFLS: per-client shift check, see client_detection.py)
    detection_client_mad_tau: float = 6.0   # client_shift: flag a client scoring > median + tau * MAD
    detection_max_flagged_client_fraction: float = 0.4  # client_shift: ceiling, not a target
    flagged_client_action: str = "heal"     # client_shift: "heal" or "drop" a flagged client's data
    gan_epochs: int = 40
    validator_epochs: int = 15
    confidence_threshold: float = 0.7
    ad_beta: float = 0.3
    ad_epochs: int = 15
    distill_epochs: int = 15
    distill_temperature: float = 2.0
    healing_mode: str = "gan"          # "gan" (HealSplit's own), "dalse", "bagan_gp", or
                                        # "hybrid_dalse_bagan" (see hybrid_escalate_classes below)
    hybrid_escalate_classes: str = ""  # comma-separated class NAMES (matching --restrict-classes'
                                        # own naming) routed to BAGAN-GP when
                                        # healing_mode="hybrid_dalse_bagan"; every other class
                                        # routes to DALSE. Ignored for other healing_modes.
    cb_loss_beta: float = 0.0          # Class-Balanced Loss (Cui et al., CVPR 2019) reweighting
                                        # strength for the Phase C training loss. 0.0 = disabled
                                        # (uniform weights). Independent of healing_mode -- a
                                        # training-time fix, not a detect/heal-pipeline one.
    la_counts: str = "full"            # class counts behind the LA prior: "full" = every sample in the round
                                        # (original), "kept" = round minus samples dropped by the defence
    la_loss_psi: float = 0.0           # Logit-Adjusted Loss (MConFL Stage 1, Yuan et al. 2026)
                                        # strength for Phase C. 0.0 = disabled. A DIFFERENT
                                        # mechanism from cb_loss_beta (shifts logits, not loss
                                        # weight) -- independently toggleable, can combine with it.
    ewc_lambda: float = 0.0            # Elastic Weight Consolidation (MConFL Stage 2) penalty
                                        # strength, protecting the CLIENT model's parameters
                                        # across rounds. 0.0 = disabled. See ewc.py.
    ewc_start_round: int = 5           # round at which EWC's penalty begins applying -- NOT
                                        # optional if ewc_lambda>0.0: applying it from round 1
                                        # caused a catastrophic collapse in the source paper's
                                        # own ablation, not just underperformance.
    ewc_fisher_num_batches: int = 5    # batches used to estimate the Fisher information
                                        # matrix, each round EWC is active.
    contrastive_lambda: float = 0.0    # FedIIC-adapted supervised contrastive refinement
                                        # strength (see contrastive.py). 0.0 = disabled. A
                                        # THIRD kind of intervention, distinct from loss-
                                        # reweighting (cb_loss_beta/la_loss_psi) and
                                        # parameter-protection (ewc_lambda) -- shapes the
                                        # representation directly, pooling smashed data
                                        # across ALL clients each round.
    contrastive_temperature: float = 0.5  # SupCon temperature; lower = sharper separation.
    contrastive_num_batches: int = 5   # pooled cross-client batches to refine on, each round.
    dalse_beta: float = 0.9            # EMA weight for active classes' drift (DALSE's Eq. 2)
    dalse_decay_lambda: float = 0.95   # decay for inactive classes' drift (DALSE's Eq. 2)
    dalse_eta: float = 0.3             # sensitivity to prolonged class inactivity (DALSE's Eq. 4)
    dalse_min_exemplars_for_active: int = 5  # min clean exemplars this round to count a class "active"
    dalse_noise_sigma_fraction: float = 0.15  # fraction of this round's clean-data spread used as
                                               # Gaussian noise on each DALSE candidate; 0.0 = fully
                                               # deterministic (original DALSE-faithful behavior)
    dalse_feedback_confidence_threshold: float = 0.85  # stricter bar than confidence_threshold --
                                               # only a healed sample clearing THIS gets remembered
    dalse_feedback_weight: float = 0.5        # a stored healed sample counts as this fraction of
                                               # one real clean sample when re-estimating a centroid
    dalse_feedback_max_share: float = 0.25    # hard cap: healed memory can never supply more than
                                               # this fraction of a class's centroid evidence, any round
    dalse_feedback_buffer_size: int = 20      # max stored healed samples per class (FIFO, oldest evicted)
    dalse_feedback_success_rate_threshold: float = 0.5  # only apply feedback to a class whose
                                               # historical cumulative success rate (from the
                                               # TransparencyLog) is below this, or has no
                                               # history yet -- classes already succeeding
                                               # reliably are left untouched
    bagan_ae_epochs: int = 20                 # autoencoder pretraining epochs, healing_mode="bagan_gp" only
    max_batches_per_client_for_detection: int = 4  # 0 = uncapped (unsafe at real FL round sizes, see defense.py)

    # --- PAMAP2 (sensor time-series) ---
    pamap2_window_size: int = 100      # samples per window (1.0s @ 100Hz). See pamap2_data.py.
    pamap2_stride: int = 50            # window stride (50% overlap). See pamap2_data.py.

    # --- Reproducibility / infra ---
    seed: int = 42
    device: str = "auto"
    results_dir: str = "./sfl_results"
    run_name: str = "run"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Split Federated Learning baseline with a label-flipping attack "
                     "(HealSplit extension track, stage 1)."
    )
    d = SFLConfig()

    sfl = p.add_argument_group("Split FL")
    sfl.add_argument("--num-clients", type=int, default=d.num_clients)
    sfl.add_argument("--num-rounds", type=int, default=d.num_rounds)
    sfl.add_argument("--local-epochs", type=int, default=d.local_epochs)
    sfl.add_argument("--batch-size", type=int, default=d.batch_size)
    sfl.add_argument("--lr-client", type=float, default=d.lr_client)
    sfl.add_argument("--lr-server", type=float, default=d.lr_server)

    data = p.add_argument_group("Data")
    data.add_argument("--dataset", type=str, default=d.dataset,
                       choices=["mnist", "fashion-mnist", "cifar10", "pamap2"])
    data.add_argument("--iid", dest="iid", action="store_true", default=d.iid)
    data.add_argument("--non-iid", dest="iid", action="store_false",
                       help="Partitions client data via a Dirichlet(alpha) distribution "
                            "instead of uniformly -- see --dirichlet-alpha. Implemented "
                            "specifically for this track (src/data.py's own partition_indices, "
                            "used for --iid, still raises NotImplementedError for non-IID -- "
                            "this flag routes to a separate implementation instead, see "
                            "dirichlet_partition.py).")
    data.add_argument("--dirichlet-alpha", type=float, default=d.dirichlet_alpha,
                       help="Non-IID skew strength, used only when --non-iid is set. Lower "
                            "values (e.g. 0.1) produce more skewed client class distributions "
                            "-- some clients may end up with very few or zero samples of a "
                            "given class; higher values (e.g. 10+) converge toward the IID "
                            "case. 0.5 (default) is a moderate first setting.")
    data.add_argument("--data-root", type=str, default=d.data_root)
    data.add_argument("--restrict-classes", type=str, default=d.restrict_classes,
                       help="Comma-separated class names to restrict the dataset to, e.g. "
                            "'automobile,truck,frog' (cifar10 only currently). Empty = no restriction, "
                            "use all of the dataset's native classes. No-op for --dataset pamap2.")
    data.add_argument("--minority-class", type=str, default=d.minority_class,
                       help="Which of --restrict-classes to subsample (must be one of them). "
                            "For mnist/fashion-mnist, pass the raw class index as a string "
                            "(e.g. '8', '6') rather than a name. No-op for --dataset pamap2 "
                            "(its minority class -- rope jumping -- is naturally rare; see "
                            "pamap2_data.py).")
    data.add_argument("--minority-keep-fraction", type=float, default=d.minority_keep_fraction,
                       help="Fraction of the minority class's original TRAIN count to keep "
                            "(test set is never subsampled -- stays full/balanced for fair evaluation). "
                            "E.g. 0.2 keeps 20%% of the minority class. No-op for --dataset pamap2.")

    attack = p.add_argument_group("Attack (label-flip or smashed-poison)")
    attack.add_argument("--attack", dest="attack", action="store_true", default=d.attack)
    attack.add_argument("--no-attack", dest="attack", action="store_false",
                         help="Disable the attack to run a clean baseline for comparison.")
    attack.add_argument("--attack-type", type=str, default=d.attack_type,
                         choices=["label_flip", "smashed_poison"],
                         help="label_flip (A1): flips labels only, z is untouched -- TAS "
                              "cannot detect this (see sfl_healsplit/README.md). "
                              "smashed_poison (A3): perturbs the smashed data z itself -- "
                              "the attack TAS is actually designed to catch.")
    attack.add_argument("--malicious-ratio", type=float, default=d.malicious_ratio)
    attack.add_argument("--label-flip-delta", type=int, default=d.label_flip_delta)
    attack.add_argument("--smashed-poison-sigma", type=float, default=d.smashed_poison_sigma,
                         help="SP noise magnitude, in multiples of the batch's own std(z).")

    defense = p.add_argument_group("HealSplit defense (stage 5)")
    defense.add_argument("--defense", dest="defense", action="store_true", default=d.defense,
                          help="Enable live detect+heal each round (stage 5). Off by default -- "
                               "without this flag, an attack runs completely undefended, matching "
                               "stages 1-2's original behavior.")
    defense.add_argument("--no-defense", dest="defense", action="store_false")
    defense.add_argument("--heal-every-n-rounds", type=int, default=d.heal_every_n_rounds,
                          help="Run the full detect->heal pipeline every N rounds; other rounds "
                               "fall back to plain (fast, undefended) training. 1 = every round.")
    defense.add_argument("--validator-mode", type=str, default=d.validator_mode,
                          choices=["standin", "distilled"],
                          help="standin: stage 3's cheap plain classifier. distilled: stage 4's "
                               "full AD Teacher + Vanilla Teacher + distillation pipeline -- much "
                               "more expensive per healing round.")
    defense.add_argument("--detection-k", type=int, default=d.detection_k)
    defense.add_argument("--detection-alpha", type=float, default=d.detection_alpha)
    defense.add_argument("--detector", type=str, default=d.detector, choices=["tas", "client_shift"],
                          help="tas: HealSplit's topology-aware detector (unchanged; flags a fixed "
                               "~20%% every round). client_shift: TrustFLS's client-level shift check "
                               "(robust MAD threshold, flags nothing when no client stands out).")
    defense.add_argument("--detection-client-mad-tau", type=float, default=d.detection_client_mad_tau,
                          help="client_shift only: flag a client whose shift score exceeds the median "
                               "client score by more than this many robust spreads (MAD).")
    defense.add_argument("--detection-max-flagged-client-fraction", type=float,
                          default=d.detection_max_flagged_client_fraction,
                          help="client_shift only: safety ceiling on the fraction of clients that can be "
                               "flagged in one round (0.4 of 5 clients = at most 2). Honest-majority guard.")
    defense.add_argument("--flagged-client-action", type=str, default=d.flagged_client_action,
                          choices=["heal", "drop"],
                          help="client_shift only: heal = replace a flagged client's smashed data with "
                               "validated candidates (server-only training); drop = skip its data.")
    defense.add_argument("--detection-percentile", type=float, default=d.detection_percentile,
                          help="Note: TAS flags roughly this percentage of data as poisoned EVERY "
                               "healing round, even with no attack present -- Eq. 5's percentile "
                               "term is an unconditional floor, not attack-conditional. Verified "
                               "during testing (see README); this is a property of the paper's own "
                               "specified method, not an artifact of this implementation.")
    defense.add_argument("--gan-epochs", type=int, default=d.gan_epochs)
    defense.add_argument("--validator-epochs", type=int, default=d.validator_epochs)
    defense.add_argument("--confidence-threshold", type=float, default=d.confidence_threshold)
    defense.add_argument("--ad-beta", type=float, default=d.ad_beta)
    defense.add_argument("--ad-epochs", type=int, default=d.ad_epochs)
    defense.add_argument("--distill-epochs", type=int, default=d.distill_epochs)
    defense.add_argument("--distill-temperature", type=float, default=d.distill_temperature)
    defense.add_argument("--healing-mode", type=str, default="gan",
                          choices=["gan", "dalse", "bagan_gp", "hybrid_dalse_bagan"],
                          help="gan: HealSplit's own GAN-based candidate generation (trains a "
                               "generator fresh each healing round, retries on rejection). "
                               "dalse: cheap, training-free, persistent drift-tracking healing "
                               "adapted from DALSE (Chang et al., CVPR-W 2026) -- see "
                               "dalse_healing.py for the full mapping. No retries, since candidates "
                               "are deterministic within a round.")
    defense.add_argument("--hybrid-escalate-classes", type=str, default=d.hybrid_escalate_classes,
                          help="Comma-separated class names (e.g. 'frog') routed to BAGAN-GP when "
                               "--healing-mode hybrid_dalse_bagan -- every other class routes to "
                               "DALSE. Names must be a subset of --restrict-classes. Ignored for "
                               "any other --healing-mode.")
    defense.add_argument("--cb-loss-beta", type=float, default=d.cb_loss_beta,
                          help="Class-Balanced Loss (Cui et al., CVPR 2019) reweighting strength "
                               "for the Phase C training loss, based on each class's effective "
                               "number of samples THIS round. 0.0 (default) = disabled, identical "
                               "to plain unweighted CrossEntropyLoss. Typical enabled values are "
                               "0.9--0.9999. Independent of --healing-mode -- a training-time fix, "
                               "not a detect/heal-pipeline one, so it can be combined with any "
                               "healing_mode.")
    defense.add_argument("--la-counts", choices=["full", "kept"], default=d.la_counts,
                          help="class counts for the LA prior: full (all round samples, default) or kept "
                               "(excluding samples the defence dropped).")
    defense.add_argument("--la-loss-psi", type=float, default=d.la_loss_psi,
                          help="Logit-Adjusted Loss strength (MConFL Stage 1, Yuan et al. 2026) "
                               "for the Phase C training loss -- shifts each class's logits by "
                               "psi*log(class_share) THIS round before softmax, forcing rarer "
                               "classes to output a larger compensating raw logit. 0.0 (default) "
                               "= disabled (zero offset). A DIFFERENT mechanism from "
                               "--cb-loss-beta (logit shift, not loss reweighting) -- "
                               "independently toggleable, and the two CAN be combined, though "
                               "each should be tested alone first. The source paper found "
                               "psi=1 optimal across all its own configurations; larger values "
                               "degraded performance monotonically.")
    defense.add_argument("--ewc-lambda", type=float, default=d.ewc_lambda,
                          help="Elastic Weight Consolidation penalty strength (MConFL Stage 2, "
                               "Yuan et al. 2026), protecting the CLIENT model's parameters "
                               "across rounds by penalizing movement away from their last "
                               "known-good values, weighted by Fisher-information importance. "
                               "0.0 (default) = disabled. WARNING from the source paper's own "
                               "hyperparameter sweep: excessively large values caused literal "
                               "NaN loss (numerical collapse), not just poor performance -- "
                               "sweep carefully, starting small (e.g. 1.0-100.0) before going "
                               "higher.")
    defense.add_argument("--ewc-start-round", type=int, default=d.ewc_start_round,
                          help="Round at which --ewc-lambda's penalty begins applying. NOT "
                               "optional if ewc_lambda>0.0: the source paper's own ablation "
                               "found applying EWC from round 1 caused a catastrophic collapse, "
                               "not just underperformance -- the model needs a chance to learn "
                               "something before its parameters get protected.")
    defense.add_argument("--ewc-fisher-num-batches", type=int, default=d.ewc_fisher_num_batches,
                          help="How many of this round's raw batches to use when estimating "
                               "the Fisher information matrix, each round EWC is active -- an "
                               "estimate, not an exact computation, kept small and bounded.")
    defense.add_argument("--contrastive-lambda", type=float, default=d.contrastive_lambda,
                          help="Supervised contrastive representation-learning refinement "
                               "strength, adapted from FedIIC (Wu et al., MICCAI 2023). 0.0 "
                               "(default) = disabled. Applied ONCE per round, after normal "
                               "training and aggregation, on smashed data POOLED across ALL "
                               "clients -- pulls same-class embeddings together and pushes "
                               "different-class ones apart. A third, distinct kind of "
                               "intervention from --cb-loss-beta/--la-loss-psi (loss "
                               "reweighting) and --ewc-lambda (parameter protection): this "
                               "shapes the representation directly.")
    defense.add_argument("--contrastive-temperature", type=float, default=d.contrastive_temperature,
                          help="SupCon temperature -- lower values produce sharper, more "
                               "confident class separation; higher values are gentler.")
    defense.add_argument("--contrastive-num-batches", type=int, default=d.contrastive_num_batches,
                          help="How many pooled, cross-client batches to refine on each round "
                               "--contrastive-lambda is active.")
    defense.add_argument("--dalse-beta", type=float, default=d.dalse_beta,
                          help="EMA weight for an active class's drift update (DALSE's Eq. 2).")
    defense.add_argument("--dalse-decay-lambda", type=float, default=d.dalse_decay_lambda,
                          help="Decay factor applied to an inactive class's drift each round it "
                               "stays inactive (DALSE's Eq. 2).")
    defense.add_argument("--dalse-eta", type=float, default=d.dalse_eta,
                          help="Sensitivity of the snapshot-anchoring blend to how long a class "
                               "has been inactive (DALSE's Eq. 4) -- higher means the blend shifts "
                               "toward the stale snapshot faster as inactivity persists.")
    defense.add_argument("--dalse-min-exemplars-for-active", type=int,
                          default=d.dalse_min_exemplars_for_active,
                          help="Minimum clean exemplars of a class this round's TAS-clean pool "
                               "needs to count that class as 'active' (i.e. reliable enough to "
                               "update its drift estimate from) this round.")
    defense.add_argument("--dalse-noise-sigma-fraction", type=float,
                          default=d.dalse_noise_sigma_fraction,
                          help="Fraction of this round's own clean-data spread used as Gaussian "
                               "noise added to each DALSE candidate on every generate/retry call. "
                               "0.0 = fully deterministic, exactly the original DALSE-faithful "
                               "behavior (every retry produces an identical candidate, so retries "
                               "have no effect). See dalse_healing.py's module docstring.")
    defense.add_argument("--dalse-feedback-confidence-threshold", type=float,
                          default=d.dalse_feedback_confidence_threshold,
                          help="A STRICTER bar than --confidence-threshold: only a successfully "
                               "healed candidate clearing this (default 0.85 vs. 0.7) gets stored "
                               "and later blended into future rounds' centroid estimates.")
    defense.add_argument("--dalse-feedback-weight", type=float, default=d.dalse_feedback_weight,
                          help="How much a single stored healed sample counts relative to one "
                               "real clean exemplar when re-estimating a class centroid (0.5 = "
                               "half weight -- never treated as equal to real evidence).")
    defense.add_argument("--dalse-feedback-max-share", type=float, default=d.dalse_feedback_max_share,
                          help="Hard cap on what fraction of a class's centroid evidence, in any "
                               "single round, can come from healed memory rather than real clean "
                               "data -- enforced by scaling healed weight down, not dropping "
                               "samples. 0.0 disables the feedback mechanism entirely.")
    defense.add_argument("--dalse-feedback-buffer-size", type=int, default=d.dalse_feedback_buffer_size,
                          help="Max past healed samples retained per class (FIFO, oldest evicted "
                               "first) -- a memory-management bound, separate from the per-round "
                               "influence cap above.")
    defense.add_argument("--dalse-feedback-success-rate-threshold", type=float,
                          default=d.dalse_feedback_success_rate_threshold,
                          help="A class only receives feedback blending if its historical "
                               "cumulative healing success rate (tracked by the TransparencyLog) "
                               "is BELOW this threshold, or has no history yet. Classes already "
                               "succeeding reliably (e.g. majority classes) are left untouched -- "
                               "added after uniform (all-classes) feedback measurably hurt "
                               "majority-class healing without a clear net benefit.")
    defense.add_argument("--bagan-ae-epochs", type=int, default=d.bagan_ae_epochs,
                          help="Autoencoder pretraining epochs for healing_mode='bagan_gp' only -- "
                               "trained once per healing round, across all classes jointly, before "
                               "the class-conditioned adversarial phase (--gan-epochs) begins.")
    defense.add_argument("--max-batches-per-client-for-detection", type=int,
                          default=d.max_batches_per_client_for_detection,
                          help="Caps how many batches per client feed the detect+heal pipeline "
                               "each healing round (0 = uncapped). Every client still trains on "
                               "ALL of its local data regardless -- this only bounds how much data "
                               "TAS/GAN/AD-Teacher/validator inspect, since the uncapped pipeline "
                               "is computationally infeasible at real FL round sizes (a K x K graph "
                               "and, for --validator-mode distilled, a K x K matrix solve -- observed "
                               "to fail outright above a few thousand samples).")

    pamap2 = p.add_argument_group("PAMAP2 (sensor time-series)")
    pamap2.add_argument("--pamap2-window-size", type=int, default=d.pamap2_window_size,
                         help="Samples per window (1.0s @ 100Hz by default). See pamap2_data.py.")
    pamap2.add_argument("--pamap2-stride", type=int, default=d.pamap2_stride,
                         help="Window stride (50%% overlap by default). See pamap2_data.py.")

    infra = p.add_argument_group("Reproducibility / infra")
    infra.add_argument("--seed", type=int, default=d.seed)
    infra.add_argument("--device", type=str, default=d.device, choices=["auto", "cpu", "cuda"])
    infra.add_argument("--results-dir", type=str, default=d.results_dir)
    infra.add_argument("--run-name", type=str, default=d.run_name)

    return p


def parse_config() -> SFLConfig:
    args = build_parser().parse_args()
    return SFLConfig(**vars(args))
