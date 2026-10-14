# Split FL + HealSplit extension track

Stage 1 of extending the FL+DP baseline (`../src/`) with ideas from
**HealSplit** (Xie & Lyu, AAAI-26) — a defense framework for Split Federated
Learning (SFL) against poisoning attacks. No official code release exists
for HealSplit (checked: no GitHub repo linked from the AAAI page or the
arXiv preprint as of this writing) — everything here is implemented from
the paper text.

**This stage (1) built:** a minimal, working Split FL system (client-side /
server-side model split, smashed data exchanged each batch, FedAvg on the
client-side weights) plus HealSplit's Label Poisoning (LP) attack, so there's
something real for the later detection/recovery stages to defend against.

**Stage 2 builds:** Topology-Aware malicious data Detection (TAS) -- Eqs.
2-5 of the paper -- plus a Smashed Poisoning (SP) attack, added specifically
because LP alone can't validate a detector that only looks at smashed data
(see "The LP-vs-SP finding" below).

**Stage 3 (this update) builds:** the recovery/"healing" pipeline --
a GAN trained on TAS-identified clean smashed data (Eq. 6), and a
consistency-validation gate that accepts or rejects each generated
candidate before it replaces a flagged sample. See "Stage 3 scoping note"
below for an important simplification: the real consistency validation
student (trained via distillation) is stage 4, not this stage.

**Stage 4 (this update) builds:** the real consistency validation student --
the Anomaly-Influence Debiasing (AD) Teacher (Eqs. 7-9), the Vanilla Teacher
(Eq. 10), and adversarial multi-teacher distillation (Eqs. 11-15) -- which
REPLACES stage 3's plain stand-in. This is the stage flagged upfront as
carrying the most genuine interpretive risk (see "Stage 4 scoping note"
below) -- the paper's own equations for the AD Teacher are underspecified
in ways that go beyond "pick a reasonable default for a missing constant."

**Stage 5 (this update) builds:** live integration -- detection and
recovery now run automatically inside the per-round training loop
(`main.py --defense`), producing an actual accuracy-under-attack-WITH-defense
trajectory across a full multi-round experiment, rather than the
single-snapshot evaluation scripts stages 2-4 validated each piece with.
See "Stage 5: live defense integration" below for the three-phase
restructuring this required, and an important false-positive-rate finding
from testing.

**Not yet built:** non-IID partitioning, Data Poisoning (A2) and Weight
Poisoning (A4) attacks, split-point configurability.

## Why this is a separate codebase from `../src/`

Flower's client abstraction (`NumPyClient.fit()`: receive full weights,
return full weights) doesn't fit Split Learning's actual communication
pattern: a live, per-batch, bidirectional exchange of smashed-data tensors
and their gradients. Forcing that into Flower's round-trip model would mean
either (a) hacky abuse of Flower's API, or (b) building a custom Flower
strategy/protocol that duplicates most of what a plain training loop already
does. So this track is plain PyTorch — see `engine.py` for the training
loop, and its module docstring for the exact SFLV1-style update schedule
this implements (the paper describes what's aggregated where in prose, not
pseudocode, so this schedule is a documented, cited design choice, not a
verbatim reproduction).

## Files

```
sfl_healsplit/
├── model.py               # ClientModel / ServerModel split of the FL+DP baseline's CNN
├── attacks.py               # LP attack + SP attack + malicious-client selection
├── detection.py              # TAS: mutual-KNN graph + PPR + KDE adaptive threshold (Eqs. 2-5)
├── gan.py                     # Vanilla GAN for generating substitute smashed data (Eq. 6)
├── consistency_student.py      # Stand-in validator (stage 3) + shared train_classifier used by the real Vanilla Teacher (stage 4)
├── multihead_model.py           # Shared trunk + 3 task heads (a/b/c), used by AD Teacher and the real student
├── gis.py                        # Gradient Interaction Score (Eq. 7)
├── ad_teacher.py                  # Inter-task influence matrix (Eq. 8) + AD Teacher loss (Eq. 9)
├── distillation.py                 # Distillation losses (Eqs. 11-12), momentum weighting (Eqs. 14-15), student loss (Eq. 13), real student training
├── recovery.py                      # Semantically Consistent Substitution: ties detection+GAN+validator together
├── defense.py                        # Stage 5: wires detection+recovery into the LIVE per-round training loop
├── config.py                         # CLI config for main.py (mirrors src/config.py's pattern)
├── engine.py                          # Split FL training loop + centralized evaluation
├── main.py                            # Entry point (trains + evaluates accuracy under attack)
├── evaluate_detection.py               # Entry point (trains briefly, then evaluates TAS detection)
├── evaluate_recovery.py                 # Entry point (evaluates the full detect->heal pipeline, stage-3 stand-in gate)
├── evaluate_distillation.py              # Entry point (evaluates the real student, stage 4, vs. stage 3's stand-in)
├── smoke_test.py                          # Fast synthetic-data sanity check (no download needed)
└── README.md
```

## Running

```bash
# From the project root (one level up from sfl_healsplit/):
python -m sfl_healsplit.smoke_test          # ~seconds, synthetic data, sanity check

# Train + evaluate accuracy under attack (stage 1 outputs):
python -m sfl_healsplit.main \
    --num-clients 5 --num-rounds 20 --malicious-ratio 0.2 \
    --attack-type label_flip --label-flip-delta 1 --seed 42 --run-name mnist_lp_attack

python -m sfl_healsplit.main \
    --num-clients 5 --num-rounds 20 --malicious-ratio 0.2 \
    --attack-type smashed_poison --smashed-poison-sigma 3.0 --seed 42 --run-name mnist_sp_attack

# Clean baseline for comparison:
python -m sfl_healsplit.main --no-attack --run-name mnist_clean

# Evaluate TAS detection (stage 2) -- trains briefly, then reports precision/recall:
python -m sfl_healsplit.evaluate_detection --attack-type smashed_poison
python -m sfl_healsplit.evaluate_detection --attack-type label_flip

# Evaluate the full recovery pipeline (stage 3) -- detect, generate, validate,
# substitute, then compare downstream accuracy: attacked vs. healed vs. clean oracle:
python -m sfl_healsplit.evaluate_recovery

# Evaluate the real consistency validation student (stage 4) -- AD Teacher +
# Vanilla Teacher + distillation, compared against stage 3's plain stand-in:
python -m sfl_healsplit.evaluate_distillation

# Full multi-round run WITH live defense active (stage 5) -- cheap stand-in
# validator, healing every round:
python -m sfl_healsplit.main --attack-type smashed_poison --defense \
    --validator-mode standin --run-name mnist_sp_defended

# Same, but with the full stage-4 distillation pipeline as the validator
# (much slower per healing round) and healing only every 3rd round:
python -m sfl_healsplit.main --attack-type smashed_poison --defense \
    --validator-mode distilled --heal-every-n-rounds 3 --run-name mnist_sp_defended_distilled
```

Outputs from `main.py` land in `sfl_results/<run_name>/`: `config.json` (includes
which client IDs were assigned malicious) and `metrics.csv` (round, test_loss,
test_accuracy, flip_confusion_rate -- the last is only meaningful under
`--attack-type label_flip`). `evaluate_detection.py` prints its results
directly rather than writing a results file, since its output (precision/
recall against known ground truth) is a diagnostic for tuning TAS's
hyperparameters, not an experiment result to log for the write-up the way
`main.py`'s accuracy curves are.

`flip_confusion_rate` is a diagnostic beyond plain accuracy: the fraction of
test predictions equal to `(true_label + delta) mod C` — i.e. how often the
global model's mistakes specifically match the attacker's label mapping,
rather than being generic confusion. This was added because in early
testing, accuracy alone could saturate (e.g. an easy task hitting ~100%
accuracy under attack) while the loss was still clearly elevated — this
metric and the loss column give a more honest picture than accuracy alone
when clean clients can outvote a minority of malicious ones via FedAvg.

## The LP-vs-SP finding (read this before interpreting detection results)

**TAS operates entirely on smashed data (z). Label Poisoning (LP) never
touches z -- only the label.** A malicious client's image goes through its
client-side model completely normally; the *only* thing wrong is the label
attached to it. TAS has no way to see a wrong label; it only sees where z
sits in feature space, which is statistically identical for a label-flipped
sample and a correctly-labeled one from the same client.

This means **TAS cannot detect LP by design, not by a bug** -- and this is
directly confirmed empirically (`evaluate_detection.py --attack-type
label_flip` lands right at chance level: precision/recall ≈ malicious_ratio,
i.e. ≈0.2 at the default 20% malicious rate). That's *why* stage 2 also adds
the Smashed Poisoning (SP) attack (`z'_j = g(x_j) + δ_z`, HealSplit's A3
threat) -- it's the attack TAS is actually built to catch, since it directly
perturbs z's position in feature space. `evaluate_detection.py
--attack-type smashed_poison` gets precision/recall ≈0.77 on synthetic
testing (see below) -- a real, working detector, just not for every attack
type in HealSplit's threat model. If your write-up evaluates TAS, evaluate
it against SP (and later DP, once that attack exists), not LP.

## Verified before handoff (stage 1)

1. **Gradients actually flow correctly across the client/server split.**
   Tested on a synthetic-but-learnable task (not random labels) — a clean
   run converges to ~100% accuracy and near-zero loss within a couple of
   rounds, confirming the detach/reattach at the smashed-data boundary
   (`engine.py::train_one_round`) is wired correctly in both directions.
2. **The label-flip attack has a measurable effect.** Same task, with 1/3
   clients malicious: final accuracy still reached 100% (the task was too
   easy / the majority of clean clients outvoted the attacker via FedAvg),
   but final loss was ~200x higher than the clean run — the attack is real
   and detectable in the loss signal even when it doesn't show up in
   accuracy.
3. **Reproducibility.** Ran twice with the same seed, diffed the output —
   identical. This track doesn't have the Flower/Ray multi-process seeding
   issue documented in `../README.md` §5.5, since it's single-process; one
   `set_global_seed()` call at the top of `main.py` is sufficient here.

## Verified before handoff (stage 2)

Three real bugs were caught and fixed while validating TAS on synthetic
data with known ground truth -- worth documenting since they'd be easy to
miss without deliberately testing against a case where the "correct"
answer is known in advance:

1. **Degree/propagation inconsistency causing PPR to diverge.** An early
   version computed `degree` from the *weighted* graph but normalized an
   *unweighted* propagation step with it. Since individual Gaussian-kernel
   edge weights are <1, this under-counts degree relative to what the
   propagation step assumes, injecting extra mass every iteration instead
   of conserving it -- raw TAS scores exploded to ~28,000 for some clean
   points. Fixed by deriving degree from whichever matrix (weighted or not)
   is actually used for propagation (`detection.py::compute_tas`).
2. **`eps` at the wrong scale.** The paper describes `eps` only as "a small
   constant to prevent division by zero," which reads like a numerical
   safety epsilon (~1e-6) -- but `v_k = 1/(d_k+eps)` means a literally
   zero-degree node gets `v_k ~ 1/eps`. At eps=1e-6 that's ~1,000,000,
   six orders of magnitude above a normally-connected node's `v_k`, which
   then dominates that node's score and inverts the intended "isolated =
   low score" rule. Fixed by using eps=1.0 (same order of magnitude as
   typical degree values) -- still prevents literal division by zero, but
   doesn't let that safety term dominate.
3. **SP attack scattering instead of clustering poisoned points.** The
   first `add_smashed_poison` implementation added *fresh* `torch.randn_like`
   noise per sample -- but the paper's own equation, `z'_j = g(x_j) + δ_z`,
   has no subscript `j` on `δ_z`: it's meant to be the SAME perturbation for
   every sample from an attacker, so poisoned points cluster together
   (matching the paper's stated premise: "high feature similarity within
   malicious samples"). Fresh-per-sample noise scatters poisoned points
   randomly instead, which (a) contradicts that premise and (b) triggers
   bug #2's failure mode again for genuinely scattered points, giving them
   *higher* TAS than clean data -- backwards. Confirmed empirically: before
   the fix, mean TAS was 0.50 for "poisoned" vs 0.26 for clean (backwards);
   after fixing to a *fixed* per-client `δ_z`, it's 0.14 vs 0.27 (correct
   direction) with precision/recall ≈0.77 on a synthetic-but-learnable task.

Also verified: stage 1 (label-flip training/accuracy) produces byte-for-byte
identical output before and after all of the above changes -- the SP-related
additions are fully additive/optional (`smashed_poison_deltas=None` by
default), so stage 1's results aren't affected by anything added in stage 2.

## Design decisions (for your write-up)

**Split point:** after the first conv block (`ClientModel` = conv1 + GroupNorm
+ ReLU + pool). A standard shallow split in the SFL literature — client
keeps minimal compute, smashed data is a feature map rather than raw pixels.
Not currently configurable; see Roadmap.

**GroupNorm, not BatchNorm:** kept for consistency with the FL+DP baseline
and to keep the door open for combining this track with DP-SGD later. Not
strictly required here (Opacus isn't in this loop), but no reason to diverge.

**SFLV1-style updates:** the server-side model is a *single* shared instance,
updated directly and sequentially by each client's batch within a round —
not aggregated after the fact. Client-side models *are* explicitly
FedAvg-aggregated after each round. See `engine.py`'s module docstring for
the full reasoning and the paper passage this is filling a gap in.

**Label-flip attack, implemented as written in the paper:** `y' = (y + δ) mod C`,
an all-to-all cyclic shift applied to every sample of every malicious
client — not a targeted single-source/single-target flip. Malicious client
IDs are chosen deterministically from `--seed`, independent of the RNG
stream used for data partitioning (`attacks.py::select_malicious_clients`).

## Stage 3 scoping note: the validator is a stand-in, not the real student

The paper's consistency validation student is trained via adversarial
multi-teacher distillation (Vanilla Teacher + AD Teacher, Eqs. 10-13) --
that's stage 4. Building stage 3's recovery pipeline meant something had to
gate the GAN's outputs *now*, so `consistency_student.py` provides a
simpler stand-in: a plain classifier trained with ordinary cross-entropy on
TAS-identified clean smashed data. That's functionally just the paper's
"Vanilla Teacher" (Eq. 10) on its own, doing double duty as the gate.

This was a deliberate choice, not an oversight: `recovery.py`'s gating
function (`validate_candidate`) takes any classifier with a `forward(z) ->
logits` signature, so when stage 4 builds the real distillation-trained
student, swapping it in is a one-line change at the call site in
`evaluate_recovery.py` -- not a rewrite of the substitution logic. Building
the simplified version now (rather than blocking stage 3 on stage 4)
meant the full detect->generate->validate->substitute loop could actually
be tested end-to-end at this stage, catching real bugs (see below) that
would otherwise have surfaced later, tangled up with distillation-specific
issues and harder to isolate.

## Design decisions (stage 3)

**Accept/reject rule** (the paper only says "high confidence and label
consistency," no formula): a candidate passes if the validator's softmax
confidence is >= `confidence_threshold` (default 0.7) AND its predicted
class matches the TRUE label of the sample being replaced -- checkable
because SP corrupts z, not y, so the label is trustworthy ground truth to
validate the candidate against, not something we're guessing at too.

**Fallback for candidates that keep failing the gate** (also unspecified by
the paper): retry with fresh noise up to `max_retries` times (default 5),
then DROP the sample from that round's training data. Dropping is safer
than the two alternatives -- keeping the original poisoned z (defeats
detection entirely) or accepting a low-confidence/wrong-label substitute
(risks injecting a *new*, GAN-shaped corruption in place of the original
one). Losing a bounded number of samples per round is a mild cost next to
either alternative.

**Generator's final activation is ReLU, not tanh/linear:** real smashed
data here is always non-negative (`ClientModel`'s forward pass ends in
ReLU -- see `model.py`). An unconstrained generator would happily learn to
output values the discriminator has never seen as "real" and still fool an
undertrained discriminator early in training. ReLU matches the actual
support of the data being modeled.

## Verified before handoff (stage 3)

Two more real bugs caught while testing the full pipeline end-to-end on
synthetic data (in addition to the three from stage 2's detection work) --
worth walking through since both would have silently produced misleading
numbers rather than crashing:

1. **Shape mismatch between detection's flattened output and the GAN/CNN's
   expected input.** `collect_round_smashed_data` (stage 2) flattened its
   output to `[N, D]` for `detect_poisoned`'s convenience, but the GAN and
   the CNN-based validator both need the real `[C, H, W]` structure --
   feeding a flat tensor into a `Conv2d` layer crashes immediately. Fixed
   by moving the flattening into `detection.py::build_mutual_knn_graph`
   itself (which now accepts any per-sample shape and flattens internally)
   and having `collect_round_smashed_data` return the natural, unflattened
   shape -- verified stage 2 still produces identical detection results
   after this change (re-ran the stage 2 validation script; same numbers).
2. **Comparing misaligned "attacked" vs. "oracle" tensors.** The first
   version of `evaluate_recovery.py` called `collect_round_smashed_data`
   twice against the same `shuffle=True` client loaders -- once under
   attack, once without -- assuming `Z_attacked[i]` and `Z_oracle[i]` were
   the same underlying sample. They weren't: shuffled loaders reorder
   independently on every iteration, so the two collections referred to
   different samples at the same index. This surfaced as *all three*
   conditions (attacked / healed / **and the clean oracle upper bound**)
   scoring near chance accuracy -- the tell that something was mis-aligned
   rather than the classifier genuinely failing, since the oracle condition
   should have been trivial on this task. Fixed with a separate,
   `shuffle=False` set of loaders used only for the collection calls (see
   `evaluate_recovery.py`), so both collections walk the same samples in
   the same order.

**What the synthetic validation confirms, and what it doesn't:** after
fixing both bugs, the mechanics all run correctly end-to-end -- TAS flags
SP-corrupted samples, the GAN trains (discriminator accuracy moves off its
initial near-100%, though doesn't fully converge to the ideal ~50%/50% in
the ~100 epochs / few hundred samples used here -- GANs are notoriously
slow/finicky to converge, and this wasn't tuned), and substitution produces
sensible replace/drop counts. What it does *not* demonstrate is a clean
"healed beats attacked" accuracy gap: the synthetic bright-patch task used
for sandbox testing (see stage 1's smoke tests) turned out too easy/robust
-- even a classifier trained on heavily-corrupted (40% malicious, large
perturbation) data still hit ceiling accuracy, because the location-based
signal survives most feature-space corruption. That's a property of the
synthetic task, not evidence the pipeline doesn't help -- but it does mean
the *quantitative benefit* claim (does healing measurably improve
downstream accuracy under attack) is still untested and worth checking with
`evaluate_recovery.py` on your real MNIST data, where the task is naturally
harder and an attack's impact should be more visible.

## Stage 4 scoping note: genuinely higher interpretive risk than stages 2-3

You chose the most faithful option for this stage (literal implementation
of Eqs. 7-9, heavily documented assumptions, over a simplified alternative)
-- worth restating what that means in practice. Stages 2-3's ambiguities
were "the paper leaves a constant unspecified" (TAS's `alpha`, the GAN's
accept/reject threshold) -- resolvable by picking a defensible default and
moving on. Stage 4's AD Teacher has ambiguity in the *mechanism itself*:
what matrix `R` refers to, what "gradient of a label" means as an
operation. Different reasonable readings of these would produce
structurally different computations, not just different numbers from the
same computation. Every interpretive choice below is a real fork, not a
default value -- flagging this explicitly rather than presenting the
result with the same confidence as stages 2-3's more mechanical fixes.

## Design decisions (stage 4)

**What `R` is (Eq. 8):** TAS (Eq. 3) is a per-node *score* (a vector), but
Eq. 8 calls `R` a *matrix*, used in a Hadamard product against a full K x K
matrix. A diagonal matrix of TAS scores would zero out everything off the
diagonal in that product, destroying essentially all the structure Eq. 8 is
trying to build. We interpret `R` as the K x K weighted mutual-KNN
adjacency matrix TAS's own graph is built from (`detection.py`'s `W`, Eq.
2) -- dimensionally consistent, and it keeps the "TAS matrix" naming
plausible (it's the graph TAS's scores come from, not a renamed copy of
the scores). See `ad_teacher.py`'s module docstring for the full reasoning.

**What "gradient of a label" means (Eq. 7):** implemented as the gradient
of a task's loss, restricted to samples carrying a specific label,
differentiated with respect to `MultiHeadModel`'s **shared trunk**
parameters (not the task-specific head, not the input). This matches the
paper's citation of multi-task gradient-alignment literature (Yu et al.
2020, "gradient surgery for multi-task learning"), where gradient alignment
is measured through parameters multiple tasks actually share. See
`gis.py`'s module docstring.

**Vanilla Teacher trained on category (task c), not poison-id (task a):**
Eq. 10 writes `L_Van = sum L_a(...)`, reusing task a's notation -- but
training a poison-identification head on data that's ALREADY been filtered
to all-clean would make every training label the same value, a degenerate
signal, and it contradicts the surrounding text ("preserves semantic
integrity by modeling the distribution of clean data"). Read as a
notational reuse/likely typo; the Vanilla Teacher is trained exactly like
stage 3's stand-in (`consistency_student.py::train_classifier`, category
task, clean data only) -- which is why stage 3's module is reused directly
here rather than duplicated.

**Both distillation losses (Eqs. 11-12) target the category head:** neither
equation specifies which of the AD Teacher's three outputs `h_TAD(z)`
refers to. Since the Vanilla Teacher only has one output (category), and
Eq. 13's student loss has no direct category-supervision term at all (the
student learns category *purely* through distillation), we target the AD
Teacher's category head too -- both teachers "vote" on category, one
trained cleanly, one trained with the anomaly-aware influence weighting.

**Eq. 9's loss weight, squashed through sigmoid (a correctness fix, not a
style choice):** see "Verified before handoff" below -- this one isn't
optional, raw `M_p` broke training.

## Verified before handoff (stage 4)

**One critical bug that would have silently produced nonsense, not a
crash:** `compute_influence_matrix`'s raw output (`M_p`) is unbounded and
can be *negative* -- observed values from -46 to +88 on a small synthetic
test batch. `ad_teacher_loss` (Eq. 9) uses `M_p` values as a multiplicative
weight on a loss term. A negative weight means the optimizer's objective is
now "*increase* this loss" for that sample -- training diverged from a
starting loss of ~19 to -2089 within 15 epochs, growing without bound.
Caught by watching the loss trajectory during testing, not by a crash (an
unwatched training run would have just silently produced a broken teacher
with no error). Fixed by squashing `M_p` through sigmoid before use as a
weight (bounded to (0,1), monotonic -- preserves relative ordering between
sample pairs). Re-verified after the fix: loss decreases smoothly from
~3.4 to ~2.3-3.0 over 15 epochs on the same test data, no divergence.

**Confirmed the mechanism actually learns, not just avoids diverging:**
tested with a mix of real-signal and pure-noise labels in the same batch
(category labels carrying genuine structure; poison-id and client-id
labels randomly assigned, no real signal). The AD Teacher reached 100%
accuracy on the real-signal task and landed within noise of chance level
on the random-label tasks (50.7% on binary, 29.3% on 5-way) -- exactly what
a correctly-functioning model should do: learn what's learnable, and not
spuriously fit noise.

**Confirmed distillation actually transfers knowledge, not just runs:**
the real student was trained with **no direct supervision on the category
task at all** (per Eq. 13's literal specification -- only tasks a and b get
a direct loss term). It still reached 100% category accuracy, purely from
the two teachers' distilled soft predictions. That's real evidence the
distillation mechanism (Eqs. 11-15) works as intended, not just that the
code runs without error.

**Confirmed plug-and-play with `recovery.py` (stage 3):** wrapped the
3-headed student in `CategoryHeadWrapper` (satisfies the single-tensor
`forward(z) -> logits` contract `recovery.py`'s gate expects) and re-ran
substitution with it in place of stage 3's stand-in -- ran without any
changes to `recovery.py` itself, confirming the "one-line swap" stage 3's
README promised.

**What wasn't demonstrated:** the same synthetic-task ceiling effect from
stage 3 shows up here too -- the stand-in, the AD Teacher alone, and the
real distilled student all reach 100% category accuracy on the sandbox's
synthetic task, so this validation confirms the *mechanism* works
end-to-end (no crashes, no divergence, genuine knowledge transfer via
distillation, correct plug-and-play integration) but doesn't show the real
student outperforming stage 3's simpler stand-in quantitatively. Per the
paper's own Table 3 ablation (MNIST: full HealSplit 93.88%, "w/o AD
Teacher" 87.34%, "w/o Distillation" 74.38%), that gap should be visible on
real, harder data -- worth checking with `evaluate_distillation.py` on your
actual MNIST setup.

## Stage 5: live defense integration

### What changed architecturally

`engine.py`'s SFLV1 loop forwards and immediately backward+updates, batch
by batch, client by client -- there's no point where "the whole round's
smashed data" exists at once for TAS to run on. `defense.py` (a new,
separate module -- `engine.py` is untouched) restructures a healing round
into three phases:

1. **Collect** (forward-only, `no_grad`): every client's every batch is
   forwarded through the current global client model to produce z, with
   the same SP injection rule as before. Both x and y are kept (not just
   z), so training can re-forward WITH gradients afterward.
2. **Detect + heal**: TAS runs on the pooled round's smashed data;
   flagged samples get GAN-generated replacements, gated by a validator
   (`--validator-mode standin` for stage 3's cheap classifier, or
   `distilled` for the full stage-4 AD Teacher + Vanilla Teacher +
   distillation pipeline).
3. **Train**: each batch splits into three groups -- normal (unflagged),
   healed (flagged + substitute accepted), dropped (flagged + rejected
   after retries). Normal samples get the standard client-forward ->
   server-forward -> backward-both treatment. **Healed samples get a
   server-only update** -- the GAN-generated z is used as an independent
   leaf tensor, deliberately *not* connected to the client's computation
   graph, since that value never came from the client's actual forward
   pass. Backpropagating "the server's gradient for a synthetic sample"
   into the client's real weights wouldn't correspond to anything
   physically meaningful. Dropped samples get no forward pass at all.

One consequence worth knowing: a client whose entire round gets
flagged-and-healed receives no client-side gradient update at all, so its
FedAvg contribution is just its unchanged copy of the incoming global
weights -- diluting the aggregate toward "no change" rather than an
arbitrary direction. Intentional, not an oversight.

### Cost controls

Running the full pipeline every round of a 20+ round experiment is
expensive (training up to four separate models' worth of epochs, every
round). Two knobs manage this:
- `--heal-every-n-rounds N`: skip detect+heal on non-healing rounds,
  falling back to `engine.py`'s plain (fast) training.
- `--validator-mode {standin, distilled}`: stage 3's cheap classifier vs.
  stage 4's full pipeline.

The paper itself supports periodic (not necessarily continuous) healing:
"HealSplit synchronizes with SFL at appropriate intervals, using only the
smashed data from the current update round to train the GAN" -- consistent
with retraining fresh each healing round rather than persisting a
generator/validator across rounds, which is what `defense.py` does.

### An important finding from testing: false positives are unconditional

Verified directly, not assumed: ran the defense loop with **no attack at
all**. TAS still flagged exactly 120 of 600 samples (20%) as poisoned,
every single round. This is not a bug -- Eq. 5's threshold,
`T = min(argmin_r f_hat(r), Q_rho({r_k}))`, applies the percentile term as
an **unconditional floor via `min()`**, not something that only activates
when a genuine bimodal valley (real evidence of an attack) is found. TAS
flags roughly `--detection-percentile`% of data as poisoned regardless of
whether an actual attack is present, and always pays the substitution/drop
cost for it. This is a real, verified property of the paper's own
specified method, not an implementation artifact -- and a legitimate point
for discussion in your write-up: HealSplit as specified always trades a
baseline false-positive cost for its detection capability, even in a fully
benign federation.

### What multi-round testing showed, and its limits

Comparing undefended vs. defended runs on the synthetic sandbox task
(2/5 clients malicious, SP attack): the **undefended** run reached ~98.5%
accuracy by round 2 on its own, because FedAvg's built-in averaging is
already fairly robust to a *minority* of malicious clients when the
underlying task is easy. The **defended** run was actually *worse* in
round 1 (8.5% vs. undefended's 49.5%) -- the unconditional 20%
false-positive rate plus an under-trained GAN/validator (limited data,
round 1) cost more than they protected against, before catching up by
round 2. This is consistent with (not contradicting) stages 3-4's findings:
the synthetic task's high signal-to-noise ratio keeps saturating every
comparison at or near ceiling, so it isn't a good testbed for showing
HealSplit's incremental value over plain FedAvg. The paper's own
experiments use real datasets, a majority-or-adaptive attack setting, and
harder architectures (ResNet-18/152, VGG16) specifically because an easy
task with a clean-majority defense-free baseline doesn't leave much room
for any defense to show a measurable improvement. Worth running
`main.py --defense` vs. without it on your real MNIST setup, ideally with
a higher malicious ratio (the paper defaults to 20%, but Fig. 4 shows their
own results across 10-50%) to see whether a genuine gap opens up.

### Verified before handoff (stage 5)

- **Reproducibility**: ran the full defended loop twice with the same
  seed -- identical output across all rounds' loss/accuracy/flagged/
  replaced/dropped counts, confirming seeding threads correctly through
  every stochastic component (GAN, validator/teacher training, detection).
- **Reduces correctly to the undefended case**: with no attack applied,
  the "normal" (unflagged) training path in `defense.py` uses the same
  client-forward -> server-forward -> backward-both logic as `engine.py`.
- **`heal_every_n_rounds` cadence control**: tested with a value of 2 --
  confirmed odd rounds fall back to plain training (`healed=False,
  flagged=0`) and even rounds run the full pipeline, exactly as configured.
- **`validator_mode="distilled"`**: ran end-to-end inside the live loop
  (not just the standalone `evaluate_distillation.py` script) -- confirmed
  the full stage-4 pipeline (AD Teacher + Vanilla Teacher + distillation +
  substitution) runs without errors as one round's healing step, not just
  as an isolated evaluation.
- **No regression**: stages 1-4's smoke tests and validation scripts were
  re-run after every change in this stage and produce identical output to
  before `defense.py` existed -- it's purely additive.

## Roadmap (subsequent stages, one at a time as requested)

1. ~~Split FL infrastructure + label-flipping attack~~ (stage 1, done)
2. ~~Topology-aware detection (TAS) + Smashed Poisoning attack~~ (stage 2,
   done).
3. ~~GAN-based recovery + consistency validation student (stand-in)~~
   (stage 3, done). Note, same pattern as stage 2: `main.py`'s live
   training loop still doesn't call detection or recovery -- both are
   validated standalone (`evaluate_detection.py`, `evaluate_recovery.py`)
   against ground truth / an oracle upper bound, not wired into a live
   multi-round experiment yet. Deliberately deferred (see "Live-training
   integration" below) rather than wired in twice -- once now with the
   stand-in validator, again after stage 4 replaces it.
4. ~~Anomaly-Influence Debiasing Teacher + adversarial multi-teacher
   distillation (the real consistency validation student)~~ (stage 4,
   done). Replaces stage 3's stand-in everywhere it was used
   (`recovery.py`'s gate) -- confirmed as a drop-in swap requiring no
   changes to `recovery.py` itself.
5. ~~Live-training integration~~ (stage 5, done). See "Stage 5: live
   defense integration" above for the three-phase restructuring, cost
   controls, and the false-positive-rate finding from testing.
6. Split-point configurability, non-IID partitioning, Data Poisoning (A2)
   and Weight Poisoning (A4) attacks, other extensions once the core
   detect→heal→train loop (stages 2–5) is validated.

## Detection-specific design decisions (TAS, stage 2)

Beyond what's in `detection.py`'s own module docstring (ambiguity in Eq. 3's
propagation term, the `eps` scale fix, the `argmin`-as-valley interpretation
in Eq. 5, and the alpha default from empirical testing — all documented
there with full reasoning), two things worth knowing before you use this in
your write-up:

- **No hyperparameters are given numeric defaults by the paper** (k, alpha,
  KDE bandwidth h, percentile ρ, num_iters). Everything here is a reasonable
  starting point validated on synthetic data, not a value carried over from
  the paper — expect to tune these against real smashed data using
  `evaluate_detection.py`'s precision/recall output.
- **TAS is run once per "collection," not continuously integrated into
  training yet.** `evaluate_detection.py` trains a few warm-up rounds (so
  smashed data reflects a partially-trained model rather than random init),
  then takes one snapshot and evaluates detection on it. Real per-round
  detection integrated into `main.py`'s training loop is stage 3's job, once
  there's a recovery action to trigger from a detection.
