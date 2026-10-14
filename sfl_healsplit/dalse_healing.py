"""
DALSE-style healing: a cheap, training-free alternative to HealSplit's
GAN-based candidate generation, adapted from DALSE (Chang et al., CVPR-W
2026, "Drift-Aware Latent State Estimation for Robust Federated Learning
under Dynamic Participation").

Design decisions -- the explicit mapping from DALSE's per-CLIENT weight
drift to per-CLASS smashed-data drift, since DALSE was built for a
different problem (client dropout in plain FL) and needs translating to fit
here (poisoned-sample substitution in Split FL):

- DALSE's "active" vs "inactive" client (does the server have a fresh
  observation this round) becomes per-CLASS active/inactive: did this
  round's TAS-clean pool contain enough exemplars of this class to update
  its drift estimate reliably (`min_exemplars_for_active`).
- DALSE's global model Theta^{t-1} (Eq. 3) becomes this round's OVERALL
  clean-data centroid -- the mean of ALL clean smashed data this round,
  the natural "current global state" analogue when there's no model being
  distributed each round, just a per-round data snapshot.
- DALSE's per-client drift Delta_i (Eq. 2: EMA of local-model-minus-global
  deviation) becomes a per-class EMA of (this class's clean centroid minus
  the overall centroid).
- DALSE's snapshot-anchored refinement (Eq. 4: blend the drift-corrected
  projection with the last-observed snapshot, weighted by how long the
  client's been inactive) becomes: blend this round's drift-corrected class
  projection with the last-known-good class centroid, weighted by how many
  consecutive rounds the class has been under-represented (`age`).

One structural difference from DALSE, worth being explicit about rather
than glossing over: DALSE produces exactly ONE reconstructed state per
absent client per round -- there's no notion of "generate several candidate
models and pick." This module originally inherited that faithfully:
`reconstruct_candidate` was fully deterministic, returning the same value
on every call within a round.

UPDATE -- deliberate departure from strict DALSE fidelity, added after
empirical testing: the fully-deterministic version turned out to have a
genuine "all-or-nothing" problem. Because every flagged sample of a class
gets the exact same candidate, a class's entire outcome for the round comes
down to one binary test -- does that single point happen to fall inside the
validator's acceptance region. Across a 5-seed test on real data, this
produced wildly inconsistent minority-class results (0%, 23%, 0%, 6%, 8%
success across otherwise-identical runs) -- consistent with candidates
often landing near the validator's decision boundary, where a single
deterministic point is a coin flip rather than a considered check.

The fix: `reconstruct_candidate` now adds fresh Gaussian noise to the
drift-corrected projection on EVERY call, scaled relative to this round's
own observed clean-data spread (not a fixed absolute magnitude, since raw
smashed data's natural scale isn't assumed -- unlike e.g. a generator's
Tanh-bounded output, this is the client model's own raw activation, whose
scale varies run to run). This means successive calls for the same class
within a round now genuinely differ, which is *why* recovery.py's DALSE
substitution path now retries on rejection, the same way the GAN path
always has -- a retry against a stochastic candidate can land somewhere
new; a retry against the old deterministic one could not have.

This is a real trade-off, not a strict improvement: it decouples per-sample
outcomes from each other (attacking the all-or-nothing symptom directly),
but there's no guarantee the *average* success rate moves in any particular
direction -- it depends on whether projections tend to sit near the
validator's boundary (where noise helps) or safely on one side of it (where
noise mostly just adds risk of a previously-safe point failing). Set
`noise_sigma_fraction=0.0` to recover the original, fully-deterministic
behavior exactly.

SECOND UPDATE -- learning from successful heals, not just clean data:
every round, `update_and_project` was fed ONLY this round's TAS-clean
(unflagged) data. A sample that was flagged, healed, and validator-approved
contributed nothing back to DALSE's own estimate -- it patched that one
sample and was discarded. This mirrors DALSE's own original framing more
closely than it first appears: in the source dynamic-participation setting,
an "active" client's real observation informs the server's state; there's a
real argument that a validator-confirmed healed sample IS a legitimate
observation, not noise to discard.

The risk, taken seriously rather than glossed over: this is structurally
similar to model-collapse in generative systems -- a system that trusts its
own outputs as future training signal can drift away from ground truth
rather than toward it, especially for a scarce class whose estimate is
already built from little real evidence. Three safeguards, all required
together, not optional extras:
  1. A STRICTER confidence bar for feedback-eligibility (`feedback_confidence
     _threshold`, default 0.85) than for healing-acceptance itself (0.7) --
     passing the normal bar isn't enough to be trusted as real evidence.
  2. DOWN-WEIGHTING (`feedback_weight`, default 0.5): a stored healed sample
     never counts as full evidence, only partial.
  3. A hard CAP (`feedback_max_share`, default 0.25) on what fraction of a
     class's centroid computation, in any single round, can come from
     healed memory -- enforced by SCALING healed samples' aggregate weight
     down when it would exceed the cap, not by dropping samples arbitrarily.
Separately, a bounded FIFO buffer (`feedback_buffer_size`, default 20 per
class) caps how many past healed samples are even retained -- a practical
memory bound, distinct from the 25% cap above, which governs per-round
INFLUENCE rather than storage.

Deliberately NOT a fixed-window "last-K samples" model, despite that being
the more obvious first instinct: DALSE's existing memory (drift + snapshot)
is an EXPONENTIALLY-WEIGHTED average, not a hard window -- with beta=0.9,
past contributions decay with an effective memory span of roughly
1/(1-beta) ~= 10 rounds, never a hard cutoff. A percentage-of-round-weight
cap fits that same decaying-influence philosophy; a fixed "just the most
recent success" would have been a different memory model bolted on
alongside the existing one, not an extension of it.

Set `feedback_max_share=0.0` (or leave the healed-sample buffer empty) to
disable this entirely and recover the prior behavior exactly.

Design decision -- persistence is a real architectural requirement, not
optional: DALSE's whole point is tracking how something evolves *while
unobserved*, across many rounds. A `DALSEHealer` instance must be created
ONCE per experiment and threaded through every healing round -- creating a
fresh one each round (the way the GAN and validator already are) would
silently defeat the entire mechanism, collapsing it into "just use this
round's centroid," with no drift-tracking or memory at all.

Adapted from the method described in Chang et al. (2026), "DALSE". Written from the
paper's equations; the mapping from per-client weight drift to per-class smashed-data drift is our own.
"""
import dataclasses
from typing import Dict, Optional, Tuple

import torch


@dataclasses.dataclass
class DALSEState:
    drift: Dict[int, torch.Tensor] = dataclasses.field(default_factory=dict)
    snapshot: Dict[int, torch.Tensor] = dataclasses.field(default_factory=dict)
    age: Dict[int, int] = dataclasses.field(default_factory=dict)


class DALSEHealer:
    """Persistent, stateful, class-level DALSE-style healer. Create ONE
    instance per experiment (not per round) -- see module docstring."""

    def __init__(self, num_classes: int, z_shape: Tuple[int, ...],
                 beta: float = 0.9, decay_lambda: float = 0.95, eta: float = 0.3,
                 min_exemplars_for_active: int = 5, noise_sigma_fraction: float = 0.15,
                 feedback_confidence_threshold: float = 0.85, feedback_weight: float = 0.5,
                 feedback_max_share: float = 0.25, feedback_buffer_size: int = 20,
                 feedback_success_rate_threshold: float = 0.5):
        self.num_classes = num_classes
        self.z_shape = z_shape
        self.beta = beta                    # Eq. 2's beta -- EMA weight for active classes
        self.decay_lambda = decay_lambda    # Eq. 2's lambda -- decay for inactive classes
        self.eta = eta                      # Eq. 4's eta -- sensitivity to prolonged inactivity
        self.min_exemplars_for_active = min_exemplars_for_active
        self.noise_sigma_fraction = noise_sigma_fraction  # fraction of this round's own
                                             # clean-data spread; see module docstring's
                                             # "UPDATE" section. 0.0 = fully deterministic,
                                             # exactly the original DALSE-faithful behavior.
        self.feedback_confidence_threshold = feedback_confidence_threshold
        self.feedback_weight = feedback_weight
        self.feedback_max_share = feedback_max_share
        self.feedback_buffer_size = feedback_buffer_size
        self.feedback_success_rate_threshold = feedback_success_rate_threshold
                                             # THIRD UPDATE -- gates WHICH classes feedback
                                             # even applies to. Uniform (all-classes) feedback
                                             # measurably hurt majority-class healing without a
                                             # clear net win on minority classes either (see
                                             # module docstring's "THIRD UPDATE" section). A
                                             # class's historical cumulative success rate --
                                             # sourced live from TransparencyLog, the same
                                             # reporting layer built for observability, now
                                             # feeding back into a healing decision -- must be
                                             # BELOW this threshold for blending to apply at
                                             # all. A class with NO history yet (None) is
                                             # treated as eligible: an untested class is more
                                             # plausibly an under-served one than a proven-fine
                                             # one, so the conservative default is to help
                                             # until evidence says otherwise.

        self.state = DALSEState()
        self._initialized = {}
        for c in range(num_classes):
            self.state.drift[c] = torch.zeros(z_shape)
            self.state.snapshot[c] = torch.zeros(z_shape)
            self.state.age[c] = 0
            self._initialized[c] = False

        self._last_projected: Dict[int, torch.Tensor] = {
            c: torch.zeros(z_shape) for c in range(num_classes)
        }
        self._round_noise_sigma: float = 0.0  # absolute magnitude, recomputed each round
                                                # in update_and_project from this round's
                                                # actual clean_z spread
        self._healed_memory: Dict[int, list] = {c: [] for c in range(num_classes)}
                                                # bounded FIFO of past validator-confirmed
                                                # healed candidates, per class -- see
                                                # module docstring's "SECOND UPDATE" section

    def record_healed_sample(self, class_idx: int, candidate: torch.Tensor, confidence: float) -> None:
        """Call once per ACCEPTED healed sample, right after
        dalse_consistent_substitution confirms it passed validation. Only
        samples clearing feedback_confidence_threshold (stricter than the
        healing-acceptance threshold itself) get stored; everything else is
        silently ignored -- this is a deliberate filter, not a bug if a
        sample that passed healing doesn't show up here."""
        if confidence < self.feedback_confidence_threshold:
            return
        buf = self._healed_memory[class_idx]
        buf.append(candidate.detach().clone())
        if len(buf) > self.feedback_buffer_size:
            buf.pop(0)  # evict oldest -- FIFO, matches the "recency" spirit of an EMA

    def _blend_with_healed_memory(self, class_idx: int, real_centroid: torch.Tensor,
                                   n_real: int, success_rate: Optional[float]) -> torch.Tensor:
        """Folds down-weighted, capped healed-sample memory into a class's
        centroid. Deliberately only called for classes already ACTIVE via
        real data this round (see update_and_project) -- healed memory
        refines an already-active class's estimate, it does NOT by itself
        make an inactive class active. That's a conservative choice: a
        class with zero real exemplars this round stays on the existing
        drift-decay/snapshot fallback rather than being kept "active"
        purely by its own past self-generated content, which would be
        exactly the circular-reinforcement risk this whole feature has to
        guard against.

        NEW GATE (see "THIRD UPDATE" in the module docstring): even for an
        active class, blending is skipped entirely if that class's
        historical success rate is already comfortably above
        feedback_success_rate_threshold -- a class reliably succeeding on
        its own doesn't need supplementary evidence, and uniformly applying
        this to EVERY class (including ones with no need for it) is exactly
        what measurably hurt majority-class healing in earlier testing.

        The 25%-style cap is enforced by SCALING healed samples' aggregate
        weight down when it would exceed feedback_max_share of the total,
        not by dropping samples arbitrarily -- so more real evidence this
        round naturally, smoothly dilutes healed memory's influence.
        """
        if success_rate is not None and success_rate >= self.feedback_success_rate_threshold:
            return real_centroid  # this class is already doing fine -- don't touch it

        healed_pool = self._healed_memory[class_idx]
        if not healed_pool or self.feedback_max_share <= 0.0 or n_real == 0:
            return real_centroid

        real_weight = float(n_real)
        max_healed_weight = (self.feedback_max_share / (1 - self.feedback_max_share)) * real_weight
        desired_healed_weight = self.feedback_weight * len(healed_pool)
        scale = min(1.0, max_healed_weight / desired_healed_weight) if desired_healed_weight > 0 else 0.0
        effective_healed_weight = desired_healed_weight * scale

        if effective_healed_weight <= 0.0:
            return real_centroid

        healed_mean = torch.stack(healed_pool).mean(dim=0)
        combined_weight = real_weight + effective_healed_weight
        return (real_weight * real_centroid + effective_healed_weight * healed_mean) / combined_weight

    def update_and_project(self, clean_z: torch.Tensor, clean_y: torch.Tensor,
                            class_success_rates: Optional[Dict[int, Optional[float]]] = None
                            ) -> Dict[int, torch.Tensor]:
        """Call ONCE per healing round, before generating any candidates
        this round. Updates persistent drift/snapshot/age state from this
        round's clean data, and returns {class_idx: projected_candidate}

        class_success_rates: optional {class_idx: historical cumulative
        success rate as of the END of the PREVIOUS round, or None if no
        history yet} -- typically sourced directly from a TransparencyLog
        instance (see defense.py's DALSE branch for exactly how). Omit
        entirely (None) to apply feedback uniformly to every active class,
        exactly the original behavior before this gate existed.
        (also cached for reconstruct_candidate to read)."""
        if clean_z.size(0) == 0:
            overall_anchor = torch.zeros(self.z_shape)
        else:
            overall_anchor = clean_z.mean(dim=0)

        # Recomputed every round from THIS round's own clean data -- a fixed
        # absolute noise magnitude would be wrong across different z_shape/
        # scale scenarios, since raw smashed data's natural spread isn't
        # assumed. If clean_z is empty this round, keep whatever sigma was
        # last observed rather than collapsing to zero.
        if clean_z.size(0) > 0:
            self._round_noise_sigma = float(clean_z.std().item()) * self.noise_sigma_fraction

        projected: Dict[int, torch.Tensor] = {}
        for c in range(self.num_classes):
            class_mask = clean_y == c
            n_exemplars = int(class_mask.sum().item())
            active = n_exemplars >= self.min_exemplars_for_active

            if active:
                real_centroid = clean_z[class_mask].mean(dim=0)
                success_rate = class_success_rates.get(c) if class_success_rates else None
                class_centroid = self._blend_with_healed_memory(c, real_centroid, n_exemplars, success_rate)
                delta_t = class_centroid - overall_anchor
                if not self._initialized[c]:
                    self.state.drift[c] = delta_t              # first observation, no prior to blend with
                else:
                    self.state.drift[c] = (
                        self.beta * self.state.drift[c] + (1 - self.beta) * delta_t
                    )  # Eq. 2, active branch
            else:
                self.state.drift[c] = self.decay_lambda * self.state.drift[c]  # Eq. 2, inactive branch

            proj = overall_anchor + self.state.drift[c]  # Eq. 3

            if not self._initialized[c]:
                final = proj  # no snapshot yet on the very first sighting of this class
            else:
                gamma = 1.0 - float(torch.exp(torch.tensor(-self.eta * self.state.age[c])))  # Eq. 4
                final = gamma * self.state.snapshot[c] + (1 - gamma) * proj

            projected[c] = final

            if active:
                self.state.snapshot[c] = final.detach().clone()
                self.state.age[c] = 0
                self._initialized[c] = True
            else:
                self.state.age[c] += 1

        self._last_projected = projected
        return projected

    def reconstruct_candidate(self, class_idx: int) -> torch.Tensor:
        """Adds FRESH Gaussian noise on every call -- see module docstring's
        "UPDATE" section for why. Two successive calls for the same class,
        within the same round, now genuinely differ (unless
        noise_sigma_fraction=0.0, which recovers the original deterministic
        behavior exactly)."""
        point = self._last_projected[class_idx]
        if self._round_noise_sigma == 0.0:
            return point.clone()
        noise = torch.randn_like(point) * self._round_noise_sigma
        return point + noise
