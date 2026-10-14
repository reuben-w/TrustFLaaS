"""
Semantically Consistent Substitution (HealSplit's "healing"/recovery step).

Pipeline: TAS flags poisoned samples -> a GAN trained on the TAS-clean
remainder generates candidate replacements -> a validator (consistency
validation student -- see consistency_student.py for what stands in for it
at this stage) gates each candidate -> accepted candidates replace the
flagged samples' smashed data; rejected candidates are retried a fixed
number of times, then the sample is dropped from training rather than kept
poisoned.

Design decision -- the accept/reject rule, since the paper doesn't give one:
"high confidence and label consistency" (the paper's only description of
this gate) is defined here as: the validator's softmax confidence on the
candidate is >= `confidence_threshold`, AND its predicted class matches the
TRUE label of the sample being replaced (not the generator's own belief
about what it made -- we know the true label already, since Smashed
Poisoning corrupts z, not y, so the label is trustworthy metadata to check
the candidate against). This is a defensible reading of "label consistency"
but not the only possible one -- documented here as our own design choice.

Design decision -- what happens to a candidate that keeps failing the gate:
Not specified by the paper. We retry generation up to `max_retries` times
(fresh noise each try), then DROP the sample from this round's training
data if it still hasn't passed. Dropping is safer than either of the two
alternatives: keeping the original poisoned z (defeats the point of
detection) or accepting a low-confidence/wrong-label substitute (could
inject a *new*, GAN-shaped corruption instead of the original one). Losing
a small number of samples per round is a mild, bounded cost; the other two
options both reintroduce exactly the problem substitution exists to solve.
"""
import dataclasses
from typing import Optional

import torch

from .gan import Generator


@dataclasses.dataclass
class SubstitutionResult:
    healed_z: torch.Tensor          # [N, ...] -- same shape as input Z, poisoned entries replaced
    keep_mask: torch.Tensor         # [N] bool -- False for samples dropped (see module docstring)
    num_replaced: int
    num_dropped: int
    num_unflagged: int
    total_attempts: int = 0         # sum of generate+validate attempts across all flagged samples --
                                     # instrumentation only (added for the DALSE-vs-GAN latency
                                     # comparison), doesn't affect accept/reject decisions at all


def validate_candidate(candidate: torch.Tensor, target_label: int, validator: torch.nn.Module,
                        confidence_threshold: float, device: torch.device) -> bool:
    """One candidate (no batch dim) -> accept/reject bool. See module
    docstring for the accept/reject rule this implements."""
    accepted, _ = _validate_candidate_with_confidence(candidate, target_label, validator,
                                                        confidence_threshold, device)
    return accepted


def _validate_candidate_with_confidence(candidate: torch.Tensor, target_label: int,
                                         validator: torch.nn.Module, confidence_threshold: float,
                                         device: torch.device) -> tuple:
    """Internal: same accept/reject rule as validate_candidate, but also
    returns the validator's raw confidence -- needed by DALSE's feedback
    mechanism (dalse_healing.py) to decide whether an accepted candidate
    clears the STRICTER feedback_confidence_threshold, separate from the
    ordinary healing-acceptance threshold. Not exposed publicly; the GAN
    and BAGAN-GP paths have no equivalent use for the raw value, so their
    call sites keep using the simpler validate_candidate above unchanged."""
    validator.eval()
    with torch.no_grad():
        logits = validator(candidate.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=1).squeeze(0)
        conf, pred = probs.max(dim=0)
    accepted = bool(conf.item() >= confidence_threshold and pred.item() == target_label)
    return accepted, float(conf.item())


def semantically_consistent_substitution(
    Z: torch.Tensor,
    y: torch.Tensor,
    flagged_poisoned: torch.Tensor,   # bool mask from detection.py::detect_poisoned
    generator: Generator,
    validator: torch.nn.Module,
    device: torch.device,
    confidence_threshold: float = 0.7,
    max_retries: int = 5,
) -> SubstitutionResult:
    """Replaces each flagged sample's z with a validator-approved GAN
    candidate for that sample's true label, or drops the sample if no
    candidate passes within max_retries attempts."""
    healed_z = Z.clone()
    keep_mask = torch.ones(Z.size(0), dtype=torch.bool)
    num_replaced = num_dropped = total_attempts = 0
    generator.eval()

    flagged_idx = torch.nonzero(flagged_poisoned, as_tuple=True)[0]
    for idx in flagged_idx.tolist():
        target_label = int(y[idx].item())
        accepted = None
        for _ in range(max_retries):
            total_attempts += 1
            with torch.no_grad():
                candidate = generator.sample(1, device)[0].cpu()
            if validate_candidate(candidate, target_label, validator, confidence_threshold, device):
                accepted = candidate
                break
        if accepted is not None:
            healed_z[idx] = accepted
            num_replaced += 1
        else:
            keep_mask[idx] = False
            num_dropped += 1

    return SubstitutionResult(
        healed_z=healed_z,
        keep_mask=keep_mask,
        num_replaced=num_replaced,
        num_dropped=num_dropped,
        num_unflagged=int((~flagged_poisoned).sum().item()),
        total_attempts=total_attempts,
    )


def bagan_gp_consistent_substitution(
    Z: torch.Tensor,
    y: torch.Tensor,
    flagged_poisoned: torch.Tensor,
    generator,   # ConditionalGenerator (bagan_gp.py) -- not type-hinted directly to avoid a circular import
    validator: torch.nn.Module,
    device: torch.device,
    confidence_threshold: float = 0.7,
    max_retries: int = 5,
) -> SubstitutionResult:
    """BAGAN-GP analogue of semantically_consistent_substitution: candidates
    come from a class-conditioned generator (bagan_gp.py) instead of the
    plain unconditioned GAN. The one structural difference is the sample
    call -- generator.sample(n, target_class, device) draws specifically
    from the flagged sample's own true-label region of the learned latent
    space, rather than a generic, untargeted sample. Retry/accept/reject
    logic is otherwise identical to semantically_consistent_substitution.
    """
    healed_z = Z.clone()
    keep_mask = torch.ones(Z.size(0), dtype=torch.bool)
    num_replaced = num_dropped = total_attempts = 0
    generator.eval()

    flagged_idx = torch.nonzero(flagged_poisoned, as_tuple=True)[0]
    for idx in flagged_idx.tolist():
        target_label = int(y[idx].item())
        accepted = None
        for _ in range(max_retries):
            total_attempts += 1
            with torch.no_grad():
                candidate = generator.sample(1, target_label, device)[0].cpu()
            if validate_candidate(candidate, target_label, validator, confidence_threshold, device):
                accepted = candidate
                break
        if accepted is not None:
            healed_z[idx] = accepted
            num_replaced += 1
        else:
            keep_mask[idx] = False
            num_dropped += 1

    return SubstitutionResult(
        healed_z=healed_z,
        keep_mask=keep_mask,
        num_replaced=num_replaced,
        num_dropped=num_dropped,
        num_unflagged=int((~flagged_poisoned).sum().item()),
        total_attempts=total_attempts,
    )


def dalse_consistent_substitution(
    Z: torch.Tensor,
    y: torch.Tensor,
    flagged_poisoned: torch.Tensor,
    healer,   # DALSEHealer (dalse_healing.py) -- not type-hinted directly to avoid a circular import
    validator: torch.nn.Module,
    device: torch.device,
    confidence_threshold: float = 0.7,
    max_retries: int = 5,
) -> SubstitutionResult:
    """DALSE-style analogue of semantically_consistent_substitution: candidates
    come from `healer.reconstruct_candidate` (persistent drift/snapshot state)
    instead of a GAN.

    UPDATE (see dalse_healing.py's module docstring for the full reasoning):
    this now retries on rejection, the same way the GAN path does. That
    wasn't meaningful under the original fully-deterministic design (a
    retry against the same fixed candidate produces the same rejection
    every time) -- it only became worthwhile once `reconstruct_candidate`
    started adding fresh noise on every call, so a retry can genuinely land
    on a different point. With `noise_sigma_fraction=0.0` on the healer,
    every retry is still identical to the first attempt, so max_retries has
    no effect and this behaves exactly as it did before -- one attempt per
    flagged sample, same as the original design.

    SECOND UPDATE: every ACCEPTED candidate is now reported back to the
    healer via record_healed_sample, along with the validator's confidence
    on it -- the healer itself decides (against its own, stricter
    feedback_confidence_threshold) whether it's trustworthy enough to
    influence future rounds. This call site doesn't apply that filter
    itself; it just reports everything accepted and lets dalse_healing.py
    own that decision, keeping the safeguard logic in one place.
    """
    healed_z = Z.clone()
    keep_mask = torch.ones(Z.size(0), dtype=torch.bool)
    num_replaced = num_dropped = total_attempts = 0

    flagged_idx = torch.nonzero(flagged_poisoned, as_tuple=True)[0]
    for idx in flagged_idx.tolist():
        target_label = int(y[idx].item())
        accepted = None
        for _ in range(max_retries):
            total_attempts += 1
            candidate = healer.reconstruct_candidate(target_label)
            passed, confidence = _validate_candidate_with_confidence(
                candidate, target_label, validator, confidence_threshold, device
            )
            if passed:
                accepted = candidate
                break
        if accepted is not None:
            healed_z[idx] = accepted
            num_replaced += 1
            if hasattr(healer, "record_healed_sample"):
                healer.record_healed_sample(target_label, accepted, confidence)
        else:
            keep_mask[idx] = False
            num_dropped += 1

    return SubstitutionResult(
        healed_z=healed_z,
        keep_mask=keep_mask,
        num_replaced=num_replaced,
        num_dropped=num_dropped,
        num_unflagged=int((~flagged_poisoned).sum().item()),
        total_attempts=total_attempts,
    )


def hybrid_consistent_substitution(
    Z: torch.Tensor,
    y: torch.Tensor,
    flagged_poisoned: torch.Tensor,
    bagan_generator,       # ConditionalGenerator (bagan_gp.py)
    dalse_healer,          # DALSEHealer (dalse_healing.py)
    validator: torch.nn.Module,
    device: torch.device,
    escalate_class_indices,   # set/list of class indices routed to BAGAN-GP; every other
                               # class routes to DALSE
    confidence_threshold: float = 0.7,
    max_retries: int = 5,
) -> SubstitutionResult:
    """Per-class ROUTER, not a new healing mechanism of its own: splits this
    round's flagged samples by class, sends escalated classes through
    bagan_gp_consistent_substitution and everything else through
    dalse_consistent_substitution, then merges the two results. Both
    underlying functions are called exactly as they already are elsewhere
    (same signatures, same behavior) -- this is composition, not a
    reimplementation of either path's accept/reject/retry logic.

    Motivation: uniform, class-blind DALSE feedback measurably hurt
    majority-class healing without a clear net win on the minority class
    either (see dalse_healing.py's module docstring, "THIRD UPDATE"), and a
    success-rate-gated version of that feedback reintroduced the exact
    all-or-nothing instability it was meant to fix, just one level up (does
    help even turn on this round, rather than does an individual heal
    pass). This sidesteps trying to make DALSE itself minority-friendly at
    all: BAGAN-GP already showed strong, low-variance minority-class
    results on its own; DALSE already showed strong majority-class results
    on its own. This just uses each one where it already won, rather than
    tuning either to try to do both jobs.
    """
    escalate_set = set(escalate_class_indices)
    escalate_mask = torch.zeros_like(flagged_poisoned)
    for c in escalate_set:
        escalate_mask |= (y == c)

    flagged_bagan = flagged_poisoned & escalate_mask
    flagged_dalse = flagged_poisoned & (~escalate_mask)

    result_bagan = bagan_gp_consistent_substitution(
        Z, y, flagged_bagan, bagan_generator, validator, device,
        confidence_threshold=confidence_threshold, max_retries=max_retries,
    )
    result_dalse = dalse_consistent_substitution(
        Z, y, flagged_dalse, dalse_healer, validator, device,
        confidence_threshold=confidence_threshold, max_retries=max_retries,
    )

    healed_z = Z.clone()
    keep_mask = torch.ones(Z.size(0), dtype=torch.bool)

    bagan_idx = torch.nonzero(flagged_bagan, as_tuple=True)[0]
    healed_z[bagan_idx] = result_bagan.healed_z[bagan_idx]
    keep_mask[bagan_idx] = result_bagan.keep_mask[bagan_idx]

    dalse_idx = torch.nonzero(flagged_dalse, as_tuple=True)[0]
    healed_z[dalse_idx] = result_dalse.healed_z[dalse_idx]
    keep_mask[dalse_idx] = result_dalse.keep_mask[dalse_idx]

    return SubstitutionResult(
        healed_z=healed_z,
        keep_mask=keep_mask,
        num_replaced=result_bagan.num_replaced + result_dalse.num_replaced,
        num_dropped=result_bagan.num_dropped + result_dalse.num_dropped,
        num_unflagged=int((~flagged_poisoned).sum().item()),
        total_attempts=result_bagan.total_attempts + result_dalse.total_attempts,
    )
