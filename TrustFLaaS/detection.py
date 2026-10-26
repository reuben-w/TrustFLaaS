"""
Topology-Aware malicious data Detection (TAS), HealSplit paper Eqs. 2-5.

Pipeline: build a mutual-KNN graph over a round's smashed data -> propagate
scores through it via Personalized PageRank (PPR) -> low scores mean
"topologically isolated" -> threshold via KDE valley-finding to flag
poisoned samples.

Design decision -- degree must match the propagation matrix, or PPR blows up:
Caught during testing: an earlier version of this file computed `degree` from the
*weighted* graph W but normalized the *unweighted* adjacency propagation
step with it. Since individual edge weights are <1 (Gaussian kernel of a
positive distance), weighted degree < unweighted neighbor count, so dividing
by the smaller weighted degree injects extra mass every iteration instead of
conserving it -- the iteration diverges geometrically instead of converging.
`compute_tas` now derives degree from whichever matrix (`W` or the 0/1
adjacency) is actually used for propagation, controlled by the same
`use_edge_weights` flag, so the two are always consistent.

Design decision -- ambiguity in Eq. 3's propagation term:
Eq. 2 defines a *weighted* graph (Gaussian-kernel edge weights W_kj), but
Eq. 3's propagation sum, `sum_{w in N(k)} r_w^(t) / d_w`, has no W_kj factor
in it -- neighbor contributions are only normalized by the neighbor's own
degree, not scaled by the specific edge weight to k. Taken literally, this
means the *edge weights* only affect the graph through which pairs count as
neighbors (and, now, through nothing else in the default mode) -- not
through how much of w's score flows specifically to k vs. w's other
neighbors. We implement this literally (`use_edge_weights=False`, the
default) since that's what's written, but expose `use_edge_weights=True` as
an easy toggle to a more conventional weighted-PPR formulation (scale each
neighbor's contribution by W_kw too, with degree switched to match) in case
the literal reading under-performs in practice -- worth an ablation if
TAS's separation looks weak on real data.

Design decision -- what "argmin" means in Eq. 5:
`T = min(argmin_r f_hat(r), Q_rho({r_k}))`. A literal global argmin over the
whole KDE curve would usually just land at the extreme edge of the score
range (density trails off to ~0 there), which isn't a meaningful "valley
between two modes." We interpret this as: find the first local minimum of
the density curve after its first (highest) peak -- i.e. the valley between
the main clean-cluster mode and a secondary poisoned-cluster mode, if one
exists. This is the standard reading of "KDE + adaptive threshold" bimodal
splitting in the anomaly-detection literature the paper cites this
technique from. If no second mode/valley is found (e.g. no attack present,
unimodal distribution), we fall back to the percentile floor alone.

Design decision -- alpha default, based on empirical testing (not the paper):
Table 1 doesn't give alpha a value, and neither does the text beyond "alpha
controls the trade-off between local and global propagation." A quick grid
search on synthetic data (one connected clean mass vs. an isolated poisoned
satellite -- see TrustFLaaS/README.md) showed alpha=0.15 (a low value,
which we tried first by analogy to PageRank's usual restart probability)
gives noticeably WEAKER separation than mid-to-high alpha: at low alpha, the
per-node teleport term v_k (a purely local, degree-based prior, not
connectivity-aware) dominates the score, drowning out the propagation term
that actually carries the "is this node reachable from the big connected
mass" signal TAS needs. We default to alpha=0.5 as a more reasonable
starting point; this is a hyperparameter you'll likely want to tune on real
smashed data (evaluate_detection.py reports precision/recall to tune
against), not a value to treat as settled.
"""
import dataclasses
from typing import Optional

import numpy as np
import torch


def build_mutual_knn_graph(Z: torch.Tensor, k: int) -> torch.Tensor:
    """Eq. 2. Z: [N, ...] smashed-data vectors -- any per-sample shape is
    accepted and flattened internally (a [N, C, H, W] feature map works
    the same as a pre-flattened [N, D] input), since distance computation
    only needs the flat representation. Returns W [N, N], the weighted
    mutual-KNN adjacency matrix."""
    Z = Z.reshape(Z.size(0), -1)
    N = Z.shape[0]
    k = min(k, N - 1)  # can't ask for more neighbors than exist
    dist = torch.cdist(Z, Z)
    dist.fill_diagonal_(float("inf"))

    knn_dist, knn_idx = torch.topk(dist, k, largest=False, dim=1)
    sigma = knn_dist.median()
    gamma = 1.0 / (2 * sigma ** 2 + 1e-12)

    neighbor_mask = torch.zeros(N, N, dtype=torch.bool)
    neighbor_mask.scatter_(1, knn_idx, True)
    mutual_mask = neighbor_mask & neighbor_mask.T  # "if z_j in N_k AND z_k in N_j"

    W = torch.zeros(N, N, dtype=Z.dtype)
    dist_sq = dist ** 2
    dist_sq[dist_sq == float("inf")] = 0.0  # diagonal, masked out anyway
    W[mutual_mask] = torch.exp(-gamma * dist_sq[mutual_mask])
    return W


def compute_tas(W: torch.Tensor, alpha: float = 0.5, num_iters: int = 20,
                 eps: float = 1.0, use_edge_weights: bool = False) -> torch.Tensor:
    """Eq. 3, via power iteration (standard way to compute PPR -- the
    iterative update in the equation converges to a fixed point; we run a
    fixed number of iterations rather than checking for convergence
    explicitly, since alpha < 1 guarantees geometric convergence and
    num_iters=20 is comfortably enough for that at typical alpha values).

    `degree` is derived HERE, from whichever matrix is used for
    propagation, not passed in separately -- see module docstring.

    Design decision -- eps scale, caught during testing:
    The paper describes eps only as "a small constant to prevent division
    by zero" (Eq. 3's context), which reads like a numerical-safety epsilon
    (e.g. 1e-6) -- but v_k = 1/(d_k+eps) means a literally zero-degree node
    (no mutual-KNN reciprocity at all -- a real occurrence at cluster
    boundaries, not just for attacks) gets v_k ~ 1/eps. With eps=1e-6 that's
    ~1,000,000, versus ~0.1 for a normally-connected node -- six orders of
    magnitude, which then dominates every downstream score through the
    (1-alpha)*v_k term and defeats the "low score = isolated = poisoned"
    threshold rule the paper states, since the *most* isolated nodes end up
    with the single largest raw scores. We use eps=1.0 instead -- the same
    order of magnitude as typical node degrees here -- so it still prevents
    exact division by zero (the paper's stated purpose) without letting
    that safety term dominate the score for genuinely disconnected nodes.
    """
    adjacency = W if use_edge_weights else (W > 0).float()
    degree = adjacency.sum(dim=1)
    d_safe = degree.clone()
    d_safe[d_safe == 0] = eps  # isolated nodes with zero mutual-KNN edges

    v = 1.0 / (degree + eps)   # Eq. 3's t=0 branch, reused as the teleport vector v_k
    r = v.clone()

    for _ in range(num_iters):
        r = alpha * (adjacency @ (r / d_safe)) + (1 - alpha) * v
    return r


def adaptive_threshold(tas_scores: torch.Tensor, bandwidth: Optional[float] = None,
                        percentile: float = 20.0, grid_size: int = 300) -> float:
    """Eqs. 4-5. Returns a scalar threshold T; scores below T are flagged
    poisoned. See module docstring for the "argmin" interpretation."""
    scores = tas_scores.detach().cpu().numpy()
    n = len(scores)

    if bandwidth is None:
        std = scores.std()
        # Silverman's rule of thumb -- a standard default KDE bandwidth,
        # not specified in the paper (Table 1 doesn't list h either).
        bandwidth = 1.06 * std * n ** (-1 / 5) if std > 1e-12 else 1e-3

    grid = np.linspace(scores.min(), scores.max(), grid_size)
    diffs = (grid[:, None] - scores[None, :]) / bandwidth
    density = np.exp(-0.5 * diffs ** 2).sum(axis=1) / (n * bandwidth * np.sqrt(2 * np.pi))

    peak_idx = int(np.argmax(density))
    valley_idx = None
    for i in range(peak_idx + 1, len(density) - 1):
        if density[i] < density[i - 1] and density[i] < density[i + 1]:
            valley_idx = i
            break

    percentile_r = float(np.percentile(scores, percentile))
    if valley_idx is not None:
        return min(float(grid[valley_idx]), percentile_r)
    return percentile_r  # no second mode found -- fall back to the percentile floor alone


@dataclasses.dataclass
class DetectionResult:
    tas_scores: torch.Tensor        # [N], raw TAS per sample
    threshold: float
    predicted_poisoned: torch.Tensor  # [N] bool, True = flagged as poisoned


def detect_poisoned(Z: torch.Tensor, k: int = 10, alpha: float = 0.5,
                     num_iters: int = 20, percentile: float = 20.0,
                     use_edge_weights: bool = False) -> DetectionResult:
    """End-to-end TAS pipeline: graph -> PPR scores -> adaptive threshold -> mask."""
    W = build_mutual_knn_graph(Z, k)
    tas_scores = compute_tas(W, alpha=alpha, num_iters=num_iters,
                              use_edge_weights=use_edge_weights)
    threshold = adaptive_threshold(tas_scores, percentile=percentile)
    predicted_poisoned = tas_scores < threshold
    return DetectionResult(tas_scores=tas_scores, threshold=threshold,
                            predicted_poisoned=predicted_poisoned)
