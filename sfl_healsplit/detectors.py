"""
Candidate detectors and threshold rules for the final TrustFLS detection step.

Everything here is pure torch/numpy/scipy and does not touch the training code, so each
candidate can be compared offline (see eval_detectors.py) before anything in defense.py changes.

Convention: a detector returns a SUSPICION score per sample, higher = more suspicious.
A threshold rule turns suspicion scores into a boolean "flagged" mask. Keeping the two apart
lets the evaluator measure score quality (AUROC) separately from the rule that picks a cut-off.

Detector inputs: Z [N, ...] smashed data, y [N] labels (the server holds them), cid [N] client ids.
"""
from typing import Callable, Dict

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .detection import build_mutual_knn_graph, compute_tas, detect_poisoned


def _flat(Z: torch.Tensor) -> torch.Tensor:
    return Z.reshape(Z.size(0), -1).float()


# ----------------------------------------------------------------------------- graphs
def knn_adjacency(Z: torch.Tensor, k: int = 10, mode: str = "union") -> torch.Tensor:
    """Boolean [N, N] adjacency of the k-nearest-neighbour graph.
    mode="mutual": edge only if each is in the other's k nearest. mode="union": either one."""
    Zf = _flat(Z)
    n = Zf.size(0)
    k = min(k, n - 1)
    dist = torch.cdist(Zf, Zf)
    dist.fill_diagonal_(float("inf"))
    idx = torch.topk(dist, k, largest=False, dim=1).indices
    a = torch.zeros(n, n, dtype=torch.bool)
    a.scatter_(1, idx, True)
    return (a & a.T) if mode == "mutual" else (a | a.T)


# ----------------------------------------------------------------------------- detectors
def det_tas_current(Z, y, cid, k=10, alpha=0.5):
    """The TAS used in every run so far. Score r falls as mutual degree rises, and the pipeline
    flags LOW r, so suspicion = -r."""
    r = compute_tas(build_mutual_knn_graph(Z, k), alpha=alpha)
    return (-r).double().numpy()


def det_tas_flipped(Z, y, cid, k=10, alpha=0.5):
    """Same scores, opposite sign: flag HIGH r (isolated, low-degree samples)."""
    r = compute_tas(build_mutual_knn_graph(Z, k), alpha=alpha)
    return r.double().numpy()


def det_ppr_uniform(Z, y, cid, k=10, alpha=0.5, iters=20, graph="union"):
    """Standard personalised PageRank with a UNIFORM teleport vector: well-connected samples get
    high mass, isolated or small groups get low mass. Suspicion = -mass."""
    a = knn_adjacency(Z, k, graph).float()
    n = a.size(0)
    deg = a.sum(1).clamp(min=1.0)
    v = torch.full((n,), 1.0 / n)
    r = v.clone()
    for _ in range(iters):
        r = alpha * (a @ (r / deg)) + (1 - alpha) * v
    return (-r * n).double().numpy()


def det_component_small(Z, y, cid, k=10, graph="union"):
    """Size of the connected component of the kNN graph that a sample sits in. A group of samples
    shifted away from the bulk forms its own small component. Suspicion = 1 - size / N."""
    a = knn_adjacency(Z, k, graph).numpy()
    _, labels = connected_components(csr_matrix(a), directed=False)
    sizes = np.bincount(labels)
    return 1.0 - sizes[labels] / len(labels)


def det_knn_dist(Z, y, cid, k=10):
    """Mean distance to the k nearest neighbours, divided by the median. Catches isolated points
    only; a shifted GROUP is as tightly packed as the bulk, so this is expected to fail on it."""
    Zf = _flat(Z)
    dist = torch.cdist(Zf, Zf)
    dist.fill_diagonal_(float("inf"))
    d = torch.topk(dist, min(k, Zf.size(0) - 1), largest=False, dim=1).values.mean(1)
    return (d / d.median()).double().numpy()


def _class_centres(Zf, y, min_n):
    """Coordinate-wise median centre and a robust spread (median distance to the centre) per class.
    Classes with fewer than min_n samples fall back to the global centre and spread."""
    g_centre = Zf.median(dim=0).values
    g_scale = (Zf - g_centre).norm(dim=1).median().clamp(min=1e-8)
    centres, scales = {}, {}
    for c in torch.unique(y).tolist():
        m = y == c
        if int(m.sum()) >= min_n:
            ctr = Zf[m].median(dim=0).values
            sc = (Zf[m] - ctr).norm(dim=1).median().clamp(min=1e-8)
        else:
            ctr, sc = g_centre, g_scale
        centres[c], scales[c] = ctr, sc
    return centres, scales


def det_class_median(Z, y, cid, min_n=5):
    """Distance from each sample to the robust centre of ITS OWN CLASS, in units of that class's
    typical spread. Uses the labels the server already holds. A fixed offset added to a minority of a
    class's samples pushes them far from the class median."""
    Zf = _flat(Z)
    centres, scales = _class_centres(Zf, y, min_n)
    out = np.zeros(len(y))
    for c in centres:
        m = y == c
        out[m.numpy()] = ((Zf[m] - centres[c]).norm(dim=1) / scales[c]).double().numpy()
    return out


def det_client_shift(Z, y, cid, min_n=3):
    """Per-client score. For each class the client holds (at least min_n samples), compare its class
    mean with the coordinate-wise median of the OTHER clients' class means, in units of the class's
    typical spread; the client score is the median over classes. Every sample gets its client's score.
    SP shifts all of a client's samples by the same offset, so the client mean moves by that offset."""
    Zf = _flat(Z)
    _, scales = _class_centres(Zf, y, 5)
    clients = torch.unique(cid).tolist()
    means = {}
    for m in clients:
        for c in torch.unique(y[cid == m]).tolist():
            sel = (cid == m) & (y == c)
            if int(sel.sum()) >= min_n:
                means[(m, c)] = Zf[sel].mean(dim=0)
    score = {m: 0.0 for m in clients}
    for m in clients:
        shifts = []
        for c in torch.unique(y).tolist():
            if (m, c) not in means:
                continue
            others = [means[(o, c)] for o in clients if o != m and (o, c) in means]
            if len(others) < 2:
                continue
            ref = torch.stack(others).median(dim=0).values
            shifts.append(float((means[(m, c)] - ref).norm() / scales[c]))
        if shifts:
            score[m] = float(np.median(shifts))
    return np.array([score[int(c)] for c in cid.tolist()])


DETECTORS: Dict[str, Callable] = {
    "tas_current": det_tas_current,
    "tas_flipped": det_tas_flipped,
    "ppr_uniform": det_ppr_uniform,
    "component_small": det_component_small,
    "knn_dist": det_knn_dist,
    "class_median": det_class_median,
    "client_shift": det_client_shift,
}


# ----------------------------------------------------------------------------- rules
def _ceiling(s: np.ndarray, flagged: np.ndarray, ceiling: float) -> np.ndarray:
    """Safety ceiling, not a target: if more than `ceiling` of the samples are flagged, keep only the
    highest-scoring ceiling*N."""
    n_max = int(np.floor(ceiling * len(s)))
    if flagged.sum() > n_max:
        keep = np.argsort(-s)[:n_max]
        out = np.zeros_like(flagged)
        out[keep] = True
        return out
    return flagged


def rule_top_fraction(s, frac=0.2, ceiling=0.5):
    """What the current fixed cap does: flag the top `frac` regardless of the data."""
    n = int(round(frac * len(s)))
    out = np.zeros(len(s), dtype=bool)
    out[np.argsort(-s)[:n]] = True
    return out


def rule_mad(s, tau=3.5, ceiling=0.5):
    """Flag s > median + tau * robust spread. Flags (almost) nothing when the scores are one cluster."""
    med = np.median(s)
    mad = 1.4826 * np.median(np.abs(s - med))
    spread = max(mad, 0.05 * abs(med), 1e-9)
    return _ceiling(s, s > med + tau * spread, ceiling)


def rule_valley(s, ceiling=0.5, grid_size=300):
    """Flag scores above the first valley of a Gaussian KDE found by scanning UP from the main peak.
    No valley means no second mode, so flag nothing. (The rule in detection.py scans up but flags
    LOW scores, so its valley can never separate a low-score group; see the notes.)"""
    n = len(s)
    std = s.std()
    if std < 1e-12:
        return np.zeros(n, dtype=bool)
    bw = 1.06 * std * n ** (-1 / 5)
    grid = np.linspace(s.min(), s.max(), grid_size)
    dens = np.exp(-0.5 * ((grid[:, None] - s[None, :]) / bw) ** 2).sum(1)
    peak = int(np.argmax(dens))
    for i in range(peak + 1, grid_size - 1):
        if dens[i] < dens[i - 1] and dens[i] < dens[i + 1]:
            return _ceiling(s, s > grid[i], ceiling)
    return np.zeros(n, dtype=bool)


RULES: Dict[str, Callable] = {
    "top20": lambda s, ceiling=0.5: rule_top_fraction(s, 0.2, ceiling),
    "mad3.5": lambda s, ceiling=0.5: rule_mad(s, 3.5, ceiling),
    "mad6": lambda s, ceiling=0.5: rule_mad(s, 6.0, ceiling),
    "valley": rule_valley,
}


def as_is_flags(Z, k=10, alpha=0.5, percentile=20.0) -> np.ndarray:
    """The detector and rule exactly as run so far (detection.detect_poisoned)."""
    return detect_poisoned(Z, k=k, alpha=alpha, percentile=percentile).predicted_poisoned.numpy()
