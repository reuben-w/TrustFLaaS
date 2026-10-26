"""
Anomaly-Influence Debiasing (AD) Teacher: inter-task influence matrix
(Eq. 8) and its loss (Eq. 9).

Design decision -- what "R" is:
Eq. 3 (detection.py) defines TAS as a per-node SCORE (a vector, r_k), but
Eq. 8 calls R a "matrix." Taking R as a per-node score turned into a
diagonal matrix doesn't work dimensionally with how it's used: Eq. 8
computes `R (Hadamard) (E G_p F^T)`, an elementwise product with a full
K x K matrix -- if R were diagonal, that product would zero out everything
except the diagonal, destroying essentially all the cross-node structure
Eq. 8 is trying to build ("a label-aware transition matrix that integrates
task-level and label-level relationships" -- the paper's own description).
We interpret R as the K x K weighted mutual-KNN adjacency matrix from TAS's
OWN graph construction (Eq. 2 / detection.py::build_mutual_knn_graph) --
the graph structure TAS's scores were computed FROM, not the scores
themselves. This is dimensionally consistent, keeps the "TAS matrix"
naming plausible (it's the matrix TAS is built on), and gives Eq. 8 a
sensible reading: "combine which nodes are topologically related (R) with
whether their labels are gradient-aligned (E G_p F^T)" -- exactly the paper's
own gloss on what M_p is for.

Design decision -- solving via torch.linalg.solve, not explicit inversion:
Eq. 8's (I_K - beta*P)^-1 is only ever used multiplied against F -- solving
the linear system (I_K - beta*P) X = F directly is both faster and more
numerically stable than forming the full inverse and multiplying, for the
same result. beta < 1 and P row-normalized (spectral radius <= 1, the same
convergence property PPR relies on in detection.py) keeps (I_K - beta*P)
comfortably invertible for the beta range used here.
"""
import torch
import torch.nn as nn

from .multihead_model import MultiHeadModel


def one_hot_matrix(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """E or F in Eq. 8: [K, num_classes] node-to-label incidence matrix."""
    K = labels.shape[0]
    M = torch.zeros(K, num_classes, device=labels.device)
    M[torch.arange(K), labels] = 1.0
    return M


def row_normalize(M: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    row_sums = M.sum(dim=1, keepdim=True).clamp(min=eps)
    return M / row_sums


def compute_influence_matrix(R: torch.Tensor, G_p: torch.Tensor,
                              E: torch.Tensor, F: torch.Tensor, beta: float = 0.3) -> torch.Tensor:
    """Eq. 8. R: [K,K] graph (see module docstring). G_p: [|Ytx|,|Yty|] from gis.py.
    E: [K,|Ytx|], F: [K,|Yty|] one-hot label incidence. Returns M_p: [|Ytx|,|Yty|]."""
    K = R.shape[0]
    EGF = (E @ G_p) @ F.T          # [K, K] -- label-alignment score, broadcast to node pairs
    weighted = R * EGF              # Hadamard product, Eq. 8
    P = row_normalize(weighted)
    I_K = torch.eye(K, device=R.device, dtype=R.dtype)
    X = torch.linalg.solve(I_K - beta * P, F)   # [K, |Yty|]
    M_p = (1 - beta) * (E.T @ X)                # [|Ytx|, |Yty|]
    return M_p


def ad_teacher_loss(logits_a: torch.Tensor, logits_b: torch.Tensor, logits_c: torch.Tensor,
                     y_a: torch.Tensor, y_b: torch.Tensor, y_c: torch.Tensor,
                     M_ab: torch.Tensor, M_ac: torch.Tensor,
                     lambda_b: float = 1.0, lambda_c: float = 1.0) -> torch.Tensor:
    """Eq. 9. Per-sample, [M_p]_{y_a,y_b} looks up the influence weight for
    THAT sample's own (task-a label, task-b label) pair, using it to scale
    how much that sample's task-b (resp. task-c) loss contributes.

    Design decision -- squashing M_p through sigmoid before using it as a
    weight, caught during testing:
    Raw M_p values from compute_influence_matrix are unbounded and can be
    NEGATIVE (observed values from -46 to +88 in testing on a small
    synthetic batch). Used directly as a multiplicative weight on a
    (non-negative) cross-entropy loss, a negative weight makes the
    optimizer's job "increase this loss" rather than decrease it --
    training diverged to a loss of -2000+ within 15 epochs when this was
    tried without squashing. The paper gives no normalization step for
    M_p, but SOME bounding into a non-negative, bounded range is a
    practical necessity for M_p to function as a loss weight at all, not
    an optional refinement. We use sigmoid (bounded to (0,1), monotonic --
    preserves M_p's relative ordering between sample pairs, which is the
    property "larger M_p = more cooperative, weight it more" actually
    depends on) rather than e.g. a hard ReLU clip, since ReLU would zero
    out and discard all the ordering information among negative values.
    """
    ce = nn.CrossEntropyLoss(reduction="none")
    loss_a = ce(logits_a, y_a)
    loss_b = ce(logits_b, y_b)
    loss_c = ce(logits_c, y_c)

    weight_b = torch.sigmoid(M_ab[y_a, y_b])   # per-sample lookup, [K], squashed to (0,1)
    weight_c = torch.sigmoid(M_ac[y_a, y_c])   # per-sample lookup, [K], squashed to (0,1)

    total = loss_a + lambda_b * weight_b * loss_b + lambda_c * weight_c * loss_c
    return total.mean()


def train_ad_teacher(
    Z: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, y_c: torch.Tensor,
    R: torch.Tensor, num_clients: int, num_classes: int, device: torch.device,
    num_epochs: int = 30, lr: float = 0.01, beta: float = 0.3,
    lambda_b: float = 1.0, lambda_c: float = 1.0, seed: int = 0,
) -> MultiHeadModel:
    """Trains the AD Teacher (Eq. 9) on one round's data. M_p is computed
    ONCE up front from the current labels/graph, not recomputed every
    epoch -- Eq. 8 depends on G_p (gis.py), which itself depends on
    gradients of a model with respect to its CURRENT weights; recomputing
    M_p during training would mean optimizing against a constantly-shifting
    target. The paper doesn't specify update frequency, and computing M_p
    once per round (like TAS itself, Eq. 3-5) is the simplest reading
    consistent with everything else in the pipeline being a per-round
    snapshot operation.
    """
    torch.manual_seed(seed)
    model = MultiHeadModel(in_channels=Z.shape[1], num_clients=num_clients,
                            num_classes=num_classes).to(device)
    Z, y_a, y_b, y_c, R = Z.to(device), y_a.to(device), y_b.to(device), y_c.to(device), R.to(device)

    from .gis import compute_gis
    G_ab = compute_gis(model, Z, y_a, model.head_a, 2, y_b, model.head_b, num_clients)
    G_ac = compute_gis(model, Z, y_a, model.head_a, 2, y_c, model.head_c, num_classes)

    E_a = one_hot_matrix(y_a, 2)
    F_b = one_hot_matrix(y_b, num_clients)
    F_c = one_hot_matrix(y_c, num_classes)
    M_ab = compute_influence_matrix(R, G_ab.to(device), E_a, F_b, beta=beta)
    M_ac = compute_influence_matrix(R, G_ac.to(device), E_a, F_c, beta=beta)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = Z.size(0)
    for epoch in range(num_epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, 32):
            idx = perm[start:start + 32]
            logits_a, logits_b, logits_c = model(Z[idx])
            loss = ad_teacher_loss(logits_a, logits_b, logits_c,
                                    y_a[idx], y_b[idx], y_c[idx],
                                    M_ab, M_ac, lambda_b, lambda_c)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model
