"""
Supervised contrastive representation learning, adapted from FedIIC (Wu, Yu,
Yang, Cheng, Yan -- "FedIIC: Towards Robust Federated Learning for
Class-Imbalanced Medical Image Classification," MICCAI 2023), itself built
on the standard supervised contrastive loss (Khosla et al., NeurIPS 2020).

Core idea, different in kind from every other intervention tried so far
(noise, feedback, Class-Balanced Loss, Logit-Adjusted Loss, EWC -- all of
which either reweight the loss surface or protect existing parameters):
this directly shapes WHAT THE REPRESENTATION LOOKS LIKE, by training the
client model so that same-class smashed-data embeddings end up close
together and different-class embeddings end up far apart, using cosine
similarity in the embedding space.

Design decision -- intra-client vs inter-client, and why this module does
neither exactly as FedIIC's own paper describes:

FedIIC's own design has two separate contrastive terms because standard FL
genuinely can't let one client see another's raw data -- "inter-client"
contrastive learning there needs prototype-exchange machinery specifically
to work around that constraint. Our split-learning setup doesn't have that
constraint: smashed data from EVERY client is already pooled at the server
every round (that's what Z_round is), so cross-client comparison is
available directly, with no exchange protocol needed.

This matters more than it might first appear for our specific problem:
FedIIC's "intra-client" term (contrastive learning within one client's own
local batch) would be close to useless for a scarce class here, since a
single client's single batch often has zero or one frog samples -- too few
to form even one positive pair, precisely for the class this is meant to
help. Pooling across all clients gives a scarce class a real chance at
having same-class positives to learn from. This module therefore applies
ONE contrastive step per round, on POOLED data spanning every client, as a
dedicated refinement of the aggregated global client model -- not mixed
into Phase C's existing per-client training loop (which trains five
separate diverging copies of the client model; there's no single "current"
model mid-Phase-C to meaningfully apply a cross-client loss to). This
happens AFTER Phase C's normal training and FedAvg aggregation, directly
refining the newly-aggregated global model.

Loss formulation: standard SupCon, "L_out" variant (Khosla et al., Eq. 2) --
for each anchor with at least one same-class sample elsewhere in the batch,
maximize the log-probability of positives against ALL other samples in the
batch (softmax over cosine similarities, temperature-scaled). Anchors with
no positives in a given batch contribute nothing to the loss, not an error
-- with a scarce class, some batches simply won't have a usable pair, and
that batch's contribution for that class is correctly zero rather than
undefined.

Implemented from the loss definition in Khosla et al., NeurIPS 2020 (Eq. 2, L_out), and from the
description of FedIIC (Wu et al., MICCAI 2023). The authors' public implementations are acknowledged
as references; any similarity to them is because we follow the same formula.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def supervised_contrastive_loss(z: torch.Tensor, y: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """z: [N, ...] embeddings (any shape per-sample, flattened internally
    and L2-normalized). y: [N] labels. Returns a scalar; 0.0 (not NaN) if no
    anchor in this batch has a same-class positive to compare against."""
    z_flat = z.flatten(1)
    z_norm = F.normalize(z_flat, dim=1)
    sim = torch.matmul(z_norm, z_norm.T) / temperature
    n = z_norm.size(0)

    sim_max = sim.max(dim=1, keepdim=True)[0].detach()
    sim = sim - sim_max  # numerical stability, standard log-sum-exp shift

    mask_self = torch.eye(n, dtype=torch.bool, device=z.device)
    mask_pos = (y.unsqueeze(0) == y.unsqueeze(1)) & (~mask_self)
    mask_valid_anchor = mask_pos.any(dim=1)

    if not mask_valid_anchor.any():
        return torch.tensor(0.0, device=z.device, requires_grad=True)

    exp_sim = torch.exp(sim).masked_fill(mask_self, 0.0)
    denom = exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-12)
    log_prob = sim - torch.log(denom)

    pos_count = mask_pos.sum(dim=1).clamp(min=1)
    mean_log_prob_pos = (mask_pos.float() * log_prob).sum(dim=1) / pos_count

    return -mean_log_prob_pos[mask_valid_anchor].mean()


def apply_contrastive_refinement(
    global_client_model: nn.Module,
    round_batches,           # List[(cid, x, y)] from Phase A -- pooled across ALL clients
    contrastive_lambda: float,
    contrastive_temperature: float,
    contrastive_num_batches: int,
    lr: float,
    device: torch.device,
) -> float:
    """Call ONCE per round, AFTER Phase C's normal per-client training and
    FedAvg aggregation -- refines the just-aggregated global_client_model
    directly, in place. Returns the mean contrastive loss observed (for
    logging/diagnostics), or 0.0 if contrastive_lambda<=0.0.

    Uses a POOLED sample of batches spanning every client (round_batches
    already mixes all clients together, in the order Phase A collected
    them) -- deliberately not scoped to one client at a time; see module
    docstring for why that's the point, not an oversight.
    """
    if contrastive_lambda <= 0.0 or not round_batches:
        return 0.0

    global_client_model = global_client_model.to(device)
    optimizer = torch.optim.SGD(global_client_model.parameters(), lr=lr)

    # Pool a sample of (x, y) across clients into fewer, larger batches --
    # gives scarce classes a better chance at a same-class positive than
    # any single client's own small batch would.
    pooled_x, pooled_y = [], []
    for _, x, y in round_batches[: max(contrastive_num_batches * 2, contrastive_num_batches)]:
        pooled_x.append(x)
        pooled_y.append(y)
    if not pooled_x:
        return 0.0
    pooled_x = torch.cat(pooled_x).to(device)
    pooled_y = torch.cat(pooled_y).to(device)

    losses = []
    batch_size = max(1, len(pooled_x) // max(1, contrastive_num_batches))
    for i in range(0, len(pooled_x), batch_size):
        x_b, y_b = pooled_x[i:i + batch_size], pooled_y[i:i + batch_size]
        if x_b.size(0) < 2:
            continue
        optimizer.zero_grad()
        z = global_client_model(x_b)
        loss = supervised_contrastive_loss(z, y_b, temperature=contrastive_temperature)
        if loss.requires_grad and loss.item() != 0.0:
            (contrastive_lambda * loss).backward()
            optimizer.step()
        losses.append(loss.item())

    return sum(losses) / len(losses) if losses else 0.0
