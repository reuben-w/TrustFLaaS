"""
Elastic Weight Consolidation (EWC), adapted from MConFL's Stage 2 (Yuan,
Wang, Xiong, Sun, Chen -- "Addressing Class Imbalance in Federated Learning:
A Perspective of Catastrophic Forgetting," Chaos, Solitons & Fractals, 2026),
itself building on the original EWC formulation (Kirkpatrick et al., PNAS
2017).

Core idea: rather than fixing class imbalance by reweighting the loss
(Class-Balanced Loss, Logit-Adjusted Loss -- both already tried, both landed
in the same "no real effect on frog, real cost to majority" place), EWC
protects the SHARED CLIENT MODEL'S PARAMETERS directly. Each round, it
identifies which parameters were important to what the model already knew
(via the Fisher information matrix -- an estimate of how sensitive the loss
is to each parameter) and penalizes moving those specific parameters too far
from their last known-good values, leaving less-important parameters free to
keep adapting.

Design decisions -- translating MConFL's setup to ours, since it differs in
real ways worth being explicit about rather than assuming a direct port:

- MConFL's own experiments run 200 communication rounds with M local steps
  between each aggregation; ours run 10 rounds with aggregation every round
  (M=1 in their notation). Their "Stage 1 for N_g rounds, then switch to
  Stage 2" schedule doesn't translate as a literal round-count formula at
  this scale -- this module instead takes a plain ROUND NUMBER
  (`ewc_start_round`) at which the penalty begins applying. Their own
  ablation is the reason this staging exists at all: applying the penalty
  from round 1 caused a catastrophic performance collapse in their own
  experiments (their Table 12/13), not just an underperformance -- so
  `ewc_start_round` is not an optional nicety, skipping it is expected to
  reproduce that same collapse here.

- MConFL computes the Fisher matrix PER CLIENT, then aggregates (weighted
  average) into a global Fisher matrix -- this is where non-IID client
  divergence actually matters in their design. For this first IID test, a
  single Fisher matrix is estimated from a pooled sample of this round's
  data across clients, rather than computed per-client and aggregated --
  a deliberate simplification that's reasonable ONLY because IID clients
  should already look similar to each other. Genuine per-client Fisher
  computation is needed once this moves to a non-IID test, since that's
  exactly the case their own design is built around.

- The anchor (theta*) is rolled forward every round EWC is active, matching
  MConFL's own Eq. 9 -- "we keep theta*=theta_t0_g...of EACH aggregation,"
  not a single fixed point from when Stage 2 begins. Each round's penalty
  pulls that round's local update back toward the IMMEDIATELY PRECEDING
  global state, not toward some earlier snapshot.

- EWC only applies to the CLIENT model's parameters, not the server
  model's -- deliberately scoped this way because the phenomenon this is
  meant to counter (RQ1's finding: the SHARED representation gets reshaped
  unevenly by imbalanced training) is specifically about the client-side
  representation, not the server-side classifier head.

- Fisher information requires an actual loss gradient with respect to the
  client model's parameters -- but the client model alone never produces a
  loss, only smashed data z. Estimating Fisher therefore needs a TEMPORARY,
  fully-connected forward pass (client -> server -> loss), separate from
  Phase C's usual detach-and-manually-inject-gradient pattern. This is a
  one-off diagnostic computation, not part of the main training step, so
  reconnecting the graph for it doesn't change anything about how the main
  training loop itself works.
  
Adapted from MConFL (Yuan et al., 2026) and the original EWC formulation (Kirkpatrick et al.,
PNAS 2017). Written from the papers, not copied from a repository.
"""
import dataclasses
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


@dataclasses.dataclass
class EWCLog:
    fisher_mean: float       # mean Fisher value across all parameters, this round -- a
                              # sanity-checkable number: should be small but nonzero
    anchor_distance: float    # ||current params - anchor|| at the moment the anchor was
                              # just refreshed -- always ~0 right after refresh, useful
                              # for confirming the anchor is actually being updated


class EWCState:
    """Persistent across rounds -- create ONCE outside the round loop, same
    lifecycle as DALSEHealer and TransparencyLog. Only meaningfully active
    once ewc_lambda > 0.0 AND the current round >= ewc_start_round; the
    caller (defense.py) is responsible for checking ewc_start_round and
    only calling refresh_anchor_and_fisher / compute_penalty once active."""

    def __init__(self, ewc_lambda: float = 0.0, fisher_num_batches: int = 5):
        self.ewc_lambda = ewc_lambda
        self.fisher_num_batches = fisher_num_batches
        self.anchor_params: Dict[str, torch.Tensor] = {}
        self.fisher: Dict[str, torch.Tensor] = {}
        self._initialized = False

    def refresh_anchor_and_fisher(
        self,
        client_model: nn.Module,
        server_model: nn.Module,
        sample_batches: List[Tuple[torch.Tensor, torch.Tensor]],
        criterion,
        device: torch.device,
    ) -> EWCLog:
        """Call ONCE per round, before Phase C's local training starts,
        using the CURRENT global client model (i.e. before this round's
        updates) -- this becomes this round's anchor. sample_batches: a
        handful of (x, y) batches (raw inputs, not smashed data) drawn from
        this round's data, used only to estimate Fisher -- kept small and
        bounded (fisher_num_batches) since this is an estimate, not an
        exact computation, and doesn't need the whole round's data."""
        client_model = client_model.to(device)
        server_model = server_model.to(device)

        # --- Anchor: snapshot of the CURRENT global client params ---
        new_anchor = {name: p.detach().clone() for name, p in client_model.named_parameters()}

        anchor_distance = 0.0
        if self._initialized:
            anchor_distance = sum(
                torch.norm(new_anchor[name] - self.anchor_params[name]).item() ** 2
                for name in new_anchor
            ) ** 0.5
        self.anchor_params = new_anchor

        # --- Fisher: average squared gradient of the task loss w.r.t. client
        # params, over a small sample of THIS round's data. Temporarily
        # reconnects the client->server graph (unlike Phase C's normal
        # detach-and-inject pattern) purely for this diagnostic computation. ---
        fisher_accum = {name: torch.zeros_like(p) for name, p in client_model.named_parameters()}
        criterion_fn = criterion if callable(criterion) else nn.CrossEntropyLoss()
        n_batches_used = 0

        client_model.zero_grad()
        for x, y in sample_batches[: self.fisher_num_batches]:
            x, y = x.to(device), y.to(device)
            client_model.zero_grad()
            server_model.zero_grad()
            z = client_model(x)               # graph stays connected -- no detach
            logits = server_model(z)
            loss = criterion_fn(logits, y)
            loss.backward()
            for name, p in client_model.named_parameters():
                if p.grad is not None:
                    fisher_accum[name] += p.grad.detach() ** 2
            n_batches_used += 1

        if n_batches_used > 0:
            self.fisher = {name: fisher_accum[name] / n_batches_used for name in fisher_accum}
        else:
            self.fisher = {name: torch.zeros_like(p) for name, p in client_model.named_parameters()}

        client_model.zero_grad()
        server_model.zero_grad()

        self._initialized = True
        fisher_mean = (
            sum(f.sum().item() for f in self.fisher.values())
            / sum(f.numel() for f in self.fisher.values())
        ) if self.fisher else 0.0
        return EWCLog(fisher_mean=fisher_mean, anchor_distance=anchor_distance)

    def compute_penalty(self, client_model: nn.Module) -> torch.Tensor:
        """Returns a scalar tensor: (lambda/2) * sum_i Fisher_i * (theta_i -
        anchor_i)^2, computed over client_model's CURRENT parameters
        (mid-training, with graph attached) against the stored anchor.
        Caller is expected to .backward() this separately, accumulating
        into whatever gradients the main task loss already produced --
        see module docstring for why this two-call pattern is correct
        here rather than combining into one loss tensor."""
        if not self._initialized or self.ewc_lambda <= 0.0:
            return torch.tensor(0.0)
        penalty = torch.tensor(0.0, device=next(client_model.parameters()).device)
        for name, p in client_model.named_parameters():
            if name in self.anchor_params:
                penalty = penalty + (self.fisher[name] * (p - self.anchor_params[name]) ** 2).sum()
        return 0.5 * self.ewc_lambda * penalty
