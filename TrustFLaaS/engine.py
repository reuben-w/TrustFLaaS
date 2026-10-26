"""
Split FL training engine.

Design decision -- SFLV1-style updates, and what "FedServer" / "main server"
map to in this code:
The HealSplit paper (following Thapa et al. 2022's SplitFed) states
"client-side updates are aggregated by the Fed server, while server-side
updates are aggregated by the main server," but doesn't spell out the exact
update schedule. SplitFed's original paper describes two variants:
SFLV1 (server-side model updated sequentially, once per client's batch,
within a round) and SFLV2 (gradients from all clients accumulated before a
single server-side step). We implement SFLV1, the simpler and more commonly
used variant in follow-up SFL literature:

  For each round:
    for each client (sequentially):
      - load the current *global* client-side weights into a local copy
      - for local_epochs, for each batch:
          forward: client-side model -> smashed data z
          forward: server-side model (the ONE shared instance) -> logits -> loss
          backward: server-side model gets its gradient and takes an
                     optimizer step immediately (this client's contribution
                     is now baked into the server-side model)
          the gradient w.r.t. z is sent back to the client, which backprops
          through its own client-side model and takes its own step
      - keep this client's resulting client-side weights for aggregation
    - FedAvg-aggregate all clients' client-side weights -> new global
      client-side model for next round
    - evaluate the (global client-side model, server-side model) pair on the
      centralized test set

Because there is only ONE server-side model instance, updated directly and
sequentially by every client in turn, there's no separate "aggregation"
step for it the way there is for client-side weights -- the sequential
online updates already are the aggregation mechanism. This is a
documented simplification consistent with SFLV1, not an oversight.

Design decision -- detaching at the smashed-data cut point:
`z_for_server = z.detach().requires_grad_(True)` breaks the autograd graph
at the client/server boundary and gives us an explicit tensor to read
`.grad` off after the server's backward pass -- this mirrors what actually
happens physically in SFL: the client sends the smashed-data *values*
(z_for_server here) over the wire, the server computes gradients with
respect to those received values, and sends the gradient *values* back.
There is never a live, differentiable path across that wire.
"""
import csv
import os
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .attacks import add_smashed_poison
from .model import ClientModel, ServerModel
from .per_class_eval import PerClassLogger


@torch.no_grad()
def estimate_smashed_data_scale(client_model: ClientModel, loader: DataLoader,
                                 device: torch.device) -> Tuple[float, Tuple[int, ...]]:
    """Runs one batch through client_model to get a representative smashed-data
    standard deviation and shape, for calibrating the SP attack's
    perturbation magnitude to whatever scale this architecture/dataset
    actually produces (see attacks.py::make_smashed_poison_deltas)."""
    client_model.to(device).eval()
    x, _ = next(iter(loader))
    z = client_model(x.to(device))
    return z.std().item(), tuple(z.shape[1:])


def fedavg_state_dicts(state_dicts_and_weights: List[Tuple[Dict[str, torch.Tensor], int]]) -> Dict[str, torch.Tensor]:
    """Weighted average of client-side state_dicts, weighted by #examples --
    identical aggregation rule to Flower's FedAvg in the plain-FL baseline,
    just applied to the (smaller) client-side submodel instead of a whole
    model, and implemented by hand since we're not using Flower here."""
    total = sum(w for _, w in state_dicts_and_weights)
    keys = state_dicts_and_weights[0][0].keys()
    avg: Dict[str, torch.Tensor] = {}
    for k in keys:
        acc = None
        for sd, w in state_dicts_and_weights:
            term = sd[k].float() * (w / total)
            acc = term if acc is None else acc + term
        avg[k] = acc.to(state_dicts_and_weights[0][0][k].dtype)
    return avg


def evaluate(client_model: ClientModel, server_model: ServerModel, loader: DataLoader,
             device: torch.device, label_flip_delta: Optional[int] = None,
             num_classes: int = 10) -> Tuple[float, float, float]:
    """Returns (loss, accuracy, flip_confusion_rate).

    flip_confusion_rate: fraction of test predictions equal to
    (true_label + label_flip_delta) mod C -- a diagnostic for how much the
    global model has been pulled toward the attacker's label mapping,
    beyond what plain accuracy alone shows. NaN if label_flip_delta is None
    (e.g. for a clean, no-attack run where this isn't a meaningful metric).
    """
    client_model.to(device).eval()
    server_model.to(device).eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct, total, confused = 0.0, 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = server_model(client_model(x))
            total_loss += criterion(logits, y).item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            if label_flip_delta is not None:
                flipped_target = (y + label_flip_delta) % num_classes
                confused += (preds == flipped_target).sum().item()
            total += y.size(0)
    flip_rate = (confused / total) if label_flip_delta is not None else float("nan")
    return total_loss / total, correct / total, flip_rate


def train_one_round(
    global_client_model: ClientModel,
    server_model: ServerModel,               # mutated in place (single shared instance)
    client_loaders: List[DataLoader],
    server_optimizer: torch.optim.Optimizer,  # persists across the whole run
    device: torch.device,
    local_epochs: int,
    lr_client: float,
    smashed_poison_deltas: Optional[Dict[int, torch.Tensor]] = None,
) -> ClientModel:
    """Runs one SFL round (all clients, sequentially) and returns the new,
    FedAvg-aggregated global client-side model.

    `smashed_poison_deltas`: optional {client_id: fixed delta_z tensor} from
    attacks.py::make_smashed_poison_deltas -- if a client's index (matching
    its position in `client_loaders`) has an entry, its smashed data z is
    perturbed by that FIXED delta right after the client-side forward pass,
    before being sent to the server. None (the default) makes this
    function's behavior identical to the original label-flip-only version
    -- label poisoning doesn't touch z at all, it's handled entirely by
    LabelFlipDataset at the data-loading level (see attacks.py), so this
    addition is purely orthogonal to that path.
    """
    criterion = nn.CrossEntropyLoss()
    client_results: List[Tuple[Dict[str, torch.Tensor], int]] = []
    smashed_poison_deltas = smashed_poison_deltas or {}

    for cid, loader in enumerate(client_loaders):
        client_model = type(global_client_model)(
            in_channels=global_client_model.conv1.in_channels
        ).to(device)
        client_model.load_state_dict(global_client_model.state_dict())
        client_optimizer = torch.optim.SGD(client_model.parameters(), lr=lr_client)
        delta_z = smashed_poison_deltas.get(cid)

        num_examples = len(loader.dataset)
        for epoch_idx in range(local_epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)

                # --- forward: client -> smashed data ---
                z = client_model(x)
                if delta_z is not None:
                    z = add_smashed_poison(z, delta_z)
                z_for_server = z.detach().requires_grad_(True)

                # --- forward + backward: server (shared instance, updated immediately) ---
                logits = server_model(z_for_server)
                loss = criterion(logits, y)
                server_optimizer.zero_grad()
                loss.backward()
                server_optimizer.step()

                # --- backward into the client using the gradient the server computed ---
                client_optimizer.zero_grad()
                z.backward(z_for_server.grad)
                client_optimizer.step()

        client_results.append((client_model.state_dict(), max(num_examples, 1)))

    new_client_state = fedavg_state_dicts(client_results)
    global_client_model.load_state_dict(new_client_state)
    return global_client_model


@torch.no_grad()
def collect_round_smashed_data(
    client_model: ClientModel,
    client_loaders: List[DataLoader],
    device: torch.device,
    smashed_poison_deltas: Optional[Dict[int, torch.Tensor]] = None,
    max_batches_per_client: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward-only pass collecting one round's worth of smashed data for
    TAS detection to evaluate, applying the same SP injection rule as
    train_one_round (so what detection.py sees matches what an attacker
    would actually send the server during real training). Not used during
    training itself -- this is a separate, read-only snapshot for detection
    evaluation (TrustFLaaS/evaluate_detection.py).

    Returns (Z [N, C, H, W] smashed data in its natural per-sample shape,
    y [N] labels, client_ids [N], is_poisoned [N] bool ground truth -- True
    if this sample came from a malicious client AND was actually perturbed
    at the smashed-data level, i.e. only meaningful/True under SP;
    label-flip poisoning doesn't touch Z at all, so ground truth here is
    deliberately about smashed-data corruption specifically, not "came from
    an attacker" in general -- see evaluate_detection.py for why that
    distinction matters).

    Design note: Z is returned in its NATURAL shape (not pre-flattened) so
    it's directly usable by the GAN and classifier in gan.py /
    consistency_student.py, which need the real [C, H, W] structure.
    detection.py::detect_poisoned flattens internally, so passing this Z
    straight to it also works without any reshaping at the call site.
    """
    smashed_poison_deltas = smashed_poison_deltas or {}
    client_model.to(device).eval()

    all_z, all_y, all_cid, all_poisoned = [], [], [], []
    for cid, loader in enumerate(client_loaders):
        delta_z = smashed_poison_deltas.get(cid)
        for batch_idx, (x, y) in enumerate(loader):
            if batch_idx >= max_batches_per_client:
                break
            x, y = x.to(device), y.to(device)
            z = client_model(x)
            if delta_z is not None:
                z = add_smashed_poison(z, delta_z)
            all_z.append(z.cpu())
            all_y.append(y.cpu())
            all_cid.append(torch.full((y.size(0),), cid, dtype=torch.long))
            all_poisoned.append(torch.full((y.size(0),), delta_z is not None, dtype=torch.bool))

    return (torch.cat(all_z), torch.cat(all_y), torch.cat(all_cid), torch.cat(all_poisoned))


def run_split_fl(
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
    label_flip_delta: Optional[int] = None,
    smashed_poison_deltas: Optional[Dict[int, torch.Tensor]] = None,
    num_classes: Optional[int] = None,
) -> None:
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    server_optimizer = torch.optim.SGD(server_model.parameters(), lr=lr_server)

    client_model.to(device)
    server_model.to(device)
    pc_logger = PerClassLogger(results_path, num_classes) if num_classes else None

    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "test_loss", "test_accuracy", "flip_confusion_rate"])

        # Round 0: evaluate the untrained initial model, same convention as
        # the FL+DP baseline's "round 0" centralized eval.
        loss, acc, flip_rate = evaluate(client_model, server_model, test_loader, device,
                                         label_flip_delta)
        writer.writerow([0, loss, acc, flip_rate])
        f.flush()
        if pc_logger is not None:
            pc_logger.log(0, client_model, server_model, test_loader, device)
        print(f"[Round 0] test_loss={loss:.4f} test_accuracy={acc:.4f} flip_confusion_rate={flip_rate:.4f}")

        for round_num in range(1, num_rounds + 1):
            client_model = train_one_round(
                client_model, server_model, client_loaders, server_optimizer,
                device, local_epochs, lr_client,
                smashed_poison_deltas=smashed_poison_deltas,
            )
            loss, acc, flip_rate = evaluate(client_model, server_model, test_loader, device,
                                             label_flip_delta)
            writer.writerow([round_num, loss, acc, flip_rate])
            f.flush()
            if pc_logger is not None:
                pc_logger.log(round_num, client_model, server_model, test_loader, device,
                              final=(round_num == num_rounds))
            print(f"[Round {round_num}] test_loss={loss:.4f} test_accuracy={acc:.4f} "
                  f"flip_confusion_rate={flip_rate:.4f}")
