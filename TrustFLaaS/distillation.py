"""
Vanilla Teacher, distillation losses (Eqs. 11-12), momentum-adaptive
weighting (Eqs. 14-15), and student loss (Eq. 13) -- the pieces that
combine AD Teacher (ad_teacher.py) and a Vanilla Teacher into the REAL
consistency validation student, replacing stage 3's plain stand-in.

Design decision -- Vanilla Teacher's loss uses task c (category), not task a:
Eq. 10 writes L_Van = sum L_a(y_hat_a, y_a), reusing the "task a"
(poison-identification) notation from the AD Teacher section. But the
surrounding text describes the Vanilla Teacher as preserving "semantic
integrity by modeling the distribution of clean data" -- which is a
category-classification framing, not poison-identification (and training a
poison-identification head on data that's ALREADY been filtered to be
all-clean would make every label the same value, a degenerate training
signal). We read Eq. 10's "L_a" as a notational reuse/likely typo and train
the Vanilla Teacher on task c (category) instead -- functionally identical
to stage 3's `consistency_student.py` stand-in, which is why we reuse that
module directly here rather than duplicating it.

Design decision -- what output both distillation losses (Eqs. 11-12) target:
Eqs. 11-12 write generic h_Tvan(z)/h_TAD(z) without specifying which head of
either teacher. We target the CATEGORY head (task c) for both: the Vanilla
Teacher only has one output (category, per the decision above); the AD
Teacher's category head still produces meaningful predictions even though
its training loss was additionally shaped by tasks a/b's influence
weighting (Eq. 9) -- it isn't a separate model, just one whose category
head has been regularized by the anomaly-aware terms. Both teachers "vote"
on category; the student's category understanding comes ENTIRELY from this
distillation (Eq. 13 has no direct category-classification term at all --
only L_a, L_b, plus the two distillation losses).
"""
import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multihead_model import MultiHeadModel


def distillation_loss(teacher_logits: torch.Tensor, student_logits: torch.Tensor,
                       temperature: float) -> torch.Tensor:
    """Eqs. 11 / 12: tau^2 * KL(LogSoftmax(teacher/tau), Softmax(student/tau)).
    Note the paper's KL argument order -- LogSoftmax on the TEACHER side,
    Softmax on the STUDENT side -- is unusual (standard distillation KL is
    the other way: KL(student_log_probs, teacher_probs), since KL divergence
    is directed and you minimize KL(student || teacher)). We implement Eq.
    11/12 exactly as written rather than "correcting" it to the more common
    convention, since silently substituting the standard form would no
    longer be testing what the paper actually specifies."""
    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=1)
    student_probs = F.softmax(student_logits / temperature, dim=1)
    # torch's kl_div(input, target) computes sum(target * (log(target) - input)),
    # matching KL(target || exp(input)) -- here input=teacher_log_probs,
    # target=student_probs, matching the paper's KL(LogSoftmax(teacher), Softmax(student))
    # read as KL(student || teacher) in torch's argument convention.
    kl = F.kl_div(teacher_log_probs, student_probs, reduction="batchmean")
    return (temperature ** 2) * kl


@dataclasses.dataclass
class MomentumWeights:
    mu: float = 0.5
    eta: float = 0.5


def update_momentum_weights(weights: MomentumWeights, L_VS: float, L_IS: float,
                             m: float = 0.9, kappa: float = 1.0, eps: float = 1e-8) -> MomentumWeights:
    """Eqs. 14-15."""
    denom = L_VS + L_IS + eps
    sigma_vs = torch.sigmoid(torch.tensor(kappa * (L_VS - L_IS) / denom)).item()
    sigma_is = torch.sigmoid(torch.tensor(kappa * (L_IS - L_VS) / denom)).item()
    new_mu = m * weights.mu + (1 - m) * sigma_vs
    new_eta = m * weights.eta + (1 - m) * sigma_is
    return MomentumWeights(mu=new_mu, eta=new_eta)


def student_loss(logits_a: torch.Tensor, logits_b: torch.Tensor, logits_c: torch.Tensor,
                  y_a: torch.Tensor, y_b: torch.Tensor,
                  vanilla_teacher_logits_c: torch.Tensor, ad_teacher_logits_c: torch.Tensor,
                  weights: MomentumWeights, temperature: float,
                  lambda_b: float = 1.0) -> torch.Tensor:
    """Eq. 13: L_Stu = L_a + lambda_b*L_b + mu*L_VS + eta*L_IS.
    No direct category (task c) term -- see module docstring."""
    ce = nn.CrossEntropyLoss()
    loss_a = ce(logits_a, y_a)
    loss_b = ce(logits_b, y_b)
    L_VS = distillation_loss(vanilla_teacher_logits_c, logits_c, temperature)
    L_IS = distillation_loss(ad_teacher_logits_c, logits_c, temperature)
    return loss_a + lambda_b * loss_b + weights.mu * L_VS + weights.eta * L_IS, L_VS.item(), L_IS.item()


class CategoryHeadWrapper(nn.Module):
    """Adapts a MultiHeadModel (3-tuple output) to the single-tensor
    `forward(z) -> logits` contract recovery.py's gating function expects
    (same contract stage 3's plain SmashedDataClassifier already satisfies).
    Exposes only the category head, since that's what the recovery gate
    cares about: does this GAN candidate look like a plausible member of
    its target category. This is the "one-line change at the call site"
    stage 3's README promised for swapping in the real student."""

    def __init__(self, multihead_model: MultiHeadModel):
        super().__init__()
        self.model = multihead_model

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        _, _, logits_c = self.model(z)
        return logits_c


def train_consistency_student(
    Z: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, y_c: torch.Tensor,
    vanilla_teacher: nn.Module, ad_teacher: MultiHeadModel,
    num_clients: int, num_classes: int, device: torch.device,
    num_epochs: int = 30, lr: float = 0.01, temperature: float = 2.0,
    lambda_b: float = 1.0, momentum: float = 0.9, seed: int = 0,
) -> MultiHeadModel:
    """Trains the REAL consistency validation student via adversarial
    multi-teacher distillation. Both teachers are frozen (already trained --
    vanilla_teacher via consistency_student.py's train_classifier, ad_teacher
    via ad_teacher.py's train_ad_teacher) and only queried for their logits,
    never updated here."""
    torch.manual_seed(seed)
    student = MultiHeadModel(in_channels=Z.shape[1], num_clients=num_clients,
                              num_classes=num_classes).to(device)
    Z, y_a, y_b, y_c = Z.to(device), y_a.to(device), y_b.to(device), y_c.to(device)
    vanilla_teacher.to(device).eval()
    ad_teacher.to(device).eval()

    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    weights = MomentumWeights()
    n = Z.size(0)

    for epoch in range(num_epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, 32):
            idx = perm[start:start + 32]
            zb, yab, ybb = Z[idx], y_a[idx], y_b[idx]

            with torch.no_grad():
                vt_logits_c = vanilla_teacher(zb)
                _, _, ad_logits_c = ad_teacher(zb)

            logits_a, logits_b, logits_c = student(zb)
            loss, L_VS, L_IS = student_loss(
                logits_a, logits_b, logits_c, yab, ybb,
                vt_logits_c, ad_logits_c, weights, temperature, lambda_b,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            weights = update_momentum_weights(weights, L_VS, L_IS, m=momentum)

    return student
