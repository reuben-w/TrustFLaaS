"""
GAN for the "Semantically Consistent Substitution" step (HealSplit Eq. 6).

Trained per round, only on smashed data TAS did NOT flag as poisoned that
round -- the generator learns to approximate the distribution of clean
smashed data, so it can synthesize plausible replacements for the samples
TAS did flag.

Design decision -- non-negative output activation:
The paper's Eq. 6 loss is the standard GAN minimax objective and doesn't
constrain the generator's output range. But real smashed data here is
always non-negative -- ClientModel's forward pass ends in ReLU
(`pool(relu(gn1(conv1(x))))`, see model.py) -- so a generator with an
unconstrained (e.g. linear or tanh) final layer would happily learn to
output negative values the discriminator has never seen as "real,"
producing implausible substitutes even if the adversarial loss looks fine.
We use ReLU as the generator's final activation to match the real data's
actual support.

Design decision -- Eq. 6 as written IS the standard "non-saturating" GAN loss:
L_G = -E[log D(z_tilde)] is already the practical alternative to the
original (saturating, vanishing-gradient-prone) minimax generator loss
E[log(1-D(z_tilde))] -- so there's no substitution to make here, Eq. 6 is
implemented literally via binary cross-entropy in both directions.

Implemented from HealSplit (Xie and Lyu, AAAI 2026), Eq. 6, and the standard non-saturating GAN
objective (Goodfellow et al., 2014). Written from the paper's description, not copied from a
repository.
"""
import dataclasses
from typing import Tuple

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noise_dim: int, z_shape: Tuple[int, ...]):
        super().__init__()
        self.noise_dim = noise_dim
        self.z_shape = z_shape
        out_dim = 1
        for d in z_shape:
            out_dim *= d

        self.net = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, out_dim),
            nn.ReLU(),  # matches real smashed data's non-negative support -- see module docstring
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        flat = self.net(noise)
        return flat.view(flat.size(0), *self.z_shape)

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        noise = torch.randn(n, self.noise_dim, device=device)
        self.eval()
        with torch.no_grad():
            out = self(noise)
        return out


class Discriminator(nn.Module):
    def __init__(self, z_shape: Tuple[int, ...]):
        super().__init__()
        in_dim = 1
        for d in z_shape:
            in_dim *= d

        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.reshape(z.size(0), -1)).squeeze(-1)


@dataclasses.dataclass
class GANTrainingLog:
    d_losses: list
    g_losses: list
    d_real_acc: list   # fraction of real samples D correctly calls "real"
    d_fake_acc: list   # fraction of fake samples D correctly calls "fake"


def train_gan(
    clean_z: torch.Tensor,
    z_shape: Tuple[int, ...],
    device: torch.device,
    noise_dim: int = 32,
    num_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 2e-4,
    seed: int = 0,
) -> Tuple[Generator, Discriminator, GANTrainingLog]:
    """Trains a vanilla GAN (Eq. 6) on `clean_z` (the TAS-identified clean
    smashed data for this round). Returns (generator, discriminator, log) --
    the log is for the caller to sanity-check convergence (see
    evaluate_recovery.py: a GAN that hasn't converged should NOT be trusted
    to generate substitutes, and d_real_acc/d_fake_acc settling near 0.5
    each is the standard signal that it has)."""
    torch.manual_seed(seed)
    G = Generator(noise_dim, z_shape).to(device)
    D = Discriminator(z_shape).to(device)
    g_opt = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    clean_z = clean_z.to(device)
    n = clean_z.size(0)
    log = GANTrainingLog([], [], [], [])

    for epoch in range(num_epochs):
        perm = torch.randperm(n)
        epoch_d_loss = epoch_g_loss = 0.0
        epoch_d_real_correct = epoch_d_fake_correct = epoch_count = 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            real = clean_z[idx]
            b = real.size(0)
            if b < 2:  # BatchNorm needs >=2 samples
                continue

            # --- train D ---
            noise = torch.randn(b, noise_dim, device=device)
            fake = G(noise).detach()
            d_real = D(real)
            d_fake = D(fake)
            d_loss = criterion(d_real, torch.ones(b, device=device)) + \
                      criterion(d_fake, torch.zeros(b, device=device))
            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # --- train G (Eq. 6: L_G = -E[log D(z_tilde)], i.e. BCE against "real") ---
            noise = torch.randn(b, noise_dim, device=device)
            fake = G(noise)
            g_loss = criterion(D(fake), torch.ones(b, device=device))
            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            epoch_d_loss += d_loss.item() * b
            epoch_g_loss += g_loss.item() * b
            epoch_d_real_correct += (d_real > 0.5).sum().item()
            epoch_d_fake_correct += (d_fake <= 0.5).sum().item()
            epoch_count += b

        if epoch_count > 0:
            log.d_losses.append(epoch_d_loss / epoch_count)
            log.g_losses.append(epoch_g_loss / epoch_count)
            log.d_real_acc.append(epoch_d_real_correct / epoch_count)
            log.d_fake_acc.append(epoch_d_fake_correct / epoch_count)

    return G, D, log
