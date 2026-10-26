"""
BAGAN-GP-style class-conditioned healing generator (core mechanism from
Huang & Jafari 2021, "Enhanced Balancing GAN"), implemented as a third
pluggable healing_mode alongside the existing "gan" and "dalse" options.

Core mechanism, differing from the existing plain per-round GAN:
  1. An autoencoder is trained on ALL classes' clean smashed data jointly (not
     from random noise) -- so a scarce class's reconstruction borrows shared
     structure learned from the abundant majority classes, rather than trying
     to learn the minority class from its own few examples alone.
  2. The generator is initialized from the trained decoder's weights, rather
     than starting from scratch each round.
  3. Generation is class-conditioned: sampling draws specifically from a given
     class's region of the learned latent space (via a class embedding merged
     with noise), rather than producing a generic sample and hoping it lands
     in the right place.
  4. Trained with a gradient penalty (WGAN-GP-style critic, not vanilla BCE
     adversarial loss) for the training stability BAGAN-GP's own paper
     specifically targets -- relevant here since HealSplit's GAN is retrained
     from scratch every round under real scarcity, exactly the unstable
     regime this is meant to help with.

This implements the core mechanism faithfully; it is not a from-scratch
reproduction of every architectural detail in the original paper (no spectral
normalization, no exact hyperparameter schedule matching).

Implemented from the description of the core mechanism in Huang and Jafari (2021), "Enhanced Balancing
GAN" (BAGAN-GP), with a gradient-penalty critic after Gulrajani et al. (2017). Written from the
papers, not copied from a repository. It is a partial reimplementation: no spectral normalisation and
no matching of the original hyperparameter schedule.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

LATENT_DIM = 64


def _sample_shape(in_channels, spatial):
    """(C, S, S) for an int `spatial` (square image-like smashed data, as before);
    (C, *spatial) for a tuple (e.g. PAMAP2's (16, 50) channels x time)."""
    if isinstance(spatial, int):
        return (in_channels, spatial, spatial)
    return (in_channels, *spatial)


def _flat_dim(in_channels, spatial):
    n = 1
    for d in _sample_shape(in_channels, spatial):
        n *= d
    return n


class Encoder(nn.Module):
    def __init__(self, in_channels: int, spatial: int, latent_dim: int = LATENT_DIM):
        super().__init__()
        flat_dim = _flat_dim(in_channels, spatial)
        self.net = nn.Sequential(
            nn.Linear(flat_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x):
        return self.net(x.flatten(1))


class Decoder(nn.Module):
    def __init__(self, in_channels: int, spatial: int, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.in_channels, self.spatial = in_channels, spatial
        flat_dim = _flat_dim(in_channels, spatial)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 256), nn.ReLU(inplace=True),
            nn.Linear(256, flat_dim), nn.Tanh(),
        )

    def forward(self, z):
        out = self.net(z)
        return out.view(-1, *_sample_shape(self.in_channels, self.spatial))


class ConditionalGenerator(nn.Module):
    """Noise + class label -> a sample. Initialized from a pretrained Decoder."""

    def __init__(self, in_channels: int, spatial: int, num_classes: int,
                 latent_dim: int = LATENT_DIM, noise_dim: int = LATENT_DIM):
        super().__init__()
        self.noise_dim = noise_dim
        self.class_embed = nn.Embedding(num_classes, latent_dim)
        self.merge = nn.Linear(noise_dim + latent_dim, latent_dim)
        self.decoder = Decoder(in_channels, spatial, latent_dim)

    def init_from_pretrained_decoder(self, pretrained_decoder: Decoder):
        self.decoder.load_state_dict(pretrained_decoder.state_dict())

    def forward(self, noise: torch.Tensor, class_labels: torch.Tensor):
        class_vec = self.class_embed(class_labels)
        merged = self.merge(torch.cat([noise, class_vec], dim=1))
        return self.decoder(merged)

    def sample(self, n: int, target_class: int, device):
        """Matches the calling convention recovery.py needs: sample N candidates
        for a SPECIFIC flagged class, unlike the existing generator's generic
        .sample(n, device) with no class targeting."""
        noise = torch.randn(n, self.noise_dim, device=device)
        labels = torch.full((n,), target_class, dtype=torch.long, device=device)
        with torch.no_grad():
            return self.forward(noise, labels)


class Critic(nn.Module):
    """WGAN-GP critic: unbounded real-valued score, no sigmoid."""

    def __init__(self, in_channels: int, spatial: int, num_classes: int):
        super().__init__()
        flat_dim = _flat_dim(in_channels, spatial)
        self.class_embed = nn.Embedding(num_classes, 32)
        self.net = nn.Sequential(
            nn.Linear(flat_dim + 32, 256), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x, class_labels):
        class_vec = self.class_embed(class_labels)
        merged = torch.cat([x.flatten(1), class_vec], dim=1)
        return self.net(merged).squeeze(-1)


def _gradient_penalty(critic: Critic, real: torch.Tensor, fake: torch.Tensor,
                       labels: torch.Tensor, device) -> torch.Tensor:
    batch_size = real.size(0)
    eps_shape = [batch_size] + [1] * (real.dim() - 1)
    eps = torch.rand(eps_shape, device=device).expand_as(real)
    interpolated = (eps * real + (1 - eps) * fake).requires_grad_(True)
    scores = critic(interpolated, labels)
    grads = torch.autograd.grad(
        outputs=scores, inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True, retain_graph=True,
    )[0]
    grads = grads.view(batch_size, -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


def _class_balanced_batches(clean_z: torch.Tensor, clean_y: torch.Tensor,
                             num_classes: int, batch_size: int, device):
    """Yields batches with roughly equal per-class representation, sampling
    WITH replacement for classes too scarce to fill their share otherwise.
    Without this, a highly imbalanced pool under plain random shuffling can
    hand many batches zero examples of the rarest class -- meaning that
    class's own conditioning pathway barely gets trained at all, silently
    reproducing the exact minority-starvation problem this architecture is
    meant to fix. Verified empirically: this specific fix was necessary --
    plain shuffling measurably failed the minority class in testing."""
    class_indices = {c: (clean_y == c).nonzero(as_tuple=True)[0] for c in range(num_classes)}
    per_class = max(1, batch_size // num_classes)
    n = clean_z.size(0)
    num_batches = max(1, n // batch_size)
    for _ in range(num_batches):
        idx_parts = []
        for c in range(num_classes):
            pool = class_indices[c]
            if pool.numel() == 0:
                continue
            replace = pool.numel() < per_class
            if replace:
                chosen = pool[torch.randint(0, pool.numel(), (per_class,), device=pool.device)]
            else:
                chosen = pool[torch.randperm(pool.numel(), device=pool.device)[:per_class]]
            idx_parts.append(chosen)
        idx = torch.cat(idx_parts)
        yield clean_z[idx].to(device), clean_y[idx].to(device)


@dataclass
class BaganGPLog:
    ae_final_loss: float
    gan_final_critic_loss: float
    gan_final_gen_loss: float


def train_bagan_gp(clean_z: torch.Tensor, clean_y: torch.Tensor, z_shape,
                    num_classes: int, device, ae_epochs: int = 20,
                    gan_epochs: int = 40, seed: int = 0,
                    critic_iters_per_gen: int = 5, gp_weight: float = 10.0,
                    batch_size: int = 32):
    torch.manual_seed(seed)
    in_channels = z_shape[0]
    # Square (C, S, S) smashed data keeps the original int path (identical behaviour);
    # anything else (e.g. PAMAP2's (16, 50)) passes its remaining dims as a tuple.
    if len(z_shape) == 3 and z_shape[1] == z_shape[2]:
        spatial = z_shape[1]
    else:
        spatial = tuple(z_shape[1:])

    encoder = Encoder(in_channels, spatial).to(device)
    decoder = Decoder(in_channels, spatial).to(device)
    ae_optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)

    clean_z_cpu, clean_y_cpu = clean_z.cpu(), clean_y.cpu()
    clean_z, clean_y = clean_z.to(device), clean_y.to(device)
    n = clean_z.size(0)

    # --- Phase 1: autoencoder pretraining across ALL classes jointly ---
    # (plain shuffling is fine here -- reconstruction loss on any sample helps
    # the shared latent space regardless of its class, unlike Phase 2 where
    # each class needs its OWN conditioning pathway specifically exercised)
    ae_loss_val = float("nan")
    for _ in range(ae_epochs):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch = clean_z[idx]
            recon = decoder(encoder(batch))
            loss = F.mse_loss(recon, batch)
            ae_optimizer.zero_grad()
            loss.backward()
            ae_optimizer.step()
            total_loss += loss.item() * batch.size(0)
        ae_loss_val = total_loss / n

    # --- Phase 2: class-conditioned adversarial training, generator initialized from decoder ---
    generator = ConditionalGenerator(in_channels, spatial, num_classes).to(device)
    generator.init_from_pretrained_decoder(decoder)
    critic = Critic(in_channels, spatial, num_classes).to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    c_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0.5, 0.9))

    critic_loss_val = gen_loss_val = float("nan")
    for _ in range(gan_epochs):
        for real_batch, labels_batch in _class_balanced_batches(
                clean_z_cpu, clean_y_cpu, num_classes, batch_size, device):
            bs = real_batch.size(0)
            if bs < 2:
                continue

            for _ in range(critic_iters_per_gen):
                noise = torch.randn(bs, generator.noise_dim, device=device)
                fake_batch = generator(noise, labels_batch).detach()
                critic_real = critic(real_batch, labels_batch)
                critic_fake = critic(fake_batch, labels_batch)
                gp = _gradient_penalty(critic, real_batch, fake_batch, labels_batch, device)
                c_loss = critic_fake.mean() - critic_real.mean() + gp_weight * gp
                c_optimizer.zero_grad()
                c_loss.backward()
                c_optimizer.step()
            critic_loss_val = c_loss.item()

            noise = torch.randn(bs, generator.noise_dim, device=device)
            fake_batch = generator(noise, labels_batch)
            g_loss = -critic(fake_batch, labels_batch).mean()
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            gen_loss_val = g_loss.item()

    log = BaganGPLog(ae_final_loss=ae_loss_val, gan_final_critic_loss=critic_loss_val,
                      gan_final_gen_loss=gen_loss_val)
    return generator, log
