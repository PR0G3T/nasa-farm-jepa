"""Free latent layer — the JEPA side of the world model.

`s_t` stays fully interpretable; `z` carries what the physics does not model
(variety, micro-relief, management history) as a learned embedding advanced
by `LatentDyn`. Training follows LeJEPA: next-latent prediction plus SIGReg,
the sketched isotropic-Gaussian regularizer that makes the latent space
well-posed for planning.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nasa_farm.dynamics.residual import N_FEATURES


class LatentDyn(nn.Module):
    """z' = z + MLP([feats, z]) — zero-initialized head ⇒ identity at init,
    same contract as ResidualNet: the model is the physics until trained."""

    def __init__(self, dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.dim = dim
        self.body = nn.Sequential(
            nn.Linear(N_FEATURES + dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.head = nn.Linear(hidden, dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.to(torch.float64)

    def forward(self, feats: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return z + self.head(self.body(torch.cat([feats, z], dim=-1)))


class ObsEncoder(nn.Module):
    """obs-vector → z. Trained so `encode(obs_t)` is the JEPA target the
    predictor `LatentDyn` chases along the rollout."""

    def __init__(self, obs_dim: int, z_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, z_dim),
        )
        self.to(torch.float64)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def sigreg(z: torch.Tensor, directions: torch.Tensor | None = None,
           n_dirs: int = 32, freqs: tuple[float, ...] = (1.0, 2.0, 4.0),
           seed: int = 0) -> torch.Tensor:
    """Sketched Isotropic Gaussian Regularization (LeJEPA).

    Mean squared error between the empirical characteristic function of z's
    projections and φ(t)=exp(-t²/2) of N(0, I), over random directions and a
    few frequencies — linear-time, stop-gradient-free Gaussianity pressure.
    z: (B, K) → scalar.
    """
    b, k = z.shape
    if directions is None:
        g = torch.Generator(device=z.device).manual_seed(seed)
        directions = torch.randn(n_dirs, k, generator=g,
                               dtype=z.dtype, device=z.device)
        directions = directions / directions.norm(dim=1, keepdim=True).clamp(min=1e-9)
    proj = z @ directions.t()                       # (B, D)
    f = torch.tensor(freqs, dtype=z.dtype, device=z.device).view(-1, 1, 1)
    phase = proj.unsqueeze(0) * f                   # (F, B, D)
    phi_re = torch.cos(phase).mean(1)               # (F, D)
    phi_im = torch.sin(phase).mean(1)
    target = torch.exp(-0.5 * f.squeeze(-1) ** 2)   # (F, 1)
    return ((phi_re - target) ** 2 + phi_im ** 2).mean()


def jepa_loss(z_obs: torch.Tensor, z_pred: torch.Tensor,
              sig_weight: float = 1.0, sig_seed: int = 0) -> torch.Tensor:
    """Predictor loss + latent regularizer.

    z_pred: predictor outputs (B, K); z_obs: encoder targets, compared with a
    stop-gradient — the encoder teaches the dynamics, never the reverse.
    """
    pred = (z_pred - z_obs.detach()).pow(2).mean()
    return pred + sig_weight * sigreg(z_obs, seed=sig_seed)
