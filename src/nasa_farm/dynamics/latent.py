"""Free latent layer — the JEPA side of the world model.

`s_t` stays fully interpretable; `z` carries what the physics does not model
(variety, micro-relief, management history) as a learned embedding advanced
by `LatentDyn`, a recurrent predictor whose hidden state `zh` persists
between days. Optional stochastic innovation (mu/logvar head + reparam)
turns it into an RSSM-style latent whose KL term is exposed for the loss.
A static site context `ctx` (e.g. foundation-model embeddings, soil
texture) conditions every transition.

Training follows LeJEPA: next-latent prediction plus SIGReg, the sketched
isotropic-Gaussian regularizer that keeps the latent space well-posed.
Zero-initialized head ⇒ identity at init, same contract as ResidualNet.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nasa_farm.dynamics.residual import N_FEATURES


LINKS: dict = {"identity": lambda t: t, "relu": torch.relu}
"""Output link on the latent update — "relu" gives nonneg sparse codes
(the geometry RDMReg targets)."""


class Dynamics(nn.Module):
    """GRU transition predictor: zh' = GRU([feats, z, ctx?], zh); z' = z + head(zh')."""

    def __init__(self, dim: int, hidden: int = 64, stochastic: bool = False,
                 context_dimension: int = 0, link: str = "identity") -> None:
        super().__init__()
        self.dim = dim
        self.hidden_dimension = hidden
        self.stochastic = stochastic
        self.context_dimension = context_dimension
        self.link = LINKS[link]
        self.gru = nn.GRUCell(N_FEATURES + dim + context_dimension, hidden)
        self.head = nn.Linear(hidden, dim * (2 if stochastic else 1))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.kl = torch.zeros((), dtype=torch.float64)
        self.to(torch.float64)

    def forward(self, features: torch.Tensor, z: torch.Tensor,
                zh: torch.Tensor | None = None,
                context: torch.Tensor | None = None,
                generation: torch.Generator | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.context_dimension:
            c = context if context is not None else torch.zeros(
                features.shape[0], self.context_dimension,
                dtype=features.dtype, device=features.device)
            x = torch.cat([features, z, c.to(features.dtype)], dim=-1)
        else:
            x = torch.cat([features, z], dim=-1)
        if zh is None:
            zh = torch.zeros(features.shape[0], self.hidden_dimension,
                             dtype=features.dtype, device=features.device)
        zh = self.gru(x, zh.to(features.dtype))
        output = self.head(zh)
        if not self.stochastic:
            self.kl = torch.zeros((), dtype=output.dtype, device=output.device)
            return self.link(z + output), zh
        mean, log_variance = output.chunk(2, dim=-1)
        log_variance = log_variance.clamp(-8.0, 4.0)
        deviation = (0.5 * log_variance).exp()
        noise = torch.randn(mean.shape, generator=generation,
                          dtype=mean.dtype, device=mean.device)
        self.kl = 0.5 * (mean.pow(2) + deviation.pow(2) - log_variance - 1.0).mean()
        return self.link(z + mean + deviation * noise), zh

    def seed(self, z: torch.Tensor) -> torch.Tensor:
        """Obs embedding → initial latent state (identity for a flat
        latent; HLatent overrides to pad upper levels)."""
        return z

    def unroll(self, feats_seq: torch.Tensor, z: torch.Tensor,
               zh: torch.Tensor | None = None, context: torch.Tensor | None = None,
               generation: torch.Generator | None = None) -> torch.Tensor:
        """Multi-step rollout of the predictor alone. feats_seq (T, B, F)."""
        zs = []
        for t in range(feats_seq.shape[0]):
            z, zh = self(feats_seq[t], z, zh, context, generation)
            zs.append(z)
        return torch.stack(zs)


class ObservationEncoder(nn.Module):
    """obs-vector → z. Trained so `encode(obs_t)` is the JEPA target the
    predictor `LatentDyn` chases along the rollout. The LayerNorm keeps the
    embedding scale O(1): the characteristic-function SIGReg saturates on
    exploded embeddings (its gradient vanishes before it can pull the
    scale back), so the bound must be architectural, not just a loss."""

    def __init__(self, observed_dimension: int, z_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observed_dimension, hidden), nn.SiLU(),
            nn.Linear(hidden, z_dim), nn.LayerNorm(z_dim),
        )
        self.to(torch.float64)

    def forward(self, observed: torch.Tensor) -> torch.Tensor:
        return self.net(observed)


def regularization(z: torch.Tensor, directions: torch.Tensor | None = None,
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


def embedding_loss(z_observed: torch.Tensor, z_pred: torch.Tensor,
              sig_weight: float = 1.0, kl: torch.Tensor | float = 0.0,
              kl_weight: float = 0.0, sig_seed: int = 0,
              z_reg: torch.Tensor | None = None) -> torch.Tensor:
    """Predictor loss + latent regularizer + stochastic-prior KL.

    z_pred: predictor outputs (B, K) or (T, B, K) for multi-step unrolls;
    z_obs: encoder targets, compared with a stop-gradient — the encoder
    teaches the dynamics, never the reverse. SIGReg applies to `z_reg`
    (the online embeddings) when given — LeJEPA regularizes the online
    side since EMA targets carry no gradient — else falls back to z_obs.
    """
    predicted = (z_pred - z_observed.detach()).pow(2).mean()
    kl_term = kl if isinstance(kl, torch.Tensor) else torch.tensor(
        float(kl), dtype=z_pred.dtype, device=z_pred.device)
    regularized = z_observed if z_reg is None else z_reg
    return predicted + sig_weight * regularization(regularized.reshape(-1, regularized.shape[-1]),
                                      seed=sig_seed) + kl_weight * kl_term


class TargetEncoder(nn.Module):
    """EMA copy of an online encoder — the non-collapsing JEPA target.

    Stop-gradient alone does not prevent representation collapse: a
    shared encoder can map everything to a constant and satisfy the
    prediction loss trivially. Chasing an exponential moving average of
    its own weights makes the target a slowly drifting manifold instead —
    the standard JEPA/BYOL stabilizer. Parameters are frozen; call
    `update_from` after every optimizer step."""

    def __init__(self, online: nn.Module, decay: float = 0.996) -> None:
        super().__init__()
        import copy
        self.net = copy.deepcopy(online).requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update_from(self, online: nn.Module) -> None:
        for t, o in zip(self.net.parameters(), online.parameters()):
            t.mul_(self.decay).add_(o, alpha=1.0 - self.decay)

    def forward(self, observed: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.net(observed)


class PatchEncoder(nn.Module):
    """Spatial observation encoder: satellite patch (B, C, H, W) → z.

    The column-scale `ObsEncoder` consumes already-aggregated channels;
    this one embeds raw multi-band imagery — the high-dimensional input
    regime where latent prediction actually adds information the physics
    does not carry (texture, intra-field heterogeneity, sensor artifacts).
    Strided convolutions + global mean pooling, so the output dimension is
    independent of patch size."""

    def __init__(self, bands: int, z_dim: int, width: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(bands, width, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(width * 2, width * 2, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(width * 2, z_dim), nn.LayerNorm(z_dim),
        )
        self.to(torch.float64)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        return self.net(patch)


class Autoencoder(nn.Module):
    """Top-k sparse autoencoder over latents — decomposes z into a sparse
    dictionary of candidate factors (mechanistic interpretability):
    `features(z)` exposes which sparse units are active, `recon` keeps the
    decomposition lossless. Top-k activation gives a fixed activation
    budget without an L1 knob to tune."""

    def __init__(self, dim: int, feature_count: int, topk: int,
                 seed: int | None = None) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.topk = topk
        self.encoder = nn.Linear(dim, feature_count)
        self.decoder = nn.Linear(feature_count, dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float64))
        with torch.no_grad():
            self.decoder.weight.div_(
                self.decoder.weight.norm(dim=0, keepdim=True) + 1e-9)
        self.to(torch.float64)

    def features(self, z: torch.Tensor) -> torch.Tensor:
        """Sparse activations (…, F): top-k ReLU units per row."""
        a = self.encoder(z - self.bias)
        v, index = a.topk(self.topk, dim=-1)
        return torch.zeros_like(a).scatter_(-1, index, v.clamp_min(0.0))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.features(z)
        return f @ self.decoder.weight.t() + self.bias, f


def fit_autoencoder(autoencoder: Autoencoder, z_stream: torch.Tensor, epochs: int = 100,
            rate: float = 1e-3, verbose: bool = False) -> dict:
    """Adam fit of the dictionary on a (N, K) latent stream; returns the
    final reconstruction error and per-unit firing rates — dead features
    are visible directly in `rate`."""
    zs = z_stream.reshape(-1, z_stream.shape[-1])
    opt = torch.optim.Adam(autoencoder.parameters(), lr=rate)
    best, history = float("inf"), []
    for evaporation in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        recon, f = autoencoder(zs)
        loss = (recon - zs).pow(2).mean()
        if not torch.isfinite(loss):
            break
        loss.backward()
        opt.step()
        with torch.no_grad():
            autoencoder.decoder.weight.div_(
                autoencoder.decoder.weight.norm(dim=0, keepdim=True) + 1e-9)
        v = float(loss.detach())
        history.append(v)
        best = min(best, v)
        if verbose and evaporation % 25 == 0:
            print(f"  epoch {evaporation}: sae {v:.5f}")
    rate = (autoencoder.features(zs) > 0).to(zs.dtype).mean(0)
    return {"best_loss": round(best, 8), "history": history,
            "rate": rate.detach()}


OPS: dict[str, tuple[int, ...]] = {
    "atmo":   (7, 8, 9, 10),      # tmin, tmax, srad, precip
    "water":  (0, 1, 10, 11),     # sw0, swr, precip, irrigation
    "bio":    (2, 4, 6, 12, 13),  # n_min, biomass, growing, n_fert, sow
    "canopy": (3, 5),             # lai, gdd
}
"""Operator-aligned factor groups over featurize() columns — one latent
sub-dynamics per physics subprocess."""


class Factor(nn.Module):
    """Factored latent: one independent sub-dynamics per feature group.

    Each factor owns a GRU over (its feature columns, its z slot[, ctx])
    plus a zero-init head — identity at init, like LatentDyn. Causal
    selectivity holds by construction: an action-channel perturbation can
    only move the factor that reads it.
    """

    def __init__(self, dim: int, parts: dict[str, tuple[int, ...]],
                 hidden: int = 64, context_dimension: int = 0) -> None:
        super().__init__()
        self.parts = dict(parts)
        self.part_dim, self.hp = dim, hidden
        self.dim, self.hidden_dimension = dim * len(parts), hidden * len(parts)
        self.context_dimension = context_dimension
        self.cell = nn.ModuleDict(
            {n: nn.GRUCell(len(c) + dim + context_dimension, hidden)
             for n, c in parts.items()})
        self.head = nn.ModuleDict(
            {n: nn.Linear(hidden, dim) for n in parts})
        for h in self.head.values():
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)
        self.kl = torch.zeros((), dtype=torch.float64)
        self.to(torch.float64)

    def _slot(self, t: torch.Tensor, i: int, w: int) -> torch.Tensor:
        return t[:, i * w:(i + 1) * w]

    def forward(self, features: torch.Tensor, z: torch.Tensor,
                zh: torch.Tensor | None = None,
                context: torch.Tensor | None = None,
                generation: torch.Generator | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        if zh is None:
            zh = torch.zeros(features.shape[0], self.hidden_dimension,
                             dtype=features.dtype, device=features.device)
        if self.context_dimension and context is None:
            context = torch.zeros(features.shape[0], self.context_dimension,
                              dtype=features.dtype, device=features.device)
        zs, hs = [], []
        for i, (name, cols) in enumerate(self.parts.items()):
            zk = self._slot(z, i, self.part_dim)
            x = torch.cat([features[:, list(cols)], zk] +
                          ([context.to(features.dtype)] if self.context_dimension else []), -1)
            hk = self.cell[name](x, self._slot(zh, i, self.hp).to(features.dtype))
            zs.append(zk + self.head[name](hk))
            hs.append(hk)
        self.kl = torch.zeros((), dtype=features.dtype, device=features.device)
        return torch.cat(zs, -1), torch.cat(hs, -1)

    def seed(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def unroll(self, feats_seq: torch.Tensor, z: torch.Tensor,
               zh: torch.Tensor | None = None, context: torch.Tensor | None = None,
               generation: torch.Generator | None = None) -> torch.Tensor:
        zs = []
        for t in range(feats_seq.shape[0]):
            z, zh = self(feats_seq[t], z, zh, context, generation)
            zs.append(z)
        return torch.stack(zs)
