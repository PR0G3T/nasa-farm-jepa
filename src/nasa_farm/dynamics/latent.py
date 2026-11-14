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


class LatentDyn(nn.Module):
    """GRU transition predictor: zh' = GRU([feats, z, ctx?], zh); z' = z + head(zh')."""

    def __init__(self, dim: int, hidden: int = 64, stochastic: bool = False,
                 ctx_dim: int = 0, link: str = "identity") -> None:
        super().__init__()
        self.dim = dim
        self.h_dim = hidden
        self.stochastic = stochastic
        self.ctx_dim = ctx_dim
        self.link = LINKS[link]
        self.gru = nn.GRUCell(N_FEATURES + dim + ctx_dim, hidden)
        self.head = nn.Linear(hidden, dim * (2 if stochastic else 1))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.kl = torch.zeros((), dtype=torch.float64)
        self.to(torch.float64)

    def forward(self, feats: torch.Tensor, z: torch.Tensor,
                zh: torch.Tensor | None = None,
                ctx: torch.Tensor | None = None,
                gen: torch.Generator | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.ctx_dim:
            c = ctx if ctx is not None else torch.zeros(
                feats.shape[0], self.ctx_dim,
                dtype=feats.dtype, device=feats.device)
            x = torch.cat([feats, z, c.to(feats.dtype)], dim=-1)
        else:
            x = torch.cat([feats, z], dim=-1)
        if zh is None:
            zh = torch.zeros(feats.shape[0], self.h_dim,
                             dtype=feats.dtype, device=feats.device)
        zh = self.gru(x, zh.to(feats.dtype))
        out = self.head(zh)
        if not self.stochastic:
            self.kl = torch.zeros((), dtype=out.dtype, device=out.device)
            return self.link(z + out), zh
        mu, logvar = out.chunk(2, dim=-1)
        logvar = logvar.clamp(-8.0, 4.0)
        sig = (0.5 * logvar).exp()
        eps = torch.randn(mu.shape, generator=gen,
                          dtype=mu.dtype, device=mu.device)
        self.kl = 0.5 * (mu.pow(2) + sig.pow(2) - logvar - 1.0).mean()
        return self.link(z + mu + sig * eps), zh

    def seed(self, z: torch.Tensor) -> torch.Tensor:
        """Obs embedding → initial latent state (identity for a flat
        latent; HLatent overrides to pad upper levels)."""
        return z

    def unroll(self, feats_seq: torch.Tensor, z: torch.Tensor,
               zh: torch.Tensor | None = None, ctx: torch.Tensor | None = None,
               gen: torch.Generator | None = None) -> torch.Tensor:
        """Multi-step rollout of the predictor alone. feats_seq (T, B, F)."""
        zs = []
        for t in range(feats_seq.shape[0]):
            z, zh = self(feats_seq[t], z, zh, ctx, gen)
            zs.append(z)
        return torch.stack(zs)


class ObsEncoder(nn.Module):
    """obs-vector → z. Trained so `encode(obs_t)` is the JEPA target the
    predictor `LatentDyn` chases along the rollout. The LayerNorm keeps the
    embedding scale O(1): the characteristic-function SIGReg saturates on
    exploded embeddings (its gradient vanishes before it can pull the
    scale back), so the bound must be architectural, not just a loss."""

    def __init__(self, obs_dim: int, z_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, z_dim), nn.LayerNorm(z_dim),
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
    pred = (z_pred - z_obs.detach()).pow(2).mean()
    kl_term = kl if isinstance(kl, torch.Tensor) else torch.tensor(
        float(kl), dtype=z_pred.dtype, device=z_pred.device)
    reg = z_obs if z_reg is None else z_reg
    return pred + sig_weight * sigreg(reg.reshape(-1, reg.shape[-1]),
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

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.net(obs)


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


class TopKSAE(nn.Module):
    """Top-k sparse autoencoder over latents — decomposes z into a sparse
    dictionary of candidate factors (mechanistic interpretability):
    `features(z)` exposes which sparse units are active, `recon` keeps the
    decomposition lossless. Top-k activation gives a fixed activation
    budget without an L1 knob to tune."""

    def __init__(self, dim: int, n_feat: int, topk: int,
                 seed: int | None = None) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.topk = topk
        self.enc = nn.Linear(dim, n_feat)
        self.dec = nn.Linear(n_feat, dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float64))
        with torch.no_grad():
            self.dec.weight.div_(
                self.dec.weight.norm(dim=0, keepdim=True) + 1e-9)
        self.to(torch.float64)

    def features(self, z: torch.Tensor) -> torch.Tensor:
        """Sparse activations (…, F): top-k ReLU units per row."""
        a = self.enc(z - self.bias)
        v, idx = a.topk(self.topk, dim=-1)
        return torch.zeros_like(a).scatter_(-1, idx, v.clamp_min(0.0))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.features(z)
        return f @ self.dec.weight.t() + self.bias, f


def sae_fit(sae: TopKSAE, z_stream: torch.Tensor, epochs: int = 100,
            lr: float = 1e-3, verbose: bool = False) -> dict:
    """Adam fit of the dictionary on a (N, K) latent stream; returns the
    final reconstruction error and per-unit firing rates — dead features
    are visible directly in `rate`."""
    zs = z_stream.reshape(-1, z_stream.shape[-1])
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    best, hist = float("inf"), []
    for ep in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        recon, f = sae(zs)
        loss = (recon - zs).pow(2).mean()
        if not torch.isfinite(loss):
            break
        loss.backward()
        opt.step()
        with torch.no_grad():
            sae.dec.weight.div_(
                sae.dec.weight.norm(dim=0, keepdim=True) + 1e-9)
        v = float(loss.detach())
        hist.append(v)
        best = min(best, v)
        if verbose and ep % 25 == 0:
            print(f"  epoch {ep}: sae {v:.5f}")
    rate = (sae.features(zs) > 0).to(zs.dtype).mean(0)
    return {"best_loss": round(best, 8), "history": hist,
            "rate": rate.detach()}


OPS: dict[str, tuple[int, ...]] = {
    "atmo":   (7, 8, 9, 10),      # tmin, tmax, srad, precip
    "water":  (0, 1, 10, 11),     # sw0, swr, precip, irrigation
    "bio":    (2, 4, 6, 12, 13),  # n_min, biomass, growing, n_fert, sow
    "canopy": (3, 5),             # lai, gdd
}
"""Operator-aligned factor groups over featurize() columns — one latent
sub-dynamics per physics subprocess."""


class FactorDyn(nn.Module):
    """Factored latent: one independent sub-dynamics per feature group.

    Each factor owns a GRU over (its feature columns, its z slot[, ctx])
    plus a zero-init head — identity at init, like LatentDyn. Causal
    selectivity holds by construction: an action-channel perturbation can
    only move the factor that reads it.
    """

    def __init__(self, dim: int, parts: dict[str, tuple[int, ...]],
                 hidden: int = 64, ctx_dim: int = 0) -> None:
        super().__init__()
        self.parts = dict(parts)
        self.part_dim, self.hp = dim, hidden
        self.dim, self.h_dim = dim * len(parts), hidden * len(parts)
        self.ctx_dim = ctx_dim
        self.cell = nn.ModuleDict(
            {n: nn.GRUCell(len(c) + dim + ctx_dim, hidden)
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

    def forward(self, feats: torch.Tensor, z: torch.Tensor,
                zh: torch.Tensor | None = None,
                ctx: torch.Tensor | None = None,
                gen: torch.Generator | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        if zh is None:
            zh = torch.zeros(feats.shape[0], self.h_dim,
                             dtype=feats.dtype, device=feats.device)
        if self.ctx_dim and ctx is None:
            ctx = torch.zeros(feats.shape[0], self.ctx_dim,
                              dtype=feats.dtype, device=feats.device)
        zs, hs = [], []
        for i, (name, cols) in enumerate(self.parts.items()):
            zk = self._slot(z, i, self.part_dim)
            x = torch.cat([feats[:, list(cols)], zk] +
                          ([ctx.to(feats.dtype)] if self.ctx_dim else []), -1)
            hk = self.cell[name](x, self._slot(zh, i, self.hp).to(feats.dtype))
            zs.append(zk + self.head[name](hk))
            hs.append(hk)
        self.kl = torch.zeros((), dtype=feats.dtype, device=feats.device)
        return torch.cat(zs, -1), torch.cat(hs, -1)

    def seed(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def unroll(self, feats_seq: torch.Tensor, z: torch.Tensor,
               zh: torch.Tensor | None = None, ctx: torch.Tensor | None = None,
               gen: torch.Generator | None = None) -> torch.Tensor:
        zs = []
        for t in range(feats_seq.shape[0]):
            z, zh = self(feats_seq[t], z, zh, ctx, gen)
            zs.append(z)
        return torch.stack(zs)
