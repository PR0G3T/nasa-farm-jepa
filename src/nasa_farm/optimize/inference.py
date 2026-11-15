"""Masked group flow for simulation-based inference — OneFlowSBI-style,
generalized to an arbitrary group structure.

A single conditional flow-matching model over the concatenated vector
z = (g_0, …, g_{G−1}) with a G-bit mask declaring which groups are *known*.
The same trained net serves every query the mask can express:

- posterior  p(θ|x)   — x-group known, θ generated
- likelihood p(x|θ)   — θ known, x generated (emulator sampling)
- hierarchical p(θ_g|θ_s, x) — global parameters given site-level
  parameters and data; stack K sites as 2K+1 groups for a multi-site
  joint posterior over shared parameters (TFMPE-flavored, flat groups)
- arbitrary conditionals / joint draws

Training is conditional flow matching: z_t = (1−t)·ε + t·z, target
velocity v = z − ε, with known groups frozen at their data value.
Sampling integrates v from t=0 (noise) to t=1 in a few Euler steps —
`steps=4` already beats a trained MDN on multimodal posteriors because
the map is shared across contexts.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nasa_farm.constants import DTYPE


class Flow(nn.Module):
    """v(z_t, t, mask, ctx) → velocity over grouped vectors.

    mask: (G,) — one bit per group; ctx carries data values on known
    groups and zeros elsewhere. dims: per-group widths."""

    def __init__(self, dims: tuple[int, ...], hidden: int = 128, depth: int = 3) -> None:
        super().__init__()
        self.dims = tuple(dims)
        total = sum(dims)
        d = total * 2 + len(dims) + 1
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, total)]
        self.net = nn.Sequential(*layers)
        self.to(DTYPE)

    @property
    def target(self) -> int:
        """Group-0 width (θ) — kept for the two-group posterior API."""
        return self.dims[0]

    @property
    def d_x(self) -> int:
        """Group-1 width (x) — kept for the two-group posterior API."""
        return self.dims[1] if len(self.dims) > 1 else 0

    def forward(
        self, z: torch.Tensor, t: torch.Tensor, mask: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        tb = t.to(DTYPE).expand(z.shape[0], 1)
        mb = mask.to(DTYPE).expand(z.shape[0], len(self.dims))
        v = self.net(torch.cat([z, tb, mb, context.to(DTYPE)], dim=-1))
        reps = torch.tensor(self.dims, device=z.device)
        return v * (1.0 - torch.repeat_interleave(mb, reps, dim=1))


def _masked_ctx(groups: list[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
    """(B, Σd_i) context: data on known groups, zeros elsewhere."""
    return torch.cat([g * mask[:, i : i + 1] for i, g in enumerate(groups)], dim=-1)


def fit_groups(
    groups: tuple[torch.Tensor, ...],
    epochs: int = 300,
    rate: float = 2e-3,
    hidden: int = 128,
    depth: int = 3,
    seed: int = 0,
) -> Flow:
    """CFM training with random known/free group masks.

    Default mask sampler draws a nonempty proper subset of groups as
    known — covering every conditional the trained net will be queried
    with. For two groups that is exactly the alternating posterior /
    likelihood pattern of the original joint flow."""
    gs = [g.to(DTYPE) for g in groups]
    dims = tuple(g.shape[-1] for g in gs)
    g_n = len(dims)
    m = Flow(dims, hidden, depth)
    torch.manual_seed(seed)
    generation = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(m.parameters(), lr=rate)
    n = gs[0].shape[0]
    for _ in range(epochs):
        i = torch.randint(0, n, (n,), generator=generation)
        z1 = torch.cat([g[i] for g in gs], -1)
        t = torch.rand(n, 1, generator=generation, dtype=DTYPE)
        eps = torch.randn(z1.shape, generator=generation, dtype=DTYPE)
        mask = (torch.rand(n, g_n, generator=generation) < 0.5).to(DTYPE)
        dead = mask.sum(1) == 0  # need ≥1 known and ≥1 free group
        full = mask.sum(1) == g_n
        mask[dead, 0] = 1.0
        mask[full, 0] = 0.0
        zt = (1.0 - t) * eps + t * z1
        context = _masked_ctx([g[i] for g in gs], mask)
        known = torch.repeat_interleave(mask, torch.tensor(dims), dim=1)
        zt = torch.where(known.bool(), z1, zt)
        predicted = m(zt, t, mask, context)
        loss = ((predicted - (z1 - eps)) ** 2 * (1.0 - known)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return m


def fit_joint(
    theta: torch.Tensor,
    x: torch.Tensor,
    epochs: int = 300,
    rate: float = 2e-3,
    hidden: int = 128,
    seed: int = 0,
) -> Flow:
    """Two-group specialization: (θ, x) joint flow."""
    return fit_groups((theta, x), epochs=epochs, rate=rate, hidden=hidden, seed=seed)


def fit_hierarchy(
    theta_g: torch.Tensor,
    theta_s: torch.Tensor,
    x: torch.Tensor,
    epochs: int = 300,
    rate: float = 2e-3,
    hidden: int = 128,
    seed: int = 0,
) -> Flow:
    """Hierarchical SBI: (θ_global, θ_site, x) three-group flow.

    Trained on single-site samples; query the global posterior with
    GroupPosterior.sample({1: θ̂_s, 2: x}) per site, or stack all sites
    into 2K+1 groups for a simultaneous multi-site fit."""
    return fit_groups((theta_g, theta_s, x), epochs=epochs, rate=rate, hidden=hidden, seed=seed)


@torch.no_grad()
def flow_sample(
    m: Flow, context: torch.Tensor, mask: torch.Tensor, n: int, steps: int = 4, seed: int = 0
) -> torch.Tensor:
    """Generate masked-off groups given `ctx` ((Σd,) or (1,·))."""
    generation = torch.Generator().manual_seed(seed)
    c = context.reshape(-1).to(DTYPE)
    z = torch.randn(n, sum(m.dims), generator=generation, dtype=DTYPE)
    known = torch.repeat_interleave(mask.to(DTYPE), torch.tensor(m.dims)).bool()
    z[:, known] = c[known]
    ctxb = (c * known.to(DTYPE)).expand(n, -1)
    for s in range(steps):
        t = torch.full((n, 1), s / steps, dtype=DTYPE)
        z = z + m(z, t, mask, ctxb) / steps
        z[:, known] = c[known]
    return z


class Posterior:
    """Trained two-group flow queried as a posterior — the SBI entry."""

    def __init__(self, flow: Flow) -> None:
        self.flow = flow

    def sample(self, x: torch.Tensor, n: int, steps: int = 4, seed: int = 0) -> torch.Tensor:
        """θ | x → (n, Dθ). x: (Dx,) or (1, Dx)."""
        context = torch.zeros(sum(self.flow.dims), dtype=DTYPE)
        context[self.flow.target : self.flow.target + self.flow.d_x] = x.reshape(-1).to(DTYPE)
        z = flow_sample(self.flow, context, torch.tensor([0.0, 1.0]), n, steps, seed)
        return z[:, : self.flow.target]

    def likelihood(
        self, theta: torch.Tensor, n: int, steps: int = 4, seed: int = 0
    ) -> torch.Tensor:
        """x | θ → (n, Dx) — emulator rollout sampling."""
        context = torch.zeros(sum(self.flow.dims), dtype=DTYPE)
        context[: self.flow.target] = theta.reshape(-1).to(DTYPE)
        z = flow_sample(self.flow, context, torch.tensor([1.0, 0.0]), n, steps, seed)
        return z[:, self.flow.target : self.flow.target + self.flow.d_x]


class GroupPosterior:
    """G-group flow queried with an arbitrary known-group assignment —
    hierarchical and multi-site conditionals."""

    def __init__(self, flow: Flow) -> None:
        self.flow = flow

    def sample(
        self, known: dict[int, torch.Tensor], n: int, steps: int = 4, seed: int = 0
    ) -> torch.Tensor:
        """known: {group_index: value}; returns (n, Σd_i) full vectors —
        generated groups live on the dims not present in `known`."""
        context = torch.zeros(sum(self.flow.dims), dtype=DTYPE)
        mask = torch.zeros(len(self.flow.dims))
        off = 0
        for i, d in enumerate(self.flow.dims):
            if i in known:
                context[off : off + d] = known[i].reshape(-1).to(DTYPE)
                mask[i] = 1.0
            off += d
        return flow_sample(self.flow, context, mask, n, steps, seed)


def fit_posterior(
    theta: torch.Tensor,
    x: torch.Tensor,
    k: int = 4,
    epochs: int = 200,
    rate: float = 3e-3,
    hidden: int = 128,
    seed: int = 0,
) -> Posterior:
    """(N, Dθ) params × (N, Dx) summaries → amortized posterior sampler."""
    return Posterior(fit_joint(theta, x, epochs=epochs, rate=rate, hidden=hidden, seed=seed))


def calibration_ranks(
    post: Posterior,
    theta: torch.Tensor,
    x: torch.Tensor,
    n: int = 64,
    steps: int = 4,
    seed: int = 0,
) -> torch.Tensor:
    """Rank of each true parameter among n posterior draws — (N, Dθ) in
    {0,…,n}; uniform iff the posterior is calibrated (Talts et al. 2018)."""
    g = torch.Generator().manual_seed(seed)
    draws = torch.stack(
        [
            post.sample(x[i], n, steps=steps, seed=int(torch.randint(2**31, (1,), generator=g)))
            for i in range(x.shape[0])
        ]
    )
    return (draws < theta.unsqueeze(1)).sum(1)
