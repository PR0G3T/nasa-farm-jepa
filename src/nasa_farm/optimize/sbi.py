"""Joint masked flow for simulation-based inference — OneFlowSBI-style.

A single conditional flow-matching model over the joint vector z=(θ, x)
with a 2-bit mask declaring which side is *known*. The same trained net
serves:

- posterior  p(θ|x)   — x masked-known, θ generated
- likelihood p(x|θ)   — θ masked-known, x generated (emulator sampling)
- arbitrary conditionals / joint draws — both bits off or partial masks

Training is conditional flow matching: z_t = (1−t)·ε + t·z, target
velocity v = z − ε, with the known side frozen at its data value. Sampling
integrates v from t=0 (noise) to t=1 in a few Euler steps — `steps=4`
already beats a trained MDN on multimodal posteriors because the map is
shared across contexts.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nasa_farm.constants import DTYPE


class JointFlow(nn.Module):
    """v(z_t, t, mask, ctx) → velocity. mask: (2,) [θ_known, x_known]."""

    def __init__(self, d_theta: int, d_x: int, hidden: int = 128,
                 depth: int = 3) -> None:
        super().__init__()
        self.d_t, self.d_x = d_theta, d_x
        d = (d_theta + d_x) * 2 + 3
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, d_theta + d_x)]
        self.net = nn.Sequential(*layers)
        self.to(DTYPE)

    def forward(self, z: torch.Tensor, t: torch.Tensor,
                mask: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        tb = t.to(DTYPE).expand(z.shape[0], 1)
        mb = mask.to(DTYPE).expand(z.shape[0], 2)
        v = self.net(torch.cat([z, tb, mb, ctx.to(DTYPE)], dim=-1))
        reps = torch.tensor([self.d_t, self.d_x], device=z.device)
        return v * (1.0 - torch.repeat_interleave(mb, reps, dim=1))


def _masked_ctx(theta: torch.Tensor, x: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
    """(B, Dθ+Dx) context: data on known dims, zeros elsewhere."""
    return torch.cat([theta * mask[:, 0:1], x * mask[:, 1:2]], dim=-1)


def fit_joint(theta: torch.Tensor, x: torch.Tensor, epochs: int = 300,
              lr: float = 2e-3, hidden: int = 128,
              seed: int = 0) -> JointFlow:
    """CFM training with random mask flips: half posterior, half likelihood."""
    th, xx = theta.to(DTYPE), x.to(DTYPE)
    d_t, d_x = th.shape[-1], xx.shape[-1]
    m = JointFlow(d_t, d_x, hidden)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    n = th.shape[0]
    for _ in range(epochs):
        i = torch.randint(0, n, (n,), generator=gen)
        z1 = torch.cat([th[i], xx[i]], -1)
        t = torch.rand(n, 1, generator=gen, dtype=DTYPE)
        eps = torch.randn(z1.shape, generator=gen, dtype=DTYPE)
        mask = (torch.rand(n, 1, generator=gen) < 0.5).to(DTYPE)
        mask = torch.cat([mask, 1.0 - mask], dim=1)        # exactly one side
        zt = (1.0 - t) * eps + t * z1
        ctx = _masked_ctx(th[i], xx[i], mask)
        # freeze known side: keep z there at its data value
        known = torch.cat([mask[:, 0:1].expand(-1, d_t),
                           mask[:, 1:2].expand(-1, d_x)], dim=1)
        zt = torch.where(known.bool(), z1, zt)
        pred = m(zt, t, mask, ctx)
        loss = ((pred - (z1 - eps)) ** 2 * (1.0 - known)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return m


@torch.no_grad()
def flow_sample(m: JointFlow, ctx: torch.Tensor, mask: torch.Tensor,
                n: int, steps: int = 4, seed: int = 0) -> torch.Tensor:
    """Generate the masked-off side given `ctx` ((Dθ+Dx,) or (1,·))."""
    gen = torch.Generator().manual_seed(seed)
    c = ctx.reshape(-1).to(DTYPE)
    z = torch.cat([
        torch.randn(n, m.d_t, generator=gen, dtype=DTYPE) * (1 - mask[0]),
        torch.randn(n, m.d_x, generator=gen, dtype=DTYPE) * (1 - mask[1])],
        dim=-1)
    known = torch.cat([mask[0].expand(m.d_t), mask[1].expand(m.d_x)]).bool()
    z[:, known] = c[known]
    ctxb = (c * torch.cat([mask[0].expand(m.d_t),
                           mask[1].expand(m.d_x)])).expand(n, -1)
    for s in range(steps):
        t = torch.full((n, 1), s / steps, dtype=DTYPE)
        z = z + m(z, t, mask, ctxb) / steps
        z[:, known] = c[known]
    return z


class Posterior:
    """Trained joint flow queried as a posterior — the SBI entry point."""

    def __init__(self, flow: JointFlow) -> None:
        self.flow = flow

    def sample(self, x: torch.Tensor, n: int, steps: int = 4,
               seed: int = 0) -> torch.Tensor:
        """θ | x → (n, Dθ). x: (Dx,) or (1, Dx)."""
        ctx = torch.zeros(self.flow.d_t + self.flow.d_x, dtype=DTYPE)
        ctx[self.flow.d_t:] = x.reshape(-1).to(DTYPE)
        z = flow_sample(self.flow, ctx, torch.tensor([0.0, 1.0]), n, steps,
                        seed)
        return z[:, :self.flow.d_t]

    def likelihood(self, theta: torch.Tensor, n: int, steps: int = 4,
                   seed: int = 0) -> torch.Tensor:
        """x | θ → (n, Dx) — emulator rollout sampling."""
        ctx = torch.zeros(self.flow.d_t + self.flow.d_x, dtype=DTYPE)
        ctx[:self.flow.d_t] = theta.reshape(-1).to(DTYPE)
        z = flow_sample(self.flow, ctx, torch.tensor([1.0, 0.0]), n, steps,
                        seed)
        return z[:, self.flow.d_t:]


def fit_posterior(theta: torch.Tensor, x: torch.Tensor, k: int = 4,
                  epochs: int = 200, lr: float = 3e-3, hidden: int = 128,
                  seed: int = 0) -> Posterior:
    """(N, Dθ) params × (N, Dx) summaries → amortized posterior sampler."""
    return Posterior(fit_joint(theta, x, epochs=epochs, lr=lr,
                               hidden=hidden, seed=seed))
