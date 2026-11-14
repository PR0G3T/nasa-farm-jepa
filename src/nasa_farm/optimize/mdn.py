"""Simulation-based inference — amortized posterior over parameters.

`MDN` is a mixture-density network q(θ|x) trained on (θ, x) pairs produced
by rolling the model forward under sampled parameters. Once fitted, the
posterior over traits/pedotransfers given an observed summary x is one
forward pass — amortized Bayesian calibration replacing thousands of
gradient-HPO restarts.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nasa_farm.constants import DTYPE


class MDN(nn.Module):
    """K-component diagonal Gaussian mixture, in_dim → out_dim."""

    def __init__(self, in_dim: int, out_dim: int, k: int = 4,
                 hidden: int = 96, depth: int = 2) -> None:
        super().__init__()
        self.k, self.out_dim = k, out_dim
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, k * (2 * out_dim + 1))
        self.to(torch.float64)

    def params(self, x: torch.Tensor):
        h = self.head(self.body(x))
        pi = torch.log_softmax(h[..., :self.k], dim=-1)
        rest = h[..., self.k:].view(*h.shape[:-1], self.k, 2 * self.out_dim)
        mu, lv = rest.chunk(2, dim=-1)
        return pi, mu, lv.clamp(-9.0, 4.0)

    def nll(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        pi, mu, lv = self.params(x)
        d = (theta.unsqueeze(-2) - mu) / (0.5 * lv).exp()
        ll = pi - 0.5 * (d ** 2).sum(-1) - 0.5 * lv.sum(-1) \
            - 0.5 * self.out_dim * torch.log(
                torch.tensor(2 * torch.pi, dtype=x.dtype))
        return -torch.logsumexp(ll, dim=-1).mean()

    def sample(self, x: torch.Tensor, n: int,
               gen: torch.Generator | None = None) -> torch.Tensor:
        """(n, out_dim) posterior samples for one conditioning vector x."""
        pi, mu, lv = self.params(x)
        c = torch.multinomial(pi.exp().squeeze(0), n, replacement=True,
                              generator=gen)
        m, s = mu[0, c], (0.5 * lv[0, c]).exp()
        return m + s * torch.randn(m.shape, generator=gen, dtype=m.dtype)


def fit_posterior(theta: torch.Tensor, x: torch.Tensor,
                  k: int = 4, epochs: int = 600, lr: float = 2e-3,
                  seed: int = 0) -> MDN:
    """Train q(θ|x) on (θ_i, x_i) simulation pairs → amortized posterior."""
    torch.manual_seed(seed)
    mdn = MDN(x.shape[-1], theta.shape[-1], k)
    opt = torch.optim.Adam(mdn.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = mdn.nll(x, theta)
        loss.backward()
        opt.step()
    return mdn
