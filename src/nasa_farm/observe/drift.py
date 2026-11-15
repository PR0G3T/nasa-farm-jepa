"""Stream drift detection — evidence for (or against) recalibration.

`psi` is the per-feature population stability index on reference-quantile
bins; `energy` is the energy distance between reference and current
windows. Together they catch marginal shifts and joint-distribution drift
before either degrades plan quality, and `drifted` turns them into the
boolean the maintenance gate consumes.
"""

from __future__ import annotations

import torch

from nasa_farm.constants import DTYPE


def psi(
    reference: torch.Tensor, current: torch.Tensor, bins: int = 10, eps: float = 1e-6
) -> torch.Tensor:
    """Population Stability Index per column: Σ (p−q)·ln(p/q).

    Bin edges come from `ref` quantiles so any shift in `cur` shows up as
    mass moving across bins. psi ≈ 0 for same-distribution windows; the
    conventional alarm level is 0.25."""
    edges = torch.quantile(
        reference.to(DTYPE),
        torch.linspace(0, 1, bins + 1, dtype=DTYPE, device=reference.device),
        dim=0,
    )
    output = torch.zeros(reference.shape[1], dtype=DTYPE, device=reference.device)
    for c in range(reference.shape[1]):
        e = torch.unique(edges[:, c])
        if e.numel() < 2:
            continue
        cuts = e[1:-1]
        p = torch.bincount(
            torch.bucketize(reference[:, c].contiguous().to(DTYPE), cuts), minlength=e.numel() - 1
        ).to(DTYPE)
        q = torch.bincount(
            torch.bucketize(current[:, c].contiguous().to(DTYPE), cuts), minlength=e.numel() - 1
        ).to(DTYPE)
        p = (p / p.sum()).clamp(min=eps)
        q = (q / q.sum()).clamp(min=eps)
        output[c] = ((p - q) * (p / q).log()).sum()
    return output


def energy(
    reference: torch.Tensor, current: torch.Tensor, n: int = 512, seed: int = 0
) -> torch.Tensor:
    """Energy distance 2E‖X−Y‖ − E‖X−X'‖ − E‖Y−Y'‖ (0 iff same law).
    Windows are subsampled to `n` rows so the pairwise cost stays bounded."""
    g = torch.Generator(device=reference.device).manual_seed(seed)

    def sub(x):
        if x.shape[0] <= n:
            return x.to(DTYPE)
        return x[torch.randperm(x.shape[0], generator=g, device=x.device)[:n]].to(DTYPE)

    x, y = sub(reference), sub(current)
    return 2.0 * torch.cdist(x, y).mean() - torch.cdist(x, x).mean() - torch.cdist(y, y).mean()


def drifted(
    reference: torch.Tensor, current: torch.Tensor, limit: float = 0.25, bins: int = 10
) -> bool:
    """True when any feature's PSI exceeds `limit` — the trigger the
    recalibration gate consumes."""
    return bool((psi(reference, current, bins=bins) > limit).any())
