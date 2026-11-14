"""Dependency-sparsity rotation — post-hoc latent disentanglement.

A JEPA identifies the latent only up to a linear map. DSReg searches the
orthogonal group for the rotation under which the observations depend on
the fewest latents: under structural diversity (distinct dependency
footprints) the recovered axes are the individual world latents, up to a
signed permutation. Solved by cyclic Jacobi sweeps over plane rotations —
deterministic, no learning rate, no decoder, no labels.
"""

from __future__ import annotations

import torch


def whiten(h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero-mean identity-covariance transform. Returns (hw, mean, W)."""
    mean = h.mean(0)
    covariance = (h - mean).t() @ (h - mean) / (h.shape[0] - 1)
    evals, evecs = torch.linalg.eigh(covariance)
    W = evecs @ torch.diag(evals.clamp(min=1e-12).rsqrt()) @ evecs.t()
    return (h - mean) @ W, mean, W


def jacobians(h: torch.Tensor, x: torch.Tensor, k: int = 16,
              lam: float = 1e-3) -> torch.Tensor:
    """Local dx/dh Jacobians via kNN ridge fits → (n, p, d)."""
    n = h.shape[0]
    k = min(k + 1, n)
    index = torch.cdist(h, h).topk(k, largest=False).indices
    nbr_h = h[index]                                     # (n, k, d)
    A = torch.cat([torch.ones(n, k, 1, dtype=h.dtype, device=h.device),
                   nbr_h], -1)
    Y = x[index]                                         # (n, k, p)
    ata = A.transpose(1, 2) @ A + lam * torch.eye(
        A.shape[-1], dtype=h.dtype, device=h.device)
    beta = torch.linalg.solve(ata, A.transpose(1, 2) @ Y)
    J = beta[:, 1:, :].transpose(1, 2)                 # (n, p, d)
    return J / J.norm(dim=-1, keepdim=True).clamp(min=1e-9)


def _rotate_pair(J: torch.Tensor, R: torch.Tensor, i: int, j: int,
                 grid: int) -> None:
    """Optimal in-plane rotation minimizing Σ|J'| — exact 1-D grid search."""
    th = torch.linspace(0, torch.pi, grid, dtype=J.dtype, device=J.device)
    c, s = th.cos(), th.sin()
    Ji, Jj = J[:, :, i], J[:, :, j]                    # (n, p)
    cost = ((Ji.unsqueeze(-1) * c - Jj.unsqueeze(-1) * s).abs().sum((0, 1))
            + (Ji.unsqueeze(-1) * s + Jj.unsqueeze(-1) * c).abs().sum((0, 1)))
    t = th[cost.argmin()]
    ct, state = t.cos(), t.sin()
    Ji2, Jj2 = Ji * ct - Jj * state, Ji * state + Jj * ct
    J[:, :, i], J[:, :, j] = Ji2, Jj2
    Ri, Rj = R[:, i].clone(), R[:, j].clone()
    R[:, i], R[:, j] = Ri * ct - Rj * state, Ri * state + Rj * ct


def rotation(h: torch.Tensor, x: torch.Tensor, k: int = 16,
          sweeps: int = 12, grid: int = 128,
          ) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Fit the sparsest-dependency rotation. Returns (R, z_tilde, report).

    h (n, d) frozen representation; x (n, p) observations it came from.
    z_tilde = whiten(h) @ R — individual latents up to sign/permutation.
    """
    hardware, _, _ = whiten(h)
    J = jacobians(hardware, x, k)
    d = hardware.shape[1]
    R = torch.eye(d, dtype=hardware.dtype, device=hardware.device)
    for _ in range(sweeps):
        for i in range(d):
            for j in range(i + 1, d):
                _rotate_pair(J, R, i, j, grid)
    loss = float(J.abs().mean())
    return R, hardware @ R, {"loss": round(loss, 8)}
