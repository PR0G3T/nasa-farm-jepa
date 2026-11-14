"""JEPA pretraining — self-supervised latent dynamics on observation streams.

The pipeline is the LeJEPA recipe anchored to physics: an online encoder
embeds observations, the recurrent `LatentDyn` predicts the next latent
inside the rollout itself, and an EMA `TargetEncoder` supplies the targets
the predictor chases. SIGReg keeps the online embeddings near an isotropic
Gaussian so the prediction task cannot collapse. Nothing is ever
reconstructed in observation space — prediction happens in latent space,
exactly the JEPA contract.

`jepa_fit` needs no labels. Two modes:
- self-dynamics (obs_tgt=None): the latent learns to forecast the model's
  own observable manifold — representation pretraining;
- gap-tracking (obs_tgt given): targets come from an external stream
  (satellite series, twin truth), so the latent is driven to carry the
  dynamics the physics cannot produce — the residual then reads that gap
  through its ctx input.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from nasa_farm.constants import DTYPE
from nasa_farm.dynamics.latent import TargetEncoder, sigreg
from nasa_farm.observe import observables
from nasa_farm.observe.trace import JEPATrace
from nasa_farm.state.field import Field
from nasa_farm.types import Actions, Forcing


def rdmreg(z: torch.Tensor, gen: torch.Generator,
           directions: int = 64) -> torch.Tensor:
    """Sliced-Wasserstein match of z to a rectified Laplacian.

    R(z) = E_c W2^2( P(c·z) ‖ P(c·g) ) over random unit directions c,
    g = ReLU(Laplace) — the max-entropy target under an E|x| constraint,
    yielding nonneg sparse mode-factored codes. 1-D W2^2 between equal-size
    samples is the mean squared difference of sorted values.
    """
    zf = z.reshape(-1, z.shape[-1])
    n, d = zf.shape
    c = torch.randn(d, directions, generator=gen, dtype=z.dtype,
                    device=z.device)
    c = c / c.norm(dim=0, keepdim=True).clamp(min=1e-9)
    e1 = torch.rand(n, d, generator=gen, dtype=z.dtype,
                    device=z.device).exponential_()
    e2 = torch.rand(n, d, generator=gen, dtype=z.dtype,
                    device=z.device).exponential_()
    g = torch.relu(e1 - e2) @ c                    # c·y, y ~ rectified GG
    return ((zf @ c).sort(0).values - g.sort(0).values).pow(2).mean()


def seeded(field: Field, z0: torch.Tensor, h_dim: int) -> Field:
    """Field copy whose state carries seed latent z0 (zh reset) — the
    prediction path enters the rollout through the state, so encoder
    gradients flow through the whole unroll."""
    s = field.state
    zh = torch.zeros(s.batch, h_dim, dtype=DTYPE, device=z0.device)
    return replace(field, state=s.replace(z=z0, zh=zh))


def jepa_fit(model, encoder, field: Field, forcing: Forcing,
             actions: Actions, channels: tuple[str, ...], *,
             obs_tgt: torch.Tensor | None = None, every: int = 5,
             ema: float = 0.996, sig_weight: float = 1.0,
             kl_weight: float = 0.0, dense: int = 0, sparse: bool = False,
             epochs: int = 60, lr: float = 3e-3,
             patience: int = 15, verbose: bool = False) -> dict:
    """Fit encoder + LatentDyn so predicted latents track the EMA-encoded
    observation manifold along the rollout. Adam + clip + early stop +
    best-parameter restore — same contract as trainer.fit plus the
    momentum-target update after each step.

    obs_tgt: (B, N, C) external target stream at the JEPATrace cadence;
    index 0 seeds the latent and is excluded from the loss."""
    if model.latent is None:
        raise ValueError("jepa_fit needs a model built with latent_dim > 0")
    k, h = model.latent.dim, model.latent.h_dim
    tgt = TargetEncoder(encoder, decay=ema)
    named = [(f"enc.{n}", p) for n, p in encoder.named_parameters()] + \
            [(f"lat.{n}", p) for n, p in model.latent.named_parameters()]
    rgen = torch.Generator().manual_seed(0)
    leaves = [p for _, p in named]
    opt = torch.optim.Adam(leaves, lr=lr)
    best, stall, history = float("inf"), 0, []
    best_params = {n: p.detach().clone() for n, p in named}

    def unroll() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(z_pred (B,N,K*), targets (B,N,C), online-embed source).
        K* = K for a flat latent, L·K packed for a hierarchical one."""
        if obs_tgt is not None:
            f = seeded(field, model.latent.seed(encoder(obs_tgt[:, 0])), h)
            tr = JEPATrace(forcing.days, k, channels, every=every)
            model.rollout(f, forcing, actions, observer=tr)
            _, z_pred = tr.result()
            return z_pred, obs_tgt, obs_tgt
        f = seeded(field, model.latent.seed(
            encoder(observables(field.state, channels))), h)
        tr = JEPATrace(forcing.days, k, channels, every=every)
        model.rollout(f, forcing, actions, observer=tr)
        obs, z_pred = tr.result()
        return z_pred, obs, obs

    def step_loss() -> torch.Tensor:
        from nasa_farm.dynamics.hierarchical import HLatent, hjepa_loss
        z_pred, targets, src = unroll()
        z_t = tgt(targets)
        if isinstance(model.latent, HLatent):
            pred = hjepa_loss(z_pred, z_t, model.latent, every)
        else:
            pred = (z_pred[:, 1:] - z_t[:, 1:]).pow(2).mean()
            # dense multi-horizon supervision: each committed latent also
            # matches targets 2..dense steps ahead, weight 1/h.
            for hh in range(2, dense + 1):
                pred = pred + (z_pred[:, :-hh]
                               - z_t[:, hh:]).pow(2).mean() / hh
        kl = model.latent.kl if kl_weight else torch.zeros((), dtype=DTYPE)
        enc = encoder(src)
        reg = (rdmreg(enc.reshape(-1, enc.shape[-1]), rgen)
               if sparse else sigreg(enc.reshape(-1, enc.shape[-1])))
        return pred + sig_weight * reg + kl_weight * kl

    for ep in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        loss = step_loss()
        if not torch.isfinite(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(leaves, 1.0)
        opt.step()
        tgt.update_from(encoder)
        v = float(loss.detach())
        history.append(v)
        if verbose and ep % 10 == 0:
            print(f"  epoch {ep}: jepa {v:.5f}")
        if v < best - 1e-9:
            best, stall = v, 0
            best_params = {n: p.detach().clone() for n, p in named}
        else:
            stall += 1
            if stall >= patience:
                break
    with torch.no_grad():
        for n, p in named:
            p.copy_(best_params[n])
        tgt.update_from(encoder)
        z_pred, targets, src = unroll()
        z_tgt = tgt(targets)
        z_std = float(encoder(src).std())
        enc = encoder(src)
        sig = float(sigreg(enc.reshape(-1, enc.shape[-1])))
    return {"best_loss": round(best, 8), "history": history,
            "epochs_run": len(history), "tgt": tgt,
            "z_pred": z_pred.detach(), "z_tgt": z_tgt.detach(),
            "obs": targets.detach(), "z_std_final": round(z_std, 6),
            "sigreg_final": round(sig, 6)}
