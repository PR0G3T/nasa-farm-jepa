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
from nasa_farm.dynamics.latent import TargetEncoder, regularization
from nasa_farm.observe import observables
from nasa_farm.observe.trace import EmbeddingTrace
from nasa_farm.state.field import Field
from nasa_farm.types import Actions, Forcing


def dissimilarity(
    z: torch.Tensor, generation: torch.Generator, directions: int = 64
) -> torch.Tensor:
    """Sliced-Wasserstein match of z to a rectified Laplacian.

    R(z) = E_c W2^2( P(c·z) ‖ P(c·g) ) over random unit directions c,
    g = ReLU(Laplace) — the max-entropy target under an E|x| constraint,
    yielding nonneg sparse mode-factored codes. 1-D W2^2 between equal-size
    samples is the mean squared difference of sorted values.
    """
    zf = z.reshape(-1, z.shape[-1])
    n, d = zf.shape
    c = torch.randn(d, directions, generator=generation, dtype=z.dtype, device=z.device)
    c = c / c.norm(dim=0, keepdim=True).clamp(min=1e-9)
    e1 = torch.rand(n, d, generator=generation, dtype=z.dtype, device=z.device).exponential_()
    e2 = torch.rand(n, d, generator=generation, dtype=z.dtype, device=z.device).exponential_()
    g = torch.relu(e1 - e2) @ c  # c·y, y ~ rectified GG
    return ((zf @ c).sort(0).values - g.sort(0).values).pow(2).mean()


def seeded(field: Field, z0: torch.Tensor, hidden_dimension: int) -> Field:
    """Field copy whose state carries seed latent z0 (zh reset) — the
    prediction path enters the rollout through the state, so encoder
    gradients flow through the whole unroll."""
    s = field.state
    zh = torch.zeros(s.batch, hidden_dimension, dtype=DTYPE, device=z0.device)
    return replace(field, state=s.replace(z=z0, zh=zh))


def embedding_fit(
    model,
    encoder,
    field: Field,
    forcing: Forcing,
    actions: Actions,
    channels: tuple[str, ...],
    *,
    targets: torch.Tensor | None = None,
    every: int = 5,
    ema: float = 0.996,
    sig_weight: float = 1.0,
    kl_weight: float = 0.0,
    dense: int = 0,
    sparse: bool = False,
    epochs: int = 60,
    rate: float = 3e-3,
    patience: int = 15,
    verbose: bool = False,
) -> dict:
    """Fit encoder + LatentDyn so predicted latents track the EMA-encoded
    observation manifold along the rollout. Adam + clip + early stop +
    best-parameter restore — same contract as trainer.fit plus the
    momentum-target update after each step. `history` logs the loss and
    its components (predict, sigreg, kl) plus the encoder's z_std each
    epoch — the collapse monitor.

    obs_tgt: (B, N, C) external target stream at the JEPATrace cadence;
    index 0 seeds the latent and is excluded from the loss."""
    if model.latent is None:
        raise ValueError("jepa_fit needs a model built with latent_dim > 0")
    k, h = model.latent.dim, model.latent.hidden_dimension
    target = TargetEncoder(encoder, decay=ema)
    named = [(f"enc.{n}", p) for n, p in encoder.named_parameters()] + [
        (f"lat.{n}", p) for n, p in model.latent.named_parameters()
    ]
    rgen = torch.Generator().manual_seed(0)
    leaves = [p for _, p in named]
    opt = torch.optim.Adam(leaves, lr=rate)
    best, stall, history = float("inf"), 0, []
    best_params = {n: p.detach().clone() for n, p in named}

    def unroll() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(z_pred (B,N,K*), targets (B,N,C), online-embed source).
        K* = K for a flat latent, L·K packed for a hierarchical one."""
        if targets is not None:
            f = seeded(field, model.latent.seed(encoder(targets[:, 0])), h)
            traits = EmbeddingTrace(forcing.days, k, channels, every=every)
            model.rollout(f, forcing, actions, observer=traits)
            _, z_pred = traits.result()
            return z_pred, targets, targets
        f = seeded(field, model.latent.seed(encoder(observables(field.state, channels))), h)
        traits = EmbeddingTrace(forcing.days, k, channels, every=every)
        model.rollout(f, forcing, actions, observer=traits)
        observed, z_pred = traits.result()
        return z_pred, observed, observed

    def step_loss() -> tuple[torch.Tensor, dict[str, float]]:
        from nasa_farm.dynamics.hierarchical import Hierarchy, hierarchical_loss

        z_pred, targets, source = unroll()
        z_t = target(targets)
        if isinstance(model.latent, Hierarchy):
            predicted = hierarchical_loss(z_pred, z_t, model.latent, every)
        else:
            predicted = (z_pred[:, 1:] - z_t[:, 1:]).pow(2).mean()
            # dense multi-horizon supervision: each committed latent also
            # matches targets 2..dense steps ahead, weight 1/h.
            for hh in range(2, dense + 1):
                predicted = predicted + (z_pred[:, :-hh] - z_t[:, hh:]).pow(2).mean() / hh
        kl = model.latent.kl if kl_weight else torch.zeros((), dtype=DTYPE)
        encoded = encoder(source)
        penalty = (
            dissimilarity(encoded.reshape(-1, encoded.shape[-1]), rgen)
            if sparse
            else regularization(encoded.reshape(-1, encoded.shape[-1]))
        )
        record = {
            "predict": round(float(predicted.detach()), 8),
            "sigreg": round(float(penalty.detach()), 8),
            "kl": round(float(kl.detach()), 8),
            "z_std": round(float(encoded.detach().std()), 6),
        }
        return predicted + sig_weight * penalty + kl_weight * kl, record

    for evaporation in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        loss, record = step_loss()
        if not torch.isfinite(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(leaves, 1.0)
        opt.step()
        target.update_from(encoder)
        v = float(loss.detach())
        history.append({"loss": v, **record})
        if verbose and evaporation % 10 == 0:
            print(f"  epoch {evaporation}: jepa {v:.5f}")
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
        target.update_from(encoder)
        z_pred, targets, source = unroll()
        z_target = target(targets)
        z_std = float(encoder(source).std())
        encoded = encoder(source)
        signal = float(regularization(encoded.reshape(-1, encoded.shape[-1])))
    return {
        "best_loss": round(best, 8),
        "history": history,
        "epochs_run": len(history),
        "tgt": target,
        "z_pred": z_pred.detach(),
        "z_tgt": z_target.detach(),
        "obs": targets.detach(),
        "z_std_final": round(z_std, 6),
        "sigreg_final": round(signal, 6),
    }
