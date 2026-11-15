"""E6 — physics-anchored JEPA ablation.

Does a self-supervised latent (predicted in embedding space by LatentDyn,
targeted by an EMA encoder, held isotropic by SIGReg) capture what the
physics misses? Four arms are scored on held-out forcing realizations:

- `prior`    : literature traits, observables vs truth — the baseline gap
- `jepa`     : JEPA-fitted latent + ridge probe z → obs (latent-space skill)
- `residual` : residual fitted on the obs loss with latent OFF (ablation)
- `hybrid`   : same residual fit but with the JEPA latent as ctx — the
               full physics-anchored JEPA arm

All targets are the truth rollout's observables; training sees them noisy
(spec.noise), evaluation is clean and out-of-sample by construction.
"""

from __future__ import annotations

import torch

from nasa_farm.dynamics.latent import ObservationEncoder
from nasa_farm.dynamics.model import WorldModel
from nasa_farm.experiments.run import register
from nasa_farm.experiments.scene import scene
from nasa_farm.experiments.specification import Specification
from nasa_farm.metrics import rmse
from nasa_farm.observe.trace import EmbeddingTrace
from nasa_farm.optimize.embedding import embedding_fit, seeded
from nasa_farm.optimize.trainer import fit
from nasa_farm.state.field import Field


def _trace(model, field: Field, forcing, actions, specification: Specification, k: int = 0):
    """Rollout with JEPATrace → (obs (B,N,C), z (B,N,K)|None)."""
    traits = EmbeddingTrace(
        forcing.days, k, specification.observed_channels, every=specification.cadence
    )
    model.rollout(field, forcing, actions, observer=traits)
    return traits.result()


def _noisy(t: torch.Tensor, noise: float, generation: torch.Generator) -> torch.Tensor:
    return t * (
        1.0 + noise * torch.randn(t.shape, generator=generation, dtype=t.dtype, device=t.device)
    )


def _ridge(z: torch.Tensor, y: torch.Tensor, lam: float = 1e-3) -> torch.Tensor:
    """Closed-form ridge z → y probe over flattened (B·N, K)/(B·N, C)."""
    zf = z.reshape(-1, z.shape[-1])
    yf = y.reshape(-1, y.shape[-1])
    a = zf.t() @ zf + lam * torch.eye(zf.shape[1], dtype=zf.dtype)
    return torch.linalg.solve(a, zf.t() @ yf)


def _obs_fit(
    model, field: Field, forcing, actions, targets: torch.Tensor, specification: Specification
) -> None:
    """Supervised obs-space fit of the residual — normalized MSE on the
    JEPATrace cadence, so JEPA and non-JEPA arms get identical budgets."""
    parameters = {f"p{i}": p for i, p in enumerate(model.residual.parameters())}
    scale = targets.std(dim=1, keepdim=True).clamp(min=1e-6)

    def loss() -> torch.Tensor:
        traits = EmbeddingTrace(
            forcing.days,
            model.latent.dim if model.latent else 0,
            specification.observed_channels,
            every=specification.cadence,
        )
        model.rollout(field, forcing, actions, observer=traits)
        observed, _ = traits.result()
        return (((observed - targets) / scale) ** 2).mean()

    fit(loss, parameters, epochs=specification.epochs, rate=specification.rate)


def _eval(
    model,
    scenario,
    specification: Specification,
    encoder=None,
    w: torch.Tensor | None = None,
    field=None,
) -> dict:
    """Held-out obs-space RMSE per channel; probe decoding when w given.
    `field` overrides the scene field (adapted z0/ctx eval arms)."""
    output: dict[str, list[float]] = {}
    k = model.latent.dim if model.latent else 0
    with torch.inference_mode():
        for i in scenario.held_indices():
            f_t = scenario.realization(i)
            obs_true, _ = _trace(
                scenario.truth, scenario.field, f_t, scenario.actions, specification
            )
            field_ = field if field is not None else scenario.field
            if encoder is not None:
                field_ = seeded(
                    field_,
                    model.latent.seed(encoder(obs_true[:, 0])),
                    model.latent.hidden_dimension,
                )
            observed_prediction, z_pred = _trace(
                model, field_, f_t, scenario.actions, specification, k
            )
            predicted = (z_pred @ w if w is not None else observed_prediction)[0]
            for j, ch in enumerate(specification.observed_channels):
                output.setdefault(ch, []).append(float(rmse(predicted[:, j], obs_true[0, :, j])))
    return {ch: round(sum(v) / len(v), 6) for ch, v in output.items()}


@register("jepa")
def run(specification: Specification) -> dict:
    torch.manual_seed(specification.seed)
    scenario = scene(specification)
    k = specification.latent_dimension
    ch = specification.observed_channels
    c = len(ch)
    generation = torch.Generator().manual_seed(specification.seed + 31)

    f0, a0 = scenario.realization(0), scenario.actions
    obs_true_cal, _ = _trace(scenario.truth, scenario.field, f0, a0, specification)
    calibration_observed = _noisy(obs_true_cal, specification.noise, generation)

    # JEPA arm: latent dynamics trained against the truth obs manifold
    m_j = WorldModel(scenario.crops, residual=True, latent_dimension=k, seed=specification.seed)
    encoder = ObservationEncoder(observed_dimension=c, z_dim=k)
    rep = embedding_fit(
        m_j,
        encoder,
        scenario.field,
        f0,
        a0,
        ch,
        every=specification.cadence,
        ema=specification.ema,
        sig_weight=specification.sig_weight,
        epochs=specification.epochs,
        targets=calibration_observed,
    )
    w = _ridge(rep["z_pred"][:, 1:], calibration_observed[:, 1:])
    ev_jep = _eval(m_j, scenario, specification, encoder=encoder, w=w)

    # Hierarchical arm (optional): same protocol, HLatent dynamics
    ev_hier = None
    if specification.hierarchy:
        m_h = WorldModel(
            scenario.crops,
            residual=True,
            latent_dimension=k,
            latent_strides=specification.hierarchy,
            seed=specification.seed,
        )
        enc_h = ObservationEncoder(observed_dimension=c, z_dim=k)
        rep_h = embedding_fit(
            m_h,
            enc_h,
            scenario.field,
            f0,
            a0,
            ch,
            every=specification.cadence,
            ema=specification.ema,
            sig_weight=specification.sig_weight,
            epochs=specification.epochs,
            targets=calibration_observed,
        )
        w_h = _ridge(rep_h["z_pred"][:, 1:], calibration_observed[:, 1:])
        ev_hier = _eval(m_h, scenario, specification, encoder=enc_h, w=w_h)

    # Supervised arms, identical obs-fit budget; the hybrid arm's residual
    # is fitted on the z-seeded field so its ctx input is live.
    m_r = WorldModel(scenario.crops, residual=True, seed=specification.seed)
    _obs_fit(m_r, scenario.field, f0, a0, calibration_observed, specification)
    ev_res = _eval(m_r, scenario, specification)
    # detached seed: the encoder is frozen during the residual obs-fit —
    # z is a fixed ctx input, not a gradient path back into the encoder.
    f_j = seeded(
        scenario.field, encoder(calibration_observed[:, 0]).detach(), m_j.latent.hidden_dimension
    )
    _obs_fit(m_j, f_j, f0, a0, calibration_observed, specification)

    ev_prior = _eval(scenario.prior, scenario, specification)
    ev_hyb = _eval(m_j, scenario, specification, encoder=encoder)
    per_ch = {
        name: {
            "prior": ev_prior[name],
            "jepa_probe": ev_jep[name],
            "residual": ev_res[name],
            "hybrid": ev_hyb[name],
            "hybrid_reduction_pct": round(100 * (ev_prior[name] - ev_hyb[name]) / ev_prior[name], 2)
            if ev_prior[name] > 0
            else 0.0,
        }
        for name in ch
    }

    z_rmse = float((rep["z_pred"][:, 1:] - rep["z_tgt"][:, 1:]).pow(2).mean().sqrt())
    output = {
        "latent": {
            "dim": k,
            "sigreg_final": rep["sigreg_final"],
            "z_std_final": rep["z_std_final"],
            "best_loss": rep["best_loss"],
            "epochs_run": rep["epochs_run"],
        },
        "z_pred_rmse": round(z_rmse, 6),
        "channels": per_ch,
    }
    if ev_hier is not None:
        output["hier"] = {
            "strides": list(specification.hierarchy),
            "channels": ev_hier,
            "best_loss": rep_h["best_loss"],
        }
    return output
