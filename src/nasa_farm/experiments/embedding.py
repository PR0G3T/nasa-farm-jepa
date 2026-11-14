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

from nasa_farm.dynamics.latent import ObsEncoder
from nasa_farm.dynamics.model import WorldModel
from nasa_farm.experiments.run import register
from nasa_farm.experiments.scene import scene
from nasa_farm.experiments.specification import Spec
from nasa_farm.metrics import rmse
from nasa_farm.observe.trace import JEPATrace
from nasa_farm.optimize.embedding import jepa_fit, seeded
from nasa_farm.optimize.trainer import fit
from nasa_farm.state.field import Field


def _trace(model, field: Field, forcing, actions, spec: Spec, k: int = 0):
    """Rollout with JEPATrace → (obs (B,N,C), z (B,N,K)|None)."""
    tr = JEPATrace(forcing.days, k, spec.obs_channels, every=spec.obs_every)
    model.rollout(field, forcing, actions, observer=tr)
    return tr.result()


def _noisy(t: torch.Tensor, noise: float, gen: torch.Generator) -> torch.Tensor:
    return t * (1.0 + noise * torch.randn(t.shape, generator=gen,
                                          dtype=t.dtype, device=t.device))


def _ridge(z: torch.Tensor, y: torch.Tensor, lam: float = 1e-3) -> torch.Tensor:
    """Closed-form ridge z → y probe over flattened (B·N, K)/(B·N, C)."""
    zf = z.reshape(-1, z.shape[-1])
    yf = y.reshape(-1, y.shape[-1])
    a = zf.t() @ zf + lam * torch.eye(zf.shape[1], dtype=zf.dtype)
    return torch.linalg.solve(a, zf.t() @ yf)


def _obs_fit(model, field: Field, forcing, actions,
             obs_tgt: torch.Tensor, spec: Spec) -> None:
    """Supervised obs-space fit of the residual — normalized MSE on the
    JEPATrace cadence, so JEPA and non-JEPA arms get identical budgets."""
    params = {f"p{i}": p for i, p in enumerate(model.residual.parameters())}
    scale = obs_tgt.std(dim=1, keepdim=True).clamp(min=1e-6)

    def loss() -> torch.Tensor:
        tr = JEPATrace(forcing.days, model.latent.dim if model.latent else 0,
                       spec.obs_channels, every=spec.obs_every)
        model.rollout(field, forcing, actions, observer=tr)
        obs, _ = tr.result()
        return (((obs - obs_tgt) / scale) ** 2).mean()

    fit(loss, params, epochs=spec.epochs, lr=spec.lr)


def _eval(model, sc, spec: Spec, encoder=None,
          w: torch.Tensor | None = None, field=None) -> dict:
    """Held-out obs-space RMSE per channel; probe decoding when w given.
    `field` overrides the scene field (adapted z0/ctx eval arms)."""
    out: dict[str, list[float]] = {}
    k = model.latent.dim if model.latent else 0
    with torch.inference_mode():
        for i in range(sc.forcing_t.batch):
            f_t = sc.realization(i)
            obs_true, _ = _trace(sc.truth, sc.field, f_t, sc.actions, spec)
            field_ = field if field is not None else sc.field
            if encoder is not None:
                field_ = seeded(field_, model.latent.seed(
                    encoder(obs_true[:, 0])), model.latent.h_dim)
            obs_pred, z_pred = _trace(model, field_, f_t, sc.actions,
                                      spec, k)
            pred = (z_pred @ w if w is not None else obs_pred)[0]
            for j, ch in enumerate(spec.obs_channels):
                out.setdefault(ch, []).append(
                    float(rmse(pred[:, j], obs_true[0, :, j])))
    return {ch: round(sum(v) / len(v), 6) for ch, v in out.items()}


@register("jepa")
def run(spec: Spec) -> dict:
    torch.manual_seed(spec.seed)
    sc = scene(spec)
    k = spec.latent_dim
    ch = spec.obs_channels
    c = len(ch)
    gen = torch.Generator().manual_seed(spec.seed + 31)

    f0, a0 = sc.realization(0), sc.actions
    obs_true_cal, _ = _trace(sc.truth, sc.field, f0, a0, spec)
    obs_cal = _noisy(obs_true_cal, spec.noise, gen)

    # JEPA arm: latent dynamics trained against the truth obs manifold
    m_j = WorldModel(sc.crops, residual=True, latent_dim=k, seed=spec.seed)
    enc = ObsEncoder(obs_dim=c, z_dim=k)
    rep = jepa_fit(m_j, enc, sc.field, f0, a0, ch,
                   every=spec.obs_every, ema=spec.ema,
                   sig_weight=spec.sig_weight, epochs=spec.epochs,
                   obs_tgt=obs_cal)
    w = _ridge(rep["z_pred"][:, 1:], obs_cal[:, 1:])
    ev_jep = _eval(m_j, sc, spec, encoder=enc, w=w)

    # Hierarchical arm (optional): same protocol, HLatent dynamics
    ev_hier = None
    if spec.hier:
        m_h = WorldModel(sc.crops, residual=True, latent_dim=k,
                         latent_strides=spec.hier, seed=spec.seed)
        enc_h = ObsEncoder(obs_dim=c, z_dim=k)
        rep_h = jepa_fit(m_h, enc_h, sc.field, f0, a0, ch,
                         every=spec.obs_every, ema=spec.ema,
                         sig_weight=spec.sig_weight, epochs=spec.epochs,
                         obs_tgt=obs_cal)
        w_h = _ridge(rep_h["z_pred"][:, 1:], obs_cal[:, 1:])
        ev_hier = _eval(m_h, sc, spec, encoder=enc_h, w=w_h)

    # Supervised arms, identical obs-fit budget; the hybrid arm's residual
    # is fitted on the z-seeded field so its ctx input is live.
    m_r = WorldModel(sc.crops, residual=True, seed=spec.seed)
    _obs_fit(m_r, sc.field, f0, a0, obs_cal, spec)
    ev_res = _eval(m_r, sc, spec)
    # detached seed: the encoder is frozen during the residual obs-fit —
    # z is a fixed ctx input, not a gradient path back into the encoder.
    f_j = seeded(sc.field, enc(obs_cal[:, 0]).detach(), m_j.latent.h_dim)
    _obs_fit(m_j, f_j, f0, a0, obs_cal, spec)

    ev_prior = _eval(sc.prior, sc, spec)
    ev_hyb = _eval(m_j, sc, spec, encoder=enc)
    per_ch = {name: {
        "prior": ev_prior[name], "jepa_probe": ev_jep[name],
        "residual": ev_res[name], "hybrid": ev_hyb[name],
        "hybrid_reduction_pct": round(
            100 * (ev_prior[name] - ev_hyb[name]) / ev_prior[name], 2)
        if ev_prior[name] > 0 else 0.0}
        for name in ch}

    z_rmse = float((rep["z_pred"][:, 1:] - rep["z_tgt"][:, 1:]).pow(2)
                   .mean().sqrt())
    out = {"latent": {"dim": k, "sigreg_final": rep["sigreg_final"],
                      "z_std_final": rep["z_std_final"],
                      "best_loss": rep["best_loss"],
                      "epochs_run": rep["epochs_run"]},
           "z_pred_rmse": round(z_rmse, 6),
           "channels": per_ch}
    if ev_hier is not None:
        out["hier"] = {"strides": list(spec.hier),
                       "channels": ev_hier,
                       "best_loss": rep_h["best_loss"]}
    return out
