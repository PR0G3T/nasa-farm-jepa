"""Multiple shooting — long-horizon training without naive BPTT.

A full-season rollout backpropagates through 365+ operator compositions;
the loss landscape is ill-conditioned and a single bad stretch poisons the
whole gradient. Multiple shooting splits the horizon into segments with
*free boundary states*: each segment is fitted to its own observation
window and a continuity penalty stitches the pieces back into one
trajectory — the prediction-error method for hybrid dynamics.

Free variables are the editable state channels at every interior boundary,
initialized at the nominal rollout values (the penalty, not the data,
decides how far they may drift).
"""

from __future__ import annotations

import torch

from nasa_farm.assimilation.kalman import STATE_VARS, _get_var
from nasa_farm.constants import DTYPE
from nasa_farm.observe import observables
from nasa_farm.optimize.trainer import fit
from nasa_farm.state.field import Field, FieldState


def _free_state(state: FieldState, x: torch.Tensor,
                names: tuple[str, ...]) -> FieldState:
    """Write free boundary channels without in-place index_put — pure
    concat/replace so autograd flows cleanly into the boundary leaves.
    x: (1, V) aligned with `names`."""
    d = dict(zip(names, x.unbind(-1)))                    # name → (1,)
    if "sw0" in d or "swr" in d:
        root = state.sw[:, 1:]
        w = root / root.sum(-1, keepdim=True).clamp(min=1e-9)
        swr = d.get("swr", root.sum(-1))
        sw = torch.cat([d.get("sw0", state.sw[:, 0]).unsqueeze(-1),
                        (swr.unsqueeze(-1) * w).clamp(min=0.0)], dim=1)
        state = state.replace(sw=sw)
    rest = {n: v.clamp(min=0.0) for n, v in d.items()
            if n not in ("sw0", "swr")}
    return state.replace(**rest) if rest else state


def ms_bounds(days: int, segments: int) -> list[tuple[int, int]]:
    """(t0, t1] day bounds of `segments` contiguous chunks covering days."""
    seg = max(1, segments)
    cuts = torch.linspace(0, days, seg + 1).long().tolist()
    return [(cuts[i], cuts[i + 1]) for i in range(seg) if cuts[i + 1] > cuts[i]]


def shoot_fit(model, field: Field, forcing, actions,
              obs, channels: tuple[str, ...], every: int = 5,
              segments: int = 6, w_cont: float = 10.0,
              var_names: tuple[str, ...] = STATE_VARS,
              epochs: int = 100, lr: float = 1e-2) -> dict:
    """Fit (params already bound on `model` by the caller) plus free segment
    boundary states to the observation series.

    obs: ObservedSeries on the `every`-cadence grid (row j ↔ day j·every).
    Returns a report dict: fit result + per-boundary continuity gap —
    large gaps flag model misspecification, the diagnostic multiple
    shooting exists for."""
    days = forcing.days
    bounds = ms_bounds(days, segments)

    # nominal boundary states = physics rollout values
    snap: dict[int, torch.Tensor] = {}
    f = field
    with torch.no_grad():
        for t in range(days):
            f, _ = model.step(f, actions.day(t), forcing.day(t), t)
            if t in {b[0] - 1 for b in bounds[1:]}:
                snap[t + 1] = torch.stack(
                    [_get_var(f.state, n)[0] for n in var_names])

    leaves = {f"b{i}": snap[t0].clone().requires_grad_(True)
              for i, (t0, _) in enumerate(bounds[1:], start=1)}

    obs_days = [j * every for j in range(obs.values.shape[0])]

    def loss() -> torch.Tensor:
        total = torch.zeros((), dtype=DTYPE)
        ends: list[tuple[torch.Tensor, FieldState]] = []
        for i, (t0, t1) in enumerate(bounds):
            f = field
            if i > 0:
                st = _free_state(f.state,
                                 leaves[f"b{i}"].unsqueeze(0), var_names)
                f = Field(state=st, soil=f.soil, lat=f.lat,
                          elev=f.elev, ctx=f.ctx)
            for t in range(t0, t1):
                f, _ = model.step(f, actions.day(t), forcing.day(t), t)
                if t in obs_days:
                    row = t // every
                    pred = observables(f.state, channels)
                    m = torch.isfinite(obs.values[row])
                    sc = obs.values[row][m].abs().mean().clamp(min=1.0)
                    total = total + (((pred[0] - torch.nan_to_num(
                        obs.values[row])) / sc) * m).pow(2).mean()
            ends.append((f.state,))
        for i, (t0, _) in enumerate(bounds[1:], start=1):
            end = torch.stack([_get_var(ends[i - 1][0], n)[0]
                               for n in var_names])
            total = total + w_cont * (end - leaves[f"b{i}"]).pow(2).mean()
        return total

    res = fit(loss, leaves, epochs=epochs, lr=lr)
    gaps = {}
    for i, (t0, _) in enumerate(bounds[1:], start=1):
        gaps[t0] = {n: float(v) for n, v in zip(
            var_names, (leaves[f"b{i}"].detach() - snap[t0]))}
    return {"best_loss": res.best_loss, "epochs_run": res.epochs_run,
            "boundary_offset": gaps}
