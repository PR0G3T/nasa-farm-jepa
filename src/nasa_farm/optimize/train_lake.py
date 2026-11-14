"""Residual training on lake observations — differentiable rollout vs
observed wetness (retrieval `sm`, or the POWER `smg` proxy when no
retrieval product covers the field).

Loss: observation-weighted MSE between model `sm_vol` root mean and the
observed channel at matching dates, over each field's forcing years.
Weights come straight from the `{var}_w` coverage shards — cloudy or
uncertain retrievals count fractionally. Optimizes the ResidualNet leaves
via `optimize.trainer.fit`; saves state_dict for serving (`--residual`).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import torch

from nasa_farm import CropLibrary, Field, WorldModel, initial_state
from nasa_farm.constants import DTYPE
from nasa_farm.observe import sm_vol
from nasa_farm.optimize.data import dataset, soil as soil_of
from nasa_farm.optimize.trainer import fit
from nasa_farm.plan import expand_actions
from nasa_farm.state.soil import SoilProfile


def _loss(model: WorldModel, field: Field, forcing, target: torch.Tensor,
          days: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    pred = []
    def obs(t, state, _fx):
        pred.append(torch.stack(
            [sm_vol(state, i) for i in range(state.sw.shape[1])]).mean())
    plan = torch.zeros(1, 1, dtype=torch.long)
    model.rollout(field, forcing,
                  expand_actions(plan, model.crop_table), observer=obs)
    r = (torch.stack(pred)[days] - target) ** 2 * w
    return r.sum() / w.sum().clamp(min=1.0)


def _day_index(d: int, start: int) -> int:
    """yyyymmdd → day-of-simulation index, −1 outside the year."""
    return (dt.date(d // 10000, d // 100 % 100, d % 100).toordinal()
            - start)


def train(root: Path | str, fids: list[str], years: list[int],
          epochs: int = 60, lr: float = 0.02,
          out: Path | str | None = None) -> dict:
    root = Path(root)
    crops = CropLibrary.default()
    model = WorldModel(crops, residual=True)
    params = {f"r{i}": p for i, p in
              enumerate(model.residual.parameters()) if p.requires_grad}
    losses = {}
    for fid in fids:
        ds = dataset(root, [fid], years)[fid]
        obs = ds["obs"].get("sm") or ds["obs"].get("smg")
        if not ds["forcings"] or obs is None:
            continue
        soil = soil_of(ds["soil"]) or SoilProfile.preset("loam")
        obs_d, obs_v, obs_w = obs
        w = np.ones_like(obs_v) if obs_w is None else np.asarray(obs_w)
        for fi, forcing in enumerate(ds["forcings"]):
            year = years[fi]
            start = dt.date(year, 1, 1).toordinal()
            keep = [(i, _day_index(int(d), start))
                    for i, d in enumerate(obs_d)
                    if 0 <= _day_index(int(d), start) < 365]
            if len(keep) < 30:
                continue
            rows, days = zip(*keep)
            target = torch.tensor(obs_v[list(rows)], dtype=DTYPE)
            wv = torch.tensor(w[list(rows)], dtype=DTYPE)
            days = torch.tensor(days, dtype=torch.long)
            field = Field(state=initial_state(1, soil), soil=soil,
                          lat=torch.tensor([0.0], dtype=DTYPE))
            res = fit(lambda: _loss(model, field, forcing, target, days,
                                    wv), params=params,
                      epochs=epochs, lr=lr, verbose=False)
            losses[f"{fid}/{year}"] = res.best_loss
    if out:
        torch.save(model.residual.state_dict(), out)
    return {"runs": len(losses), "loss": losses}


if __name__ == "__main__":
    import sys
    from nasa_farm.lake import Lake
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "lake")
    with Lake(root) as L:
        fids = L.fids()
    print(train(root, fids, list(range(2021, 2026)), out=root / "residual.pt"))
