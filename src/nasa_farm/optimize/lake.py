"""Residual training on lake observations — differentiable rollout vs
observed wetness (retrieval `sm`).

Loss: observation-weighted MSE between model `sm_vol` root mean and the
observed channel at matching dates, over each field's forcing years.
Weights come straight from the `{var}_w` coverage shards — cloudy or
uncertain retrievals count fractionally. Optimizes the ResidualNet leaves
via `optimize.trainer.fit`; saves state_dict for serving (`--residual`).
"""

from __future__ import annotations

import datetime as dt
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as numpy
import torch

from nasa_farm import CropLibrary, Field, WorldModel, initial_state
from nasa_farm.constants import DTYPE
from nasa_farm.observe import sm_vol
from nasa_farm.optimize.data import dataset, soil as soil_of
from nasa_farm.optimize.trainer import fit_batch
from nasa_farm.plan import expand_actions
from nasa_farm.state.soil import SoilProfile

MIN_OBS = 30


def _loss(model: WorldModel, field: Field, forcing, target: torch.Tensor,
          days: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    predicted = []
    def observed(t, state, _fx):
        predicted.append(torch.stack(
            [sm_vol(state, i) for i in range(state.sw.shape[1])]).mean())
    plan = torch.zeros(1, 1, dtype=torch.long)
    model.rollout(field, forcing,
                  expand_actions(plan, model.crop_table), observer=observed)
    r = (torch.stack(predicted)[days] - target) ** 2 * w
    return r.sum() / w.sum().clamp(min=1.0)


def _day_index(d: int, start: int) -> int:
    """yyyymmdd → day-of-simulation index, −1 outside the year."""
    return (dt.date(d // 10000, d // 100 % 100, d % 100).toordinal()
            - start)


def _samples(root: Path | str, fids: list[str], years: list[int]) -> list[dict]:
    """(fid, year) training samples — each with ≥ MIN_OBS observed days."""
    ds = dataset(root, fids, years)
    out = []
    for fid in fids:
        d = ds[fid]
        observed = d["obs"].get("sm")
        if not d["forcings"] or observed is None:
            continue
        soil = soil_of(d["soil"]) or SoilProfile.preset("loam")
        obs_d, obs_v, obs_w = observed
        w = numpy.ones_like(obs_v) if obs_w is None else numpy.asarray(obs_w)
        for fi, forcing in enumerate(d["forcings"]):
            start = dt.date(years[fi], 1, 1).toordinal()
            keep = [(i, _day_index(int(d_), start))
                    for i, d_ in enumerate(obs_d)
                    if 0 <= _day_index(int(d_), start) < 365]
            if len(keep) < MIN_OBS:
                continue
            rows, days = zip(*keep)
            out.append({"fid": fid, "year": years[fi], "soil": soil,
                        "forcing": forcing,
                        "target": torch.tensor(obs_v[list(rows)], dtype=DTYPE),
                        "w": torch.tensor(w[list(rows)], dtype=DTYPE),
                        "days": torch.tensor(days, dtype=torch.long)})
    return out


# --- data-parallel worker -------------------------------------------------
# `fork` inherits `_PRELOAD` copy-on-write: the whole dataset is shared with
# workers for zero bytes. One intra-op thread each — N replicas, not N×OMP.
_PRELOAD: list[dict] = []
_WORKER: dict = {}


def _init() -> None:
    global _WORKER
    torch.set_num_threads(1)
    _WORKER = {"model": WorldModel(CropLibrary.default(), residual=True),
               "samples": _PRELOAD}


def _grad(payload: tuple[int, dict]) -> tuple[int, float, dict]:
    """One sample: load shared params, rollout, return (loss, grads)."""
    i, state = payload
    model = _WORKER["model"]
    with torch.no_grad():
        for name, p in model.residual.named_parameters():
            if name in state:
                p.copy_(state[name])
        model.residual.zero_grad(set_to_none=True)
    s = _WORKER["samples"][i]
    field = Field(state=initial_state(1, s["soil"]), soil=s["soil"],
                  latitude=torch.zeros(1, dtype=DTYPE))
    loss = _loss(model, field, s["forcing"], s["target"], s["days"], s["w"])
    if not torch.isfinite(loss):
        return i, float("nan"), {}
    loss.backward()
    return i, float(loss.detach()), {
        n: p.grad.detach().clone() if p.grad is not None
        else torch.zeros_like(p)
        for n, p in model.residual.named_parameters() if p.requires_grad}


def train(root: Path | str, fids: list[str], years: list[int],
          epochs: int = 60, rate: float = 0.02, workers: int | None = None,
          output: Path | str | None = None) -> dict:
    """Joint full-batch training: every epoch evaluates every sample's
    gradient at the shared parameters — in a fork pool when workers > 1 —
    reduces the mean, and takes one Adam step. One residual serving the
    whole lake, not a per-field curriculum."""
    global _PRELOAD
    crops = CropLibrary.default()
    model = WorldModel(crops, residual=True)
    parameters = {n: p for n, p in model.residual.named_parameters()
                  if p.requires_grad}
    samples = _samples(root, fids, years)
    if not samples:
        if output:
            torch.save(model.residual.state_dict(), output)
        return {"runs": 0, "loss": {}, "best_loss": None, "epochs": 0}
    n_workers = max(1, min(workers or os.cpu_count() or 1, len(samples)))
    _PRELOAD = samples
    if n_workers > 1:
        ctx = multiprocessing.get_context("fork")
        pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                                   initializer=_init)
        run = lambda tasks: list(pool.map(_grad, tasks, chunksize=1))
    else:
        _init()
        run = lambda tasks: [_grad(t) for t in tasks]
    try:
        def evaluate() -> tuple[float, dict]:
            state = {n: p.detach().clone() for n, p in parameters.items()}
            out = run([(i, state) for i in range(len(samples))])
            valid = [r for r in out if r[2]]
            if not valid:
                return float("nan"), {}
            return (sum(v for _, v, _ in valid) / len(valid),
                    {n: torch.stack([g[n] for _, _, g in valid]).mean(0)
                     for n in parameters})
        result = fit_batch(evaluate, parameters, epochs=epochs, rate=rate)
        final = run([(i, {n: p.detach().clone() for n, p in
                          parameters.items()}) for i in range(len(samples))])
    finally:
        _PRELOAD = []
        if n_workers > 1:
            pool.shutdown()
    losses = {f"{s['fid']}/{s['year']}": v
              for s, (i, v, _) in zip(samples, final)}
    if output:
        torch.save(model.residual.state_dict(), output)
    return {"runs": len(losses), "loss": losses,
            "best_loss": result.best_loss, "epochs": result.epochs_run}


if __name__ == "__main__":
    import sys
    from nasa_farm.lake import Lake
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "lake")
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    with Lake(root) as L:
        fids = L.fids()
    print(train(root, fids, list(range(2021, 2026)), workers=workers,
                output=root / "residual.pt"))
