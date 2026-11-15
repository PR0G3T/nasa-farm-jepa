"""Residual training on lake observations — differentiable rollout vs
every retrieval channel the field carries (SMAP `sm`, HLS `ndvi`,
ECOSTRESS `et`/`lst`, GLDAS `tws`…).

Loss: per-channel observation-weighted MSE between the model's matching
observation operator and the retrieval at matching dates, summed across
channels after per-channel MAD normalization so mm and °C and unitless
vegetation indices coexist without manual scale tuning. Weights come
straight from the `{var}_w` coverage shards — cloudy or uncertain
retrievals count fractionally. Optimizes the ResidualNet leaves via
`optimize.trainer.fit`; saves state_dict for serving (`--residual`).
"""

from __future__ import annotations

import datetime as dt
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from nasa_farm import CropLibrary, Field, WorldModel, initial_state
from nasa_farm.constants import DTYPE
from nasa_farm.observe import CHANNELS, sm_vol
from nasa_farm.optimize.data import dataset
from nasa_farm.optimize.data import soil as soil_of
from nasa_farm.optimize.trainer import fit_batch
from nasa_farm.plan import expand_actions
from nasa_farm.state.soil import SoilProfile

MIN_OBS = 30
CHANNEL_ORDER = ("sm", "sme2", "et", "lst", "ndvi", "fapar", "tws", "vwc")


def _sm(state) -> torch.Tensor:
    """SMAP's surface+root retrieval ≈ depth-weighted column VSM."""
    return torch.stack(
        [sm_vol(state, i) for i in range(state.sw.shape[1])]
    ).mean(0)


def _operator(name: str):
    """Obs var → per-day state observable (scalar at batch 1)."""
    if name == "sm":
        return _sm
    return CHANNELS[name]


def _loss(model: WorldModel, field: Field, forcing, channels: dict) -> torch.Tensor:
    predicted = {name: [] for name in channels}

    def observed(t, state, _fx):
        for name in channels:
            predicted[name].append(_operator(name)(state).mean())

    plan = torch.zeros(1, 1, dtype=torch.long)
    model.rollout(field, forcing, expand_actions(plan, model.crop_table), observer=observed)
    total = torch.zeros((), dtype=DTYPE)
    for name, (target, days, w, scale) in channels.items():
        r = (torch.stack(predicted[name])[days] - target) / scale
        total = total + (r * r * w).sum() / w.sum().clamp(min=1.0)
    return total / len(channels)


def _day_index(d: int, start: int) -> int:
    """yyyymmdd → day-of-simulation index, −1 outside the year."""
    return dt.date(d // 10000, d // 100 % 100, d % 100).toordinal() - start


def _samples(
    root: Path | str, fids: list[str], years: list[int], channels: tuple[str, ...]
) -> list[dict]:
    """(fid, year) samples — ≥MIN_OBS primary-channel days, every other
    available channel joined on its own dates. Per-channel `scale` is the
    retrieval MAD so residuals are dimensionless."""
    ds = dataset(root, fids, years)
    out = []
    for fid in fids:
        d = ds[fid]
        if not d["forcings"] or not d["obs"]:
            continue
        soil = soil_of(d["soil"]) or SoilProfile.preset("loam")
        primary = next((v for v in channels if v in d["obs"]), None)
        if primary is None:
            continue
        for fi, forcing in enumerate(d["forcings"]):
            start = dt.date(years[fi], 1, 1).toordinal()

            def pack(v):
                obs = d["obs"].get(v)
                if obs is None:
                    return None
                od, ov, ow = obs
                w = np.ones_like(ov) if ow is None else np.asarray(ow)
                keep = [
                    (i, _day_index(int(d_), start))
                    for i, d_ in enumerate(od)
                    if 0 <= _day_index(int(d_), start) < 365
                ]
                if not keep:
                    return None
                rows, days = zip(*keep, strict=True)
                values = np.asarray(ov)[list(rows)]
                scale = max(float(np.abs(values - np.median(values)).mean()) * 1.4826, 1e-3)
                return (
                    torch.tensor(values, dtype=DTYPE),
                    torch.tensor(days, dtype=torch.long),
                    torch.tensor(w[list(rows)], dtype=DTYPE),
                    scale,
                )

            joined = {v: p for v in channels if (p := pack(v)) is not None}
            if primary not in joined or len(joined[primary][1]) < MIN_OBS:
                continue
            out.append(
                {
                    "fid": fid,
                    "year": years[fi],
                    "soil": soil,
                    "forcing": forcing,
                    "channels": joined,
                }
            )
    return out


# --- data-parallel worker -------------------------------------------------
# `fork` inherits `_PRELOAD` copy-on-write: the whole dataset is shared with
# workers for zero bytes. One intra-op thread each — N replicas, not N×OMP.
_PRELOAD: list[dict] = []
_WORKER: dict = {}


def _init() -> None:
    global _WORKER
    torch.set_num_threads(1)
    _WORKER = {"model": WorldModel(CropLibrary.default(), residual=True), "samples": _PRELOAD}


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
    field = Field(
        state=initial_state(1, s["soil"]), soil=s["soil"], latitude=torch.zeros(1, dtype=DTYPE)
    )
    loss = _loss(model, field, s["forcing"], s["channels"])
    if not torch.isfinite(loss):
        return i, float("nan"), {}
    loss.backward()
    return (
        i,
        float(loss.detach()),
        {
            n: p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
            for n, p in model.residual.named_parameters()
            if p.requires_grad
        },
    )


def train(
    root: Path | str,
    fids: list[str],
    years: list[int],
    channels: tuple[str, ...] = ("sm",),
    holdout_years: tuple[int, ...] = (),
    holdout_fids: tuple[str, ...] = (),
    epochs: int = 60,
    rate: float = 0.02,
    workers: int | None = None,
    seed: int | None = 0,
    output: Path | str | None = None,
) -> dict:
    """Joint full-batch training: every epoch evaluates every sample's
    gradient at the shared parameters — in a fork pool when workers > 1 —
    reduces the mean, and takes one Adam step. One residual serving the
    whole lake, not a per-field curriculum. `seed` makes init and the
    whole trajectory reproducible; None leaves RNG untouched.

    `channels` picks which lake obs vars join the loss (first present is
    primary for the MIN_OBS gate). Samples whose year is in
    `holdout_years` or fid in `holdout_fids` never see an update — they
    are evaluated once at the final parameters and reported under
    `held`, the honest spatial/temporal transfer metric."""
    global _PRELOAD
    if seed is not None:
        torch.manual_seed(seed)
    crops = CropLibrary.default()
    model = WorldModel(crops, residual=True)
    parameters = {n: p for n, p in model.residual.named_parameters() if p.requires_grad}
    samples = _samples(root, fids, years, channels)
    held_out = {
        i
        for i, s in enumerate(samples)
        if s["year"] in holdout_years or s["fid"] in holdout_fids
    }
    trained = [s for i, s in enumerate(samples) if i not in held_out]
    if not trained:
        if output:
            torch.save(model.residual.state_dict(), output)
        return {"runs": 0, "loss": {}, "held": {}, "best_loss": None, "epochs": 0}
    n_workers = max(1, min(workers or os.cpu_count() or 1, len(trained)))
    _PRELOAD = samples
    pool: ProcessPoolExecutor | None = None
    if n_workers > 1:
        pool = ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=multiprocessing.get_context("fork"),
            initializer=_init,
        )

        def run(tasks):
            return list(pool.map(_grad, tasks, chunksize=1))
    else:
        _init()

        def run(tasks):
            return [_grad(t) for t in tasks]

    trained_index = [i for i in range(len(samples)) if i not in held_out]
    try:

        def evaluate() -> tuple[float, dict]:
            state = {n: p.detach().clone() for n, p in parameters.items()}
            out = run([(i, state) for i in trained_index])
            valid = [r for r in out if r[2]]
            if not valid:
                return float("nan"), {}
            return (
                sum(v for _, v, _ in valid) / len(valid),
                {n: torch.stack([g[n] for _, _, g in valid]).mean(0) for n in parameters},
            )

        result = fit_batch(evaluate, parameters, epochs=epochs, rate=rate)
        final = run(
            [
                (i, {n: p.detach().clone() for n, p in parameters.items()})
                for i in range(len(samples))
            ]
        )
    finally:
        _PRELOAD = []
        if pool is not None:
            pool.shutdown()
    scored = {i: v for i, v, _ in final}
    losses = {
        f"{samples[i]['fid']}/{samples[i]['year']}": scored[i]
        for i in range(len(samples))
        if i not in held_out
    }
    held = {
        f"{samples[i]['fid']}/{samples[i]['year']}": scored[i] for i in held_out
    }
    if output:
        torch.save(model.residual.state_dict(), output)
    return {
        "runs": len(losses),
        "loss": losses,
        "held": held,
        "best_loss": result.best_loss,
        "epochs": result.epochs_run,
        "history": result.history,
    }
