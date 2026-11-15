"""Real-data lake experiments — NASA retrieval streams as ground truth.

loco   multi-channel residual cross-validation: leave-one-county-out
       folds (spatial transfer) + leave-one-year-out (temporal
       transfer) over the crawled lake; reports trained vs held
       weighted-MSE — the multi-ecoregion evidence the single-county
       fit could not give.
stream real-stream JEPA: the actual retrieval stream (missing days
       kept missing, forward-filled encode, masked predict loss)
       drives `embedding_fit` gap-tracking per field; the held-year
       latent-target gap measures whether the latent tracked the
       physics-retrieval discrepancy.
tws    regional water: GLDAS-DA daily TWS + GRACE-FO mascon monthly
       anomalies (the only honest scale for ~3-degree mascons) calibrate
       `model.k_tws`; reports anomaly correlation across the grid.
imerg  forcing provenance: POWER daily precip vs IMERG per field-year —
       agreement statistics plus the downstream yield deltas of
       swapping the source; forcing uncertainty quantified, not assumed.

Every runner reads `spec.lake`; management is unobserved, so rollouts
use the field's CSB crop with default actions — stated, not hidden.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path

import numpy as np
import torch

from nasa_farm import CropLibrary, Field, WorldModel, initial_state
from nasa_farm.constants import DTYPE
from nasa_farm.dynamics.latent import ObservationEncoder
from nasa_farm.experiments.run import register
from nasa_farm.experiments.specification import Specification
from nasa_farm.lake import Lake
from nasa_farm.optimize.data import dataset
from nasa_farm.optimize.data import soil as soil_of
from nasa_farm.optimize.embedding import embedding_fit
from nasa_farm.optimize.lake import train as lake_train
from nasa_farm.plan import expand_actions
from nasa_farm.state.soil import SoilProfile

LAKE_CHANNEL = {
    "sm": "sm1",
    "sme2": "sme2",
    "ndvi": "ndvi",
    "fapar": "fapar",
    "et": "et",
    "lst": "lst",
    "tws": "tws",
    "vwc": "vwc",
}
"""lake obs var → model observables channel. SMAP 0-5 cm VSM ≈ layer-1
`sm1`; the rest share names."""


def _root(specification: Specification) -> Path:
    if not specification.lake:
        raise ValueError("experiment needs --lake")
    return Path(specification.lake)


def _index(root: Path) -> dict[str, dict]:
    """fid → {county, kind, soil, lat} from lake metadata; county falls
    back to the `County N` name stem fields are minted with."""
    out = {}
    with Lake(root, readonly=True) as lake:
        for f in lake.fields():
            meta = lake.metadata(f["id"])
            out[f["id"]] = {
                "county": meta.get("county") or f["name"].rsplit(" ", 1)[0] or "unknown",
                "kind": meta.get("kind"),
                "soil": meta.get("soil"),
                "lat": f["lat"],
            }
    return out


def _years(root: Path, fid: str, specification: Specification) -> list[int]:
    """spec.years, else years the lake actually holds forcing for."""
    if specification.years:
        return list(specification.years)
    with Lake(root, readonly=True) as lake:
        series = lake.get(fid, "forcing", "tmin")
    return sorted({int(d) // 10000 for d in series[0]}) if series else []


def _field(soil_dict: dict | None, latitude: float) -> Field:
    soil = soil_of(soil_dict) or SoilProfile.preset("loam")
    return Field(initial_state(1, soil), soil, torch.full((1,), latitude, dtype=DTYPE))


def _plan(entry: dict, crops: CropLibrary) -> torch.Tensor:
    """(1,1) crop plan — the field's CSB kind when the library has it,
    maize otherwise."""
    try:
        return torch.tensor([[crops.index(entry["kind"])]], dtype=torch.long)
    except (KeyError, TypeError, ValueError):
        return torch.zeros(1, 1, dtype=torch.long)


def _correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    """Pearson r, None when either side is degenerate."""
    if len(a) < 6 or a.std() < 1e-9 or b.std() < 1e-9:
        return None
    return round(float(np.corrcoef(a, b)[0, 1]), 4)


def _frames(
    dates: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray | None,
    year: int,
    count: int,
    every: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Obs dates → (value, confidence) per cadence frame; days off the
    grid drop. Multiple obs in one frame → confidence-weighted mean."""
    start = dt.date(year, 1, 1).toordinal()
    buckets: dict[int, list] = {}
    for i, d in enumerate(dates):
        index = (
            dt.date(int(d) // 10000, int(d) // 100 % 100, int(d) % 100).toordinal() - start
        ) // every
        if 0 <= index < count:
            buckets.setdefault(index, []).append(
                (values[i], 1.0 if weights is None else float(weights[i]))
            )
    filled = np.full(count, np.nan)
    confidence = np.zeros(count)
    for index, rows in buckets.items():
        w = np.asarray([r[1] for r in rows])
        filled[index] = np.average([r[0] for r in rows], weights=w)
        confidence[index] = w.mean()
    return filled, confidence


def _fill(column: np.ndarray) -> np.ndarray:
    """Forward-fill then back-fill leading NaN — the standard async
    multi-rate stream view; the mask keeps track of what was real."""
    finite = np.isfinite(column)
    if not finite.any():
        return column
    order = np.where(finite, np.arange(len(column)), 0)
    np.maximum.accumulate(order, out=order)
    filled = column[order]
    return np.where(np.isfinite(filled), filled, column[finite][0])


def _stream(
    observed: dict, lake_vars: tuple[str, ...], year: int, every: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Field-year retrieval stream → (1, N, C) filled targets + (1, N)
    real-observation mask — the JEPA's honest view of an asynchronous
    multi-sensor cadence."""
    count = 366 // every + 1
    targets = np.full((count, len(lake_vars)), np.nan)
    seen = np.zeros(count)
    for j, variable in enumerate(lake_vars):
        if variable not in observed:
            continue
        dates, values, weights = observed[variable]
        filled, confidence = _frames(dates, values, weights, year, count, every)
        targets[:, j] = filled
        seen = np.maximum(seen, confidence)
    targets = np.stack([_fill(targets[:, j]) for j in range(len(lake_vars))], axis=-1)
    return (
        torch.from_numpy(targets).unsqueeze(0).to(DTYPE),
        torch.from_numpy(seen > 0).unsqueeze(0),
    )


@register("loco")
def loco(specification: Specification) -> dict:
    """Leave-one-county-out + leave-one-year-out residual CV."""
    root = _root(specification)
    index = _index(root)
    counties = sorted({v["county"] for v in index.values()})
    fids = list(index)
    years = list(specification.years) or sorted(
        {y for f in fids for y in _years(root, f, specification)}
    )
    channels = tuple(v for v in specification.observed_channels if v in LAKE_CHANNEL) or (
        "sm",
    )
    keywords = {
        "channels": channels,
        "epochs": specification.epochs,
        "rate": specification.rate,
        "seed": specification.seed,
    }
    folds = {}
    for county in counties:
        held = [f for f, v in index.items() if v["county"] == county]
        result = lake_train(root, fids, years, holdout_fids=held, **keywords)
        folds[county] = {
            "held_fields": len(held),
            "held_mean": round(float(np.mean(list(result["held"].values()))), 6)
            if result["held"]
            else None,
            "trained_mean": round(float(np.mean(list(result["loss"].values()))), 6)
            if result["loss"]
            else None,
            "held": result["held"],
        }
    temporal = lake_train(root, fids, years, holdout_years=(years[-1],), **keywords)
    return {
        "years": years,
        "channels": list(channels),
        "fields": len(fids),
        "counties": folds,
        "loyo": {
            "year": years[-1],
            "held_mean": round(float(np.mean(list(temporal["held"].values()))), 6)
            if temporal["held"]
            else None,
            "held": temporal["held"],
        },
    }


@register("stream")
def stream(specification: Specification) -> dict:
    """Real-stream JEPA gap-tracking on the lake's retrieval cadence."""
    root = _root(specification)
    index = _index(root)
    crops = CropLibrary.default()
    fids = list(index)[: max(1, specification.members)]
    lake_vars = tuple(v for v in ("ndvi", "sm", "et", "lst") if v in LAKE_CHANNEL)
    channels = tuple(LAKE_CHANNEL[v] for v in lake_vars)
    every = specification.cadence
    torch.manual_seed(specification.seed)
    rows = {}
    eval_year = None
    for fid in fids:
        entry = index[fid]
        years = _years(root, fid, specification)
        if not years:
            continue
        data = dataset(root, [fid], years)[fid]
        if not data["forcings"] or not data["obs"]:
            continue
        eval_year = years[-1]
        field = _field(data["soil"], entry["lat"])
        actions = expand_actions(_plan(entry, crops), crops.table)
        model = WorldModel(
            crops,
            residual=True,
            latent_dimension=specification.latent_dimension,
            seed=specification.seed,
        )
        encoder = ObservationEncoder(len(channels), specification.latent_dimension)
        fitted = None
        for y, forcing in zip(years, data["forcings"], strict=False):
            if y == eval_year:
                continue
            targets, mask = _stream(data["obs"], lake_vars, y, every)
            if mask.sum() < 4:
                continue
            fitted = embedding_fit(
                model,
                encoder,
                field,
                forcing,
                actions,
                channels=channels,
                targets=targets,
                mask=mask,
                every=every,
                ema=specification.ema,
                sig_weight=specification.sig_weight,
                epochs=specification.epochs,
                rate=specification.latent_rate,
            )
        if fitted is None or eval_year >= len(years) + years[0]:
            continue
        targets, mask = _stream(data["obs"], lake_vars, eval_year, every)
        if mask.sum() < 2:
            continue
        model_channels = channels
        z = fitted["z_pred"][0].detach().numpy()
        encoded = fitted["z_tgt"][0].detach().numpy()
        m = mask[0].numpy().astype(bool)
        rows[fid] = {
            "train_years": len(years) - 1,
            "frames": int(m.sum()),
            "latent_gap": round(float(np.abs(encoded[m] - z[m]).mean()), 6),
            "best_loss": fitted["best_loss"],
            "channels": model_channels,
        }
    return {
        "fields": rows,
        "every": every,
        "channels": list(channels),
        "lake_vars": list(lake_vars),
        "eval_year": eval_year,
        "fields_fit": len(rows),
    }


@register("tws")
def tws(specification: Specification) -> dict:
    """GRACE-FO mascon + GLDAS-DA regional water → k_tws calibration."""
    root = _root(specification)
    index = _index(root)
    crops = CropLibrary.default()

    def model_monthly(entry: dict, soil_dict, years: list[int], k_tws: float) -> dict:
        field = _field(soil_dict, entry["lat"])
        actions = expand_actions(_plan(entry, crops), crops.table)
        model = WorldModel(crops, residual=True)
        with torch.no_grad():
            model.k_tws.fill_(k_tws)
        out = {}
        for y, forcing in zip(years, dataset(root, [entry["fid"]], years)[entry["fid"]]["forcings"], strict=False):
            daily = []

            def remember(t, state, _fx):
                daily.append(float((state.tws.mean() + state.sw.sum(-1).mean()).detach()))

            model.rollout(field, forcing, actions, observer=remember)
            series = np.asarray(daily)
            months = np.array(
                [(dt.date(y, 1, 1) + dt.timedelta(i)).month for i in range(len(series))]
            )
            monthly = np.array(
                [series[months == m].mean() for m in range(1, 13)]
            )
            out[y] = monthly - monthly.mean()
        return out

    grid = (5e-4, 1e-3, 2e-3, 5e-3, 1e-2)
    scores = {k: [] for k in grid}
    correlations = {k: [] for k in grid}
    covered = 0
    for fid, entry in index.items():
        years = _years(root, fid, specification)
        data = dataset(root, [fid], years)[fid]
        mascon = data["obs"].get("mascon")
        if not years or mascon is None:
            continue
        covered += 1
        mdates, mvals, _ = mascon
        observed = {
            (int(d) // 10000, int(d) // 100 % 100): float(v)
            for d, v in zip(mdates, mvals, strict=True)
        }
        gldas = data["obs"].get("tws")
        for k in grid:
            monthly = model_monthly(
                {**entry, "fid": fid}, data["soil"], years, k
            )
            for y, series in monthly.items():
                pairs = [
                    (series[m - 1], observed[y, m])
                    for m in range(1, 13)
                    if (y, m) in observed
                ]
                if len(pairs) >= 6:
                    pm, pv = np.array(pairs).T
                    pv = pv - pv.mean()
                    scores[k].append(float(np.sqrt(np.mean((pm - pv) ** 2))))
                    if (r := _correlation(pm, pv)) is not None:
                        correlations[k].append(r)
            if gldas is not None:
                gdates, gvals, _ = gldas
                for y, series in monthly.items():
                    pairs = [
                        (series[int(d) // 100 % 100 - 1], float(v))
                        for d, v in zip(gdates, gvals, strict=True)
                        if int(d) // 10000 == y
                    ]
                    if len(pairs) >= 8:
                        pm, pv = np.array(pairs).T
                        pv = pv - pv.mean()
                        if (r := _correlation(pm, pv)) is not None:
                            correlations[k].append(r)
    best = min(grid, key=lambda k: np.nanmean(scores[k]) if scores[k] else np.inf)
    return {
        "k_tws": best,
        "k_tws_default": 2e-3,
        "fields": covered,
        "grid": {
            f"{k:g}": {
                "mascon_rmse_mm": round(float(np.nanmean(scores[k])), 3)
                if scores[k]
                else None,
                "anomaly_corr": round(float(np.nanmean(correlations[k])), 4)
                if correlations[k]
                else None,
                "n": len(scores[k]),
            }
            for k in grid
        },
    }


@register("imerg")
def imerg(specification: Specification) -> dict:
    """POWER vs IMERG precip — agreement + downstream yield deltas."""
    root = _root(specification)
    index = _index(root)
    crops = CropLibrary.default()
    rows, deltas = {}, []
    with Lake(root, readonly=True) as lake:
        for fid, entry in index.items():
            series = lake.get(fid, "forcing", "precip_gpm")
            years = _years(root, fid, specification)
            if series is None or not years:
                continue
            data = dataset(root, [fid], years)[fid]
            field = _field(data["soil"], entry["lat"])
            actions = expand_actions(_plan(entry, crops), crops.table)
            gdates, gvals = series
            for y, forcing in zip(years, data["forcings"], strict=False):
                start = dt.date(y, 1, 1).toordinal()
                imerg_daily = np.full(365, np.nan)
                for d, v in zip(gdates, gvals, strict=True):
                    i = (
                        dt.date(int(d) // 10000, int(d) // 100 % 100, int(d) % 100).toordinal()
                        - start
                    )
                    if 0 <= i < 365:
                        imerg_daily[i] = v
                power = forcing.precip.numpy().reshape(-1)[:365]
                both = np.isfinite(imerg_daily) & np.isfinite(power)
                if both.sum() < 200:
                    continue
                model = WorldModel(crops, residual=False)
                model.rollout(field, forcing, actions)
                power_yield = float(field.state.cum_yield.mean())
                alt = dataclasses.replace(
                    forcing, precip=torch.as_tensor(np.nan_to_num(imerg_daily), dtype=DTYPE).unsqueeze(0)
                )
                field2 = _field(data["soil"], entry["lat"])
                model.rollout(field2, alt, actions)
                delta = float(field2.state.cum_yield.mean()) - power_yield
                deltas.append(delta)
                rows[f"{fid}/{y}"] = {
                    "corr": _correlation(power[both], imerg_daily[both]),
                    "rmse_mm": round(float(np.sqrt(np.mean((power[both] - imerg_daily[both]) ** 2))), 4),
                    "bias_mm": round(float(np.mean(imerg_daily[both] - power[both])), 4),
                    "yield_delta_kg_ha": round(delta, 2),
                }
    corr = [r["corr"] for r in rows.values() if r["corr"] is not None]
    return {
        "n": len(rows),
        "corr_median": round(float(np.median(corr)), 4) if corr else None,
        "yield_delta_median_kg_ha": round(float(np.median(deltas)), 2) if deltas else None,
        "years": rows,
    }
