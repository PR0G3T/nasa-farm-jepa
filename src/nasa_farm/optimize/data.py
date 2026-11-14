"""Lake → training tensors — the shared contract with the live path.

Series come straight out of `lake.db` packed blobs into numpy arrays;
`vars_to_forcing` stays the single source of truth for fill/clamp so
train-on-lake and serve-on-live produce identical forcings by
construction. Reads open the lake read-only — no files are created, and
a missing `lake.db` raises `FileNotFoundError` (callers' documented
fallback to live sources).
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from nasa_farm.acquire.power import FORCING_VARS, vars_to_forcing
from nasa_farm.constants import DTYPE
from nasa_farm.lake import Lake
from nasa_farm.state.soil import SoilProfile
from nasa_farm.types import Forcing

OBS_VARS = ("sm", "sme2", "et", "lst", "ndvi")
PROXY_VARS = ("smg",)


@contextmanager
def _lk(x):
    if isinstance(x, Lake):
        yield x
    else:
        with Lake(x, ro=True) as L:
            yield L


def _param_map(L: Lake, fid: str) -> dict[str, dict[str, float]]:
    """Canonical-name → {yyyymmdd: value} as vars_to_forcing expects."""
    out: dict[str, dict[str, float]] = {}
    for p in FORCING_VARS:
        if (s := L.get(fid, "forcing", p)) is not None:
            out[p] = {str(d): float(v) for d, v in zip(*s)}
    return out


def _forcings(L: Lake, fid: str, years: list[int]) -> list[Forcing]:
    data = _param_map(L, fid)
    return [
        vars_to_forcing(
            {p: {k: v for k, v in s.items() if k.startswith(str(y))}
             for p, s in data.items()}, y)
        for y in years
    ]


def forcings(root: Lake | Path | str, fid: str,
             years: list[int]) -> list[Forcing]:
    """Rebuild per-year forcings from lake series."""
    with _lk(root) as L:
        return _forcings(L, fid, years)


def _wet(L: Lake, fid: str) -> np.ndarray | None:
    for v in ("sm", "smg"):
        if (s := L.get(fid, "obs", v)) is not None:
            return s[1]
    return None


def wetness(root: Lake | Path | str, fid: str) -> np.ndarray | None:
    """Wetness series: observed sm, else POWER proxy smg."""
    with _lk(root) as L:
        return _wet(L, fid)


def _targets(L: Lake, fid: str) -> dict:
    out = {}
    for v in OBS_VARS:
        if (s := L.get(fid, "obs", v)) is not None:
            w = L.get(fid, "obs", f"{v}_w")
            out[v] = (s[0], s[1], None if w is None else w[1])
    return out


def targets(root: Lake | Path | str, fid: str) -> dict:
    """Observed targets: {var: (dates, values, weights|None)}."""
    with _lk(root) as L:
        return _targets(L, fid)


def emb(root: Lake | Path | str, fid: str, name: str = "aef"):
    """Foundation-model embedding as a flat f32 array, or None."""
    with _lk(root) as L:
        return L.vec(fid, name)


def dataset(root: Lake | Path | str, fids: list[str],
            years: list[int]) -> dict:
    """Full training batch for `fids` — one lake open for all of them."""
    with _lk(root) as L:
        return {fid: {"forcings": _forcings(L, fid, years),
                      "wet": _wet(L, fid),
                      "obs": _targets(L, fid),
                      "emb": L.vec(fid),
                      "soil": (m := L.meta(fid)).get("soil"),
                      "soil_lo": m.get("soil_lo"),
                      "soil_hi": m.get("soil_hi")}
                for fid in fids}


def soil(d: dict) -> SoilProfile:
    """SoilGrid attrs dict → SoilProfile (shared with live crawl)."""
    return SoilProfile(
        clay=float(d["clay"]), sand=float(d["sand"]), bd=float(d["bd"]),
        soc=float(d.get("soc", 15.0)), ph=float(d.get("ph", 6.5)),
        depth=float(d.get("depth", 150.0)),
    )


def soil_sample(lo: dict, hi: dict, n: int,
                seed: int = 0) -> list[SoilProfile]:
    """n profiles sampled uniformly within the [lo, hi] uncertainty box —
    stochastic soil ensemble for risk-aware objectives."""
    rng = np.random.default_rng(seed)
    keys = ("clay", "sand", "bd", "soc", "ph")
    out = []
    for i in range(n):
        d = {"depth": lo.get("depth", 150.0)}
        for k in keys:
            a, b = float(lo.get(k, np.nan)), float(hi.get(k, np.nan))
            d[k] = float(a + (b - a) * rng.random()) if a == a and b == b \
                else float(lo.get(k, {"clay": 30, "sand": 30, "bd": 1.35,
                                      "soc": 15.0, "ph": 6.5}[k]))
        out.append(soil(d))
    return out


def to_tensors(obs: dict) -> dict[str, torch.Tensor]:
    """{var: (dates, values, w)} → {var: value tensor} — for the loss."""
    return {v: torch.as_tensor(np.asarray(x[1]), dtype=DTYPE)
            for v, x in obs.items()}
