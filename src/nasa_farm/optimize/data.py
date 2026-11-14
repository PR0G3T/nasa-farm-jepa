"""Lake → training tensors — the shared contract with the live path.

Forcing series rebuild through `vars_to_forcing` — the same converter
the live path uses — so train and serve can never diverge. Observation
series land under canonical names as (dates, values, weights) triples —
weights come from the `{var}_w` coverage shards the crawler writes next
to every footprint-sampled obs, so QA propagates into the loss rather
than a plain mask.

Reads open `lake.db` read-only (no files created); a missing lake raises
`FileNotFoundError` — the callers' documented fallback to live sources.
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
"""Model-produced context series — never a training target."""


@contextmanager
def _lk(x):
    if isinstance(x, Lake):
        yield x
    else:
        with Lake(x, ro=True) as L:
            yield L


def _param_map(L: Lake, fid: str) -> dict[str, dict[str, float]]:
    """Lake forcing shards → canonical {var: {yyyymmdd: v}} dict."""
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
    with _lk(root) as L:
        return _forcings(L, fid, years)


def _wet(L: Lake, fid: str) -> np.ndarray | None:
    for v in ("sm", "smg"):
        if (s := L.get(fid, "obs", v)) is not None:
            return s[1]
    return None


def wetness(root: Lake | Path | str, fid: str) -> np.ndarray | None:
    """Observed root-zone wetness; falls back to the POWER model proxy."""
    with _lk(root) as L:
        return _wet(L, fid)


def _targets(L: Lake, fid: str) -> dict:
    out = {}
    for v in OBS_VARS:
        if (s := L.get(fid, "obs", v)) is not None:
            w = L.get(fid, "obs", f"{v}_w")
            out[v] = (s[0], s[1], None if w is None else w[1])
    return out


def targets(root: Lake | Path | str, fid: str
            ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]]:
    """Observed channel targets — (dates, values, weights) per channel."""
    with _lk(root) as L:
        return _targets(L, fid)


def emb(root: Lake | Path | str, fid: str, name: str = "aef"
        ) -> np.ndarray | None:
    """Foundation-model embedding vector, or None."""
    with _lk(root) as L:
        return L.vec(fid, name)


def soil(d: dict | None) -> SoilProfile | None:
    """Stored profile dict → SoilProfile; None passes through."""
    if d is None:
        return None
    return SoilProfile(**{k: torch.as_tensor(np.asarray(v), dtype=DTYPE)
                          for k, v in d.items()})


def soil_sample(d_lo: dict | None, d_hi: dict | None, n: int, seed: int = 0
                ) -> SoilProfile | None:
    """Stored Q05/Q95 dicts → n-member uncertainty ensemble."""
    lo, hi = soil(d_lo), soil(d_hi)
    return None if lo is None or hi is None \
        else SoilProfile.sample(lo, hi, n, seed)


def dataset(root: Lake | Path | str, fids: list[str], years: list[int]
            ) -> dict[str, dict]:
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


def to_tensors(obs: dict[str, tuple]) -> dict[str, tuple]:
    return {v: (torch.from_numpy(np.asarray(d)),
                torch.from_numpy(np.asarray(x)),
                None if w is None else torch.from_numpy(np.asarray(w)))
            for v, (d, x, w) in obs.items()}
