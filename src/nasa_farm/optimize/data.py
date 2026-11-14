"""Lake → training tensors — the shared contract with the live path.

Forcing series are stored under source-native names, so they rebuild
through the same `series_to_forcing`/`series_to_wet` converters the live
path uses: train and serve can never diverge. Observation series land
under canonical `observe/` channel names as (dates, values, weights)
triples — weights come from the `{var}_w` coverage shards the crawler
writes next to every footprint-sampled obs, so QA propagates into the
loss rather than a plain mask.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from nasa_farm.acquire.power import FORCING_VARS, vars_to_forcing
from nasa_farm.constants import DTYPE
from nasa_farm.lake import store
from nasa_farm.state.soil import SoilProfile
from nasa_farm.types import Forcing

OBS_VARS = ("sm", "sme2", "et", "lst", "ndvi")
PROXY_VARS = ("smg",)
"""Model-produced context series — never a training target."""


def _param_map(root: Path, fid: str) -> dict[str, dict[str, float]]:
    """Lake forcing shards → canonical {var: {yyyymmdd: v}} dict."""
    out: dict[str, dict[str, float]] = {}
    for p in FORCING_VARS:
        if (s := store.get(root, fid, "forcing", p)) is not None:
            out[p] = {str(d): float(v) for d, v in zip(*s)}
    return out


def forcings(root: Path | str, fid: str, years: list[int]) -> list[Forcing]:
    root = Path(root)
    data = _param_map(root, fid)
    return [
        vars_to_forcing(
            {p: {k: v for k, v in s.items() if k.startswith(str(y))}
             for p, s in data.items()}, y)
        for y in years
    ]


def wetness(root: Path | str, fid: str) -> np.ndarray | None:
    """Observed root-zone wetness; falls back to the POWER model proxy."""
    for v in ("sm", "smg"):
        if (s := store.get(Path(root), fid, "obs", v)) is not None:
            return s[1]
    return None


def targets(root: Path | str, fid: str
            ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]]:
    """Observed channel targets — (dates, values, weights) per channel."""
    out = {}
    for v in OBS_VARS:
        if (s := store.get(Path(root), fid, "obs", v)) is not None:
            w = store.get(Path(root), fid, "obs", f"{v}_w")
            out[v] = (s[0], s[1], None if w is None else w[1])
    return out


def emb(root: Path | str, fid: str, name: str = "aef"
        ) -> np.ndarray | None:
    """Foundation-model embedding vector stored under `emb/{name}`."""
    s = store.get(Path(root), fid, "emb", name)
    return None if s is None else s[1].reshape(-1)


def soil(d: dict | None) -> SoilProfile | None:
    """Stored profile dict → SoilProfile; None passes through."""
    if d is None:
        return None
    return SoilProfile(**{k: torch.as_tensor(v, dtype=DTYPE)
                          for k, v in d.items()})


def soil_sample(d_lo: dict | None, d_hi: dict | None, n: int, seed: int = 0
                ) -> SoilProfile | None:
    """Stored Q05/Q95 dicts → n-member uncertainty ensemble."""
    lo, hi = soil(d_lo), soil(d_hi)
    return None if lo is None or hi is None \
        else SoilProfile.sample(lo, hi, n, seed)


def dataset(root: Path | str, fids: list[str], years: list[int]
            ) -> dict[str, dict]:
    meta = lambda fid: store.meta(Path(root), fid)
    return {fid: {"forcings": forcings(root, fid, years),
                  "wet": wetness(root, fid),
                  "obs": targets(root, fid),
                  "emb": emb(root, fid),
                  "soil": (m := meta(fid)).get("soil"),
                  "soil_lo": m.get("soil_lo"),
                  "soil_hi": m.get("soil_hi")}
            for fid in fids}


def to_tensors(obs: dict[str, tuple]) -> dict[str, tuple]:
    return {v: (torch.from_numpy(d), torch.from_numpy(x),
                None if w is None else torch.from_numpy(w))
            for v, (d, x, w) in obs.items()}
