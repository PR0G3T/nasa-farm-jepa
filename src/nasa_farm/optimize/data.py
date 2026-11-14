"""Lake → training tensors — the shared contract with the live path.

Forcing series are stored under source-native names, so they rebuild
through the same `series_to_forcing`/`series_to_wet` converters the live
path uses: train and serve can never diverge. Observation series land
under canonical `observe/` channel names as (dates, values) pairs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from nasa_farm.acquire.power import FORCING_VARS, vars_to_forcing
from nasa_farm.lake import store
from nasa_farm.types import Forcing

OBS_VARS = ("sm", "et", "lst", "ndvi")


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
    if (s := store.get(Path(root), fid, "obs", "sm")) is not None:
        return s[1]
    return None


def targets(root: Path | str, fid: str
            ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Observed channel targets — what the model must reproduce."""
    return {v: s for v in OBS_VARS
            if (s := store.get(Path(root), fid, "obs", v)) is not None}


def dataset(root: Path | str, fids: list[str], years: list[int]
            ) -> dict[str, dict]:
    return {fid: {"forcings": forcings(root, fid, years),
                  "wet": wetness(root, fid),
                  "obs": targets(root, fid),
                  "soil": store.meta(Path(root), fid).get("soil")}
            for fid in fids}


def to_tensors(obs: dict[str, tuple[np.ndarray, np.ndarray]]
               ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    return {v: (torch.from_numpy(d), torch.from_numpy(x))
            for v, (d, x) in obs.items()}
