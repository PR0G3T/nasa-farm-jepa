"""ISRIC SoilGrids v2 REST client — no key. Maps soil properties to SoilProfile.

Endpoint: https://rest.isric.org/soilgrids/v2.0/properties/query
Units returned are scaled integers (e.g. clay in g/kg → /10 for %).

One request fetches the full prediction distribution per property
(mean / Q0.05 / Q0.50 / Q0.95): `bounds` turns the 90 % prediction
interval into (lo, mid, hi) SoilProfiles, which `SoilProfile.sample`
expands into the parameter-uncertainty ensembles the rollout batch
consumes — free, calibrated soil spread instead of a point estimate.
"""

from __future__ import annotations

import httpx2 as httpx
import torch

from nasa_farm.constants import DTYPE
from nasa_farm.state.soil import SoilProfile

BASE_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
PROPERTIES = ("clay", "sand", "soc")
STATS = ("mean", "Q0.05", "Q0.5", "Q0.95")
UNIT_SCALE = {"clay": 10.0, "sand": 10.0, "soc": 10.0}  # → percent
SOC_TO_OM = 1.724  # organic matter ≈ SOC × 1.724 (van Bemmelen)


def request_params(lat: float, lon: float, depth: str = "0-30cm",
                   values: tuple[str, ...] = ("mean",)) -> list[tuple[str, str]]:
    params: list[tuple[str, str]] = [("lon", str(lon)), ("lat", str(lat)),
                                     ("depth", depth)]
    params += [("value", v) for v in values]
    params += [("property", p) for p in PROPERTIES]
    return params


def parse_soilgrids(payload: dict, stat: str = "mean") -> dict[str, float]:
    """One statistic per property at requested depth (percent)."""
    out: dict[str, float] = {}
    for layer in payload["properties"]["layers"]:
        name = layer["name"]
        for d in layer["depths"]:
            v = d["values"].get(stat)
            if v is not None:
                out[name] = v / UNIT_SCALE[name]
                break
    return out


def parse_dist(payload: dict) -> dict[str, dict[str, float]]:
    """All statistics per property: {prop: {mean, Q0.05, Q0.5, Q0.95}}."""
    return {s: parse_soilgrids(payload, s) for s in STATS}


def _profile(p: dict[str, float]) -> SoilProfile:
    return SoilProfile.from_texture(
        torch.tensor([p["sand"]], dtype=DTYPE),
        torch.tensor([p["clay"]], dtype=DTYPE),
        torch.tensor([p.get("soc", 1.0) * SOC_TO_OM], dtype=DTYPE))


class SoilGridsClient:
    name = "isric-soilgrids"
    requires_auth = False

    def __init__(self, client: httpx.Client | None = None, timeout: float = 60.0) -> None:
        self._client = client or httpx.Client(timeout=timeout)

    def fetch(self, aoi: None, temporal: None, *,
              lat: float, lon: float, depth: str = "0-30cm",
              values: tuple[str, ...] = ("mean",)) -> dict[str, dict[str, float]]:
        resp = self._client.get(
            BASE_URL, params=request_params(lat, lon, depth, values))
        resp.raise_for_status()
        return parse_dist(resp.json())

    def dist(self, lat: float, lon: float) -> dict[str, dict[str, float]]:
        """Full prediction distribution in one request."""
        return self.fetch(None, None, lat=lat, lon=lon, values=STATS)

    def bounds(self, lat: float, lon: float,
               fallback_texture: str = "loam"
               ) -> tuple[SoilProfile, SoilProfile, SoilProfile]:
        """(lo Q0.05, mid Q0.5, hi Q0.95) profiles for sampling."""
        try:
            d = self.dist(lat, lon)
            q = lambda s: {p: d[s].get(p, d["mean"][p]) for p in PROPERTIES}
            return _profile(q("Q0.05")), _profile(q("Q0.5")), \
                _profile(q("Q0.95"))
        except (httpx.HTTPError, KeyError, TypeError):
            p = SoilProfile.preset(fallback_texture)
            return p, p, p

    def profile(self, lat: float, lon: float, fallback_texture: str = "loam") -> SoilProfile:
        """Median profile (Q0.5) — the robust central estimate."""
        return self.bounds(lat, lon, fallback_texture)[1]

    def profiles(self, lat: float, lon: float, n: int, seed: int = 0,
                 fallback_texture: str = "loam") -> SoilProfile:
        """Uncertainty ensemble: n profiles sampled inside the 90 % PI."""
        lo, _, hi = self.bounds(lat, lon, fallback_texture)
        return SoilProfile.sample(lo, hi, n, seed)
