"""ISRIC SoilGrids v2 REST client — no key. Maps soil properties to SoilProfile.

Endpoint: https://rest.isric.org/soilgrids/v2.0/properties/query
Units returned are scaled integers (e.g. clay in g/kg → /10 for %).
"""

from __future__ import annotations

import httpx2 as httpx
import torch

from nasa_farm.constants import DTYPE
from nasa_farm.state.soil import SoilProfile

BASE_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
PROPERTIES = ("clay", "sand", "soc")
UNIT_SCALE = {"clay": 10.0, "sand": 10.0, "soc": 10.0}  # → percent
SOC_TO_OM = 1.724  # organic matter ≈ SOC × 1.724 (van Bemmelen)


def request_params(lat: float, lon: float, depth: str = "0-30cm") -> list[tuple[str, str]]:
    params: list[tuple[str, str]] = [("lon", str(lon)), ("lat", str(lat)),
                                     ("depth", depth), ("value", "mean")]
    params += [("property", p) for p in PROPERTIES]
    return params


def parse_soilgrids(payload: dict) -> dict[str, float]:
    """Mean values per property at requested depth (percent)."""
    out: dict[str, float] = {}
    for layer in payload["properties"]["layers"]:
        name = layer["name"]
        for d in layer["depths"]:
            mean = d["values"].get("mean")
            if mean is not None:
                out[name] = mean / UNIT_SCALE[name]
                break
    return out


class SoilGridsClient:
    name = "isric-soilgrids"
    requires_auth = False

    def __init__(self, client: httpx.Client | None = None, timeout: float = 60.0) -> None:
        self._client = client or httpx.Client(timeout=timeout)

    def fetch(self, aoi: None, temporal: None, *,
              lat: float, lon: float, depth: str = "0-30cm") -> dict[str, float]:
        resp = self._client.get(BASE_URL, params=request_params(lat, lon, depth))
        resp.raise_for_status()
        return parse_soilgrids(resp.json())

    def profile(self, lat: float, lon: float, fallback_texture: str = "loam") -> SoilProfile:
        try:
            p = self.fetch(None, None, lat=lat, lon=lon)
            sand = torch.tensor([p["sand"]], dtype=DTYPE)
            clay = torch.tensor([p["clay"]], dtype=DTYPE)
            om = torch.tensor([p.get("soc", 1.0) * SOC_TO_OM], dtype=DTYPE)
            return SoilProfile.from_texture(sand, clay, om)
        except (httpx.HTTPError, KeyError, TypeError):
            return SoilProfile.preset(fallback_texture)
