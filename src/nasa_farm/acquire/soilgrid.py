"""ISRIC SoilGrids v2 REST client — no key.

Endpoint: https://rest.isric.org/soilgrids/v2.0/properties/query
Values are scaled integers; one request fetches the full prediction
distribution (mean / Q0.05 / Q0.50 / Q0.95). `sample` flattens it into
the dict the lake stores per field — median texture + 90 % prediction
bounds on clay/sand/bd — and `profile` maps the same dict to a
SoilProfile, so live serving and lake attrs can never diverge.
"""

from __future__ import annotations

import httpx2 as httpx
import torch

from nasa_farm.constants import DTYPE
from nasa_farm.state.soil import SoilProfile

BASE_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
PROPERTIES = ("clay", "sand", "bdod", "soc", "phh2o")
STATS = ("mean", "Q0.05", "Q0.5", "Q0.95")
UNIT = {"clay": 10.0, "sand": 10.0, "bdod": 100.0,
        "soc": 10.0, "phh2o": 10.0}
# clay/sand % · bd g/cm³ · soc g/kg · ph ×1
SOC_TO_OM = 1.724  # organic matter ≈ SOC × 1.724 (van Bemmelen)

DEFAULTS = {"clay": 22.0, "sand": 42.0, "bd": 1.35,
            "soc": 15.0, "ph": 6.5, "depth": 30.0}


def request_params(latitude: float, longitude: float, depth: str = "0-30cm",
                   values: tuple[str, ...] = ("mean",)) -> list[tuple[str, str]]:
    parameters: list[tuple[str, str]] = [("lon", str(longitude)), ("lat", str(latitude)),
                                     ("depth", depth)]
    parameters += [("value", v) for v in values]
    parameters += [("property", p) for p in PROPERTIES]
    return parameters


def parse(payload: dict) -> dict[str, dict[str, float]]:
    """{prop: {mean, Q0.05, Q0.5, Q0.95}} at requested depth, unit-scaled."""
    output = {s: {} for s in STATS}
    for layer in payload["properties"]["layers"]:
        name = layer["name"]
        for d in layer["depths"]:
            for s in STATS:
                if (v := d["values"].get(s)) is not None:
                    output[s][name] = v / UNIT[name]
            break
    return output


def _t(v) -> torch.Tensor:
    return torch.tensor([float(v)], dtype=DTYPE)


def profile(d: dict) -> SoilProfile:
    """Flat soil dict → SoilProfile (texture + SOC→OM pedotransfer)."""
    organic = d.get("soc")
    return SoilProfile.from_texture(
        _t(d["sand"]), _t(d["clay"]),
        None if organic is None else _t(organic / 10.0 * SOC_TO_OM))


class SoilGridClient:
    name = "isric-soilgrid"
    requires_auth = False

    def __init__(self, client: httpx.Client | None = None,
                 timeout: float = 60.0) -> None:
        self._client = client or httpx.Client(timeout=timeout)

    def dist(self, latitude: float, longitude: float,
             depth: str = "0-30cm") -> dict[str, dict[str, float]]:
        """Full prediction distribution in one request."""
        response = self._client.get(
            BASE_URL, parameters=request_params(latitude, longitude, depth, STATS))
        response.raise_for_status()
        return parse(response.json())

    def sample(self, latitude: float, longitude: float) -> dict:
        """Flat field dict: median texture + Q05/Q95 bounds (lake attrs)."""
        try:
            d = self.dist(latitude, longitude)
            q = lambda s, p: d[s].get(p)
            med = lambda p: q("Q0.5", p) if q("Q0.5", p) is not None \
                else d["mean"].get(p)
            output = {"clay": med("clay") or DEFAULTS["clay"],
                   "sand": med("sand") or DEFAULTS["sand"],
                   "bd": med("bdod") or DEFAULTS["bd"],
                   "soc": med("soc") or DEFAULTS["soc"],
                   "ph": med("phh2o") or DEFAULTS["ph"],
                   "depth": DEFAULTS["depth"]}
            for p, key in (("clay", "clay"), ("sand", "sand"),
                           ("bdod", "bd")):
                output[f"{key}_lo"] = q("Q0.05", p) or output[key] * 0.7
                output[f"{key}_hi"] = q("Q0.95", p) or output[key] * 1.3
            return output
        except (httpx.HTTPError, KeyError, TypeError):
            low = dict(DEFAULTS, clay_lo=DEFAULTS["clay"] * 0.7,
                      sand_lo=DEFAULTS["sand"] * 0.7,
                      bd_lo=DEFAULTS["bd"] * 0.8,
                      clay_hi=DEFAULTS["clay"] * 1.3,
                      sand_hi=DEFAULTS["sand"] * 1.3,
                      bd_hi=DEFAULTS["bd"] * 1.2)
            return low

    def profile(self, latitude: float, longitude: float) -> SoilProfile:
        """Median-profile estimate for the live serving path."""
        return profile(self.sample(latitude, longitude))

    def bounds(self, latitude: float, longitude: float
               ) -> tuple[SoilProfile, SoilProfile, SoilProfile]:
        """(Q05, Q50, Q95) profiles for ensemble sampling."""
        d = self.sample(latitude, longitude)
        low = dict(d, clay=d["clay_lo"], sand=d["sand_lo"], bd=d["bd_lo"])
        high = dict(d, clay=d["clay_hi"], sand=d["sand_hi"], bd=d["bd_hi"])
        return profile(low), profile(d), profile(high)

    def profiles(self, latitude: float, longitude: float, n: int,
                 seed: int = 0) -> SoilProfile:
        """Uncertainty ensemble: n profiles inside the 90 % PI."""
        low, _, high = self.bounds(latitude, longitude)
        return SoilProfile.sample(low, high, n, seed)
