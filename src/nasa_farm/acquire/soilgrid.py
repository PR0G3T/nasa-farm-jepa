"""ISRIC SoilGrids v2 REST client — no key.

Endpoint: https://rest.isric.org/soilgrids/v2.0/properties/query
Values are scaled integers; one request fetches the full prediction
distribution (mean / Q0.05 / Q0.50 / Q0.95). Depths must be standard
intervals: "0-30cm" exists only for `ocs` — for clay/sand/bdod/soc/phh2o
it returns `"layers": []` (checked live 2026-11-14), which used to fall
through to DEFAULTS silently. The topsoil value is therefore the
thickness-weighted mean of 0-5, 5-15 and 15-30 cm. `sample` flattens it into
the dict the lake stores per field — median texture + 90 % prediction
bounds on clay/sand/bd — and `profile` maps the same dict to a
SoilProfile, so live serving and lake attrs can never diverge.
"""

from __future__ import annotations

import logging

import httpx2 as httpx
import torch

from nasa_farm.constants import DTYPE
from nasa_farm.state.soil import SoilProfile

log = logging.getLogger(__name__)

BASE_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
DEPTHS = ("0-5cm", "5-15cm", "15-30cm")
PROPERTIES = ("clay", "sand", "bdod", "soc", "phh2o")
STATS = ("mean", "Q0.05", "Q0.5", "Q0.95")
UNIT = {"clay": 10.0, "sand": 10.0, "bdod": 100.0,
        "soc": 10.0, "phh2o": 10.0}
# clay/sand % · bd g/cm³ · soc g/kg · ph ×1
SOC_TO_OM = 1.724  # organic matter ≈ SOC × 1.724 (van Bemmelen)

DEFAULTS = {"clay": 22.0, "sand": 42.0, "bd": 1.35,
            "soc": 15.0, "ph": 6.5, "depth": 30.0}


def request_params(lat: float, lon: float,
                   depths: tuple[str, ...] | str = DEPTHS,
                   values: tuple[str, ...] = ("mean",)) -> list[tuple[str, str]]:
    if isinstance(depths, str):
        depths = (depths,)
    params: list[tuple[str, str]] = [("lon", str(lon)), ("lat", str(lat))]
    params += [("depth", d) for d in depths]
    params += [("value", v) for v in values]
    params += [("property", p) for p in PROPERTIES]
    return params


def _thickness(d: dict) -> float:
    r = d.get("range") or {}
    try:
        return float(r["bottom_depth"]) - float(r["top_depth"])
    except (KeyError, TypeError, ValueError):
        return 1.0


def parse(payload: dict) -> dict[str, dict[str, float]]:
    """{stat: {prop: value}} unit-scaled, thickness-weighted over the
    returned depths. Averaging quantiles across depths approximates the
    quantile of the 0–30 cm mean (slightly too wide)."""
    out = {s: {} for s in STATS}
    for layer in payload["properties"]["layers"]:
        name = layer["name"]
        for s in STATS:
            num = den = 0.0
            for d in layer["depths"]:
                if (v := d["values"].get(s)) is not None:
                    t = _thickness(d)
                    num, den = num + v * t, den + t
            if den > 0:
                out[s][name] = num / den / UNIT[name]
    return out


def _t(v) -> torch.Tensor:
    return torch.tensor([float(v)], dtype=DTYPE)


def profile(d: dict) -> SoilProfile:
    """Flat soil dict → SoilProfile (texture + SOC→OM pedotransfer)."""
    om = d.get("soc")
    return SoilProfile.from_texture(
        _t(d["sand"]), _t(d["clay"]),
        None if om is None else _t(om / 10.0 * SOC_TO_OM))


class SoilGridClient:
    name = "isric-soilgrid"
    requires_auth = False

    def __init__(self, client: httpx.Client | None = None,
                 timeout: float = 60.0) -> None:
        self._client = client or httpx.Client(timeout=timeout)

    def dist(self, lat: float, lon: float,
             depths: tuple[str, ...] = DEPTHS) -> dict[str, dict[str, float]]:
        """Full prediction distribution in one request."""
        resp = self._client.get(
            BASE_URL, params=request_params(lat, lon, depths, STATS))
        resp.raise_for_status()
        return parse(resp.json())

    def sample(self, lat: float, lon: float) -> dict:
        """Flat field dict: median texture + Q05/Q95 bounds (lake attrs)."""
        try:
            d = self.dist(lat, lon)
            if not d["Q0.5"] and not d["mean"]:
                log.warning("SoilGrids returned no layers at (%s, %s) → "
                            "DEFAULTS", lat, lon)
            q = lambda s, p: d[s].get(p)
            med = lambda p: q("Q0.5", p) if q("Q0.5", p) is not None \
                else d["mean"].get(p)
            out = {"clay": med("clay") or DEFAULTS["clay"],
                   "sand": med("sand") or DEFAULTS["sand"],
                   "bd": med("bdod") or DEFAULTS["bd"],
                   "soc": med("soc") or DEFAULTS["soc"],
                   "ph": med("phh2o") or DEFAULTS["ph"],
                   "depth": DEFAULTS["depth"]}
            for p, key in (("clay", "clay"), ("sand", "sand"),
                           ("bdod", "bd")):
                out[f"{key}_lo"] = q("Q0.05", p) or out[key] * 0.7
                out[f"{key}_hi"] = q("Q0.95", p) or out[key] * 1.3
            return out
        except (httpx.HTTPError, KeyError, TypeError) as e:
            log.warning("SoilGrids failed at (%s, %s): %r → DEFAULTS",
                        lat, lon, e)
            lo = dict(DEFAULTS, clay_lo=DEFAULTS["clay"] * 0.7,
                      sand_lo=DEFAULTS["sand"] * 0.7,
                      bd_lo=DEFAULTS["bd"] * 0.8,
                      clay_hi=DEFAULTS["clay"] * 1.3,
                      sand_hi=DEFAULTS["sand"] * 1.3,
                      bd_hi=DEFAULTS["bd"] * 1.2)
            return lo

    def profile(self, lat: float, lon: float) -> SoilProfile:
        """Median-profile estimate for the live serving path."""
        return profile(self.sample(lat, lon))

    def bounds(self, lat: float, lon: float
               ) -> tuple[SoilProfile, SoilProfile, SoilProfile]:
        """(Q05, Q50, Q95) profiles for ensemble sampling."""
        d = self.sample(lat, lon)
        lo = dict(d, clay=d["clay_lo"], sand=d["sand_lo"], bd=d["bd_lo"])
        hi = dict(d, clay=d["clay_hi"], sand=d["sand_hi"], bd=d["bd_hi"])
        return profile(lo), profile(d), profile(hi)

    def profiles(self, lat: float, lon: float, n: int,
                 seed: int = 0) -> SoilProfile:
        """Uncertainty ensemble: n profiles inside the 90 % PI."""
        lo, _, hi = self.bounds(lat, lon)
        return SoilProfile.sample(lo, hi, n, seed)
