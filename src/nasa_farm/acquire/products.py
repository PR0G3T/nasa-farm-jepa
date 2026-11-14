"""Non-CMR product adapters — declared once, lazy deps.

- `GeeSource`: Google Earth Engine image collections (WorldCereal,
  AlphaEarth embeddings, Fields of The World) — needs `earthengine-api`
  and an authenticated EE session at fetch time.
- `OpenETClient`: OpenET v2 REST API — needs an API key (free tier).
Both keep `DataSource`'s contract: fetch returns lightweight descriptors
(file URIs or EE export tasks), not bulky arrays — callers decide what to
materialize, keeping memory minimal.
"""

from __future__ import annotations

lazy import ee

import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import xarray as xr

from nasa_farm.acquire.base import BBox, Temporal
from nasa_farm.acquire.earthdata import (
    ECOSTRESS_MET, ECOSTRESS_SEB, NISAR_SME, SMAP_L4, EarthdataSource,
    granule_day,
)
from nasa_farm.acquire.qc import FLAG_QC, SMAP_QC, QcSpec
from nasa_farm.acquire.support import coverage as _cov, sample as _smp

WORLD = (-180.0, -90.0, 180.0, 90.0)


@dataclass(frozen=True, slots=True)
class Spec:
    """One satellite product → one lake `obs` channel.

    `vars` are candidate dataset names resolved in order (products rename
    bands between versions); `radius_km` is the nominal footprint for the
    extraction kernel (0 = nearest); `qc` yields the confidence map that
    weights extraction and lands as the `{obs}_w` series.
    """

    source: EarthdataSource
    obs: str
    vars: tuple[str, ...]
    radius_km: float = 0.0
    qc: QcSpec | None = None
    workers: int = 4
    limit: int = 10_000
    bbox: tuple[float, float, float, float] = WORLD
    span: tuple[str, str] = ("2015-01-01", "2030-01-01")

    def var(self, ds: xr.Dataset) -> str | None:
        return next((v for v in self.vars if v in ds), None)

    def _conf(self, ds: xr.Dataset):
        return self.qc.mask(ds) if self.qc else None

    def read(self, ds: xr.Dataset, lat: float, lon: float) -> float:
        """Footprint-weighted value at the point; NaN when unsupported."""
        if (v := self.var(ds)) is None:
            return float("nan")
        return _smp(ds, v, lat, lon, self.radius_km, self._conf(ds))

    def coverage(self, ds: xr.Dataset, lat: float, lon: float) -> float:
        """Kernel-mass fraction over valid cells — the stored obs weight."""
        if (v := self.var(ds)) is None:
            return 0.0
        return _cov(ds, v, lat, lon, self.radius_km, self._conf(ds))

    def day_of(self, ds: xr.Dataset) -> int:
        """yyyymmdd — parsed from the granule id fetch() stashed in attrs."""
        return granule_day(ds.attrs.get("granule_id", ""))


SMAP_SM = Spec(SMAP_L4, obs="sm", vars=("soil_moisture",),
               radius_km=18.0, qc=SMAP_QC)
NISAR_SM = Spec(NISAR_SME, obs="sme2", radius_km=0.2, qc=FLAG_QC,
                vars=("soil_moisture", "soilMoisture", "soil_moisture_pmi",
                      "PMI", "DSG", "TSR"),
                span=("2025-06-01", "2030-01-01"))
ECO_ET = Spec(ECOSTRESS_SEB, obs="et", vars=("ETdaily", "EVAPOTRANSPIRATION"),
              radius_km=0.5, qc=FLAG_QC, span=("2018-07-01", "2030-01-01"))
ECO_LST = Spec(ECOSTRESS_MET, obs="lst", vars=("LST", "LST_Day", "LST_Night"),
               radius_km=0.5, qc=FLAG_QC, span=("2018-07-01", "2030-01-01"))

SPECS = {"smap": SMAP_SM, "nisar": NISAR_SM,
         "ecostress-seb": ECO_ET, "ecostress-met": ECO_LST}


@dataclass(frozen=True, slots=True)
class GeeSource:
    """One EE ImageCollection: name + the band(s) the caller wants."""

    collection: str
    bands: tuple[str, ...]
    name: str = "gee"
    requires_auth: bool = True

    def fetch(self, aoi: BBox | None, temporal: Temporal, *,
              count: int = 10) -> list[dict[str, Any]]:
        ee.Initialize()
        coll = ee.ImageCollection(self.collection).select(*self.bands) \
            .filterDate(temporal.start, temporal.end).limit(count)
        if aoi is not None:
            coll = coll.filterBounds(ee.Geometry.BBox(*aoi.as_tuple()))
        return coll.getInfo()["features"]


WORLDCEREAL = GeeSource("ESA/WorldCereal/2021_MARKERS_v100",
                        ("classification",), name="worldcereal-cropmask")
ALPHA_EARTH = GeeSource("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
                        tuple(f"A{i:02d}" for i in range(64)),
                        name="alphaearth-embeddings")
FTW = GeeSource("projects/sat-io/open-datasets/FTW/field_boundaries",
                ("b1",), name="ftw-boundaries")


@dataclass(frozen=True, slots=True)
class OpenETClient:
    """OpenET v2 REST client — ensemble ET rasters/point series."""

    api_key: str | None = None
    base_url: str = "https://openet-api.org"
    name: str = "openet"
    requires_auth: bool = True
    _headers: dict[str, str] = field(default_factory=dict)

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        import json
        q = urllib.parse.urlencode(params)
        req = urllib.request.Request(f"{self.base_url}/{path}?{q}",
                                     headers={"Authorization": self.api_key or ""})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)

    def fetch(self, aoi: BBox | None, temporal: Temporal, *,
              model: str = "Ensemble", variable: str = "ET",
              interval: str = "daily", units: str = "mm",
              **kw: Any) -> Any:
        if aoi is None:
            raise ValueError("OpenET point queries require a bbox centroid")
        lon, lat = (aoi.west + aoi.east) / 2, (aoi.south + aoi.north) / 2
        return self._get("raster/timeseries/point", {
            "lon": lon, "lat": lat, "model": model, "variable": variable,
            "start_date": temporal.start, "end_date": temporal.end,
            "interval": interval, "units": units, **kw})
