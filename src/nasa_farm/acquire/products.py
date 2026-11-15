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

import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import xarray as xarray
lazy import ee

from nasa_farm.acquire.base import Bounds, Temporal
from nasa_farm.acquire.earthdata import (
    ECOSTRESS_JET,
    ECOSTRESS_LSTE,
    GLDAS_DA,
    GPM_IMERG_DAILY,
    HLS_L30_VI,
    HLS_S30_VI,
    NISAR_SME,
    SMAP_L4,
    Earthdata,
    granule_day,
)
from nasa_farm.acquire.quality import (
    CLOUD_QC,
    FMASK_QC,
    FLAG_QC,
    SMAP_QC,
    Quality,
)
from nasa_farm.acquire.support import (
    at as _at,
)
from nasa_farm.acquire.support import (
    coverage as _cov,
)
from nasa_farm.acquire.support import (
    prep as _prep,
)
from nasa_farm.acquire.support import (
    sample as _smp,
)

WORLD = (-180.0, -90.0, 180.0, 90.0)


@dataclass(frozen=True, slots=True)
class Specification:
    """One satellite product → one lake `obs` channel.

    `vars` are candidate dataset names resolved in order (products rename
    bands between versions); `radius_km` is the nominal footprint for the
    extraction kernel (0 = nearest); `qc` yields the confidence map that
    weights extraction and lands as the `{obs}_w` series.
    """

    source: Earthdata
    observed: str
    variables: tuple[str, ...]
    radius_km: float = 0.0
    quality: Quality | None = None
    workers: int = 4
    limit: int = 10_000
    bbox: tuple[float, float, float, float] = WORLD
    span: tuple[str, str] = ("2015-01-01", "2030-01-01")
    engine: str = "h5"
    """"h5" opens the granule as one HDF/netCDF dataset; "cog" treats it as
    a bundle of per-band GeoTIFFs read through `acquire.cog` (HLS-VI,
    ECOSTRESS tiled)."""
    group: str = "obs"
    """Lake group the series lands under — "obs" for retrievals,
    "forcing" for alternate forcing inputs (IMERG precip)."""
    scale: float = 1.0
    offset: float = 0.0
    months: tuple[int, ...] = ()
    """Calendar months to keep (1-12); empty = whole year. Growing-season
    products skip winter granules — no crop signal, pure cost."""
    shared: bool = False
    """True when several specs read the same granules into different obs
    vars — granule marks are namespaced per obs so one spec never marks
    the granule done for its siblings."""

    def variable(self, ds: xarray.Dataset) -> str | None:
        return next((v for v in self.variables if v in ds), None)

    def _conf(self, ds: xarray.Dataset):
        return self.quality.mask(ds) if self.quality else None

    def prep(
        self, ds: xarray.Dataset, bbox: tuple[float, float, float, float] | None = None
    ) -> tuple | None:
        """Materialized grid for this product — one read, many points.
        `bbox` crops the read to the local window (footprint margin must
        already be folded in by the caller); the QA band is sliced with
        the data var, then converted to confidence on the small window."""
        if (v := self.variable(ds)) is None:
            return None
        band = self.quality.band(ds) if self.quality else None
        values, glat, glon, raw = _prep(ds, v, ds[band] if band else None, bbox)
        conf = (
            None
            if raw is None or self.quality is None
            else np.asarray(self.quality.to_conf(raw), np.float64)
        )
        return values, glat, glon, conf

    def at(self, grid: tuple, latitude: float, longitude: float) -> tuple[float, float]:
        """(footprint-weighted value, coverage) on a `prep` grid."""
        return _at(grid, latitude, longitude, self.radius_km)

    def read(self, ds: xarray.Dataset, latitude: float, longitude: float) -> float:
        """Footprint-weighted value at the point; NaN when unsupported."""
        if (v := self.variable(ds)) is None:
            return float("nan")
        return _smp(ds, v, latitude, longitude, self.radius_km, self._conf(ds))

    def coverage(self, ds: xarray.Dataset, latitude: float, longitude: float) -> float:
        """Kernel-mass fraction over valid cells — the stored obs weight."""
        if (v := self.variable(ds)) is None:
            return 0.0
        return _cov(ds, v, latitude, longitude, self.radius_km, self._conf(ds))

    def day_of(self, ds: xarray.Dataset) -> int:
        """yyyymmdd — parsed from the granule id fetch() stashed in attrs."""
        return granule_day(ds.attrs.get("granule_id", ""))


SMAP_SM = Specification(
    SMAP_L4,
    observed="sm",
    variables=("soil_moisture",),
    radius_km=18.0,
    quality=SMAP_QC,
    workers=12,
)
NISAR_SM = Specification(
    NISAR_SME,
    observed="sme2",
    radius_km=0.2,
    quality=FLAG_QC,
    variables=("soil_moisture", "soilMoisture", "soil_moisture_pmi", "PMI", "DSG", "TSR"),
    span=("2025-06-01", "2030-01-01"),
)
ECO_ET = Specification(
    ECOSTRESS_JET,
    observed="et",
    variables=("ETdaily", "ETinst", "EVAPOTRANSPIRATION"),
    radius_km=0.5,
    quality=CLOUD_QC,
    span=("2018-07-01", "2030-01-01"),
    engine="cog",
    months=(4, 5, 6, 7, 8, 9, 10),
)
ECO_LST = Specification(
    ECOSTRESS_LSTE,
    observed="lst",
    variables=("LST", "LST_Day", "LST_Night"),
    radius_km=0.5,
    quality=CLOUD_QC,
    span=("2018-07-01", "2030-01-01"),
    engine="cog",
    offset=-273.15,
    months=(4, 5, 6, 7, 8, 9, 10),
)


def _hls(source: Earthdata) -> Specification:
    """HLS *_VI granules: NDVI precomputed at 30 m, Fmask QA, growing
    season only — the dense real optical stream the JEPA reads."""
    return Specification(
        source,
        observed="ndvi",
        variables=("NDVI",),
        radius_km=0.15,
        quality=FMASK_QC,
        engine="cog",
        months=(4, 5, 6, 7, 8, 9, 10),
    )


GPM_PR = Specification(
    GPM_IMERG_DAILY,
    observed="precip_gpm",
    variables=("precipitation", "precipitationCal"),
    radius_km=11.0,
    span=("2000-06-01", "2030-01-01"),
    group="forcing",
    scale=24.0,  # IMERG daily reports the day-mean rate (mm/h) → mm/day
    shared=True,
)
GLDAS_TWS = Specification(
    GLDAS_DA,
    observed="tws",
    variables=("TWS_tavg",),
    radius_km=28.0,
    span=("2003-02-01", "2030-01-01"),
    shared=True,
)
GLDAS_SMP = Specification(
    GLDAS_DA,
    observed="smp",
    variables=("SoilMoist_P_tavg",),
    radius_km=28.0,
    span=("2003-02-01", "2030-01-01"),
    shared=True,
)

SPECS = {
    "smap": SMAP_SM,
    "nisar": NISAR_SM,
    "ecostress-et": ECO_ET,
    "ecostress-lst": ECO_LST,
    "hls-l30": _hls(HLS_L30_VI),
    "hls-s30": _hls(HLS_S30_VI),
    "gpm": GPM_PR,
    "gldas-tws": GLDAS_TWS,
    "gldas-smp": GLDAS_SMP,
}


@dataclass(frozen=True, slots=True)
class EarthEngine:
    """One EE ImageCollection: name + the band(s) the caller wants."""

    collection: str
    bands: tuple[str, ...]
    name: str = "gee"
    requires_auth: bool = True

    def fetch(
        self, area: Bounds | None, temporal: Temporal, *, count: int = 10
    ) -> list[dict[str, Any]]:
        ee.Initialize()
        coll = (
            ee.ImageCollection(self.collection)
            .select(*self.bands)
            .filterDate(temporal.start, temporal.end)
            .limit(count)
        )
        if area is not None:
            coll = coll.filterBounds(ee.Geometry.Bounds(*area.as_tuple()))
        return coll.getInfo()["features"]


WORLDCEREAL = EarthEngine(
    "ESA/WorldCereal/2021_MARKERS_v100", ("classification",), name="worldcereal-cropmask"
)
ALPHA_EARTH = EarthEngine(
    "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
    tuple(f"A{i:02d}" for i in range(64)),
    name="alphaearth-embeddings",
)
FTW = EarthEngine(
    "projects/sat-io/open-datasets/FTW/field_boundaries", ("b1",), name="ftw-boundaries"
)


@dataclass(frozen=True, slots=True)
class OpenET:
    """OpenET v2 REST client — ensemble ET rasters/point series."""

    api_key: str | None = None
    base_url: str = "https://openet-api.org"
    name: str = "openet"
    requires_auth: bool = True
    _headers: dict[str, str] = field(default_factory=dict)

    def _get(self, path: str, parameters: dict[str, Any]) -> Any:
        import json

        q = urllib.parse.urlencode(parameters)
        request = urllib.request.Request(
            f"{self.base_url}/{path}?{q}", headers={"Authorization": self.api_key or ""}
        )
        with urllib.request.urlopen(request, timeout=30) as r:
            return json.load(r)

    def fetch(
        self,
        area: Bounds | None,
        temporal: Temporal,
        *,
        model: str = "Ensemble",
        variable: str = "ET",
        interval: str = "daily",
        units: str = "mm",
        **keywords: Any,
    ) -> Any:
        if area is None:
            raise ValueError("OpenET point queries require a bbox centroid")
        longitude, latitude = (area.west + area.east) / 2, (area.south + area.north) / 2
        return self._get(
            "raster/timeseries/point",
            {
                "lon": longitude,
                "lat": latitude,
                "model": model,
                "variable": variable,
                "start_date": temporal.start,
                "end_date": temporal.end,
                "interval": interval,
                "units": units,
                **keywords,
            },
        )
