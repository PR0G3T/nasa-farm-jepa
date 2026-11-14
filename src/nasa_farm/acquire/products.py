"""Non-CMR product adapters — declared once, lazy deps.

- `GeeSource`: Google Earth Engine image collections (WorldCereal,
  AlphaEarth embeddings, Fields of The World) — needs `earthengine-api`
  and an authenticated EE session at fetch time.
- `OpenETClient`: OpenET v2 REST API — needs an API key (free tier).
- `StacSource`: Microsoft Planetary Computer STAC (no key; signed URLs).
All keep `DataSource`'s contract: fetch returns lightweight descriptors
(file URIs or EE export tasks), not bulky arrays — callers decide what to
materialize, keeping memory minimal.
"""

from __future__ import annotations

lazy import ee

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any

import xarray as xr

from nasa_farm.acquire.base import BBox, Temporal
from nasa_farm.acquire.earthdata import (
    ECOSTRESS_ET, ECOSTRESS_LST, NISAR_SME, SMAP_L3, SMAP_L4,
    EarthdataSource, granule_day,
)
from nasa_farm.acquire.qc import CLOUD_QC, FLAG_QC, SMAP_QC, QcSpec
from nasa_farm.acquire.support import coverage as _cov, sample as _smp

WORLD = (-180.0, -90.0, 180.0, 90.0)


@dataclass(frozen=True, slots=True)
class Spec:
    """One satellite product → one lake `obs` channel.

    `vars` are candidate dataset names resolved in order (products rename
    bands between versions); `radius_km` is the nominal footprint for the
    extraction kernel (0 = nearest); `qc` yields the confidence map that
    weights extraction and lands as the `{obs}_w` series.

    How the granule is read: `reader` "h5" (HDF5/netCDF via h5netcdf) or
    "cog" (one GeoTIFF per variable, ECOSTRESS tiles); `assets` = file
    suffixes to fetch (skips XML, browse images, unused bands);
    `root_coords` = lat/lon datasets in the HDF5 root when data sit in a
    group (SMAP L4); `pick` = regex a granule id must match (one of the
    eight 3-hourly SMAP L4 files per day); `stream` = read HDF5 in place
    over HTTPS (`earthaccess.open`) — SMAP files are 146–600 MB but only
    a few variables are read; `offset` is added to values (K → °C).
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
    reader: str = "h5"
    assets: tuple[str, ...] = (".h5", ".nc", ".nc4")
    root_coords: tuple[str, str] | None = None
    pick: str | None = None
    stream: bool = False
    offset: float = 0.0

    def var(self, ds: xr.Dataset) -> str | None:
        return next((v for v in self.vars if v in ds), None)

    def _conf(self, ds: xr.Dataset):
        return self.qc.mask(ds) if self.qc else None

    def read(self, ds: xr.Dataset, lat: float, lon: float) -> float:
        """Footprint-weighted value at the point; NaN when unsupported."""
        if (v := self.var(ds)) is None:
            return float("nan")
        return _smp(ds, v, lat, lon, self.radius_km, self._conf(ds)) \
            + self.offset

    def coverage(self, ds: xr.Dataset, lat: float, lon: float) -> float:
        """Kernel-mass fraction over valid cells — the stored obs weight."""
        if (v := self.var(ds)) is None:
            return 0.0
        return _cov(ds, v, lat, lon, self.radius_km, self._conf(ds))

    def day_of(self, ds: xr.Dataset) -> int:
        """yyyymmdd — parsed from the granule id fetch() stashed in attrs."""
        return granule_day(ds.attrs.get("granule_id", ""))


SMAP_SM = Spec(SMAP_L3, obs="sm", vars=("soil_moisture",),
               radius_km=18.0, qc=SMAP_QC, assets=(".h5",), stream=True)
"""SPL3SMP_E surface (0–5 cm) retrieval, AM pass — kept as `sm` for the
existing training loop."""
SMAP_RZ = Spec(SMAP_L4, obs="sm_rz", vars=("sm_rootzone",), radius_km=18.0,
               assets=(".h5",), root_coords=("cell_lat", "cell_lon"),
               pick=r"T133000", stream=True)
"""SPL4SMGP root zone (0–100 cm), 12:00–15:00 UTC mean — 1 of 8 files/day."""
NISAR_SM = Spec(NISAR_SME, obs="sme2", radius_km=0.2, qc=FLAG_QC,
                vars=("soil_moisture", "soilMoisture", "soil_moisture_pmi",
                      "PMI", "DSG", "TSR"),
                span=("2025-06-01", "2030-01-01"))
ECO_ET = Spec(ECOSTRESS_ET, obs="et", vars=("ETdaily",), radius_km=0.5,
              reader="cog", assets=("_ETdaily.tif",),
              span=("2018-07-01", "2030-01-01"))
"""ECO_L3T_JET daily ET (mm/day). ETdaily is not in SEB (Rn, SWin)."""
ECO_LST = Spec(ECOSTRESS_LST, obs="lst", vars=("LST",), radius_km=0.5,
               qc=CLOUD_QC, reader="cog", assets=("_LST.tif", "_cloud.tif"),
               offset=-273.15, span=("2018-07-01", "2030-01-01"))
"""ECO_L2T_LSTE, K → °C (observe.lst is °C). LST is not in MET (Ta, RH)."""

SPECS = {"smap": SMAP_SM, "smap-l4": SMAP_RZ, "nisar": NISAR_SM,
         "ecostress-et": ECO_ET, "ecostress-lst": ECO_LST}
ALIASES = {"ecostress-seb": "ecostress-et", "ecostress-met": "ecostress-lst"}
"""Old CLI names — they pointed at products without ETdaily / LST."""


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
    """OpenET v2 REST client — ensemble ET point series.

    The API takes POST + JSON body (docs: openet.gitbook.io, raster
    endpoints); a GET with query parameters is not part of the API.
    Key: `api_key` or env `OPENET_API_KEY`.
    """

    api_key: str | None = None
    base_url: str = "https://openet-api.org"
    name: str = "openet"
    requires_auth: bool = True

    def _post(self, path: str, body: dict[str, Any]) -> Any:
        key = self.api_key or os.environ.get("OPENET_API_KEY")
        if not key:
            raise RuntimeError("OpenET needs an API key (OPENET_API_KEY)")
        req = urllib.request.Request(
            f"{self.base_url}/{path}", data=json.dumps(body).encode(),
            method="POST", headers={"Authorization": key,
                                    "Content-Type": "application/json",
                                    "accept": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)

    def fetch(self, aoi: BBox | None, temporal: Temporal, *,
              model: str = "Ensemble", variable: str = "ET",
              interval: str = "daily", units: str = "mm",
              reference_et: str = "gridMET", **kw: Any) -> Any:
        if aoi is None:
            raise ValueError("OpenET point queries require a bbox centroid")
        lon, lat = (aoi.west + aoi.east) / 2, (aoi.south + aoi.north) / 2
        return self._post("raster/timeseries/point", {
            "date_range": [temporal.start, temporal.end],
            "interval": interval, "geometry": [lon, lat], "model": model,
            "variable": variable, "reference_et": reference_et,
            "units": units, "file_format": "JSON", **kw})


@dataclass(frozen=True, slots=True)
class StacSource:
    """One Planetary Computer collection → signed STAC items (descriptors;
    the caller reads only the COG windows it needs)."""

    collection: str
    name: str = "planetary-computer"
    requires_auth: bool = False
    url: str = "https://planetarycomputer.microsoft.com/api/stac/v1"

    def fetch(self, aoi: BBox | None, temporal: Temporal, *,
              count: int = 500, query: dict | None = None) -> list[Any]:
        import planetary_computer
        import pystac_client
        cat = pystac_client.Client.open(
            self.url, modifier=planetary_computer.sign_inplace)
        search = cat.search(
            collections=[self.collection],
            bbox=None if aoi is None else aoi.as_tuple(),
            datetime=f"{temporal.start}/{temporal.end}",
            query=query, max_items=count)
        return list(search.items())


# Checked live over Ames 2026-11-14 (see claude/download-access.md).
S1_RTC = StacSource("sentinel-1-rtc", name="pc-sentinel-1-rtc")
S2_L2A = StacSource("sentinel-2-l2a", name="pc-sentinel-2-l2a")  # −1000 DN offset ≥ baseline 04.00
HLS2_L30 = StacSource("hls2-l30", name="pc-hls-l30")   # 2013–
HLS2_S30 = StacSource("hls2-s30", name="pc-hls-s30")   # 2020– only on PC
DEM_3DEP = StacSource("3dep-seamless", name="pc-3dep")
GNATSGO = StacSource("gnatsgo-rasters", name="pc-gnatsgo")  # 2020 vintage, NCCPI
MODIS_NBAR_PC = StacSource("modis-43A4-061", name="pc-modis-nbar")
