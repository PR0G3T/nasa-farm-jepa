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

import urllib.request
from dataclasses import dataclass, field
from typing import Any

from nasa_farm.acquire.base import BBox, Temporal


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
