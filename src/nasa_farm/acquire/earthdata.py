"""NASA Earthdata (CMR) source via `earthaccess` — PEP 810 lazy import keeps
the package usable (and import-fast) without the heavy dep or credentials;
search needs no login, downloads do.
"""

from __future__ import annotations

lazy import earthaccess

from dataclasses import dataclass
from typing import Any

from nasa_farm.acquire.base import BBox, Temporal


@dataclass(frozen=True, slots=True)
class EarthdataSource:
    """Generic CMR granule source for one product short name."""

    short_name: str
    name: str = "earthdata"
    requires_auth: bool = True  # search is open; download needs Earthdata Login

    def fetch(self, aoi: BBox | None, temporal: Temporal, *,
              count: int = 10, cloud_cover: tuple[int, int] | None = None,
              download_dir: str | None = None) -> list[Any]:
        kw: dict[str, Any] = {
            "short_name": self.short_name,
            "temporal": (temporal.start, temporal.end),
            "count": count,
        }
        if aoi is not None:
            kw["bounding_box"] = aoi.as_tuple()
        if cloud_cover is not None:
            kw["cloud_cover"] = cloud_cover
        results = earthaccess.search_data(**kw)
        if download_dir is not None:
            earthaccess.login()
            return earthaccess.download(results, download_dir)
        return results


# Challenge-relevant products, declared once.
SMAP_L4 = EarthdataSource("SPL3SMP_E", name="smap-l4-soil-moisture")
GPM_IMERG_DAILY = EarthdataSource("GPM_3IMERGDF", name="gpm-imerg-daily")
HLS_L30 = EarthdataSource("HLSL30", name="hls-landsat-30m")
ECOSTRESS_ET = EarthdataSource("ECO_L3T_JET", name="ecostress-et")
GRACE_MASCON = EarthdataSource("TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4",
                               name="grace-fo-mascon")
