"""NASA Earthdata (CMR) source via `earthaccess` — PEP 810 lazy import keeps
the package usable (and import-fast) without the heavy dep or credentials;
search needs no login, downloads do.
"""

from __future__ import annotations

lazy import earthaccess

import re
from dataclasses import dataclass
from typing import Any

from nasa_farm.acquire.base import BBox, Temporal


@dataclass(frozen=True, slots=True)
class Granule:
    """One CMR granule — id for dedup, raw result for open/download."""

    id: str
    raw: Any


def granule_id(res: Any) -> str:
    """GranuleUR / ProducerGranuleId from a CMR search result."""
    for k in ("GranuleUR", "ProducerGranuleId"):
        try:
            return str(res["umm"][k])
        except (KeyError, TypeError):
            continue
    return str(res)


_DATE = re.compile(r"(?<!\d)(?:19|20)\d{6}(?!\d)")
_DOY = re.compile(r"(?<!\d)((?:19|20)\d{2})(\d{3})(?!\d)")


def granule_day(gid: str) -> int:
    """yyyymmdd int from a granule id — ISO date or yyyyddd day-of-year."""
    if m := _DATE.search(gid):
        return int(m.group())
    import datetime as dt
    m = _DOY.search(gid)
    d = dt.date(int(m[1]), 1, 1) + dt.timedelta(int(m[2]) - 1)
    return int(d.strftime("%Y%m%d"))


@dataclass(frozen=True, slots=True)
class EarthdataSource:
    """Generic CMR granule source for one product short name.

    `group` is the HDF5 group holding data vars (None = root); `span` is
    the product's native temporal window passed to open calls.
    """

    short_name: str
    name: str = "earthdata"
    requires_auth: bool = True  # search is open; download needs Earthdata Login
    group: str | None = None

    def search(self, aoi: BBox | None, temporal: Temporal,
               limit: int = 10_000) -> list[Granule]:
        """CMR query → lightweight granule descriptors (no download)."""
        kw: dict[str, Any] = {
            "short_name": self.short_name,
            "temporal": (temporal.start, temporal.end),
            "count": limit,
        }
        if aoi is not None:
            kw["bounding_box"] = aoi.as_tuple()
        return [Granule(granule_id(r), r)
                for r in earthaccess.search_data(**kw)]

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
            earthaccess.login(strategy="environment")
            return earthaccess.download(results, download_dir)
        return results


# Challenge-relevant products, declared once.
SMAP_L4 = EarthdataSource("SPL3SMP_E", name="smap-l4-soil-moisture",
                          group="Soil_Moisture_Retrieval_Data_AM")
GPM_IMERG_DAILY = EarthdataSource("GPM_3IMERGDF", name="gpm-imerg-daily")
HLS_L30 = EarthdataSource("HLSL30", name="hls-landsat-30m")
ECOSTRESS_ET = EarthdataSource("ECO_L3T_JET", name="ecostress-et")
ECOSTRESS_SEB = EarthdataSource("ECO_L3G_SEB", name="ecostress-seb")
ECOSTRESS_MET = EarthdataSource("ECO_L3G_MET", name="ecostress-met")
NLDAS2_FORCING = EarthdataSource("NLDAS_FORA0125_H", name="nldas2-forcing")
GRACE_MASCON = EarthdataSource("TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4",
                               name="grace-fo-mascon")
NISAR_GCOV = EarthdataSource("NISAR_L2_GCOV_BETA_V1", name="nisar-l2-gcov")
NISAR_SME = EarthdataSource("NISAR_L3_SME2_BETA_V1",
                            name="nisar-l3-soil-moisture")
