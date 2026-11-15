"""NASA Earthdata (CMR) source via `earthaccess` — PEP 810 lazy import keeps
the package usable (and import-fast) without the heavy dep or credentials;
search needs no login, downloads do.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

lazy import earthaccess

from nasa_farm.acquire.base import Bounds, Temporal


@dataclass(frozen=True, slots=True)
class Granule:
    """One CMR granule — id for dedup, raw result for open/download."""

    id: str
    raw: Any


def granule(result: Any) -> str:
    """GranuleUR / ProducerGranuleId from a CMR search result."""
    for k in ("GranuleUR", "ProducerGranuleId"):
        try:
            return str(result["umm"][k])
        except KeyError, TypeError:
            continue
    return str(result)


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
class Earthdata:
    """Generic CMR granule source for one product short name.

    `group` is the HDF5 group holding data vars (None = root); `span` is
    the product's native temporal window passed to open calls.
    """

    short_name: str
    name: str = "earthdata"
    requires_auth: bool = True  # search is open; download needs Earthdata Login
    group: str | None = None

    def search(self, area: Bounds | None, temporal: Temporal, limit: int = 10_000) -> list[Granule]:
        """CMR query → lightweight granule descriptors (no download)."""
        keywords: dict[str, Any] = {
            "short_name": self.short_name,
            "temporal": (temporal.start, temporal.end),
            "count": limit,
        }
        if area is not None:
            keywords["bounding_box"] = area.as_tuple()
        return [Granule(granule(r), r) for r in earthaccess.search_data(**keywords)]

    def fetch(
        self,
        area: Bounds | None,
        temporal: Temporal,
        *,
        count: int = 10,
        cloud_cover: tuple[int, int] | None = None,
        download_dir: str | None = None,
    ) -> list[Any]:
        keywords: dict[str, Any] = {
            "short_name": self.short_name,
            "temporal": (temporal.start, temporal.end),
            "count": count,
        }
        if area is not None:
            keywords["bounding_box"] = area.as_tuple()
        if cloud_cover is not None:
            keywords["cloud_cover"] = cloud_cover
        results = earthaccess.search_data(**keywords)
        if download_dir is not None:
            earthaccess.login(strategy="environment")
            return earthaccess.download(results, download_dir)
        return results


# Challenge-relevant products, declared once.
SMAP_L4 = Earthdata(
    "SPL3SMP_E", name="smap-l4-soil-moisture", group="Soil_Moisture_Retrieval_Data_AM"
)
GPM_IMERG_DAILY = Earthdata("GPM_3IMERGDF", name="gpm-imerg-daily")
HLS_L30 = Earthdata("HLSL30", name="hls-landsat-30m")
ECOSTRESS_ET = Earthdata("ECO_L3T_JET", name="ecostress-et")
ECOSTRESS_SEB = Earthdata("ECO_L3G_SEB", name="ecostress-seb")
ECOSTRESS_MET = Earthdata("ECO_L3G_MET", name="ecostress-met")
NLDAS2_FORCING = Earthdata("NLDAS_FORA0125_H", name="nldas2-forcing")
GRACE_MASCON = Earthdata("TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4", name="grace-fo-mascon")
NISAR_GCOV = Earthdata("NISAR_L2_GCOV_BETA_V1", name="nisar-l2-gcov")
NISAR_SME = Earthdata("NISAR_L3_SME2_BETA_V1", name="nisar-l3-soil-moisture")
