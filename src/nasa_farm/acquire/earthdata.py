"""NASA Earthdata (CMR) source via `earthaccess` — PEP 810 lazy import keeps
the package usable (and import-fast) without the heavy dep or credentials;
search needs no login, downloads do.

Every product is pinned to a CMR (short_name, version): without a version,
CMR returns granules of *all* versions (ECOSTRESS 002 + 003, GEDI 002 + 003,
GLDAS 2.0 + 2.1, MODIS 061 + 6.1NRT…), so series silently mix versions.
IDs below were checked against CMR on 2026-11-14 (project doc
`claude/download-access.md`).
"""

from __future__ import annotations

lazy import earthaccess

import datetime as dt
import functools
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
_DOY = re.compile(r"(?<!\d)((?:19|20)\d{2})(\d{3})(?=\D|\d{6}(?!\d)|$)")
"""yyyyddd, alone or heading a yyyydddhhmmss stamp (GEDI)."""


def granule_day(gid: str) -> int | None:
    """yyyymmdd int from a granule id — ISO date or yyyyddd day-of-year;
    None when the id carries no date (use `umm_day` then)."""
    if m := _DATE.search(gid):
        return int(m.group())
    m = _DOY.search(gid)
    if m is None or not 1 <= int(m[2]) <= 366:
        return None
    d = dt.date(int(m[1]), 1, 1) + dt.timedelta(int(m[2]) - 1)
    return int(d.strftime("%Y%m%d"))


def umm_day(res: Any) -> int | None:
    """Fallback date: start of the granule's CMR temporal extent."""
    try:
        t = res["umm"]["TemporalExtent"]
        t = (t.get("RangeDateTime", {}).get("BeginningDateTime")
             or t["SingleDateTime"])
        return int(t[:10].replace("-", ""))
    except (KeyError, TypeError, AttributeError, ValueError):
        return None


@functools.cache
def login() -> Any:
    """One Earthdata login per process: env vars first, then ~/.netrc.
    Never falls back to an interactive prompt (crawls run unattended)."""
    for strategy in ("environment", "netrc"):
        try:
            auth = earthaccess.login(strategy=strategy)
        except Exception:  # raises or returns, depending on the version
            continue
        if auth is not None and getattr(auth, "authenticated", False):
            return auth
    raise RuntimeError("Earthdata login failed: set EARTHDATA_USERNAME/"
                       "EARTHDATA_PASSWORD or add urs.earthdata.nasa.gov "
                       "to ~/.netrc")


@dataclass(frozen=True, slots=True)
class EarthdataSource:
    """Generic CMR granule source for one product short name.

    `group` is the HDF5 group holding data vars (None = root); `version`
    pins the CMR collection version (None = every version — avoid).
    """

    short_name: str
    name: str = "earthdata"
    requires_auth: bool = True  # search is open; download needs Earthdata Login
    group: str | None = None
    version: str | None = None

    def _query(self, aoi: BBox | None, temporal: Temporal,
               count: int) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "short_name": self.short_name,
            "temporal": (temporal.start, temporal.end),
            "count": count,
        }
        if self.version is not None:
            kw["version"] = self.version
        if aoi is not None:
            kw["bounding_box"] = aoi.as_tuple()
        return kw

    def search(self, aoi: BBox | None, temporal: Temporal,
               limit: int = 10_000) -> list[Granule]:
        """CMR query → lightweight granule descriptors (no download)."""
        kw = self._query(aoi, temporal, limit)
        return [Granule(granule_id(r), r)
                for r in earthaccess.search_data(**kw)]

    def fetch(self, aoi: BBox | None, temporal: Temporal, *,
              count: int = 10, cloud_cover: tuple[int, int] | None = None,
              download_dir: str | None = None) -> list[Any]:
        kw = self._query(aoi, temporal, count)
        if cloud_cover is not None:
            kw["cloud_cover"] = cloud_cover
        results = earthaccess.search_data(**kw)
        if download_dir is not None:
            login()
            return earthaccess.download(results, download_dir)
        return results


def _src(short_name: str, version: str | None, name: str,
         group: str | None = None) -> EarthdataSource:
    return EarthdataSource(short_name, name=name, group=group,
                           version=version)


# --- Soil moisture / freeze-thaw / carbon (NSIDC) -------------------------
SMAP_L3 = _src("SPL3SMP_E", "006", "smap-l3-soil-moisture",
               group="Soil_Moisture_Retrieval_Data_AM")  # retrieval, daily
SMAP_L4 = _src("SPL4SMGP", "008", "smap-l4-soil-moisture",
               group="Geophysical_Data")                  # model+DA, 3-hourly
SMAP_FT = _src("SPL3FTP", "004", "smap-l3-freeze-thaw")
SMAP_L4C = _src("SPL4CMDL", "008", "smap-l4-carbon")

# --- ECOSTRESS 70 m tiled COGs (LP DAAC) ----------------------------------
ECOSTRESS_LST = _src("ECO_L2T_LSTE", "002", "ecostress-lst")
ECOSTRESS_ET = _src("ECO_L3T_JET", "002", "ecostress-et")
ECOSTRESS_SEB = _src("ECO_L3T_SEB", "002", "ecostress-seb")  # Rn, SWin
ECOSTRESS_MET = _src("ECO_L3T_MET", "002", "ecostress-met")  # Ta, RH
ECOSTRESS_SM = _src("ECO_L3T_SM", "002", "ecostress-sm")
ECOSTRESS_ESI = _src("ECO_L4T_ESI", "002", "ecostress-esi")
ECOSTRESS_WUE = _src("ECO_L4T_WUE", "002", "ecostress-wue")

# --- Forcing / land-surface models (GES DISC) -----------------------------
NLDAS2_FORCING = _src("NLDAS_FORA0125_H", "2.0", "nldas2-forcing")
NLDAS2_NOAH = _src("NLDAS_NOAH0125_H", "2.0", "nldas2-noah")
GLDAS_DA = _src("GLDAS_CLSM025_DA1_D", "2.2", "gldas22-grace-da")
GLDAS_NOAH = _src("GLDAS_NOAH025_3H", "2.1", "gldas21-noah")
GRACE_DA_US = _src("GRACEDADM_CLSM0125US_7D", "4.0", "grace-da-drought-us")
GPM_IMERG_DAILY = _src("GPM_3IMERGDF", "07", "gpm-imerg-daily")
GRACE_MASCON = _src("TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4",
                    "RL06.3Mv04", "grace-fo-mascon")

# --- Optical reflectance / indices (LP DAAC) ------------------------------
HLS_L30 = _src("HLSL30", "2.0", "hls-landsat-30m")
HLS_S30 = _src("HLSS30", "2.0", "hls-sentinel-30m")
HLS_L30_VI = _src("HLSL30_VI", "2.0", "hls-landsat-vi")
HLS_S30_VI = _src("HLSS30_VI", "2.0", "hls-sentinel-vi")
MODIS_SR_DAILY = _src("MOD09GA", "061", "modis-sr-daily")
MODIS_SR_8D = _src("MOD09A1", "061", "modis-sr-8day")
MODIS_NBAR = _src("MCD43A4", "061", "modis-nbar")
MODIS_LAI = _src("MCD15A3H", "061", "modis-lai-4day")
MODIS_GPP = _src("MOD17A2HGF", "061", "modis-gpp-8day")
MODIS_NPP = _src("MOD17A3HGF", "061", "modis-npp-annual")
MODIS_PHENO = _src("MCD12Q2", "061", "modis-phenology")
VIIRS_LST_DAY = _src("VNP21A1D", "002", "viirs-lst-day")
VIIRS_LST_NIGHT = _src("VNP21A1N", "002", "viirs-lst-night")
VIIRS_SR = _src("VNP09GA", "002", "viirs-sr")

# --- Structure / water / SAR ----------------------------------------------
GEDI_L2A = _src("GEDI02_A", "003", "gedi-l2a")
GEDI_L2B = _src("GEDI02_B", "003", "gedi-l2b")
GEDI_L4A = _src("GEDI_L4A_AGB_Density_V3_2508", "3", "gedi-l4a")
GEDI_L3 = _src("GEDI_L3_LandSurface_Metrics_V3_2525", "3", "gedi-l3")
OPERA_DSWX = _src("OPERA_L3_DSWX-HLS_V1", "1.0", "opera-dswx-hls")
OPERA_RTC_S1 = _src("OPERA_L2_RTC-S1_V1", "1", "opera-rtc-s1")
NISAR_GCOV = _src("NISAR_L2_GCOV_BETA_V1", None, "nisar-l2-gcov")
NISAR_SME = _src("NISAR_L3_SME2_BETA_V1", None, "nisar-l3-soil-moisture")

PRODUCTS: dict[str, EarthdataSource] = {
    s.name: s for s in list(globals().values())
    if isinstance(s, EarthdataSource)}
"""name → source, for CLI lookup and bulk crawls."""
