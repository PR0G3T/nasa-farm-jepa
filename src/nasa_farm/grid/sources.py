"""CMR collections used by the 1 km build — pinned (short_name, version).

Kept here rather than in `acquire.earthdata` on purpose: the grid build
must not change when that module is edited, and a CMR search without a
version returns granules of *every* version (ECOSTRESS 002 + 003, GEDI
002 + 003, GLDAS 2.0 + 2.1, MODIS 061 + 6.1NRT…), silently mixing them.
IDs from `claude/download-access.md` (checked against CMR 2026-11-14).
"""

from __future__ import annotations

import datetime as dt
import functools
import os
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Collection:
    short_name: str
    version: str


# soil moisture / freeze–thaw / carbon (NSIDC)
SMAP_L4 = Collection("SPL4SMGP", "008")  # L4 — not SPL3SMP_E (L3)
SMAP_FREEZE_THAW = Collection("SPL3FTP", "004")
SMAP_CARBON = Collection("SPL4CMDL", "008")
# ECOSTRESS 70 m tiled COGs (LP DAAC) — tiled L3T/L4T, not global L3G
ECOSTRESS_LST = Collection("ECO_L2T_LSTE", "002")
ECOSTRESS_SEB = Collection("ECO_L3T_SEB", "002")
ECOSTRESS_MET = Collection("ECO_L3T_MET", "002")
ECOSTRESS_SM = Collection("ECO_L3T_SM", "002")
ECOSTRESS_ESI = Collection("ECO_L4T_ESI", "002")
ECOSTRESS_WUE = Collection("ECO_L4T_WUE", "002")
# forcing / land-surface models (GES DISC)
NLDAS2_FORCING = Collection("NLDAS_FORA0125_H", "2.0")
NLDAS2_NOAH = Collection("NLDAS_NOAH0125_H", "2.0")
GLDAS_ASSIMILATION = Collection("GLDAS_CLSM025_DA1_D", "2.2")
GLDAS_NOAH = Collection("GLDAS_NOAH025_3H", "2.1")
GRACE_PERCENTILES = Collection("GRACEDADM_CLSM0125US_7D", "4.0")
# optical / thermal (LP DAAC)
HLS_L30_INDICES = Collection("HLSL30_VI", "2.0")
VIIRS_LST_DAY = Collection("VNP21A1D", "002")
VIIRS_LST_NIGHT = Collection("VNP21A1N", "002")
# NOAA-20 twins of the Suomi-NPP LST products (same algorithm and layers):
# fallback when S-NPP has a gap (e.g. h11v04, 2024-07-11 → 07-16)
VIIRS_LST_DAY_N20 = Collection("VJ121A1D", "002")
VIIRS_LST_NIGHT_N20 = Collection("VJ121A1N", "002")
VIIRS_REFLECTANCE = Collection("VNP09GA", "002")
# structure / water
GEDI_L2A = Collection("GEDI02_A", "003")
GEDI_L2B = Collection("GEDI02_B", "003")
GEDI_L4A = Collection("GEDI_L4A_AGB_Density_V3_2508", "3")
GEDI_L3 = Collection("GEDI_L3_LandSurface_Metrics_V3_2525", "3")
OPERA_WATER = Collection("OPERA_L3_DSWX-HLS_V1", "1.0")


# --- granule helpers ----------------------------------------------------------
def granule_id(result: Any) -> str:
    for k in ("GranuleUR", "ProducerGranuleId"):
        try:
            return str(result["umm"][k])
        except KeyError, TypeError:
            continue
    return str(result)


_DATE = re.compile(r"(?<!\d)(?:19|20)\d{6}(?!\d)")
_DAY_OF_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(\d{3})(?=\D|\d{6}(?!\d)|$)")


def granule_day(identifier: str, result: Any = None) -> int | None:
    """yyyymmdd from an id (ISO date or yyyyddd), else from the CMR
    temporal extent, else None — never raises."""
    if m := _DATE.search(identifier):
        return int(m.group())
    m = _DAY_OF_YEAR.search(identifier)
    if m and 1 <= int(m[2]) <= 366:
        d = dt.date(int(m[1]), 1, 1) + dt.timedelta(int(m[2]) - 1)
        return int(d.strftime("%Y%m%d"))
    try:
        t = result["umm"]["TemporalExtent"]
        t = t.get("RangeDateTime", {}).get("BeginningDateTime") or t["SingleDateTime"]
        return int(t[:10].replace("-", ""))
    except KeyError, TypeError, AttributeError, ValueError:
        return None


@functools.cache
def login() -> Any:
    """One Earthdata login per process: environment first, then ~/.netrc;
    never an interactive prompt (runs are unattended)."""
    import earthaccess

    for strategy in ("environment", "netrc"):
        if strategy == "environment" and not os.environ.get("EARTHDATA_USERNAME"):
            continue
        try:
            auth = earthaccess.login(strategy=strategy)
        except Exception:  # noqa: S112 — raises or returns, depending on the version
            continue
        if auth is not None and getattr(auth, "authenticated", False):
            return auth
    raise RuntimeError(
        "Earthdata login failed: set EARTHDATA_USERNAME/"
        "EARTHDATA_PASSWORD or add urs.earthdata.nasa.gov to ~/.netrc"
    )
