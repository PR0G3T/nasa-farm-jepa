"""USDA NASS QuickStats — county-level yield ground truth.

Trait calibration used hard-coded Story County targets; QuickStats serves
actual county statistics for every county the crawler covers, so trait
fitting follows the fields wherever they land. Bushels per acre convert
to kg/ha per crop (USDA standard weights); silage units are skipped —
the model yields grain.
"""

from __future__ import annotations

import os

import httpx2 as httpx

API = "https://quickstats.nass.usda.gov/api/api_GET"
"""Standard commodity mass per bushel → kg/ha = bu/ac × factor."""
BUSHEL_KG_HA = {"maize": 62.7725, "soybean": 67.2549, "wheat": 67.2549}
NASS_CROPS = {"maize": "CORN", "soybean": "SOYBEANS", "wheat": "WHEAT"}


def yields(
    county: str,
    state_alpha: str,
    crop: str,
    years: tuple[int, int] = (2015, 2024),
    key: str | None = None,
) -> dict[int, float]:
    """{year: kg/ha} of county yield, grain units only, years inclusive."""
    key = key or os.environ.get("QUICKSTATS_API_KEY", "")
    if not key:
        raise RuntimeError("QUICKSTATS_API_KEY not set")
    if crop not in NASS_CROPS:
        return {}
    payload = (
        httpx.get(
            API,
            params={
                "key": key,
                "source_desc": "SURVEY",
                "sector_desc": "CROPS",
                "group_desc": "FIELD CROPS",
                "commodity_desc": NASS_CROPS[crop],
                "statisticcat_desc": "YIELD",
                "agg_level_desc": "COUNTY",
                "state_alpha": state_alpha,
                "county_name": county.upper(),
                "year__GE": years[0],
                "format": "JSON",
            },
            timeout=30,
        )
        .json()
        .get("data", [])
    )
    out = {}
    for row in payload:
        year = int(row["year"])
        if year > years[1] or "BU / ACRE" not in row.get("unit_desc", ""):
            continue
        out[year] = float(row["Value"].replace(",", "")) * BUSHEL_KG_HA[crop]
    return out
