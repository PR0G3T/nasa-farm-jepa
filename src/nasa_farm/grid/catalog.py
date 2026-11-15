"""One entry per product of `claude/download-access.md`: fetch one file
near the reference date and put it on the common grid.

Each function receives the run context (grid, date, cache dir) and an
`Out` to fill; it may raise `Skip` (credential/dependency absent) or any
error (reported as failed, the run continues). `conf` is the confidence in
the *route* as tagged in the access doc (V verified live, D official docs,
M medium/inferred) — the regridding itself is tested offline.

Choices that differ from the access doc's "best route" (and why):
- MODIS → Planetary Computer COGs first, AppEEARS for the Jul–Oct 2025
  PC hole: LP DAAC MODIS is HDF4, which neither the rasterio 3.15t wheel
  (no HDF4 driver) nor pyhdf (does not build) can read here.
- MCD12Q2 phenology → AppEEARS only (HDF4, not on PC).
- CDL → per-year 30 m COG on Source Cooperative instead of the Icechunk
  store: no extra dependency, windowed read, and one consistent 30 m
  series (the 10 m 2024–25 layers would make the series inconsistent).
- Crop Frequency → recomputed from CDL 2008→year (doc's own suggestion).
- One file per CMR collection: alternates of a row (VJ121, VNP21A2,
  MCD15A3H, MOD09GA/GQ, HLSS30_VI, ECOSTRESS v003) are listed in
  `ALTERNATES`, not downloaded.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from nasa_farm.grid import sources as cmr
from nasa_farm.grid.fetch import (
    REQUESTER_PAYS, AppEEARS, Context, Skip, cds_retrieve, download, earthdata_download,
    earthdata_granules, earthdata_open, gcs_href, gcs_list, lfps_job, openet_export,
    planetary_assets, planetary_href, planetary_items, s3_list, sda_query, session, usgs_items,
)
from nasa_farm.grid.output import Out
from nasa_farm.grid.readers import (
    NoOverlap, read_ease2, read_gedi, read_latlon, read_sinusoidal_h5,
    read_tif,
)
from nasa_farm.grid.regrid import (
    Raster, bin_points, rasterize_cover, rasterize_values, regrid_continuous,
)


@dataclass(frozen=True)
class Product:
    key: str
    section: int           # variable section of products-by-variable.md
    title: str
    route: str
    needs: tuple[str, ...]
    conf: str
    fn: Callable[[Context, Out], None]


REGISTRY: dict[str, Product] = {}


def product(key: str, section: int, title: str, route: str,
            needs: Sequence[str] = (), conf: str = "D"):
    def deco(fn):
        REGISTRY[key] = Product(key, section, title, route, tuple(needs), conf, fn)
        return fn
    return deco


ALTERNATES = {
    "viirs-lst-day": "VJ121A1D v002 (NOAA-20), VNP21A2 v002 (8-day)",
    "modis-sr-8day": "MOD09GA / MOD09GQ v061 daily (HDF4 → AppEEARS)",
    "modis-lai": "MCD15A3H v061 4-day (PC modis-15A3H-061)",
    "hls-vi": "HLSS30_VI v2.0 (Sentinel-2 side)",
    "ecostress-lst": "ECO_L2T_LSTE v003; ESI/WUE also exist in v003",
    "landsat-dswe": "OPERA DSWx-HLS (entry opera-dswx)",
}


# --- small helpers -------------------------------------------------------------
def _h5(f):
    try:
        import h5py
    except ImportError as e:
        raise Skip("needs h5py") from e
    return h5py.File(f, "r")


def _xr(path_or_file):
    import xarray as xr
    return xr.open_dataset(path_or_file, engine="h5netcdf",
                           decode_timedelta=False)


def _attr_str(d, k: str) -> str:
    v = d.attrs.get(k, "")
    return v.decode() if isinstance(v, bytes) else str(v)


def _first(h5, names: Sequence[str]) -> str:
    for n in names:
        if n in h5:
            return n
    raise KeyError(f"none of {names}; top-level: {list(h5)[:30]}")


def _match(mask: Raster, like: Raster) -> np.ndarray:
    """Categorical mask resampled (nearest) onto another raster's grid."""
    if mask.data.shape[1:] == like.data.shape[1:] and \
            mask.transform == like.transform:
        return mask.data[0]
    from rasterio.warp import Resampling, reproject
    dst = np.zeros(like.data.shape[1:], np.int32) - 1
    reproject(mask.data[0].astype(np.int32), dst, src_transform=mask.transform,
              src_crs=mask.crs, dst_transform=like.transform,
              dst_crs=like.crs, resampling=Resampling.nearest,
              src_nodata=None, dst_nodata=-1)
    return dst


def _stack(rs: Sequence[Raster]) -> Raster:
    """Same-grid single-band rasters → one multi-band raster."""
    r0 = rs[0]
    for r in rs[1:]:
        if r.data.shape[1:] != r0.data.shape[1:] or r.transform != r0.transform:
            raise ValueError("bands on different grids")
    return Raster(np.concatenate([r.data for r in rs]), r0.transform, r0.crs,
                  sum((r.bands for r in rs), []),
                  units={k: v for r in rs for k, v in r.units.items()})


def _mask(r: Raster, bad: np.ndarray) -> Raster:
    r.data[:, bad] = np.nan
    return r


def _bits(a: np.ndarray, *bits: int) -> np.ndarray:
    a = a.astype(np.int64)
    out = np.zeros(a.shape, bool)
    for b in bits:
        out |= ((a >> b) & 1) == 1
    return out


# ============================================================================
# 1–3  Weather
# ============================================================================
NLDAS3_DAILY = ("https://nasa-waterinsight.s3.amazonaws.com/NLDAS3/forcing/daily/"
                "{ym}/NLDAS_FOR0010_D.A{day}.030.beta.nc")
NLDAS3_VARS = ("Tair", "Tair_min", "Tair_max", "Qair", "PSurf", "Wind_E", "Wind_N",
               "SWdown", "LWdown", "Rainf")


@product("nldas3-forcing", 1, "NLDAS-3 forcing (beta), daily, 0.01°",
         "public S3 over HTTPS (range reads)", conf="D")
def nldas3(context: Context, out: Out) -> None:
    """Whole-domain daily file (~0.7 GB) read lazily: only the chunks under
    the AOI travel. Units are kept as in the file (`units` attribute);
    conversion to model forcing belongs to the consumer."""
    import fsspec
    url = NLDAS3_DAILY.format(ym=context.date.strftime("%Y%m"), day=context.date.strftime("%Y%m%d"))
    context.note(url)
    with fsspec.open(url).open() as f, _xr(f) as ds:
        r = read_latlon(ds, context.grid, [v for v in NLDAS3_VARS if v in ds], buffer_deg=0.05)
    out.continuous(r, valid=None)


@product("ecostress-met", 2, "ECOSTRESS L3T MET (Ta, RH), 70 m",
         "earthaccess", needs=("earthdata",), conf="V")
def ecostress_met(context, out):
    _ecostress(cmr.ECOSTRESS_MET, {"Ta": "Ta", "RH": "RH"}, context, out)


@product("smap-ft", 2, "SMAP L3 freeze/thaw SPL3FTP v004, 36 km",
         "earthaccess", needs=("earthdata",), conf="V")
def smap_ft(context, out):
    g = earthdata_granules(cmr.SMAP_FREEZE_THAW, context, days=2, max_tiles=1)
    [f] = earthdata_download(g, context)
    with _h5(f) as h5:
        grp = next((k for k in h5 if "Freeze_Thaw" in k and "Global" in k),
                   None) or next(k for k in h5 if "Freeze_Thaw" in k)
        G = h5[grp]
        var = _first(G, ["freeze_thaw", "freeze_thaw_state"])
        lat = _first(G, ["latitude", "latitude_centroid"])
        lon = _first(G, ["longitude", "longitude_centroid"])
        r = read_ease2(G, context.grid, {"ft_am": var, "ft_pm": var}, lat, lon,
                       select={"ft_am": 0, "ft_pm": 1})
    for k, band in enumerate(r.bands):
        rr = Raster(np.nan_to_num(r.data[k:k + 1], nan=255).astype(np.uint8),
                    r.transform, r.crs, [band], kind="categorical",
                    nodata=255, classes=[0, 1],
                    class_names={0: "thawed", 1: "frozen"})
        out.categorical(rr, band)


@product("nldas2-forcing", 1, "NLDAS-2 forcing FORA0125_H v2.0, hourly, 0.125°",
         "earthaccess", needs=("earthdata",), conf="V")
def nldas2_forcing(context, out):
    _latlon_granule(cmr.NLDAS2_FORCING, context, out, days=1, pick=r"\.1800\.")


@product("ecostress-seb", 3, "ECOSTRESS L3T SEB (Rn, SWin), 70 m",
         "earthaccess", needs=("earthdata",), conf="V")
def ecostress_seb(context, out):
    _ecostress(cmr.ECOSTRESS_SEB, {"Rn": "Rn", "SWin": "SWin", "Rg": "Rg"},
               context, out)


# ============================================================================
# 4–7  Soil moisture, ET, stress
# ============================================================================
SMAP_L4_VARS = ("sm_surface", "sm_rootzone", "sm_profile", "sm_surface_wetness",
                "sm_rootzone_wetness", "surface_temp", "soil_temp_layer1",
                "land_evapotranspiration_flux", "heat_flux_latent",
                "heat_flux_sensible", "net_downward_shortwave_flux",
                "net_downward_longwave_flux", "precipitation_total_surface_flux",
                "overland_runoff_flux", "baseflow_flux")


@product("smap-l4", 4, "SMAP L4 SPL4SMGP v008 (surface + root-zone SM, fluxes), 9 km",
         "earthaccess (HDF5 streamed)", needs=("earthdata",), conf="V")
def smap_l4(context, out):
    g = earthdata_granules(cmr.SMAP_L4, context, days=1, pick=r"T133000", max_tiles=1)
    [f] = earthdata_open(g, suffixes=(".h5",))
    with _h5(f) as h5:
        grp = h5["Geophysical_Data"]
        vs = {v: f"Geophysical_Data/{v}" for v in SMAP_L4_VARS if v in grp}
        r = read_ease2(h5, context.grid, vs, "cell_lat", "cell_lon")
        r.units = {v: _attr_str(grp[v], "units") for v in vs}
    out.continuous(r)


@product("ecostress-sm", 4, "ECOSTRESS L3T SM, 70 m (GEOS-5 resampled)",
         "earthaccess", needs=("earthdata",), conf="V")
def ecostress_sm(context, out):
    _ecostress(cmr.ECOSTRESS_SM, {"SM": "SM"}, context, out)


@product("sentinel1-rtc", 4, "Sentinel-1 RTC (VV, VH linear γ0), 10 m",
         "Planetary Computer", conf="V")
def s1_rtc(context, out):
    items = planetary_items("sentinel-1-rtc", context, days=6)
    rs = []
    for it in items:
        bands = [read_tif(planetary_href(it, a), context.grid, name=a) for a in ("vv", "vh")
                 if a in it.assets]
        rs.append(_stack(bands))
    # averaging is done in linear power (never in dB)
    out.continuous(rs)


@product("openet", 6, "OpenET ensemble daily ET, 30 m",
         "OpenET REST /raster/export/stack", needs=("openet",), conf="M")
def openet(context, out):
    p = openet_export(context, context.bbox(1.0), context.date)
    out.continuous(read_tif(str(p), context.grid, name="et_ensemble", units="mm/day"))


@product("ecostress-esi", 7, "ECOSTRESS L4T ESI v002 (ESI, PET), 70 m",
         "earthaccess", needs=("earthdata",), conf="V")
def ecostress_esi(context, out):
    _ecostress(cmr.ECOSTRESS_ESI, {"ESI": "ESI", "PET": "PET"}, context, out)


@product("ecostress-wue", 7, "ECOSTRESS L4T WUE v002 (WUE, GPP), 70 m",
         "earthaccess", needs=("earthdata",), conf="V")
def ecostress_wue(context, out):
    _ecostress(cmr.ECOSTRESS_WUE, {"WUE": "WUE", "GPP": "GPP"}, context, out)


# ============================================================================
# 8–9  Water storage, streamflow, surface water
# ============================================================================
@product("gldas-grace-da", 8, "GLDAS-2.2 CLSM with GRACE DA, daily, 0.25°",
         "earthaccess", needs=("earthdata",), conf="V")
def gldas_da(context, out):
    _latlon_granule(cmr.GLDAS_ASSIMILATION, context, out, days=1, buffer_deg=0.6)


@product("grace-da-percentiles", 8, "GRACE-DA drought percentiles CONUS 0.125°, weekly",
         "earthaccess", needs=("earthdata",), conf="V")
def grace_da(context, out):
    _latlon_granule(cmr.GRACE_PERCENTILES, context, out, days=7, buffer_deg=0.3)


@product("usgs-discharge", 9, "USGS daily discharge + gage height (gauges in AOI)",
         "USGS OGC API", conf="V")
def usgs_discharge(context, out):
    import pandas as pd
    b = context.bbox(2.0)
    rows = []
    for code, name in (("00060", "discharge_cfs"), ("00065", "gage_height_ft")):
        feats = usgs_items("daily", {
            "bbox": f"{b.west},{b.south},{b.east},{b.north}",
            "parameter_code": code, "statistic_id": "00003",
            "time": f"{context.date}/{context.date}"})
        for f in feats:
            p, geom = f["properties"], f.get("geometry") or {}
            lon, lat = (geom.get("coordinates") or [np.nan, np.nan])[:2]
            rows.append({"site": p.get("monitoring_location_id"), "var": name,
                         "value": float(p["value"]) if p.get("value") not in (None, "") else np.nan,
                         "lon": lon, "lat": lat, "time": p.get("time")})
    _points(context, out, pd.DataFrame(rows), pivot=True)


@product("usgs-water-quality", 9, "USGS samples (nitrate, phosphorus…) in AOI, year of date",
         "USGS Samples API", conf="D")
def usgs_wq(context, out):
    import pandas as pd
    b = context.bbox(5.0)
    y = context.date.year
    p = context.raw / f"samples_{y}.csv"
    download("https://api.waterdata.usgs.gov/samples-data/results/fullphyschem", p,
             params={"mimeType": "text/csv",
                     "boundingBox": f"{b.west:.4f},{b.south:.4f},{b.east:.4f},{b.north:.4f}",
                     "activityStartDateLower": f"{y}-01-01",
                     "activityStartDateUpper": f"{y}-12-31",
                     "characteristicGroup": "Nutrient"})
    df = pd.read_csv(p, low_memory=False)
    col = lambda *c: next((x for x in df.columns if x in c), None)
    lat, lon = col("Location_Latitude", "ActivityLocation_Latitude"), \
        col("Location_Longitude", "ActivityLocation_Longitude")
    val, ch = col("Result_Measure", "ResultMeasureValue"), \
        col("Result_Characteristic", "CharacteristicName")
    site = col("Location_Identifier", "MonitoringLocationIdentifier")
    t = col("Activity_StartDate", "ActivityStartDate")
    if not all((lat, lon, val, ch, site)):
        raise KeyError(f"unexpected Samples columns: {list(df.columns)[:25]}")
    d = pd.DataFrame({"site": df[site], "var": df[ch], "lon": df[lon],
                      "lat": df[lat], "time": df[t] if t else None,
                      "value": pd.to_numeric(df[val], errors="coerce")})
    _points(context, out, d, pivot=False)


@product("nldas2-noah", 9, "NLDAS-2 Noah LSM (runoff, soil moisture…), hourly, 0.125°",
         "earthaccess", needs=("earthdata",), conf="V")
def nldas2_noah(context, out):
    _latlon_granule(cmr.NLDAS2_NOAH, context, out, days=1, pick=r"\.1800\.")


@product("gldas-noah", 9, "GLDAS-2.1 Noah 3-hourly, 0.25°",
         "earthaccess", needs=("earthdata",), conf="V")
def gldas_noah(context, out):
    _latlon_granule(cmr.GLDAS_NOAH, context, out, days=1, pick=r"\.1800\.",
                    buffer_deg=0.6)


ERA5L_VARS = ["2m_temperature", "2m_dewpoint_temperature", "total_precipitation",
              "surface_solar_radiation_downwards", "10m_u_component_of_wind",
              "10m_v_component_of_wind", "volumetric_soil_water_layer_1",
              "volumetric_soil_water_layer_2", "volumetric_soil_water_layer_3",
              "volumetric_soil_water_layer_4", "skin_temperature",
              "soil_temperature_level_1", "runoff", "total_evaporation",
              "leaf_area_index_high_vegetation", "leaf_area_index_low_vegetation"]


@product("era5-land", 9, "ERA5-Land hourly (24 steps of the date), 0.1°",
         "CDS API", needs=("cds",), conf="D")
def era5_land(context, out):
    b = context.bbox(30.0)
    d = context.date
    p = cds_retrieve(context, "reanalysis-era5-land", {
        "variable": ERA5L_VARS, "year": f"{d.year}", "month": f"{d.month:02d}",
        "day": f"{d.day:02d}", "time": [f"{h:02d}:00" for h in range(24)],
        "area": [round(b.north, 2), round(b.west, 2), round(b.south, 2),
                 round(b.east, 2)],
        "data_format": "netcdf", "download_format": "unarchived"},
        f"era5land_{d}.nc")
    with _xr(p) as ds:
        r = read_latlon(ds, context.grid, buffer_deg=0.3)
    out.continuous(r, valid=None)
    context.note("accumulated vars (tp, ssrd, ro, e) are since 00 UTC: "
             "daily total = value at 00 UTC of the next day")


def _usgs_l3(context: Context, out: Out, collection: str, asset_rx: str, name: str,
             classes, class_names, nodata=255, days=16):
    """USGS Landsat Collection 2 Level-3 via the LandsatLook STAC; assets
    live in the requester-pays `usgs-landsat` bucket (read via /vsis3)."""
    import pystac_client
    cat = pystac_client.Client.open("https://landsatlook.usgs.gov/stac-server")
    w = context.window(days)
    items = list(cat.search(collections=[collection], bbox=context.bbox().as_tuple(),
                            datetime=f"{w.start}/{w.end}", max_items=50).items())
    if not items:
        raise FileNotFoundError(f"{collection}: no item within ±{days} d")
    items.sort(key=lambda i: abs((i.datetime.date() - context.date).days))
    best = items[0].datetime.date()
    same = [i for i in items if i.datetime.date() == best][:4]
    context.note(f"{collection}: " + ", ".join(i.id for i in same))
    rs = []
    for it in same:
        key = next((k for k in it.assets if re.search(asset_rx, k, re.I)), None)
        if key is None:
            raise KeyError(f"{it.id}: no asset ~{asset_rx}; has {list(it.assets)}")
        a = it.assets[key]
        alt = (a.extra_fields.get("alternate", {}).get("s3", {}) or {}).get("href")
        href = alt.replace("s3://", "/vsis3/") if alt else a.href
        rs.append(read_tif(href, context.grid, name=name, kind="categorical",
                           classes=classes, class_names=class_names,
                           env=REQUESTER_PAYS if alt else None))
    for r in rs:
        r.nodata = nodata
    out.categorical(rs, name)


DSWE = {0: "not_water", 1: "water_high", 2: "water_moderate",
        3: "potential_wetland", 4: "water_or_wetland_low", 9: "cloud_shadow_snow"}


@product("landsat-dswe", 9, "Landsat C2 L3 DSWE (surface water classes), 30 m",
         "USGS STAC + requester-pays S3", needs=("aws",), conf="M")
def landsat_dswe(context, out):
    _usgs_l3(context, out, "landsat-c2l3-dswe", r"inwm|intr", "dswe",
             list(DSWE), DSWE)


OPERA_WTR = {0: "not_water", 1: "open_water", 2: "partial_water",
             252: "snow_ice", 253: "cloud_shadow"}


@product("opera-dswx", 9, "OPERA DSWx-HLS water classes, 30 m",
         "earthaccess", needs=("earthdata",), conf="V")
def opera_dswx(context, out):
    g = earthdata_granules(cmr.OPERA_WATER, context, days=10)
    rs = []
    for gr in g:
        [f] = earthdata_download([gr], context, ["_B01_WTR.tif"])
        r = read_tif(str(f), context.grid, name="wtr", kind="categorical",
                     classes=list(OPERA_WTR), class_names=OPERA_WTR)
        r.nodata = 255
        rs.append(r)
    out.categorical(rs, "wtr")


# ============================================================================
# 10–12  LST, vegetation indices, reflectance
# ============================================================================
@product("ecostress-lst", 10, "ECOSTRESS L2T LSTE v002 (LST, error, emissivity), 70 m",
         "earthaccess", needs=("earthdata",), conf="V")
def ecostress_lst(context, out):
    _ecostress(cmr.ECOSTRESS_LST, {"LST": "LST", "LST_err": "LST_err",
                                  "EmisWB": "EmisWB"}, context, out)


def _viirs_lst(src, context, out):
    g = earthdata_granules(src, context, days=1)
    files = earthdata_download(g, context, [".h5"])
    bad = lambda qc: np.isin(np.asarray(qc).astype(np.int64) & 3, (2, 3))
    rs = []
    for f in files:
        with _h5(f) as h5:
            rs += read_sinusoidal_h5(
                h5, context.grid, ["LST_1KM", "Emis_14", "Emis_15", "Emis_16", "View_Time"],
                qc={"LST_1KM": ("QC", bad)})
    out.continuous(rs)


@product("viirs-lst-day", 10, "VIIRS VNP21A1D v002 daytime LST, 1 km",
         "earthaccess", needs=("earthdata",), conf="V")
def viirs_lst_day(context, out):
    _viirs_lst(cmr.VIIRS_LST_DAY, context, out)


@product("viirs-lst-night", 10, "VIIRS VNP21A1N v002 night-time LST, 1 km",
         "earthaccess", needs=("earthdata",), conf="V")
def viirs_lst_night(context, out):
    _viirs_lst(cmr.VIIRS_LST_NIGHT, context, out)


HLS_VI = ("NDVI", "EVI", "SAVI", "MSAVI", "NDMI", "NDWI", "NBR", "NBR2", "TVI")


def _fmask_bad(fm: np.ndarray) -> np.ndarray:
    """HLS v2 Fmask: bit1 cloud, bit2 adjacent, bit3 shadow, bit4 snow/ice."""
    return _bits(fm, 1, 2, 3, 4) | (fm == 255)


@product("hls-vi", 11, "HLS L30 vegetation indices v2.0 (NDVI, EVI, …), 30 m",
         "earthaccess (COG)", needs=("earthdata",), conf="V")
def hls_vi(context, out):
    g = earthdata_granules(cmr.HLS_L30_INDICES, context, days=10, cloud_max=40)
    rs = []
    for gr in g:
        files = earthdata_download([gr], context, [f".{v}.tif" for v in HLS_VI] + [".Fmask.tif"])
        fm = next(f for f in files if f.name.endswith(".Fmask.tif"))
        m = read_tif(str(fm), context.grid, name="Fmask", kind="categorical")
        bands = [read_tif(str(f), context.grid, name=v, scale=1e-4, valid_range=(-10000, 10000))
                 for v in HLS_VI for f in files if f.name.endswith(f".{v}.tif")]
        r = _stack(bands)
        rs.append(_mask(r, _fmask_bad(_match(m, r))))
    out.continuous(rs)


def _hls_pc(collection: str, context: Context, out: Out):
    items = planetary_items(collection, context, days=10, query={"eo:cloud_cover": {"lt": 40}})
    rs = []
    for it in items:
        keys = planetary_assets(it, r"B\d\d|B8A")
        bands = [read_tif(planetary_href(it, k), context.grid, name=k, scale=1e-4,
                          valid_range=(-100, 16000)) for k in keys]
        r = _stack(bands)
        m = read_tif(planetary_href(it, "Fmask"), context.grid, name="Fmask", kind="categorical")
        rs.append(_mask(r, _fmask_bad(_match(m, r))))
    out.continuous(rs)


@product("hls-l30", 12, "HLS L30 v2.0 surface reflectance, 30 m",
         "Planetary Computer (hls2-l30)", conf="V")
def hls_l30(context, out):
    _hls_pc("hls2-l30", context, out)


@product("hls-s30", 12, "HLS S30 v2.0 surface reflectance, 30 m (PC: 2020+)",
         "Planetary Computer (hls2-s30)", conf="V")
def hls_s30(context, out):
    if context.date.year < 2020:
        raise Skip("PC hls2-s30 starts in 2020: use LP DAAC HLSS30 for 2015–2019")
    _hls_pc("hls2-s30", context, out)


S2_10M = ("B02", "B03", "B04", "B08")
S2_20M = ("B05", "B06", "B07", "B8A", "B11", "B12")
SCL = {2: "dark", 3: "cloud_shadow", 4: "vegetation", 5: "bare_soil",
       6: "water", 7: "unclassified", 8: "cloud_medium", 9: "cloud_high",
       10: "thin_cirrus", 11: "snow"}


@product("sentinel2-l2a", 12, "Sentinel-2 L2A reflectance + scene classes, 10/20 m",
         "Planetary Computer", conf="V")
def s2_l2a(context, out):
    items = planetary_items("sentinel-2-l2a", context, days=10, query={"eo:cloud_cover": {"lt": 30}})
    r10, r20, scl = [], [], []
    for it in items:
        base = float(str(it.properties.get("s2:processing_baseline", "0")) or 0)
        off = -0.1 if base >= 4.0 else None   # (DN − 1000) / 10000 from 04.00
        s = read_tif(planetary_href(it, "SCL"), context.grid, name="SCL", kind="categorical",
                     classes=list(SCL), class_names=SCL)
        s.nodata = 0
        scl.append(s)
        for names, acc in ((S2_10M, r10), (S2_20M, r20)):
            r = _stack([read_tif(planetary_href(it, b), context.grid, name=b, scale=1e-4,
                                 offset=off) for b in names])
            bad = np.isin(_match(s, r), (0, 1, 3, 8, 9, 10))
            acc.append(_mask(r, bad))
    out.continuous(r10, valid="valid_frac_10m")
    out.continuous(r20, valid="valid_frac_20m")
    out.categorical(scl, "scl")


def _modis(context: Context, out: Out, pc_collection: str, layers: dict[str, dict],
           *, days: int = 8, datetime: str | None = None,
           appeears: str | None = None, qa: Callable | None = None,
           qa_asset: str | None = None) -> None:
    """MODIS 500 m: PC COG tiles (mosaic) → per-layer scale/valid range,
    optional QA mask. Falls back to AppEEARS when PC has no item (2025
    Jul–Oct hole)."""
    try:
        items = planetary_items(pc_collection, context, days, datetime=datetime)
        tiles = [{k: planetary_href(it, k) for k in list(layers) + ([qa_asset] if qa_asset else [])
                  if k in it.assets} for it in items]
        if not any(set(layers) & set(t) for t in tiles):
            raise KeyError(f"{pc_collection}: assets {list(layers)} not in "
                           f"{sorted(items[0].assets)[:30]}")
    except FileNotFoundError:
        if not appeears:
            raise
        start, end = context.date - dt.timedelta(days), context.date + dt.timedelta(days)
        files = AppEEARS().area(context, appeears, list(layers) + ([qa_asset] if qa_asset else []),
                                start, end)
        files = sorted(files, key=lambda f: -_doy_dist(f.name, context.date))  # closest last wins
        tiles = [{k: str(f) for k in list(layers) + ([qa_asset] if qa_asset else [])
                  for f in files if f"_{k}_" in f.name}]
        context.note(f"PC {pc_collection} empty near {context.date} → AppEEARS {appeears}")
    rs = []
    for t in tiles:
        bands = [read_tif(t[k], context.grid, name=spec.get("name", k), scale=spec.get("scale"),
                          valid_range=spec.get("valid"), units=spec.get("units"))
                 for k, spec in layers.items() if k in t]
        r = _stack(bands)
        if qa and qa_asset in t:
            q = read_tif(t[qa_asset], context.grid, name="qa", kind="categorical")
            r = _mask(r, qa(_match(q, r)))
        rs.append(r)
    out.continuous(rs)


def _doy_dist(name: str, when: dt.date) -> int:
    m = re.search(r"doy(\d{4})(\d{3})", name)
    if not m:
        return 10 ** 6
    d = dt.date(int(m[1]), 1, 1) + dt.timedelta(int(m[2]) - 1)
    return abs((d - when).days)


def _sr_state_bad(q):
    q = q.astype(np.int64)
    return np.isin(q & 3, (1, 2)) | (((q >> 2) & 1) == 1)


@product("modis-sr-8day", 12, "MODIS MOD09A1 v061 8-day surface reflectance, 500 m",
         "Planetary Computer → AppEEARS fallback", conf="V")
def modis_sr(context, out):
    _modis(context, out, "modis-09A1-061",
           {f"sur_refl_b0{b}": {"scale": 1e-4, "valid": (-100, 16000)} for b in range(1, 8)},
           appeears="MOD09A1.061", qa=_sr_state_bad, qa_asset="sur_refl_state_500m")


@product("modis-nbar", 12, "MODIS MCD43A4 v061 NBAR daily, 500 m",
         "Planetary Computer → AppEEARS fallback", conf="V")
def modis_nbar(context, out):
    _modis(context, out, "modis-43A4-061",
           {f"Nadir_Reflectance_Band{b}": {"scale": 1e-4, "valid": (0, 32766)}
            for b in range(1, 8)}, days=2, appeears="MCD43A4.061")


@product("viirs-sr", 12, "VIIRS VNP09GA v002 daily surface reflectance (M 1 km, I 500 m)",
         "earthaccess", needs=("earthdata",), conf="V")
def viirs_sr(context, out):
    g = earthdata_granules(cmr.VIIRS_REFLECTANCE, context, days=1)
    files = earthdata_download(g, context, [".h5"])
    rs: dict[float, list[Raster]] = {}
    for f in files:
        with _h5(f) as h5:
            for r in read_sinusoidal_h5(h5, context.grid, [r"SurfReflect_[MI]\d+(_1)?"]):
                rs.setdefault(round(abs(r.transform.a)), []).append(r)
    for res, group in sorted(rs.items()):
        out.continuous(group, valid=f"valid_frac_{res}m")
    context.note("VIIRS SR: no cloud mask applied (QF layers not decoded)")


# ============================================================================
# 13–14  LAI, productivity, phenology, structure
# ============================================================================
@product("modis-lai", 13, "MODIS MOD15A2H v061 LAI/FPAR 8-day, 500 m",
         "Planetary Computer → AppEEARS fallback", conf="V")
def modis_lai(context, out):
    _modis(context, out, "modis-15A2H-061",
           {"Lai_500m": {"name": "LAI", "scale": 0.1, "valid": (0, 100), "units": "m2/m2"},
            "Fpar_500m": {"name": "FPAR", "scale": 0.01, "valid": (0, 100), "units": "1"}},
           appeears="MOD15A2H.061")


@product("modis-gpp", 13, "MODIS MOD17A2HGF v061 GPP 8-day, 500 m",
         "Planetary Computer", conf="V")
def modis_gpp(context, out):
    _modis(context, out, "modis-17A2HGF-061",
           {"Gpp_500m": {"name": "GPP", "scale": 1e-4, "valid": (0, 30000), "units": "kgC/m2/8d"},
            "PsnNet_500m": {"name": "PsnNet", "scale": 1e-4, "valid": (-30000, 30000),
                            "units": "kgC/m2/8d"}}, appeears="MOD17A2HGF.061")


@product("modis-npp", 13, "MODIS MOD17A3HGF v061 annual NPP, 500 m",
         "Planetary Computer", conf="V")
def modis_npp(context, out):
    y = context.date.year
    _modis(context, out, "modis-17A3HGF-061",
           {"Npp_500m": {"name": "NPP", "scale": 1e-4, "valid": (-30000, 32700),
                         "units": "kgC/m2/yr"}},
           days=200, datetime=f"{y}-01-01/{y}-12-31", appeears="MOD17A3HGF.061")


@product("smap-l4c", 13, "SMAP L4 carbon SPL4CMDL v008 (GPP, NEE, Rh, SOC), 9 km",
         "earthaccess", needs=("earthdata",), conf="V")
def smap_l4c(context, out):
    g = earthdata_granules(cmr.SMAP_CARBON, context, days=1, max_tiles=1)
    [f] = earthdata_download(g, context, [".h5"])   # skip the .h5.iso.xml sidecar
    with _h5(f) as h5:
        want = {"gpp": ["GPP/gpp_mean"], "nee": ["NEE/nee_mean"],
                "rh": ["RH/rh_mean"], "soc": ["SOC/soc_mean"]}
        vs = {k: p for k, ps in want.items() for p in ps if p in h5}
        lat = _first(h5, ["GEO/latitude", "cell_lat", "latitude"])
        lon = _first(h5, ["GEO/longitude", "cell_lon", "longitude"])
        r = read_ease2(h5, context.grid, vs, lat, lon)
    out.continuous(r)


PHENO = ("NumCycles", "Greenup", "MidGreenup", "Maturity", "Peak", "Senescence",
         "MidGreendown", "Dormancy", "EVI_Minimum", "EVI_Amplitude", "EVI_Area")


@product("modis-phenology", 13, "MODIS MCD12Q2 v061 phenology (year of date), 500 m",
         "AppEEARS (HDF4 unreadable on 3.15t)", needs=("earthdata",), conf="D")
def modis_pheno(context, out):
    y = context.date.year
    a = AppEEARS()
    have = set(a.layers("MCD12Q2.061"))
    layers = [l for l in PHENO if l in have] or sorted(have)[:12]
    files = a.area(context, "MCD12Q2.061", layers, dt.date(y, 1, 1), dt.date(y, 12, 31))
    epoch = (dt.date(y, 1, 1) - dt.date(1970, 1, 1)).days
    rs = []
    for f in files:
        lay = next((l for l in sorted(layers, key=len, reverse=True) if f"_{l}_" in f.name), None)
        if lay is None:
            continue
        cyc = re.search(r"_(\d)_doy", f.name)
        name = lay + (f"_c{int(cyc.group(1)) + 1}" if cyc else "")
        r = read_tif(str(f), context.grid, name=name, valid_range=(0, 32766))
        if lay not in ("NumCycles",) and not lay.startswith("EVI"):
            r.data = r.data - epoch + 1          # days since 1970 → day of year
            r.units = {name: f"day of year {y}"}
        rs.append(r)
    if not rs:
        raise FileNotFoundError("AppEEARS returned no MCD12Q2 layer")
    for r in rs:
        out.continuous(r, valid=None)


def _gedi(src, context: Context, out: Out, fields: dict, *, quality: str,
          lat="lat_lowestmode", lon="lon_lowestmode"):
    g = earthdata_granules(src, context, days=120, max_tiles=1)
    [f] = earthdata_open(g, suffixes=(".h5",))
    with _h5(f) as h5:
        plon, plat, vals = read_gedi(h5, context.grid, fields, lat=lat, lon=lon,
                                     quality=quality)
    b = bin_points(plon, plat, vals, context.grid)
    out.add("n_shots", b.pop("count"), method="count of quality shots per cell")
    for k, v in b.items():
        out.add(k, v, method="mean of quality shots", native_res_m=25.0)
    context.note(f"{len(plon)} quality shots in AOI")


@product("gedi-l2a", 14, "GEDI L2A V003 canopy height (RH50, RH98), footprints → 1 km",
         "earthaccess (HDF5 streamed)", needs=("earthdata",), conf="V")
def gedi_l2a(context, out):
    _gedi(cmr.GEDI_L2A, context, out, {"rh98": ("rh", 98), "rh50": ("rh", 50),
                                  "elev_lowestmode": ("elev_lowestmode", None)},
          quality="quality_flag")


@product("gedi-l2b", 14, "GEDI L2B V003 cover, PAI, FHD, footprints → 1 km",
         "earthaccess (HDF5 streamed)", needs=("earthdata",), conf="V")
def gedi_l2b(context, out):
    _gedi(cmr.GEDI_L2B, context, out, {"cover": ("cover", None), "pai": ("pai", None),
                                  "fhd_normal": ("fhd_normal", None)},
          quality="l2b_quality_flag", lat="geolocation/lat_lowestmode",
          lon="geolocation/lon_lowestmode")


@product("gedi-l4a", 14, "GEDI L4A V3 aboveground biomass density, footprints → 1 km",
         "earthaccess (HDF5 streamed)", needs=("earthdata",), conf="V")
def gedi_l4a(context, out):
    _gedi(cmr.GEDI_L4A, context, out, {"agbd": ("agbd", None), "agbd_se": ("agbd_se", None)},
          quality="l4_quality_flag")


@product("gedi-l3", 14, "GEDI L3 V3 gridded land-surface metrics, 1 km (2019–2025)",
         "earthaccess (COG)", needs=("earthdata",), conf="V")
def gedi_l3(context, out):
    import earthaccess
    from nasa_farm.grid.sources import login
    res = earthaccess.search_data(short_name=cmr.GEDI_L3.short_name,
                                  bounding_box=context.bbox().as_tuple(), count=50)
    if not res:
        raise FileNotFoundError("GEDI L3: no granule")
    login()
    urls = [u for r in res for u in r.data_links(access="external") if u.endswith(".tif")]
    context.note(f"GEDI L3: {len(urls)} GeoTIFF(s)")
    files = []
    for u in urls[:40]:
        files.append(download(u, context.raw / u.rsplit("/", 1)[-1],
                              sess=earthaccess.get_requests_https_session()))
    for f in files:
        try:
            out.continuous(read_tif(str(f), context.grid, name=f.stem), valid=None)
        except NoOverlap:
            continue


@product("landfire", 14, "LANDFIRE EVT, EVH, EVC, BPS, disturbance (LF 2024), 30 m",
         "LFPS REST", needs=("lfps",), conf="M")
def landfire(context, out):
    layers = os.environ.get("LANDFIRE_LAYERS", "240EVT;240EVH;240EVC;240BPS;240Dist").split(";")
    z = lfps_job(context, layers)
    d = context.raw / z.stem
    if not d.exists():
        zipfile.ZipFile(z).extractall(d)
    tifs = sorted(d.rglob("*.tif"))
    if not tifs:
        raise FileNotFoundError(f"no GeoTIFF in {z.name}")
    for t in tifs:
        lay = next((l for l in layers if l.lower() in t.name.lower()), t.stem)
        name = re.sub(r"^\d+", "", lay).lower()
        r = read_tif(str(t), context.grid, name=name, kind="categorical")
        r.classes = _top_classes(r, 15)
        out.categorical(r, name)


def _top_classes(r: Raster, n: int) -> list[int]:
    """The n classes covering most of the window (others pooled in "other").
    AOI-dependent by construction: fine for EVT/BPS whose ~1000 codes have
    no fixed short list; prefer a fixed list (CDL, NLCD) where one exists."""
    a = r.data[0]
    v = a[a != r.nodata] if r.nodata is not None else a.ravel()
    u, c = np.unique(v, return_counts=True)
    return [int(x) for x in u[np.argsort(-c)][:n]]


# ============================================================================
# 15–16  Crop type, rotation, land cover
# ============================================================================
CDL_CLASSES = {1: "corn", 5: "soybeans", 24: "winter_wheat", 36: "alfalfa",
               37: "other_hay", 61: "fallow", 111: "water", 121: "dev_open",
               122: "dev_low", 123: "dev_med", 124: "dev_high", 141: "deciduous_forest",
               176: "grass_pasture", 190: "woody_wetlands", 195: "herb_wetlands"}
CDL_URL = "/vsicurl/https://data.source.coop/tyler/usda-nass-cdl/{y}/{y}_30m_cdls.tif"


def _cdl(context: Context, year: int) -> Raster:
    r = read_tif(CDL_URL.format(y=year), context.grid, name="cdl", kind="categorical",
                 classes=list(CDL_CLASSES), class_names=CDL_CLASSES)
    r.nodata = 0
    return r


@product("cdl", 15, "Cropland Data Layer (year of date), 30 m",
         "Source Cooperative COG (windowed)", conf="D")
def cdl(context, out):
    y = min(context.date.year, 2025)
    out.categorical(_cdl(context, y), "cdl")
    out.attrs["cdl_year"] = y


CROP_GROUPS = {"corn": {1, 225, 226, 237, 241}, "soybeans": {5, 26, 239, 240, 241, 254},
               "wheat": {22, 23, 24, 26, 225, 230, 234, 236, 238},
               "cotton": {2, 232, 238, 239}}


@product("crop-frequency", 15, "Crop frequency 2008→year (years in corn/soy/wheat/cotton)",
         "derived from CDL COGs", conf="D")
def crop_frequency(context, out):
    """NASS Crop Frequency Layers recomputed: per 30 m pixel the number of
    years in each crop group (double crops count for both), then the cell
    mean. Identical to summing each year's 1 km class fraction."""
    y1 = min(context.date.year, 2025)
    tot = {k: np.zeros(context.grid.shape, np.float32) for k in CROP_GROUPS}
    years = 0
    for y in range(2008, y1 + 1):
        r = _cdl(context, y)
        a = r.data[0]
        for k, codes in CROP_GROUPS.items():
            m = Raster(np.where(a == 0, np.nan, np.isin(a, list(codes))).astype(np.float32)[None],
                       r.transform, r.crs, [k])
            v, _, _ = regrid_continuous(m, context.grid, "average")
            tot[k] += np.nan_to_num(v[0])
        years += 1
    for k, v in tot.items():
        out.add(f"years_{k}", v, units=f"years of {years} (2008–{y1})",
                method="sum over years of 1 km class fraction")


@product("csb", 15, "Crop Sequence Boundaries 2018–2025 (fields + 8-year rotation)",
         "NASS national zip (large) + pyogrio bbox read", conf="M")
def csb(context, out):
    try:
        import pyogrio
        import shapely
    except ImportError as e:
        raise Skip("CSB needs pyogrio + shapely (uv add pyogrio shapely)") from e
    url = os.environ.get("CSB_URL", "https://www.nass.usda.gov/Research_and_Science/"
                         "Crop-Sequence-Boundaries/datasets/NationalCSB_2018-2025_rev23.zip")
    z = download(url, context.raw / url.rsplit("/", 1)[-1], timeout=7200)
    names = zipfile.ZipFile(z).namelist()
    src = next((n.split(".gdb/")[0] + ".gdb" for n in names if ".gdb/" in n), None) \
        or next((n for n in names if n.endswith((".gpkg", ".parquet", ".shp"))), None)
    if src is None:
        raise FileNotFoundError(f"no vector dataset in {z.name}: {names[:10]}")
    path = f"/vsizip/{z}/{src}"
    from pyproj import CRS, Transformer
    info = pyogrio.read_info(path)
    from rasterio.warp import transform_bounds
    g = context.grid
    bb = transform_bounds(g.crs, info["crs"], g.west, g.south, g.east, g.north)
    meta, tbl = pyogrio.read_arrow(path, bbox=bb)
    gcol = meta.get("geometry_name") or "wkb_geometry"
    geoms = shapely.from_wkb(tbl.column(gcol).to_numpy(zero_copy_only=False))
    if not CRS.from_user_input(info["crs"]).equals(CRS.from_user_input(g.crs)):
        tr = Transformer.from_crs(info["crs"], g.crs, always_xy=True)
        geoms = shapely.transform(geoms, lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1])))
    cols = {c.upper(): c for c in tbl.column_names}
    yrs = sorted(c for c in cols if re.fullmatch(r"CDL\d{4}", c))
    cdl = np.column_stack([tbl.column(cols[c]).to_numpy(zero_copy_only=False) for c in yrs]) \
        if yrs else np.zeros((len(geoms), 0))
    corn = np.isin(cdl, list(CROP_GROUPS["corn"])).sum(1)
    soy = np.isin(cdl, list(CROP_GROUPS["soybeans"])).sum(1)
    n = max(len(yrs), 1)
    rot = ((corn + soy == n) & (corn >= n // 2 - 1) & (soy >= n // 2 - 1)).astype(float)
    area_ha = shapely.area(geoms) / 1e4
    out.add("field_cover", rasterize_cover(list(geoms), g), units="1",
            method="area fraction of cell in CSB fields (100 m sub-grid)")
    cx, cy = shapely.get_coordinates(shapely.centroid(geoms)).T
    row, col, inside = g.cell_index(cx, cy)
    cnt = np.zeros(g.shape, np.float32)
    np.add.at(cnt, (row[inside], col[inside]), 1)
    out.add("n_fields", cnt, method="field centroids per cell")
    for name, v, u in (("field_area_ha", area_ha, "ha"),
                       ("corn_years", corn, f"years of {n}"),
                       ("soy_years", soy, f"years of {n}"),
                       ("corn_soy_rotation", rot, "1")):
        out.add(name, rasterize_values(list(geoms), v, g), units=u,
                method="area-weighted mean over field polygons")
    context.note(f"CSB {src}: {len(geoms)} fields, years {yrs[:1]}–{yrs[-1:]}")


NLCD = {11: "water", 12: "ice", 21: "dev_open", 22: "dev_low", 23: "dev_med",
        24: "dev_high", 31: "barren", 41: "deciduous", 42: "evergreen",
        43: "mixed_forest", 52: "shrub", 71: "grassland", 81: "pasture_hay",
        82: "cultivated_crops", 90: "woody_wetlands", 95: "herb_wetlands"}


@product("nlcd", 16, "Annual NLCD Collection 1 land cover (year of date), 30 m",
         "USGS S3 COG (requester pays)", needs=("aws",), conf="D")
def nlcd(context, out):
    y = min(context.date.year, 2024)
    names = s3_list("usgs-landcover/annual-nlcd/c1/v0/cu/mosaic/",
                    rf"LndCov_{y}.*\.tif$", requester_pays=True)
    if not names:
        raise FileNotFoundError(f"no Annual NLCD LndCov {y} under the S3 prefix")
    r = read_tif(f"/vsis3/{names[0]}", context.grid, name="nlcd", kind="categorical",
                 classes=list(NLCD), class_names=NLCD, env=REQUESTER_PAYS)
    r.nodata = 250 if r.nodata is None else r.nodata
    out.categorical(r, "nlcd")
    context.note(names[0])


BA = {0: "unburned", 1: "burned"}


@product("landsat-ba", 16, "Landsat C2 L3 burned area classification, 30 m",
         "USGS STAC + requester-pays S3", needs=("aws",), conf="M")
def landsat_ba(context, out):
    _usgs_l3(context, out, "landsat-c2l3-ba", r"^bc$|burn_?class", "burn", list(BA), BA)


@product("landsat-fsca", 16, "Landsat C2 L3 fractional snow-covered area, 30 m",
         "USGS STAC + requester-pays S3", needs=("aws",), conf="M")
def landsat_fsca(context, out):
    import pystac_client
    cat = pystac_client.Client.open("https://landsatlook.usgs.gov/stac-server")
    w = context.window(16)
    items = list(cat.search(collections=["landsat-c2l3-fsca"], bbox=context.bbox().as_tuple(),
                            datetime=f"{w.start}/{w.end}", max_items=20).items())
    if not items:
        raise FileNotFoundError("landsat-c2l3-fsca: no item")
    it = min(items, key=lambda i: abs((i.datetime.date() - context.date).days))
    key = next(k for k in it.assets if re.search(r"snow|fsca|^vs$", k, re.I))
    a = it.assets[key]
    alt = (a.extra_fields.get("alternate", {}).get("s3", {}) or {}).get("href")
    r = read_tif(alt.replace("s3://", "/vsis3/") if alt else a.href, context.grid,
                 name="fsca", valid_range=(0, 1000), scale=0.001,
                 env=REQUESTER_PAYS if alt else None)
    out.continuous(r)
    context.note(it.id)


# ============================================================================
# 17–20  Terrain, soils, suitability
# ============================================================================
@product("3dep", 17, "USGS 3DEP 10 m DEM → mean elevation, slope, roughness",
         "Planetary Computer", conf="V")
def dep3(context, out):
    from rasterio.crs import CRS
    items = planetary_items("3dep-seamless", context, None, query={"gsd": {"eq": 10}})
    rs, zref = [], None
    for it in items:
        r = read_tif(planetary_href(it, "data"), context.grid, name="elevation", units="m")
        z = r.data[0].astype(np.float64)
        if zref is None:
            zref = float(np.nanmean(z))   # centring keeps z² precise in float32
        dx, dy = abs(r.transform.a), abs(r.transform.e)
        if CRS.from_user_input(r.crs).is_geographic:   # 3DEP seamless: NAD83 lon/lat
            lat = r.transform.f + r.transform.e * (np.arange(z.shape[0]) + 0.5)
            dxm = (dx * 111_320.0 * np.cos(np.radians(lat)))[:, None]
            dym = dy * 110_540.0
        else:
            dxm, dym = dx, dy
        gy, gx = np.gradient(z)
        slope = np.degrees(np.arctan(np.hypot(gx / dxm, gy / dym)))
        rs.append(Raster(np.stack([z, slope, (z - zref) ** 2]).astype(np.float32),
                         r.transform, r.crs, ["elevation", "slope_deg", "_zc2"],
                         units={"elevation": "m", "slope_deg": "degree"}))
    out.continuous(rs)
    zc2 = out.vars.pop("_zc2")[1].astype(np.float64)
    zm = out.vars["elevation"][1].astype(np.float64) - zref
    out.add("elevation_std", np.sqrt(np.maximum(zc2 - zm ** 2, 0)), units="m",
            method="sqrt(mean(z²) − mean(z)²) over 10 m pixels")
    context.note("slope computed at 10 m before averaging (the slope of a 1 km "
             "mean DEM would be biased low)")


VALU1 = ("aws0_100", "aws0_150", "soc0_30", "soc0_100", "rootznemc", "rootznaws",
         "droughty", "nccpi3all", "nccpi3corn", "nccpi3soy", "pctearthmc")


@product("gssurgo-valu1", 18, "gSSURGO Valu1 (current SDA values on the 30 m map-unit raster)",
         "PC gnatsgo mukey raster + Soil Data Access", conf="M")
def gssurgo_valu1(context, out):
    [it] = planetary_items("gnatsgo-rasters", context, None, max_tiles=1)
    mk = read_tif(planetary_href(it, "mukey"), context.grid, name="mukey", kind="categorical")
    a = mk.data[0].astype(np.int64)
    keys = sorted(int(k) for k in np.unique(a) if k > 0 and k != mk.nodata)
    rows = []
    for i in range(0, len(keys), 500):
        chunk = ",".join(str(k) for k in keys[i:i + 500])
        rows += sda_query(f"SELECT mukey, {', '.join(VALU1)} FROM valu1 "
                          f"WHERE mukey IN ({chunk})")
    lut = {int(r["mukey"]): r for r in rows}
    miss = np.isin(a, [k for k in keys if k not in lut]).mean()
    context.note(f"{len(keys)} map units in AOI; {miss:.1%} of pixels have a mukey "
             "absent from current SDA (raster 2020 vintage → those stay NaN)")
    idx = np.searchsorted(np.array(keys), a).clip(0, max(len(keys) - 1, 0))
    bands = []
    for col in VALU1:
        vals = np.array([float(lut[k][col]) if k in lut and lut[k][col] not in (None, "")
                         else np.nan for k in keys], np.float32)
        b = vals[idx] if len(keys) else np.full(a.shape, np.nan, np.float32)
        b[(a <= 0) | ~np.isin(a, keys)] = np.nan
        bands.append(b)
    out.continuous(Raster(np.stack(bands), mk.transform, mk.crs, list(VALU1)))


@product("statsgo2", 18, "STATSGO2 map-unit aggregates (SDA spatial query), 1:250 000",
         "Soil Data Access", conf="M")
def statsgo2(context, out):
    try:
        import shapely
        from shapely import wkt
    except ImportError as e:
        raise Skip("STATSGO2 needs shapely") from e
    b = context.bbox(1.0)
    poly = (f"POLYGON(({b.west} {b.south},{b.east} {b.south},{b.east} {b.north},"
            f"{b.west} {b.north},{b.west} {b.south}))")
    rows = sda_query(
        "SELECT mp.mukey, mp.mupolygongeo.STAsText() AS wkt, ma.aws0150wta, "
        "ma.slopegraddcp, ma.wtdepannmin, ma.brockdepmin "
        "FROM mupolygon mp JOIN muaggatt ma ON ma.mukey = mp.mukey "
        "WHERE mp.areasymbol = 'US' AND mp.mupolygongeo.STIntersects("
        f"geometry::STGeomFromText('{poly}', 4326)) = 1")
    if not rows:
        raise FileNotFoundError("SDA returned no STATSGO2 polygon (areasymbol 'US')")
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", context.grid.crs, always_xy=True)
    geoms = [shapely.transform(wkt.loads(r["wkt"]),
                               lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1])))
             for r in rows]
    for col, u in (("aws0150wta", "cm"), ("slopegraddcp", "%"),
                   ("wtdepannmin", "cm"), ("brockdepmin", "cm")):
        v = [float(r[col]) if r[col] not in (None, "") else np.nan for r in rows]
        out.add(col, rasterize_values(geoms, v, context.grid), units=u,
                method="area-weighted mean over map-unit polygons")
    context.note(f"{len(rows)} STATSGO2 polygons")


SOLUS_PROPS = ("claytotal", "sandtotal", "soc", "ph1to1h2o", "dbovendry", "cec7")
SOLUS_DEPTHS = (0, 30)


@product("solus100", 18, "SOLUS100 soil properties (prediction), 100 m",
         "public GCS bucket (COG)", conf="D")
def solus100(context, out):
    names = gcs_list("solus100pub", "", r"\.tif$")
    if not names:
        raise FileNotFoundError("solus100pub listing empty")
    picked = []
    for p in SOLUS_PROPS:
        for d in SOLUS_DEPTHS:
            rx = re.compile(rf"(^|/){p}_{d}_cm_p\.tif$")
            n = next((x for x in names if rx.search(x)), None)
            if n:
                picked.append((f"{p}_{d}cm", n))
    if not picked:
        raise FileNotFoundError("no SOLUS file matched <prop>_<depth>_cm_p.tif; "
                                f"examples: {names[:8]}")
    for name, n in picked:
        out.continuous(read_tif(gcs_href("solus100pub", n), context.grid, name=name), valid=None)
    context.note(f"SOLUS100: {len(picked)} layers")


SOILGRIDS = {"clay": 0.1, "sand": 0.1, "soc": 0.1, "phh2o": 0.1, "bdod": 0.01, "cec": 0.1}
"""prop → factor to conventional units (%, %, g/kg, pH, g/cm³, cmol(c)/kg)."""
IGH = "+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"


@product("soilgrids", 18, "SoilGrids v2 mean properties 0–5 and 15–30 cm, 250 m",
         "ISRIC WCS", conf="V")
def soilgrids(context, out):
    from pyproj import Transformer
    g = context.grid.buffered(2.0)
    t = Transformer.from_crs(g.crs, IGH, always_xy=True)
    xs, ys = t.transform([g.west, g.east, g.west, g.east], [g.south, g.south, g.north, g.north])
    s = session()
    for prop, f in SOILGRIDS.items():
        for depth in ("0-5cm", "15-30cm"):
            cov = f"{prop}_{depth}_mean"
            p = context.raw / f"{cov}.tif"
            crs = "http://www.opengis.net/def/crs/EPSG/0/152160"
            download("https://maps.isric.org/mapserv", p, sess=s, params=[
                ("map", f"/map/{prop}.map"), ("SERVICE", "WCS"), ("VERSION", "2.0.1"),
                ("REQUEST", "GetCoverage"), ("COVERAGEID", cov), ("FORMAT", "image/tiff"),
                ("SUBSET", f"X({min(xs):.0f},{max(xs):.0f})"),
                ("SUBSET", f"Y({min(ys):.0f},{max(ys):.0f})"),
                ("SUBSETTINGCRS", crs), ("OUTPUTCRS", crs)])
            r = read_tif(str(p), context.grid, name=f"{prop}_{depth.replace('-', '_')}",
                         scale=f, valid_range=(0, 32000))
            out.continuous(r, valid=None)


@product("nccpi", 19, "NCCPI v3 (corn, soy, small grains, all) — gNATSGO 2020, 30 m",
         "Planetary Computer", conf="V")
def nccpi(context, out):
    [it] = planetary_items("gnatsgo-rasters", context, None, max_tiles=1)
    for k in ("nccpi3corn", "nccpi3soy", "nccpi3sg", "nccpi3all"):
        out.continuous(read_tif(planetary_href(it, k), context.grid, name=k, valid_range=(0, 1)),
                       valid=None)


@product("gaez-v5", 19, "GAEZ v5 suitability / attainable yield (maize, soy, wheat), 30″",
         "public GCS bucket", conf="M")
def gaez(context, out):
    rx = os.environ.get("GAEZ_PATTERN", r"(?i)/(RES05)[^/]*/.*(MZE|MAIZ|SOY|WHE).*\.tif$")
    names = gcs_list("fao-gismgr-gaez-v5-data", "DATA/GAEZ-V5/MAPSET/", rx)
    if not names:
        sample = gcs_list("fao-gismgr-gaez-v5-data", "DATA/GAEZ-V5/MAPSET/", limit=40)
        raise FileNotFoundError("no GAEZ v5 file matched GAEZ_PATTERN; first objects: "
                                + "; ".join(sample[:12]))
    for n in names[:8]:
        out.continuous(read_tif(gcs_href("fao-gismgr-gaez-v5-data", n), context.grid,
                                name=Path(n).stem), valid=None)
    context.note(f"GAEZ v5: {min(len(names), 8)} of {len(names)} matching files")


# ============================================================================
# shared patterns
# ============================================================================
def _ecostress(src, wanted: dict[str, str], context: Context, out: Out, days: int = 20) -> None:
    """ECOSTRESS tiled COGs: the wanted layers of the acquisition closest to
    the date (all MGRS tiles of that pass), cloud mask applied when the
    tile ships one (`_cloud.tif`, 1 = cloud)."""
    g = earthdata_granules(src, context, days)
    rasters = []
    for gr in g:
        files = earthdata_download([gr], context, [f"_{s}.tif" for s in wanted] + ["_cloud.tif"])
        layers = [(wanted[s], f) for s in wanted for f in files
                  if f.name.endswith(f"_{s}.tif")]
        if not layers:
            continue
        r = _stack([read_tif(str(f), context.grid, name=n) for n, f in layers])
        cloud = next((f for f in files if f.name.endswith("_cloud.tif")), None)
        if cloud is not None:
            c = read_tif(str(cloud), context.grid, name="cloud", kind="categorical")
            r = _mask(r, _match(c, r) == 1)
        rasters.append(r)
    if not rasters:
        raise FileNotFoundError(f"{src.short_name}: none of {list(wanted)} in granule")
    out.continuous(rasters)


def _latlon_granule(src, context: Context, out: Out, *, days: int, pick: str | None = None,
                    buffer_deg: float = 0.3) -> None:
    g = earthdata_granules(src, context, days, pick=pick, max_tiles=1)
    [f] = earthdata_download(g, context)
    with _xr(f) as ds:
        r = read_latlon(ds, context.grid, buffer_deg=buffer_deg,
                        skip=("time_bnds", "lat_bnds", "lon_bnds"))
    out.continuous(r, valid=None)


def _points(context: Context, out: Out, df, *, pivot: bool) -> None:
    """Point table → `out.points` (+ per-cell site count on the grid)."""
    from pyproj import Transformer
    if df.empty:
        raise FileNotFoundError("no site with data in the AOI on that date")
    sites = df.groupby("site")[["lon", "lat"]].first().reset_index()
    x, y = Transformer.from_crs("EPSG:4326", context.grid.crs, always_xy=True).transform(
        sites["lon"].to_numpy(float), sites["lat"].to_numpy(float))
    row, col, inside = context.grid.cell_index(np.asarray(x), np.asarray(y))
    sites = sites.assign(row=row, col=col, inside=inside)
    if pivot:
        wide = df.pivot_table(index="site", columns="var", values="value", aggfunc="mean")
        table = sites.merge(wide.reset_index(), on="site", how="left")
    else:
        table = df.merge(sites[["site", "row", "col", "inside"]], on="site", how="left")
    out.points = table
    cnt = np.zeros(context.grid.shape, np.float32)
    s = sites[sites.inside]
    np.add.at(cnt, (s.row.to_numpy(), s.col.to_numpy()), 1)
    out.add("n_sites", cnt, method="sites per cell")
