"""Fetchers — "get one file of product P for date D over the grid".

One helper per access family of `claude/download-access.md` §1:
Earthdata (earthaccess), AppEEARS, Planetary Computer STAC, public
HTTPS / GCS / S3 (incl. requester-pays), and the REST APIs (OpenET,
LANDFIRE LFPS, Soil Data Access, USGS, CDS). All downloads are cached
under `context.raw/<product>/` and skipped when already present, so re-runs
only regrid.

Credentials are read from the usual places, never from code:
Earthdata `~/.netrc` or EARTHDATA_USERNAME/PASSWORD · OPENET_API_KEY ·
`~/.cdsapirc` · AWS default chain (requester-pays buckets) ·
LFPS_EMAIL · USGS_API_KEY (optional, raises rate limits).
"""

from __future__ import annotations

import datetime as dt
import json
import netrc
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from nasa_farm.acquire.base import Bounds, Temporal
from nasa_farm.grid.target import TargetGrid

if TYPE_CHECKING:
    from nasa_farm.grid.sources import Collection

UA = {"User-Agent": "nasa-farm/0.1 (Space Apps 2026)"}


@dataclass
class Context:
    """Run context shared by every product function."""

    grid: TargetGrid
    date: dt.date
    raw_root: Path
    key: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def raw(self) -> Path:
        p = self.raw_root / self.key
        p.mkdir(parents=True, exist_ok=True)
        return p

    def bbox(self, buffer_km: float = 0.0) -> Bounds:
        return self.grid.bbox_ll(buffer_km)

    def window(self, days: int) -> Temporal:
        return Temporal(str(self.date - dt.timedelta(days)),
                        str(self.date + dt.timedelta(days)))

    def note(self, msg: str) -> None:
        self.notes.append(msg)


class Skip(RuntimeError):
    """Product cannot run here (missing credential / dependency) — reported
    as `skipped`, not `failed`."""


# --- credentials -----------------------------------------------------------
def _netrc_login(host: str) -> tuple[str, str] | None:
    try:
        a = netrc.netrc().authenticators(host)
    except (FileNotFoundError, netrc.NetrcParseError):
        return None
    return (a[0], a[2]) if a else None


def earthdata_userpass() -> tuple[str, str] | None:
    u, p = os.environ.get("EARTHDATA_USERNAME"), os.environ.get("EARTHDATA_PASSWORD")
    return (u, p) if u and p else _netrc_login("urs.earthdata.nasa.gov")


def available() -> dict[str, bool]:
    """Which credential families are configured on this machine."""
    aws = bool(os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE")
               or (Path.home() / ".aws" / "credentials").exists())
    return {
        "earthdata": earthdata_userpass() is not None,
        "openet": bool(os.environ.get("OPENET_API_KEY")),
        "cds": (Path.home() / ".cdsapirc").exists() or bool(os.environ.get("CDSAPI_KEY")),
        "aws": aws,
        "lfps": bool(os.environ.get("LFPS_EMAIL")),
    }


# --- generic HTTP ------------------------------------------------------------
def session():
    import requests
    s = requests.Session()
    s.headers.update(UA)
    return s


def download(url: str, dest: Path, *, sess=None, params=None, headers=None,
             method: str = "GET", json_body=None, timeout: int = 600) -> Path:
    """Stream a URL to `dest` (cached: skipped if it exists and is non-empty)."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    s = sess or session()
    with s.request(method, url, params=params, headers=headers, json=json_body,
                   stream=True, timeout=timeout) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        tmp.rename(dest)
    return dest


def download_ranged(url: str, dest: Path, *, sess=None, chunk: int = 64 << 20,
                    timeout: int = 300, context: "Context | None" = None) -> Path:
    """Large file fetched as successive HTTP Range requests, resumable.

    For servers that fail a single whole-file GET but serve byte ranges
    (NASS answers HTTP 500 to a plain GET of the 3.8 GB CSB zip, 206 to
    ranges). Each chunk is retried on transient errors and length-checked;
    the `.part` file is kept on failure so the next run resumes from it."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    s = sess or session()
    h = s.head(url, allow_redirects=True, timeout=60)
    h.raise_for_status()
    total = int(h.headers.get("Content-Length") or 0)
    if not total or h.headers.get("Accept-Ranges", "").lower() != "bytes":
        return download(url, dest, sess=s, timeout=timeout)   # no ranges: plain GET
    tmp = dest.with_suffix(dest.suffix + ".part")
    done = tmp.stat().st_size if tmp.exists() else 0
    if done > total:                                  # stale partial of another file
        tmp.unlink()
        done = 0
    if done and context is not None:
        context.note(f"resuming {dest.name} at {done / total:.0%}")
    with open(tmp, "ab") as f:
        while done < total:
            lo, hi = done, min(done + chunk, total) - 1

            def get(lo=lo, hi=hi) -> bytes:
                r = s.get(url, headers={"Range": f"bytes={lo}-{hi}"}, timeout=timeout)
                r.raise_for_status()
                if r.status_code != 206:
                    raise RuntimeError(f"{url}: server ignored the Range header "
                                       f"(HTTP {r.status_code})")
                if len(r.content) != hi - lo + 1:
                    raise ConnectionError(f"short range read {len(r.content)} of {hi - lo + 1} B")
                return r.content

            data = retry_transient(get, attempts=5, wait=10, what=f"{dest.name} @{lo}",
                                   context=context)
            f.write(data)
            f.flush()
            done += len(data)
    if tmp.stat().st_size != total:
        raise ConnectionError(f"{dest.name}: {tmp.stat().st_size} of {total} B")
    tmp.rename(dest)
    return dest


def find_urls(obj: Any) -> list[str]:
    """Every http(s) URL anywhere in a JSON payload (for APIs whose result
    key is not documented consistently)."""
    out: list[str] = []
    if isinstance(obj, str):
        out += re.findall(r"https?://[^\s\"']+", obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            out += find_urls(v)
    elif isinstance(obj, list):
        for v in obj:
            out += find_urls(v)
    return out


def poll(fn, *, every: float = 10.0, timeout: float = 1800.0, what: str = "job"):
    """Call fn() until it returns a truthy value."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        v = fn()
        if v:
            return v
        time.sleep(every)
    raise TimeoutError(f"{what} not ready after {timeout:.0f} s")


def is_transient(e: BaseException) -> bool:
    """Server-side or network hiccup worth retrying: HTTP 429/5xx (aiohttp,
    as raised by fsspec streams, or requests) and dropped connections."""
    import asyncio
    status = getattr(e, "status", None)
    if status is None and getattr(e, "response", None) is not None:
        status = getattr(e.response, "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    try:
        import requests
        if isinstance(e, (requests.ConnectionError, requests.Timeout,
                          requests.exceptions.ChunkedEncodingError)):
            return True
    except ImportError:
        pass
    try:
        import aiohttp
        if isinstance(e, (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError)):
            return True
    except ImportError:
        pass
    try:
        from fsspec.exceptions import FSTimeoutError
        if isinstance(e, FSTimeoutError):
            return True
    except ImportError:
        pass
    return isinstance(e, (asyncio.TimeoutError, ConnectionError, TimeoutError))


def retry_transient(fn, *, attempts: int = 4, wait: float = 30.0,
                    what: str = "read", context: "Context | None" = None):
    """fn() retried with linear backoff on transient errors only; any other
    error (missing variable, bad path…) is raised at once. For long
    streamed reads, where one 502 among hundreds of range requests would
    otherwise fail the whole product."""
    for k in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — filtered just below
            if not is_transient(e) or k + 1 == attempts:
                raise
            if context is not None:
                context.note(f"{what}: {type(e).__name__} "
                             f"{getattr(e, 'status', '')} → retry {k + 1}/{attempts - 1}")
            time.sleep(wait * (k + 1))


# --- granule choice ------------------------------------------------------------
_STAMP = re.compile(r"(?<!\d)((?:19|20)\d{5}T\d{4}|(?:19|20)\d{6}T\d{4})")


def _day_of(gid: str, raw: Any = None) -> int | None:
    from nasa_farm.grid.sources import granule_day
    return granule_day(gid, raw)


def _to_date(yyyymmdd: int) -> dt.date:
    s = str(yyyymmdd)
    return dt.date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def pick_closest(items: Sequence[Any], ids: Sequence[str], when: dt.date, *,
                 days: Sequence[int | None] | None = None,
                 cloud: Sequence[float | None] | None = None,
                 max_tiles: int = 4) -> list[int]:
    """Indices of the acquisition closest to `when` — all tiles sharing its
    time stamp (mosaic), least cloudy first on ties."""
    if not items:
        return []
    days = days or [_day_of(i) for i in ids]
    keys = []
    for k, (gid, d) in enumerate(zip(ids, days)):
        m = _STAMP.search(gid)
        stamp = m.group(1) if m else str(d)
        dist = abs((_to_date(d) - when).days) if d else 10 ** 6
        cc = (cloud[k] if cloud and cloud[k] is not None else 0.0)
        keys.append((dist, cc, stamp, k))
    best = min(keys)
    same = [k for (dist, cc, stamp, k) in sorted(keys) if stamp == best[2]]
    return same[:max_tiles]


# --- NASA Earthdata ------------------------------------------------------------
def earthdata_granules(src: "Collection", context: Context, days: int, *, pick: str | None = None,
                cloud_max: int | None = None, max_tiles: int = 4,
                buffer_km: float = 0.0) -> list[Any]:
    """CMR search over the AOI within ±days → the closest acquisition."""
    import earthaccess
    w = context.window(days)
    kw: dict[str, Any] = {"short_name": src.short_name,
                          "temporal": (w.start, w.end),
                          "bounding_box": context.bbox(buffer_km).as_tuple(),
                          "count": 500}
    if src.version:
        kw["version"] = src.version
    if cloud_max is not None:
        kw["cloud_cover"] = (0, cloud_max)
    res = earthaccess.search_data(**kw)
    from nasa_farm.grid.sources import granule_id
    ids = [granule_id(r) for r in res]
    if pick:
        keep = [k for k, g in enumerate(ids) if re.search(pick, g)]
        res, ids = [res[k] for k in keep], [ids[k] for k in keep]
    if not res:
        raise FileNotFoundError(f"{src.short_name} v{src.version}: no granule "
                                f"within ±{days} d of {context.date}")
    cc = []
    for r in res:
        try:
            cc.append(float(r["umm"].get("CloudCover")))
        except (TypeError, ValueError, KeyError):
            cc.append(None)
    sel = pick_closest(res, ids, context.date,
                       days=[_day_of(g, r) for g, r in zip(ids, res)],
                       cloud=cc, max_tiles=max_tiles)
    chosen = [res[k] for k in sel]
    context.note(f"{src.short_name} v{src.version}: " +
             ", ".join(ids[k] for k in sel))
    return chosen


def earthdata_download(granules: Iterable[Any], context: Context,
                suffixes: Sequence[str] | None = None) -> list[Path]:
    """Download the granules' data files (only those ending with one of
    `suffixes`, e.g. ECOSTRESS "_LST.tif") into context.raw."""
    import earthaccess
    from nasa_farm.grid.sources import login
    login()
    urls = []
    for g in granules:
        for u in g.data_links(access="external"):
            if suffixes is None or any(u.endswith(s) for s in suffixes):
                urls.append(u)
    if not urls:
        raise FileNotFoundError(f"no data link ending with {suffixes}")
    todo = [u for u in urls if not (context.raw / u.rsplit("/", 1)[-1]).exists()]
    if todo:
        earthaccess.download(todo, str(context.raw))
    return [context.raw / u.rsplit("/", 1)[-1] for u in urls]


def earthdata_open(granules: Iterable[Any],
                   suffixes: Sequence[str] | None = None) -> list[Any]:
    """HTTPS file-likes (fsspec) — HDF5 read in place, only the chunks of
    the requested window travel (SMAP L4 146 MB → a few MB).

    A granule usually carries sidecar files (SPL4SMGP: `.h5`, `.qa`,
    `.h5.iso.xml`), and `earthaccess.open` opens all of them: pass `suffixes`
    (e.g. (".h5",)) to open only the data files, one handle per match."""
    import earthaccess
    from nasa_farm.grid.sources import login
    login()
    if suffixes is None:
        return earthaccess.open(list(granules))
    urls = [u for g in granules for u in g.data_links(access="external")
            if any(u.endswith(s) for s in suffixes)]
    if not urls:
        raise FileNotFoundError(f"no data link ending with {tuple(suffixes)}")
    return earthaccess.open(urls)


# --- AppEEARS -----------------------------------------------------------------
class AppEEARS:
    """NASA AppEEARS area subsets (GeoTIFF). Used where the granule format
    cannot be read here (MODIS HDF4 on CPython 3.15t: no HDF4 driver in the
    rasterio wheel, pyhdf does not build) or where whole tiles are wasteful.
    Asynchronous: a task takes minutes to an hour."""

    base = "https://appeears.earthdatacloud.nasa.gov/api"
    _token: str | None = None          # one login per process, shared by products
    _failed: str | None = None         # …and one failure: later products fail fast
    _token_lock = threading.Lock()

    def __init__(self) -> None:
        self.s = session()
        self.s.headers["Authorization"] = f"Bearer {self._login()}"

    @classmethod
    def _login(cls, attempts: int = 3) -> str:
        """POST /login, retried: the endpoint intermittently answers HTTP 504
        (gateway timeout at 60 s) while the rest of the API is up."""
        import requests
        with cls._token_lock:
            if cls._token:
                return cls._token
            if cls._failed:
                raise ConnectionError(cls._failed)
            up = earthdata_userpass()
            if up is None:
                raise Skip("AppEEARS needs Earthdata credentials")
            last = ""
            for k in range(attempts):
                try:
                    r = session().post(f"{cls.base}/login", auth=up, timeout=120)
                    if r.ok:
                        cls._token = r.json()["token"]
                        return cls._token
                    if r.status_code in (401, 403):
                        raise PermissionError(f"AppEEARS login refused: HTTP {r.status_code} "
                                              "(check the Earthdata credentials)")
                    last = f"HTTP {r.status_code}"
                except (requests.Timeout, requests.ConnectionError) as e:
                    last = type(e).__name__
                if k + 1 < attempts:
                    time.sleep(30 * (k + 1))
            cls._failed = (f"AppEEARS /login failed {attempts}× ({last}). Bad credentials get "
                           "a fast 401, so a 504 means the account was accepted but not set up "
                           "for AppEEARS: sign in once at https://appeears.earthdatacloud.nasa.gov "
                           "(authorizes the app in Earthdata Login), then retry")
            raise ConnectionError(cls._failed)

    def layers(self, product: str) -> list[str]:
        r = self.s.get(f"{self.base}/product/{product}", timeout=60)
        r.raise_for_status()
        return list(r.json())

    def area(self, context: Context, product: str, layers: Sequence[str],
             start: dt.date, end: dt.date, name: str | None = None,
             timeout: float = 3600) -> list[Path]:
        cache = context.raw / f"appeears_{product}_{start}_{end}.json"
        if cache.exists():
            files = [context.raw / f for f in json.loads(cache.read_text())]
            if all(f.exists() for f in files):
                return files
        b = context.bbox(buffer_km=2.0)
        poly = [[b.west, b.south], [b.east, b.south], [b.east, b.north],
                [b.west, b.north], [b.west, b.south]]
        task = {
            "task_type": "area",
            "task_name": name or f"nasa-farm-{product}-{start}",
            "params": {
                "dates": [{"startDate": start.strftime("%m-%d-%Y"),
                           "endDate": end.strftime("%m-%d-%Y")}],
                "layers": [{"product": product, "layer": l} for l in layers],
                "geo": {"type": "FeatureCollection", "features": [{
                    "type": "Feature", "properties": {},
                    "geometry": {"type": "Polygon", "coordinates": [poly]}}]},
                "output": {"format": {"type": "geotiff"},
                           "projection": "geographic"},
            },
        }
        r = self.s.post(f"{self.base}/task", json=task, timeout=120)
        r.raise_for_status()
        tid = r.json()["task_id"]
        context.note(f"AppEEARS task {tid} ({product})")

        def done():
            st = self.s.get(f"{self.base}/task/{tid}", timeout=60).json()
            if st.get("status") == "error":
                raise RuntimeError(f"AppEEARS task {tid}: {st}")
            return st.get("status") == "done"

        poll(done, every=30, timeout=timeout, what=f"AppEEARS {tid}")
        files = self.s.get(f"{self.base}/bundle/{tid}", timeout=60).json()["files"]
        out = []
        for f in files:
            if not f["file_name"].endswith(".tif"):
                continue
            dest = context.raw / Path(f["file_name"]).name
            download(f"{self.base}/bundle/{tid}/{f['file_id']}", dest, sess=self.s)
            out.append(dest)
        cache.write_text(json.dumps([p.name for p in out]))
        return out


# --- Microsoft Planetary Computer ------------------------------------------------
PLANETARY_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


def planetary_items(collection: str, context: Context, days: int | None, *,
             query: dict | None = None, max_items: int = 200,
             datetime: str | None = None, max_tiles: int = 4) -> list[Any]:
    """STAC search → the acquisition closest to context.date (all its tiles),
    least cloudy first. days=None → no date filter (static collections)."""
    import planetary_computer
    import pystac_client
    cat = pystac_client.Client.open(PLANETARY_STAC, modifier=planetary_computer.sign_inplace)
    if datetime is None and days is not None:
        w = context.window(days)
        datetime = f"{w.start}/{w.end}"
    items = list(cat.search(collections=[collection],
                            bbox=context.bbox().as_tuple(), datetime=datetime,
                            query=query, max_items=max_items).items())
    if not items:
        raise FileNotFoundError(f"PC {collection}: no item near {context.date}")
    if days is None:
        return items[:max_tiles]
    d = [int(i.datetime.strftime("%Y%m%d")) if i.datetime else
         int(str(i.properties.get("start_datetime", ""))[:10].replace("-", "") or 0) or None
         for i in items]
    cc = [i.properties.get("eo:cloud_cover") for i in items]
    # tiles of one acquisition share a date, not always a time stamp
    sel = pick_closest(items, [f"{x}" for x in d], context.date, days=d, cloud=cc,
                       max_tiles=max_tiles)
    chosen = [items[k] for k in sel]
    context.note(f"PC {collection}: " + ", ".join(i.id for i in chosen))
    return chosen


def planetary_href(item, asset: str) -> str:
    if asset not in item.assets:
        raise KeyError(f"{item.id}: no asset {asset!r}; has "
                       f"{sorted(item.assets)[:40]}")
    return item.assets[asset].href  # already signed by sign_inplace


def planetary_assets(item, pattern: str) -> list[str]:
    rx = re.compile(pattern)
    keys = [k for k in item.assets if rx.fullmatch(k)]
    if not keys:
        raise KeyError(f"{item.id}: no asset matching {pattern!r}; has "
                       f"{sorted(item.assets)[:40]}")
    return sorted(keys)


# --- public buckets ------------------------------------------------------------
def gcs_list(bucket: str, prefix: str, pattern: str | None = None,
             limit: int = 20000) -> list[str]:
    """Object names of a public GCS bucket (JSON API, anonymous)."""
    s, out, tok = session(), [], None
    while len(out) < limit:
        p = {"prefix": prefix, "fields": "items(name),nextPageToken",
             "maxResults": 1000}
        if tok:
            p["pageToken"] = tok
        r = s.get(f"https://storage.googleapis.com/storage/v1/b/{bucket}/o",
                  params=p, timeout=60)
        r.raise_for_status()
        j = r.json()
        out += [i["name"] for i in j.get("items", [])]
        tok = j.get("nextPageToken")
        if not tok:
            break
    if pattern:
        rx = re.compile(pattern)
        out = [n for n in out if rx.search(n)]
    return out


def gcs_href(bucket: str, name: str) -> str:
    return f"/vsicurl/https://storage.googleapis.com/{bucket}/{name}"


REQUESTER_PAYS = {"AWS_REQUEST_PAYER": "requester"}
"""GDAL env for /vsis3 reads of requester-pays buckets: only the window's
bytes are billed (cents for a 30 km AOI)."""


def s3_list(prefix: str, pattern: str | None = None,
            requester_pays: bool = False, anon: bool = False) -> list[str]:
    import s3fs
    fs = s3fs.S3FileSystem(anon=anon, requester_pays=requester_pays)
    names = fs.find(prefix)
    if pattern:
        rx = re.compile(pattern)
        names = [n for n in names if rx.search(n)]
    return names


# --- REST APIs -------------------------------------------------------------------
SDA_URL = "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest"


def sda_query(sql: str) -> list[dict[str, Any]]:
    """USDA Soil Data Access SQL → list of row dicts."""
    r = session().post(SDA_URL, json={"query": sql, "format": "JSON+COLUMNNAME"},
                       timeout=300)
    r.raise_for_status()
    t = r.json().get("Table", [])
    if not t:
        return []
    cols = t[0]
    return [dict(zip(cols, row)) for row in t[1:]]


USGS_OGC = "https://api.waterdata.usgs.gov/ogcapi/v0"


def usgs_items(collection: str, params: dict[str, Any]) -> list[dict]:
    """USGS Water Data OGC API → GeoJSON features (all pages)."""
    s = session()
    if os.environ.get("USGS_API_KEY"):
        s.headers["X-Api-Key"] = os.environ["USGS_API_KEY"]
    url = f"{USGS_OGC}/collections/{collection}/items"
    feats, p = [], dict(params, f="json", limit=10000)
    while url:
        r = s.get(url, params=p, timeout=120)
        r.raise_for_status()
        j = r.json()
        feats += j.get("features", [])
        url = next((l["href"] for l in j.get("links", []) if l.get("rel") == "next"), None)
        p = None
    return feats


OPENET_MAX_ACRES = 45_000.0
"""Per-request area cap. The API refuses > 50,000 acres on the default tier
(HTTP 422 "Single query area limit exceeded"); the margin absorbs the
difference between our geodesic estimate and the server's."""


def _ll_tiles(b: Bounds, max_acres: float) -> list[Bounds]:
    """Split a lon/lat box into k × k equal tiles, each under `max_acres`."""
    import math
    from pyproj import Geod
    area_m2, _ = Geod(ellps="WGS84").polygon_area_perimeter(
        [b.west, b.east, b.east, b.west], [b.south, b.south, b.north, b.north])
    k = max(1, math.ceil(math.sqrt(abs(area_m2) / 4046.8564224 / max_acres)))
    dx, dy = (b.east - b.west) / k, (b.north - b.south) / k
    return [Bounds(b.west + i * dx, b.south + j * dy,
                   b.west + (i + 1) * dx, b.south + (j + 1) * dy)
            for j in range(k) for i in range(k)]


def openet_export(context: Context, bbox: Bounds, day: dt.date, *, variable: str = "ET",
                  model: str = "Ensemble", interval: str = "daily",
                  max_acres: float = OPENET_MAX_ACRES) -> Path:
    """OpenET `/raster/geotiff/stack` → one merged 30 m GeoTIFF (EPSG:4326).

    The endpoint answers synchronously with `{date: link}`; the link (valid
    ~5 min) serves a ZIP holding one `<model>_<variable>_<date>.tif`. The
    AOI is split into tiles under the per-request area cap and the tiles,
    which share OpenET's pixel grid, are merged before any regridding — so
    no 1 km cell is averaged from a partial tile. Each tile costs one query
    of the monthly quota (100 on the free tier)."""
    import zipfile
    import rasterio
    from rasterio.merge import merge
    key = os.environ.get("OPENET_API_KEY")
    if not key:
        raise Skip("OpenET needs OPENET_API_KEY")
    dest = context.raw / f"openet_{model}_{variable}_{interval}_{day}.tif"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    s = session()
    s.headers["Authorization"] = key
    tiles = _ll_tiles(bbox, max_acres)
    parts = []
    for k, t in enumerate(tiles):
        zp = context.raw / f"{dest.stem}_tile{k}.zip"
        if not (zp.exists() and zp.stat().st_size > 0):
            body = {"date_range": [str(day), str(day)], "interval": interval,
                    "geometry": [t.west, t.north, t.east, t.north, t.east, t.south,
                                 t.west, t.south],
                    "model": model, "variable": variable,
                    "reference_et": "gridMET", "units": "mm"}
            import requests
            for attempt in range(4):        # transient resets (rate limit: 20/min)
                try:
                    r = s.post("https://openet-api.org/raster/geotiff/stack",
                               json=body, timeout=300)
                    break
                except requests.ConnectionError:
                    if attempt == 3:
                        raise
                    time.sleep(15 * (attempt + 1))
            if not r.ok:
                raise RuntimeError(f"OpenET tile {k}: HTTP {r.status_code} {r.text[:300]}")
            urls = find_urls(r.json())
            if not urls:
                raise FileNotFoundError(f"OpenET: no raster for {day} ({r.text[:200]})")
            download(urls[0], zp, sess=session())   # signed link: no API key header
        with zipfile.ZipFile(zp) as z:
            tif = next((n for n in z.namelist() if n.lower().endswith((".tif", ".tiff"))), None)
        if tif is None:
            raise FileNotFoundError(f"OpenET tile {k}: no GeoTIFF in {zp.name}")
        parts.append(f"/vsizip/{zp}/{tif}")
    context.note(f"OpenET {model} {variable} {day}: {len(tiles)} tile(s) ≤ {max_acres:.0f} ac")
    srcs = [rasterio.open(p) for p in parts]
    try:
        nodata = srcs[0].nodata if srcs[0].nodata is not None else float("nan")
        data, transform = merge(srcs, nodata=nodata)
        profile = srcs[0].profile | {"driver": "GTiff", "height": data.shape[1],
                                     "width": data.shape[2], "transform": transform,
                                     "nodata": nodata, "compress": "deflate"}
    finally:
        for src in srcs:
            src.close()
    tmp = dest.with_suffix(".part.tif")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(data)
    tmp.rename(dest)
    return dest


LFPS = "https://lfps.usgs.gov/api"


def lfps_job(context: Context, layers: Sequence[str], timeout: float = 3600) -> Path:
    """LANDFIRE Product Service: submit → poll → zip of GeoTIFFs.
    Output kept in the native 30 m Albers (no server-side resampling, so
    our own fractions/mode apply). Confidence D for the flow, M for the
    exact status keys (read defensively)."""
    email = os.environ.get("LFPS_EMAIL")
    if not email:
        raise Skip("LANDFIRE LFPS needs LFPS_EMAIL")
    dest = context.raw / f"landfire_{'_'.join(layers)}.zip"
    if dest.exists():
        return dest
    b = context.bbox(buffer_km=1.0)
    s = session()
    r = s.get(f"{LFPS}/job/submit", params={
        "Email": email, "Layer_List": ";".join(layers),
        "Area_of_Interest": f"{b.west} {b.south} {b.east} {b.north}",
        "Output_Projection": "5070"}, timeout=120)
    r.raise_for_status()
    j = r.json()
    job = j.get("jobId") or j.get("JobId") or j.get("job_id")
    if not job:
        raise RuntimeError(f"LFPS submit: {r.text[:300]}")
    context.note(f"LFPS job {job}")

    def done():
        st = s.get(f"{LFPS}/job/status", params={"JobId": job}, timeout=60).json()
        state = str(st.get("status") or st.get("jobStatus") or "")
        if state.lower() in ("failed", "esrijobfailed", "cancelled"):
            raise RuntimeError(f"LFPS job {job}: {st}")
        zips = [u for u in find_urls(st) if u.endswith(".zip")]
        return zips[0] if zips else None

    url = poll(done, every=20, timeout=timeout, what=f"LFPS {job}")
    return download(url, dest)


def cds_retrieve(context: Context, dataset: str, request: dict, name: str) -> Path:
    try:
        import cdsapi
    except ImportError as e:
        raise Skip("CDS route needs `cdsapi` (uv add cdsapi)") from e
    if not available()["cds"]:
        raise Skip("CDS needs ~/.cdsapirc (free key)")
    dest = context.raw / name
    if not dest.exists():
        cdsapi.Client().retrieve(dataset, request, str(dest))
    return dest
