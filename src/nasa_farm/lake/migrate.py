"""Legacy `*.zarr` dirs + `catalog.db` → `lake.db` (idempotent).

`Lake(root)` runs this automatically the first time it opens a root that
still has the legacy layout and no `lake.db` yet: every series is
re-packed through `Lake.put` (POWER-native names canonicalized, source
provenance replayed from `catalog.series`), fields and granule marks
replayed from `catalog.db` — all inside one transaction, so a failed
import can never leave a half-migrated lake. `clean=True` removes the
legacy files after the import lands.
"""

from __future__ import annotations

import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np

from nasa_farm.acquire.power import CANON


def _legacy_series(root: Path):
    import zarr
    for d in sorted(root.glob("*.zarr")):
        fid = d.stem
        try:
            g = zarr.open_group(d, mode="r")
        except Exception:
            continue                          # stray artifact, not a group
        if attrs := dict(g.attrs):
            yield fid, None, attrs
        for grp in g.group_keys():
            for var in g[grp].group_keys():
                node = g[grp][var]
                if "dates" in node and "values" in node:
                    yield fid, (grp, var), (node["dates"][:],
                                            node["values"][:])


def _catalog(root: Path):
    """(fields, granules, series-source map) from a legacy catalog.db."""
    cat = root / "catalog.db"
    fields, granules, src = [], [], {}
    if cat.exists():
        with closing(sqlite3.connect(
                f"file:{cat}?mode=ro", uri=True)) as c:
            fields = c.execute(
                "SELECT id, lat, lon, name FROM fields").fetchall()
            granules = c.execute(
                "SELECT id, product FROM granules").fetchall()
            for fid, grp, var, s in c.execute(
                    "SELECT fid, grp, var, source FROM series"):
                src[fid, grp, var] = s or ""
    return fields, granules, src


def migrate_into(lake) -> dict:
    """Import every legacy store under `lake.root` in one transaction."""
    stats = {"fields": 0, "series": 0, "emb": 0, "granules": 0}
    fields, granules, src = _catalog(lake.root)
    with lake.tx():
        for fid, key, data in _legacy_series(lake.root):
            if key is None:
                lake.attrs(fid, **data)
            elif key[0] == "emb":
                lake.put_vec(fid, key[1], np.asarray(data[1]).reshape(-1))
                stats["emb"] += 1
            else:
                grp, var = key
                if grp == "forcing":
                    var = CANON.get(var, var)
                lake.put(fid, grp, var, data[0], data[1],
                         src.get(key, ""))
                stats["series"] += 1
        for fid, lat, lon, name in fields:
            lake.field(fid, lat, lon, name or "")
            stats["fields"] += 1
        for gid, product in granules:
            lake.mark_granule(gid, product)
            stats["granules"] += 1
    return stats


def migrate(root: Path | str, clean: bool = False) -> dict:
    """Import the legacy layout under `root`; `clean` removes it after."""
    root = Path(root)
    from nasa_farm.lake.db import Lake
    with Lake(root) as lake:
        stats = lake.migrated or migrate_into(lake)
    if clean:
        for d in root.glob("*.zarr"):
            shutil.rmtree(d)
        for f in root.glob("catalog.db*"):
            f.unlink()
    return stats
