"""Legacy `*.zarr` dirs + `catalog.db` → `lake.db` (idempotent).

`Lake(root)` runs this automatically the first time it opens a root that
still has the legacy layout and no `lake.db` yet: every series is
re-packed through `Lake.put`, fields and granule marks replayed from
`catalog.db`, all inside one transaction — a failed import can never
leave a half-migrated lake. `clean=True` removes the legacy files after
a verified import.
"""

from __future__ import annotations

import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np


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


def migrate_into(lake) -> dict:
    """Import every legacy store under `lake.root` in one transaction."""
    stats = {"fields": 0, "series": 0, "emb": 0, "granules": 0}
    with lake.tx():
        for fid, key, data in _legacy_series(lake.root):
            if key is None:
                lake.attrs(fid, **data)
            elif key[0] == "emb":
                lake.put_vec(fid, key[1], np.asarray(data[1]).reshape(-1))
                stats["emb"] += 1
            else:
                lake.put(fid, key[0], key[1], data[0], data[1])
                stats["series"] += 1
        cat = lake.root / "catalog.db"
        if cat.exists():
            with closing(sqlite3.connect(
                    f"file:{cat}?mode=ro", uri=True)) as c:
                for fid, lat, lon, name in c.execute(
                        "SELECT id, lat, lon, name FROM fields"):
                    lake.field(fid, lat, lon, name or "")
                    stats["fields"] += 1
                for gid, product in c.execute(
                        "SELECT id, product FROM granules"):
                    lake.mark_granule(gid, product)
                    stats["granules"] += 1
    return stats


def migrate(root: Path | str, clean: bool = False) -> dict:
    """Import the legacy layout under `root`; `clean` removes it after."""
    root = Path(root)
    with Lake(root) as lake:
        stats = lake.migrated or migrate_into(lake)
    if clean:
        for d in root.glob("*.zarr"):
            shutil.rmtree(d)
        for f in root.glob("catalog.db*"):
            f.unlink()
    return stats
