"""Data lake — one SQLite file (`lake.db`) for every crawled observation.

`series` packs each `{grp}/{var}` time series as zstd'd int32-ordinal +
float32 blobs with coverage columns, `fields` keeps sites + JSONB attrs
(soil profiles, names), `granules` makes crawls resumable, `emb` stores
foundation-model vectors. `Lake` auto-migrates a legacy `*.zarr` +
`catalog.db` layout on first open; `migrate` runs it on demand.
"""

from pathlib import Path

from nasa_farm.lake.db import Lake
from nasa_farm.lake.migrate import migrate

__all__ = ["Lake", "get", "migrate", "put"]


def put(root: Path | str, fid: str, grp: str, var: str, dates, values,
        source: str = "") -> int:
    """One-shot series upsert (embeddings route to the `emb` table)."""
    import numpy as np
    with Lake(root) as L:
        if grp == "emb":
            return L.put_vec(fid, var, np.asarray(values).reshape(-1)) or 0
        return L.put(fid, grp, var, dates, values, source)


def get(root: Path | str, fid: str, grp: str, var: str):
    """One-shot series read — (dates yyyymmdd, vals) or None."""
    with Lake(root, ro=True) as L:
        return L.get(fid, grp, var)
