"""Data lake — one SQLite file (`lake.db`) for every crawled observation.

`series` packs each `{grp}/{var}` time series as zstd'd int32-ordinal +
float32 blobs with coverage columns, `fields` keeps sites + JSONB attrs
(soil profiles, names), `granules` makes crawls resumable, `emb` stores
foundation-model vectors. `Lake` auto-migrates a legacy `*.zarr` +
`catalog.db` layout on first open; `migrate` runs it on demand.
"""

from nasa_farm.lake.db import Lake
from nasa_farm.lake.migrate import migrate

__all__ = ["Lake", "migrate"]
