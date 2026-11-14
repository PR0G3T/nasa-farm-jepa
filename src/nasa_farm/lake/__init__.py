"""Data lake — crawled observations stored once, trained from forever.

Layout: one zarr group per field, series as `{group}/{var}/{dates,values}`.
`forcing/` keeps source-native names (raw dump, feeds `series_to_forcing`);
`obs/` keeps canonical `observe/` channel names (sm, sme2, et, lst, ndvi)
with a `{var}_w` coverage-weight shard beside every footprint-sampled
series; `smg` holds the POWER root-zone proxy — context, not a target.
`emb/` stores foundation-model embedding vectors (2-D values).
The sqlite catalog tracks coverage and processed granules — resumable.
"""

from nasa_farm.lake.catalog import Catalog
from nasa_farm.lake.store import get, put

__all__ = ["Catalog", "get", "put"]
