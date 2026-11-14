"""Granule → xr.Dataset over authenticated HTTPS range reads.

`earthaccess.open` hands back fsspec handles, so a granule is read the
same way as the NLDAS-3 S3 files — the sampler's spatial slice costs
kilobytes of range requests instead of a full-file download, and nothing
ever touches the local disk. The dataset carries `granule_id` in attrs
for day extraction and catalog marking.
"""

from __future__ import annotations

import xarray as xarray

from nasa_farm.acquire.base import Bounds, Temporal
from nasa_farm.acquire.earthdata import EarthdataSource, Granule


def fetch(source: EarthdataSource, granule: Granule,
          area: Bounds | None = None, temporal: Temporal | None = None
          ) -> xarray.Dataset:
    """Open one granule lazily; closing the dataset frees the handles."""
    import earthaccess
    earthaccess.login(strategy="environment")
    h = earthaccess.open([granule.raw])[0]
    f = h.open() if hasattr(h, "open") else h
    ds = xarray.open_dataset(f, engine="h5netcdf", group=source.group,
                         phony_dims="access")
    ds.attrs["granule_id"] = granule.id
    return ds
