"""Granule → xr.Dataset over authenticated HTTPS range reads.

`earthaccess.open` hands back fsspec handles, so a granule is read the
same way as the NLDAS-3 S3 files — the sampler's spatial slice costs
kilobytes of range requests instead of a full-file download, and nothing
ever touches the local disk. The dataset carries `granule_id` in attrs
for day extraction and catalog marking.

The h5netcdf backend does NOT own the fsspec file object — closing the
dataset alone leaves the handle, its block cache (~25 MB per granule)
and the earthaccess session alive. `fetch` is therefore a context
manager: on exit it closes the dataset, then the fsspec file and the
earthaccess handle it came from.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import xarray as xarray

from nasa_farm.acquire.base import Bounds, Temporal
from nasa_farm.acquire.earthdata import Earthdata, Granule


@contextlib.contextmanager
def fetch(
    source: Earthdata,
    granule: Granule,
    area: Bounds | None = None,
    temporal: Temporal | None = None,
) -> Iterator[xarray.Dataset]:
    """Open one granule lazily; on exit, dataset and handles all close."""
    import earthaccess

    earthaccess.login(strategy="environment")
    h = earthaccess.open([granule.raw])[0]
    f = h.open() if hasattr(h, "open") else h
    ds = xarray.open_dataset(f, engine="h5netcdf", group=source.group, phony_dims="access")
    ds.attrs["granule_id"] = granule.id
    try:
        yield ds
    finally:
        ds.close()
        for handle in (f, h):
            with contextlib.suppress(Exception):
                handle.close()
