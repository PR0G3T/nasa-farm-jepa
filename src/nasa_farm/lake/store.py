"""Zarr series shards — one group per field, date-keyed upsert.

Every series is `{group}/{var}/{dates,values}`: dates as int32 yyyymmdd,
values as float32. `put` merges by date (dedupe, sort) so re-crawls are
idempotent; series are small (≤ ~2000 points) so merge-in-place beats
chunk tuning.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr
from zarr.errors import GroupNotFoundError


def _group(root: Path, fid: str) -> zarr.Group:
    return zarr.open_group(root / f"{fid}.zarr", mode="a")


def put(root: Path, fid: str, group: str, var: str,
        dates: np.ndarray, values: np.ndarray) -> int:
    """Upsert one series, return total count after merge."""
    g = _group(root, fid)
    dates = np.asarray(dates, dtype=np.int32)
    values = np.asarray(values, dtype=np.float32)
    if (old := get(root, fid, group, var)) is not None:
        od, ov = old
        dates, values = np.concatenate([od, dates]), np.concatenate([ov, values])
    order = np.argsort(dates, kind="stable")
    dates, values = dates[order], values[order]
    keep = np.concatenate([[True], dates[1:] != dates[:-1]])
    node = g.require_group(f"{group}/{var}")
    node.create_array("dates", data=dates[keep], overwrite=True)
    node.create_array("values", data=values[keep], overwrite=True)
    return int(keep.sum())


def get(root: Path, fid: str, group: str, var: str
        ) -> tuple[np.ndarray, np.ndarray] | None:
    path = root / f"{fid}.zarr" / group / var
    try:
        node = zarr.open_group(path, mode="r")
        return node["dates"][:], node["values"][:]
    except (KeyError, FileNotFoundError, GroupNotFoundError):
        return None



def attrs(root: Path, fid: str, **kw) -> None:
    """Set field-level metadata (soil profile, name, ...)."""
    _group(root, fid).attrs.update(kw)


def meta(root: Path, fid: str) -> dict:
    try:
        return dict(_group(root, fid).attrs)
    except FileNotFoundError:
        return {}
