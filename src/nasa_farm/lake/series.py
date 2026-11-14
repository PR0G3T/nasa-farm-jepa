"""Series merge + blob codec — pure array logic, no I/O.

On-disk layout per (fid, grp, var): dates as packed int32 proleptic
ordinals, values as packed float32, both zstd-compressed via the stdlib
`compression.zstd` (zero extra deps). `merge` is a stable sorted-unique
union — last write wins on duplicate dates, so a re-crawl with fresher
retrievals converges to the newest data, and an identical re-put is a
provable no-op.
"""

from __future__ import annotations

from compression import zstd

import numpy as numpy

from nasa_farm.lake.calendar import to_ordinal, to_date


def merge(old: tuple[numpy.ndarray, numpy.ndarray] | None,
          dates: numpy.ndarray, values: numpy.ndarray
          ) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Sorted-unique union of old + new; newest input wins duplicates."""
    d = numpy.asarray(dates, dtype=numpy.int32).reshape(-1)
    v = numpy.asarray(values, dtype=numpy.float32).reshape(-1)
    if old is not None:
        d = numpy.concatenate([old[0], d])
        v = numpy.concatenate([old[1], v])
    if not d.size:
        return d, v
    order = numpy.argsort(d, kind="stable")
    d, v = d[order], v[order]
    keep = numpy.concatenate([d[1:] != d[:-1], [True]])
    return d[keep], v[keep]


def encode(dates: numpy.ndarray, values: numpy.ndarray) -> tuple[bytes, bytes]:
    """yyyymmdd + f32 → (zstd int32-ordinal blob, zstd f32 blob)."""
    return (zstd.compress(to_ordinal(dates).tobytes()),
            zstd.compress(numpy.asarray(values, numpy.float32).tobytes()))


def decode(dates: bytes, values: bytes) -> tuple[numpy.ndarray, numpy.ndarray]:
    """(zstd ord blob, zstd f32 blob) → yyyymmdd int32 + f32."""
    d = numpy.frombuffer(zstd.decompress(dates), dtype=numpy.int32)
    v = numpy.frombuffer(zstd.decompress(values), dtype=numpy.float32)
    return to_date(d), v
