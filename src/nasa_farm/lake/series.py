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

import numpy as np

from nasa_farm.lake.calendar import to_ord, to_ymd


def merge(old: tuple[np.ndarray, np.ndarray] | None,
          dates: np.ndarray, values: np.ndarray
          ) -> tuple[np.ndarray, np.ndarray]:
    """Sorted-unique union of old + new; newest input wins duplicates."""
    d = np.asarray(dates, dtype=np.int32).reshape(-1)
    v = np.asarray(values, dtype=np.float32).reshape(-1)
    if old is not None:
        d = np.concatenate([old[0], d])
        v = np.concatenate([old[1], v])
    if not d.size:
        return d, v
    order = np.argsort(d, kind="stable")
    d, v = d[order], v[order]
    keep = np.concatenate([d[1:] != d[:-1], [True]])
    return d[keep], v[keep]


def encode(dates: np.ndarray, values: np.ndarray) -> tuple[bytes, bytes]:
    """yyyymmdd + f32 → (zstd int32-ordinal blob, zstd f32 blob)."""
    return (zstd.compress(to_ord(dates).tobytes()),
            zstd.compress(np.asarray(values, np.float32).tobytes()))


def decode(dates: bytes, values: bytes) -> tuple[np.ndarray, np.ndarray]:
    """(zstd ord blob, zstd f32 blob) → yyyymmdd int32 + f32."""
    d = np.frombuffer(zstd.decompress(dates), dtype=np.int32)
    v = np.frombuffer(zstd.decompress(values), dtype=np.float32)
    return to_ymd(d), v
