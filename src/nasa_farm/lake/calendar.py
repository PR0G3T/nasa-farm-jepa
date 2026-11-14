"""yyyymmdd ↔ proleptic ordinal — vectorized civil-date math.

Hinnant's days-from-civil algorithms run elementwise on int arrays: no
datetime objects, no per-point Python. Ordinals are the on-disk date
domain (monotone, gap-free — delta-1 runs compress to ~1 B/day); yyyymmdd
is the API domain.
"""

from __future__ import annotations

import numpy as numpy

_SHIFT = 305  # days_from_civil counts from 1970-01-01; ordinal = z + 719163


def to_ordinal(yyyymmdd: numpy.ndarray) -> numpy.ndarray:
    """int yyyymmdd array → int32 proleptic ordinals (date.toordinal)."""
    a = numpy.asarray(yyyymmdd, dtype=numpy.int64)
    y, m = a // 10000, a // 100 % 100
    d = a - (a // 100) * 100
    y = y - (m <= 2)
    era = numpy.where(y >= 0, y, y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m + numpy.where(m > 2, -3, 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return (era * 146097 + doe - _SHIFT).astype(numpy.int32)


def to_date(ord_: numpy.ndarray) -> numpy.ndarray:
    """int32 proleptic ordinals → int yyyymmdd array."""
    z = numpy.asarray(ord_, dtype=numpy.int64) + _SHIFT
    era = numpy.where(z >= 0, z, z - 146096) // 146097
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = mp + numpy.where(mp < 10, 3, -9)
    return ((y + (m <= 2)) * 10000 + m * 100 + d).astype(numpy.int32)
