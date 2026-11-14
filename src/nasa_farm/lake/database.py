"""Lake — one SQLite file for fields, series, granules and embeddings.

`lake.db` (WAL) replaces the per-field zarr trees + catalog.db: one
transactional row per series (`dates` and `vals` can never diverge),
coverage columns (`first`, `last`, `n`, `source`, `rev`) maintained by
`put` itself, and every crawl unit lands in a single `tx()` — a crash
can never leave data written but unmarked. Writes outside `tx()` commit
per call; `put` is a no-op when the merged series is unchanged, so
re-crawls are write-free and idempotent.

Read-only opens (`ro=True`) create no files; a missing `lake.db` under a
legacy `*.zarr`/`catalog.db` layout is auto-migrated once through a
writable open, then served read-only.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as numpy

from nasa_farm.lake import series

DB_NAME = "lake.db"
_SCHEMA = 1

DDL = """
CREATE TABLE IF NOT EXISTS fields(
    fid TEXT PRIMARY KEY, lat REAL, lon REAL,
    name TEXT NOT NULL DEFAULT '', attrs BLOB) STRICT;
CREATE INDEX IF NOT EXISTS fields_ll ON fields(lat, lon);
CREATE TABLE IF NOT EXISTS series(
    fid TEXT NOT NULL, grp TEXT NOT NULL, var TEXT NOT NULL,
    dates BLOB NOT NULL, vals BLOB NOT NULL,
    n INTEGER NOT NULL, first INTEGER NOT NULL, last INTEGER NOT NULL,
    source TEXT NOT NULL DEFAULT '', rev INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY(fid, grp, var)) STRICT;
CREATE TABLE IF NOT EXISTS granules(
    id TEXT PRIMARY KEY, product TEXT NOT NULL DEFAULT '',
    at TEXT NOT NULL DEFAULT (datetime('now'))) STRICT;
CREATE TABLE IF NOT EXISTS emb(
    fid TEXT NOT NULL, name TEXT NOT NULL, vec BLOB NOT NULL,
    PRIMARY KEY(fid, name)) STRICT;
"""

_UPSERT = """
INSERT INTO series(fid, grp, var, dates, vals, n, first, last, source)
VALUES (?,?,?,?,?,?,?,?,?)
ON CONFLICT(fid, grp, var) DO UPDATE SET
    dates = excluded.dates, vals = excluded.vals, n = excluded.n,
    first = excluded.first, last = excluded.last,
    source = CASE WHEN excluded.source = '' THEN series.source
                  ELSE excluded.source END,
    rev = series.rev + 1
"""

_FIELD = """
INSERT INTO fields(fid, lat, lon, name) VALUES (?,?,?,?)
ON CONFLICT(fid) DO UPDATE SET
    lat = COALESCE(excluded.lat, fields.lat),
    lon = COALESCE(excluded.lon, fields.lon),
    name = CASE WHEN excluded.name = '' THEN fields.name
               ELSE excluded.name END
"""


def _serializable(o):
    if isinstance(o, dict):
        return {k: _serializable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_serializable(v) for v in o]
    if isinstance(o, numpy.generic):
        return o.item()
    return o.tolist() if hasattr(o, "tolist") \
        and not isinstance(o, (str, float, int)) else o


class Lake:
    """One-file lake handle — batch crawl writes inside `tx()`."""

    def __init__(self, root: Path | str, readonly: bool = False) -> None:
        self.root = Path(root)
        self.db_path = self.root / DB_NAME
        self.readonly = readonly
        self.migrated: dict | None = None
        self._depth = 0
        if readonly and not self.db_path.exists():
            if any(self.root.glob("*.zarr")) or (self.root / "catalog.db").exists():
                Lake(root).close()      # writable open auto-migrates legacy
            if not self.db_path.exists():
                raise FileNotFoundError(self.db_path)
        if readonly:
            self.database = sqlite3.connect(
                f"file:{self.db_path}?mode=ro&immutable=1", uri=True,
                isolation_level=None)
            self.database.execute("PRAGMA busy_timeout=5000")
            try:
                self.database.execute("PRAGMA mmap_size=268435456")
            except sqlite3.OperationalError:
                pass
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            fresh = not self.db_path.exists()
            self.database = sqlite3.connect(self.db_path, isolation_level=None)
            self.database.execute("PRAGMA journal_mode=WAL")
            self.database.execute("PRAGMA wal_autocheckpoint=1")
            self.database.execute("PRAGMA synchronous=NORMAL")
            self.database.execute("PRAGMA mmap_size=268435456")
            self.database.execute("PRAGMA busy_timeout=5000")
            if self.database.execute("PRAGMA user_version").fetchone()[0] < _SCHEMA:
                self.database.executescript(DDL)
                self.database.execute(f"PRAGMA user_version={_SCHEMA}")
            if fresh:
                from nasa_farm.lake.migrate import migrate_into
                self.migrated = migrate_into(self)

    # -- lifecycle -----------------------------------------------------

    @contextmanager
    def transaction(self):
        """One commit per crawl unit; nested calls join the outer tx."""
        outer = self._depth == 0
        if outer:
            self.database.execute("BEGIN IMMEDIATE")
        self._depth += 1
        try:
            yield self
        except BaseException:
            if outer:
                self.database.execute("ROLLBACK")
            self._depth -= 1
            raise
        self._depth -= 1
        if outer:
            self.database.execute("COMMIT")

    def _w(self):
        """Autocommit outside tx; inside, statements join the open tx."""
        return nullcontext()

    def close(self) -> None:
        if self._depth:
            self._depth = 0
            self.database.execute("ROLLBACK")
        if not self.readonly:
            self.database.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.database.close()

    def __enter__(self) -> "Lake":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # -- fields --------------------------------------------------------

    def field(self, fid: str, latitude: float | None = None,
              longitude: float | None = None, name: str = "") -> None:
        with self._w():
            self.database.execute(_FIELD, (fid, latitude, longitude, name))

    def fields(self, bbox: tuple[float, float, float, float] | None = None):
        """All fields, or those inside (west, south, east, north)."""
        q, arguments = "SELECT fid, lat, lon, name FROM fields", ()
        if bbox is not None:
            w, s, e, n = bbox
            q += " WHERE lon BETWEEN ? AND ? AND lat BETWEEN ? AND ?"
            arguments = (w, e, s, n)
        return [{"id": f, "lat": la, "lon": low, "name": nm}
                for f, la, low, nm
                in self.database.execute(q + " ORDER BY fid", arguments)]

    def fids(self) -> list[str]:
        """Every field id with a row or any series."""
        return [r[0] for r in self.database.execute(
            "SELECT fid FROM fields UNION SELECT fid FROM series"
            " ORDER BY 1")]

    def attrs(self, fid: str, **keywords) -> None:
        """Merge JSON-able metadata onto a field row (created if absent)."""
        with self._w():
            self.database.execute(
                "INSERT INTO fields(fid, attrs) VALUES (?, jsonb(?)) "
                "ON CONFLICT(fid) DO UPDATE SET attrs = jsonb_patch("
                "COALESCE(fields.attrs, jsonb('{}')), excluded.attrs)",
                (fid, json.dumps(_serializable(keywords))))

    def metadata(self, fid: str) -> dict:
        r = self.database.execute(
            "SELECT json(attrs) FROM fields WHERE fid=?", (fid,)).fetchone()
        return {} if r is None or r[0] is None else json.loads(r[0])

    # -- series --------------------------------------------------------

    def _row(self, fid: str, group: str, var: str):
        r = self.database.execute(
            "SELECT dates, vals, source FROM series"
            " WHERE fid=? AND grp=? AND var=?", (fid, group, var)).fetchone()
        return None if r is None else (*series.decode(r[0], r[1]), r[2])

    def get(self, fid: str, group: str, var: str):
        """(dates int32 yyyymmdd, vals f32) or None."""
        row = self._row(fid, group, var)
        return None if row is None else (row[0], row[1])

    def put(self, fid: str, group: str, var: str,
            dates: numpy.ndarray, values: numpy.ndarray,
            source: str = "") -> int:
        """Upsert a series (sorted-unique, newest wins); returns its length.

        A no-op when the merge is unchanged — re-crawls write nothing.
        """
        with self._w():
            old = self._row(fid, group, var)
            d, v = series.merge(old, dates, values)
            if old is not None and numpy.array_equal(old[0], d) \
                    and numpy.array_equal(old[1], v) \
                    and (not source or source == old[2]):
                return int(d.size)
            database, vb = series.encode(d, v)
            self.database.execute(_UPSERT, (fid, group, var, database, vb, int(d.size),
                                      int(d[0]) if d.size else 0,
                                      int(d[-1]) if d.size else 0, source))
            return int(d.size)

    def coverage(self, fid: str | None = None):
        """(fid, grp, var, first, last, n, source, rev) per series."""
        q = ("SELECT fid, grp, var, first, last, n, source, rev"
             " FROM series" + (" WHERE fid=?" if fid else "")
             + " ORDER BY fid, grp, var")
        return self.database.execute(q, (fid,) if fid else ()).fetchall()

    # -- granules ------------------------------------------------------

    def has_granule(self, gid: str) -> bool:
        return self.database.execute(
            "SELECT 1 FROM granules WHERE id=?", (gid,)).fetchone() is not None

    def mark_granule(self, gid: str, product: str = "") -> None:
        with self._w():
            self.database.execute(
                "INSERT OR IGNORE INTO granules(id, product) VALUES (?,?)",
                (gid, product))

    # -- embeddings ----------------------------------------------------

    def put_vec(self, fid: str, name: str, vector: numpy.ndarray) -> None:
        """Raw f32 blob — dense floats don't zstd."""
        b = numpy.ascontiguousarray(vector, dtype=numpy.float32).tobytes()
        with self._w():
            self.database.execute(
                "INSERT INTO emb VALUES (?,?,?) ON CONFLICT(fid, name)"
                " DO UPDATE SET vec = excluded.vec", (fid, name, b))

    def vector(self, fid: str, name: str = "aef") -> numpy.ndarray | None:
        r = self.database.execute(
            "SELECT vec FROM emb WHERE fid=? AND name=?",
            (fid, name)).fetchone()
        return None if r is None else numpy.frombuffer(r[0], numpy.float32)
