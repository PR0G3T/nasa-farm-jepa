"""Lake catalog — sqlite, stdlib only.

Three relations: `fields` (AOIs), `granules` (processed product files —
dedupe makes crawls resumable), `series` (coverage per field/var for gap
planning). One file, WAL mode, transactional.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DDL = """
CREATE TABLE IF NOT EXISTS fields(
    id TEXT PRIMARY KEY, lat REAL, lon REAL, name TEXT);
CREATE TABLE IF NOT EXISTS granules(
    id TEXT PRIMARY KEY, product TEXT, at TEXT DEFAULT (datetime('now')));
CREATE TABLE IF NOT EXISTS series(
    fid TEXT, grp TEXT, var TEXT, first INT, last INT, n INT,
    source TEXT, PRIMARY KEY(fid, grp, var));
"""


class Catalog:
    def __init__(self, path: Path | str) -> None:
        self.db = sqlite3.connect(Path(path))
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(DDL)

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> Catalog:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def field(self, fid: str, lat: float, lon: float, name: str = "") -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO fields VALUES(?,?,?,?)",
            (fid, lat, lon, name))
        self.db.commit()

    def fields(self) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM fields").fetchall()

    def has_granule(self, gid: str) -> bool:
        return self.db.execute(
            "SELECT 1 FROM granules WHERE id=?", (gid,)).fetchone() is not None

    def mark_granule(self, gid: str, product: str) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO granules(id, product) VALUES(?,?)",
            (gid, product))
        self.db.commit()

    def set_series(self, fid: str, group: str, var: str, first: int,
                   last: int, n: int, source: str) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO series VALUES(?,?,?,?,?,?,?)",
            (fid, group, var, first, last, n, source))
        self.db.commit()

    def series(self, fid: str | None = None) -> list[tuple]:
        q = "SELECT * FROM series" + (" WHERE fid=?" if fid else "")
        return self.db.execute(q, (fid,) if fid else ()).fetchall()
