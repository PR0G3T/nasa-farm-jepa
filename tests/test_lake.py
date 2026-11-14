"""Lake — packed series, catalog, transactions, migration."""

import datetime as dt
import sqlite3
import subprocess
import sys
import threading
import tracemalloc

import numpy as numpy
import pytest
from hypothesis import given, settings, strategies as state

from nasa_farm.lake import Lake, migrate
from nasa_farm.lake.calendar import to_ordinal, to_date
from nasa_farm.lake.series import decode, encode, merge


def _ymd(d: dt.date) -> int:
    return d.year * 10000 + d.month * 100 + d.day


class TestCal:
    @settings(max_examples=300)
    @given(state.dates(min_value=dt.date(1200, 1, 1),
                    max_value=dt.date(9999, 12, 30)))
    def test_matches_datetime(self, d):
        assert to_ordinal(numpy.array([_ymd(d)]))[0] == d.toordinal()
        assert to_date(numpy.array([d.toordinal()]))[0] == _ymd(d)

    def test_edges(self):
        days = [10101, 19700101, 20000229, 20240229, 21000301, 99991231]
        assert to_date(to_ordinal(numpy.array(days))).tolist() == days

    def test_monotone(self):
        sequence = [19991231, 20000101, 20231231, 20240101,
               20240228, 20240229, 20240301]
        assert numpy.all(numpy.diff(to_ordinal(numpy.array(sequence))) > 0)
        run = [20240227, 20240228, 20240229, 20240301, 20240302]
        assert numpy.all(numpy.diff(to_ordinal(numpy.array(run))) == 1)


class TestSeries:
    def test_merge_sorted_new_wins(self):
        d, v = merge(None, numpy.array([20240102, 20240101]),
                     numpy.array([20.0, 10.0]))
        assert d.tolist() == [20240101, 20240102]
        d, v = merge((d, v), numpy.array([20240102, 20240103]),
                     numpy.array([99.0, 30.0]))
        assert d.tolist() == [20240101, 20240102, 20240103]
        assert v.tolist() == [10.0, 99.0, 30.0]

    def test_merge_empty(self):
        d, v = merge(None, numpy.array([]), numpy.array([]))
        assert d.size == 0 and v.size == 0

    @given(state.lists(state.integers(20200101, 20201231), max_size=64))
    def test_idempotent(self, days):
        random = numpy.random.default_rng(0)
        d = numpy.array(days, numpy.int32)
        v = random.random(len(days), dtype=numpy.float32)
        m1 = merge(None, d, v)
        m2 = merge(m1, *m1)
        assert numpy.array_equal(m1[0], m2[0]) and numpy.array_equal(m1[1], m2[1])

    @given(state.lists(state.integers(20200101, 20201231), min_size=1, max_size=48),
           state.lists(state.integers(20200101, 20201231), min_size=1, max_size=48))
    def test_merge_model(self, a, b):
        random = numpy.random.default_rng(0)
        data, database = numpy.array(a, numpy.int32), numpy.array(b, numpy.int32)
        va = random.random(len(a), dtype=numpy.float32)
        vb = random.random(len(b), dtype=numpy.float32)
        model = dict(zip(data.tolist(), va.tolist()))
        model.update(zip(database.tolist(), vb.tolist()))   # newest wins
        d, v = merge((data, va), database, vb)
        assert numpy.all(numpy.diff(d) > 0)               # strictly sorted unique
        assert d.tolist() == sorted(model)
        assert numpy.array_equal(v, numpy.array(
            [model[k] for k in sorted(model)], numpy.float32))

    @given(state.lists(
        state.dates(min_value=dt.date(2020, 1, 1),
                 max_value=dt.date(2020, 12, 31)),
        min_size=1, max_size=48, unique=True))
    def test_codec_roundtrip(self, days):
        d = numpy.array(sorted(_ymd(x) for x in days), numpy.int32)
        v = numpy.arange(len(d), dtype=numpy.float32)
        d2, v2 = decode(*encode(d, v))
        assert numpy.array_equal(d2, d) and numpy.array_equal(v2, v)


class TestLake:
    def test_roundtrip(self, tmp_path):
        with Lake(tmp_path) as L:
            L.put("F1", "forcing", "tmin",
                  numpy.array([20240101, 20240102]), numpy.array([4.0, 5.0]))
            d, v = L.get("F1", "forcing", "tmin")
        assert d.tolist() == [20240101, 20240102]
        assert numpy.array_equal(v, numpy.array([4.0, 5.0], numpy.float32))

    def test_merge_dedup_new_wins(self, tmp_path):
        with Lake(tmp_path) as L:
            L.put("F1", "obs", "sm", numpy.array([20240101, 20240103]),
                  numpy.array([0.5, 0.7]))
            L.put("F1", "obs", "sm", numpy.array([20240102, 20240103]),
                  numpy.array([0.6, 0.9]))
            d, v = L.get("F1", "obs", "sm")
        assert d.tolist() == [20240101, 20240102, 20240103]
        assert numpy.array_equal(v, numpy.array([0.5, 0.6, 0.9], numpy.float32))

    def test_missing_none(self, tmp_path):
        with Lake(tmp_path) as L:
            assert L.get("F1", "obs", "sm") is None

    def test_attrs_meta(self, tmp_path):
        with Lake(tmp_path) as L:
            L.attrs("F1", soil={"clay": numpy.float64(30.0)})
            assert L.metadata("F1") == {"soil": {"clay": 30.0}}
            L.attrs("F1", name2={"x": [1, 2]})
            m = L.metadata("F1")
            assert L.metadata("NOPE") == {}
        assert m["soil"] == {"clay": 30.0} and m["name2"] == {"x": [1, 2]}

    def test_fields_bbox(self, tmp_path):
        with Lake(tmp_path) as L:
            L.field("A", 40.0, -100.0, "ne")
            L.field("B", 48.0, 2.0, "fr")
            L.field("C", -30.0, -60.0, "br")
            assert {f["id"] for f in L.fields()} == {"A", "B", "C"}
            assert [f["id"] for f in L.fields(bbox=(-105, 35, -95, 45))] == ["A"]
            assert L.fids() == ["A", "B", "C"]

    def test_fids_union(self, tmp_path):
        with Lake(tmp_path) as L:
            L.field("A", 1.0, 2.0)
            L.put("B", "obs", "sm", numpy.array([20240101]), numpy.array([0.5]))
            assert L.fids() == ["A", "B"]

    def test_granule_dedup(self, tmp_path):
        with Lake(tmp_path) as L:
            assert not L.has_granule("G1")
            L.mark_granule("G1", "smap")
            L.mark_granule("G1")
            assert L.has_granule("G1")

    def test_coverage_auto(self, tmp_path):
        with Lake(tmp_path) as L:
            L.put("F1", "obs", "sm", numpy.array([20240102, 20240101]),
                  numpy.array([0.5, 0.4]), "smap")
            (row,) = L.coverage("F1")
            assert row == ("F1", "obs", "sm", 20240101, 20240102, 2, "smap", 1)
            L.put("F1", "obs", "sm", numpy.array([20240103]), numpy.array([0.9]))
            (row,) = L.coverage("F1")
            assert row[3:6] == (20240101, 20240103, 3)  # source kept, rev bumped
            assert row[6] == "smap" and row[7] == 2

    def test_reput_writes_nothing(self, tmp_path):
        with Lake(tmp_path) as L:
            L.put("F1", "obs", "sm", numpy.array([20240101]), numpy.array([0.5]), "smap")
            assert L.coverage("F1")[0][-1] == 1
            L.put("F1", "obs", "sm", numpy.array([20240101]), numpy.array([0.5]))
            assert L.coverage("F1")[0][-1] == 1   # identical merge → no write

    def test_vec(self, tmp_path):
        x = numpy.arange(64, dtype=numpy.float32)
        with Lake(tmp_path) as L:
            L.put_vec("F1", "aef", x)
            assert numpy.array_equal(L.vector("F1", "aef"), x)
            assert L.vector("F1", "nope") is None
            L.put_vec("F1", "aef", x + 1)
            assert numpy.array_equal(L.vector("F1", "aef"), x + 1)

    def test_tx_atomic(self, tmp_path):
        with Lake(tmp_path) as L:
            with pytest.raises(RuntimeError):
                with L.transaction():
                    L.put("F1", "obs", "sm", numpy.array([20240101]),
                          numpy.array([0.5]))
                    L.mark_granule("G1", "smap")
                    raise RuntimeError("boom")
            assert L.get("F1", "obs", "sm") is None
            assert not L.has_granule("G1")

    def test_tx_single_commit(self, tmp_path):
        with Lake(tmp_path) as L:
            commits = []
            L.database.set_trace_callback(
                lambda s: commits.append(1) if s.strip() == "COMMIT" else None)
            L.put("F1", "obs", "sm", numpy.array([20240101]), numpy.array([0.5]))
            assert commits == []                # autocommit: no explicit COMMIT
            with L.transaction():
                L.put("F1", "obs", "et", numpy.array([20240101]), numpy.array([2.0]))
                L.put("F1", "obs", "lst", numpy.array([20240101]), numpy.array([300.0]))
                L.mark_granule("G", "p")
            assert commits == [1]               # one commit for the whole unit

    def test_tx_nested(self, tmp_path):
        with Lake(tmp_path) as L:
            seen = []
            L.database.set_trace_callback(lambda s: seen.append(s.strip()))
            with L.transaction():
                with L.transaction():
                    L.put("F1", "obs", "sm", numpy.array([20240101]),
                          numpy.array([0.5]))
            assert seen.count("COMMIT") == 1
            assert seen.count("BEGIN IMMEDIATE") == 1

    def test_snapshot_isolation(self, tmp_path):
        with Lake(tmp_path) as L:
            L.put("F1", "obs", "sm", numpy.array([20240101]), numpy.array([0.1]))
            with L.transaction():
                L.put("F1", "obs", "sm", numpy.array([20240102]), numpy.array([0.2]))
                with Lake(tmp_path, readonly=True) as R:
                    d, _ = R.get("F1", "obs", "sm")
                    assert d.tolist() == [20240101]   # pre-tx snapshot
            with Lake(tmp_path, readonly=True) as R:
                assert R.get("F1", "obs", "sm")[0].tolist() \
                    == [20240101, 20240102]

    def test_read_only_no_files(self, tmp_path):
        with Lake(tmp_path) as L:
            L.put("F1", "obs", "sm", numpy.array([20240101]), numpy.array([0.5]))
        with Lake(tmp_path, readonly=True) as R:
            assert R.get("F1", "obs", "sm") is not None
        assert {p.name for p in tmp_path.iterdir()} <= \
            {"lake.db", "lake.db-shm", "lake.db-wal"}  # sqlite wal-index only
        with Lake(tmp_path):                        # next writer reclaims it
            pass
        assert [p.name for p in tmp_path.iterdir()] == ["lake.db"]

    def test_missing_ro(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Lake(tmp_path / "nowhere", readonly=True)

    def test_one_file(self, tmp_path):
        with Lake(tmp_path) as L:
            L.field("F1", 1.0, 2.0)
            L.put("F1", "obs", "sm", numpy.array([20240101]), numpy.array([0.5]))
            L.mark_granule("G", "p")
            L.put_vec("F1", "aef", numpy.ones(4, numpy.float32))
        assert [p.name for p in tmp_path.iterdir()] == ["lake.db"]


class TestMigrate:
    def _legacy(self, root):
        import zarr
        g = zarr.open_group(root / "F1.zarr", mode="w")
        g.attrs["soil"] = {"clay": 30}
        for group, var, d, v in [
            ("forcing", "tmin", [20240101, 20240102], [4.0, 5.0]),
            ("obs", "sm", [20240102], [0.4]),
            ("emb", "aef", [20240000],
             numpy.arange(4, dtype=numpy.float32).reshape(1, 4)),
        ]:
            node = g.require_group(f"{group}/{var}")
            node.create_array("dates", data=numpy.array(d, numpy.int32))
            node.create_array("values", data=numpy.asarray(v, numpy.float32))
        c = sqlite3.connect(root / "catalog.db")
        c.executescript(
            "CREATE TABLE fields(id TEXT PRIMARY KEY, lat REAL, lon REAL,"
            " name TEXT);"
            "CREATE TABLE granules(id TEXT PRIMARY KEY, product TEXT,"
            " at TEXT DEFAULT (datetime('now')));"
            "CREATE TABLE series(fid TEXT, grp TEXT, var TEXT, first INT,"
            " last INT, n INT, source TEXT, PRIMARY KEY(fid, grp, var));")
        c.execute("INSERT INTO fields VALUES('F1', 48.0, 1.5, 'parcelle')")
        c.execute("INSERT INTO granules VALUES('G1', 'smap', datetime('now'))")
        c.execute("INSERT INTO series VALUES('F1', 'obs', 'sm', 20240102,"
                  " 20240102, 1, 'smap')")
        c.commit()
        c.close()

    def test_migrate_parity(self, tmp_path):
        self._legacy(tmp_path)
        stats = migrate(tmp_path)
        assert stats["series"] == 2 and stats["granules"] == 1
        assert (tmp_path / "F1.zarr").exists()      # non-destructive default
        with Lake(tmp_path) as L:
            d, v = L.get("F1", "forcing", "tmin")
            assert d.tolist() == [20240101, 20240102]
            assert numpy.array_equal(v, numpy.array([4.0, 5.0], numpy.float32))
            assert numpy.allclose(L.get("F1", "obs", "sm")[1], [0.4])
            assert L.vector("F1", "aef").tolist() == [0, 1, 2, 3]
            assert L.metadata("F1")["soil"] == {"clay": 30}
            assert L.has_granule("G1")
            assert L.fields()[0]["lat"] == 48.0
            assert ("F1", "obs", "sm", 20240102, 20240102, 1, "smap", 1) \
                in L.coverage("F1")

    def test_auto_migrate_on_open(self, tmp_path):
        self._legacy(tmp_path)
        with Lake(tmp_path) as L:                   # no explicit migrate call
            assert L.get("F1", "forcing", "tmin") is not None
            assert L.migrated["series"] == 2

    def test_migrate_idempotent(self, tmp_path):
        self._legacy(tmp_path)
        migrate(tmp_path)
        with Lake(tmp_path) as L:
            rev = L.coverage("F1")[0][-1]
        migrate(tmp_path)                           # second run: no writes
        with Lake(tmp_path) as L:
            assert L.coverage("F1")[0][-1] == rev

    def test_migrate_clean(self, tmp_path):
        self._legacy(tmp_path)
        migrate(tmp_path, clean=True)
        assert not list(tmp_path.glob("*.zarr"))
        assert not (tmp_path / "catalog.db").exists()
        with Lake(tmp_path) as L:
            assert L.get("F1", "obs", "sm") is not None

    def test_fresh_empty(self, tmp_path):
        with Lake(tmp_path) as L:
            assert L.fids() == [] and L.migrated == {
                "fields": 0, "series": 0, "emb": 0, "granules": 0}


_KILL_MID_TX = """
import os, sys, numpy
from nasa_farm.lake import Lake
L = Lake(sys.argv[1])
with L.transaction():
    L.put("F1", "obs", "sm", numpy.array([20240101]), numpy.array([.5]))
    L.mark_granule("G1", "smap")
    os._exit(0)                    # dead before COMMIT
"""

_KILL_BETWEEN_UNITS = """
import os, sys, numpy
from nasa_farm.lake import Lake
L = Lake(sys.argv[1])
with L.transaction():
    L.put("F1", "obs", "sm", numpy.array([20240101]), numpy.array([.5]))
    L.mark_granule("G1", "smap")
with L.transaction():                       # unit 2 begins
    L.put("F1", "obs", "sm", numpy.array([20240102]), numpy.array([.6]))
    L.mark_granule("G2", "smap")
    os._exit(0)                             # dead before COMMIT
"""


class TestFault:
    def test_kill_mid_tx_leaves_nothing(self, tmp_path):
        subprocess.run([sys.executable, "-c", _KILL_MID_TX, str(tmp_path)])
        with Lake(tmp_path) as L:
            assert L.get("F1", "obs", "sm") is None
            assert not L.has_granule("G1")

    def test_kill_between_units_resumes(self, tmp_path):
        subprocess.run(
            [sys.executable, "-c", _KILL_BETWEEN_UNITS, str(tmp_path)])
        with Lake(tmp_path) as L:
            assert L.get("F1", "obs", "sm")[0].tolist() == [20240101]
            assert L.has_granule("G1") and not L.has_granule("G2")
            with L.transaction():           # resumable: replay unit 2
                L.put("F1", "obs", "sm", numpy.array([20240102]),
                      numpy.array([.6]))
                L.mark_granule("G2", "smap")
            d, v = L.get("F1", "obs", "sm")
            assert d.tolist() == [20240101, 20240102]
            assert L.has_granule("G2")

    def test_readers_never_see_partial(self, tmp_path):
        with Lake(tmp_path) as L:
            L.field("F1", 1.0, 2.0)
        errors, done = [], threading.Event()

        def read():
            while not done.is_set():
                try:
                    with Lake(tmp_path, readonly=True) as R:
                        for fid, grp, var, _, _, n, _, _ in R.coverage():
                            got = R.get(fid, grp, var)
                            if got is None:
                                errors.append("coverage but no series")
                            else:
                                d, v = got
                                if len(d) != len(v) or len(d) != n \
                                        or numpy.diff(d).min() <= 0:
                                    errors.append("torn series")
                except sqlite3.OperationalError:
                    pass                    # db mid-schema: retry

        threads = [threading.Thread(target=read) for _ in range(4)]
        for t in threads:
            t.start()
        with Lake(tmp_path) as L:
            for i in range(20):
                with L.transaction():
                    L.put("F1", "obs", f"s{i}",
                          numpy.array([20240101 + j for j in range(50)]),
                          numpy.arange(50, dtype=numpy.float32))
                    L.mark_granule(f"G{i}", "p")
        done.set()
        for t in threads:
            t.join()
        assert errors == []

    def test_load_memory_budget(self, tmp_path):
        d = numpy.array(sorted(numpy.random.default_rng(0).choice(
            to_date(numpy.arange(
                to_ordinal(numpy.array([20200101]))[0],
                to_ordinal(numpy.array([20251231]))[0] + 1)),
            size=2000, replace=False)), numpy.int32)
        raw = d.size * 8 * 40                        # dates+vals, 40 series
        with Lake(tmp_path) as L:
            for i in range(40):
                L.put("F1", "forcing", f"v{i}", d,
                      numpy.random.default_rng(i).random(
                          d.size, dtype=numpy.float32))
        tracemalloc.start()
        with Lake(tmp_path, readonly=True) as L:
            out = [L.get("F1", "forcing", f"v{i}") for i in range(40)]
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        assert all(x is not None for x in out)
        assert peak < raw * 2                        # no per-point objects
