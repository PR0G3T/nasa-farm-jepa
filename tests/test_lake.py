"""Lake — packed series, catalog, transactions, migration."""

import datetime as dt
import sqlite3

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from nasa_farm.lake import Lake, migrate
from nasa_farm.lake.calendar import to_ord, to_ymd
from nasa_farm.lake.series import decode, encode, merge


def _ymd(d: dt.date) -> int:
    return d.year * 10000 + d.month * 100 + d.day


class TestCal:
    @settings(max_examples=300)
    @given(st.dates(min_value=dt.date(1200, 1, 1),
                    max_value=dt.date(9999, 12, 30)))
    def test_matches_datetime(self, d):
        assert to_ord(np.array([_ymd(d)]))[0] == d.toordinal()
        assert to_ymd(np.array([d.toordinal()]))[0] == _ymd(d)

    def test_edges(self):
        days = [10101, 19700101, 20000229, 20240229, 21000301, 99991231]
        assert to_ymd(to_ord(np.array(days))).tolist() == days

    def test_monotone(self):
        seq = [19991231, 20000101, 20231231, 20240101,
               20240228, 20240229, 20240301]
        assert np.all(np.diff(to_ord(np.array(seq))) > 0)
        run = [20240227, 20240228, 20240229, 20240301, 20240302]
        assert np.all(np.diff(to_ord(np.array(run))) == 1)


class TestSeries:
    def test_merge_sorted_new_wins(self):
        d, v = merge(None, np.array([20240102, 20240101]),
                     np.array([20.0, 10.0]))
        assert d.tolist() == [20240101, 20240102]
        d, v = merge((d, v), np.array([20240102, 20240103]),
                     np.array([99.0, 30.0]))
        assert d.tolist() == [20240101, 20240102, 20240103]
        assert v.tolist() == [10.0, 99.0, 30.0]

    def test_merge_empty(self):
        d, v = merge(None, np.array([]), np.array([]))
        assert d.size == 0 and v.size == 0

    @given(st.lists(st.integers(20200101, 20201231), max_size=64))
    def test_idempotent(self, days):
        rng = np.random.default_rng(0)
        d = np.array(days, np.int32)
        v = rng.random(len(days), dtype=np.float32)
        m1 = merge(None, d, v)
        m2 = merge(m1, *m1)
        assert np.array_equal(m1[0], m2[0]) and np.array_equal(m1[1], m2[1])

    @given(st.lists(st.integers(20200101, 20201231), min_size=1, max_size=48),
           st.lists(st.integers(20200101, 20201231), min_size=1, max_size=48))
    def test_merge_model(self, a, b):
        rng = np.random.default_rng(0)
        da, db = np.array(a, np.int32), np.array(b, np.int32)
        va = rng.random(len(a), dtype=np.float32)
        vb = rng.random(len(b), dtype=np.float32)
        model = dict(zip(da.tolist(), va.tolist()))
        model.update(zip(db.tolist(), vb.tolist()))   # newest wins
        d, v = merge((da, va), db, vb)
        assert np.all(np.diff(d) > 0)               # strictly sorted unique
        assert d.tolist() == sorted(model)
        assert np.array_equal(v, np.array(
            [model[k] for k in sorted(model)], np.float32))

    @given(st.lists(
        st.dates(min_value=dt.date(2020, 1, 1),
                 max_value=dt.date(2020, 12, 31)),
        min_size=1, max_size=48, unique=True))
    def test_codec_roundtrip(self, days):
        d = np.array(sorted(_ymd(x) for x in days), np.int32)
        v = np.arange(len(d), dtype=np.float32)
        d2, v2 = decode(*encode(d, v))
        assert np.array_equal(d2, d) and np.array_equal(v2, v)


class TestLake:
    def test_roundtrip(self, tmp_path):
        with Lake(tmp_path) as L:
            L.put("F1", "forcing", "tmin",
                  np.array([20240101, 20240102]), np.array([4.0, 5.0]))
            d, v = L.get("F1", "forcing", "tmin")
        assert d.tolist() == [20240101, 20240102]
        assert np.array_equal(v, np.array([4.0, 5.0], np.float32))

    def test_merge_dedup_new_wins(self, tmp_path):
        with Lake(tmp_path) as L:
            L.put("F1", "obs", "sm", np.array([20240101, 20240103]),
                  np.array([0.5, 0.7]))
            L.put("F1", "obs", "sm", np.array([20240102, 20240103]),
                  np.array([0.6, 0.9]))
            d, v = L.get("F1", "obs", "sm")
        assert d.tolist() == [20240101, 20240102, 20240103]
        assert np.array_equal(v, np.array([0.5, 0.6, 0.9], np.float32))

    def test_missing_none(self, tmp_path):
        with Lake(tmp_path) as L:
            assert L.get("F1", "obs", "sm") is None

    def test_attrs_meta(self, tmp_path):
        with Lake(tmp_path) as L:
            L.attrs("F1", soil={"clay": np.float64(30.0)})
            assert L.meta("F1") == {"soil": {"clay": 30.0}}
            L.attrs("F1", name2={"x": [1, 2]})
            m = L.meta("F1")
            assert L.meta("NOPE") == {}
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
            L.put("B", "obs", "sm", np.array([20240101]), np.array([0.5]))
            assert L.fids() == ["A", "B"]

    def test_granule_dedup(self, tmp_path):
        with Lake(tmp_path) as L:
            assert not L.has_granule("G1")
            L.mark_granule("G1", "smap")
            L.mark_granule("G1")
            assert L.has_granule("G1")

    def test_coverage_auto(self, tmp_path):
        with Lake(tmp_path) as L:
            L.put("F1", "obs", "sm", np.array([20240102, 20240101]),
                  np.array([0.5, 0.4]), "smap")
            (row,) = L.coverage("F1")
            assert row == ("F1", "obs", "sm", 20240101, 20240102, 2, "smap", 1)
            L.put("F1", "obs", "sm", np.array([20240103]), np.array([0.9]))
            (row,) = L.coverage("F1")
            assert row[3:6] == (20240101, 20240103, 3)  # source kept, rev bumped
            assert row[6] == "smap" and row[7] == 2

    def test_reput_writes_nothing(self, tmp_path):
        with Lake(tmp_path) as L:
            L.put("F1", "obs", "sm", np.array([20240101]), np.array([0.5]), "smap")
            assert L.coverage("F1")[0][-1] == 1
            L.put("F1", "obs", "sm", np.array([20240101]), np.array([0.5]))
            assert L.coverage("F1")[0][-1] == 1   # identical merge → no write

    def test_vec(self, tmp_path):
        x = np.arange(64, dtype=np.float32)
        with Lake(tmp_path) as L:
            L.put_vec("F1", "aef", x)
            assert np.array_equal(L.vec("F1", "aef"), x)
            assert L.vec("F1", "nope") is None
            L.put_vec("F1", "aef", x + 1)
            assert np.array_equal(L.vec("F1", "aef"), x + 1)

    def test_tx_atomic(self, tmp_path):
        with Lake(tmp_path) as L:
            with pytest.raises(RuntimeError):
                with L.tx():
                    L.put("F1", "obs", "sm", np.array([20240101]),
                          np.array([0.5]))
                    L.mark_granule("G1", "smap")
                    raise RuntimeError("boom")
            assert L.get("F1", "obs", "sm") is None
            assert not L.has_granule("G1")

    def test_tx_single_commit(self, tmp_path):
        with Lake(tmp_path) as L:
            commits = []
            L.db.set_trace_callback(
                lambda s: commits.append(1) if s.strip() == "COMMIT" else None)
            L.put("F1", "obs", "sm", np.array([20240101]), np.array([0.5]))
            assert commits == []                # autocommit: no explicit COMMIT
            with L.tx():
                L.put("F1", "obs", "et", np.array([20240101]), np.array([2.0]))
                L.put("F1", "obs", "lst", np.array([20240101]), np.array([300.0]))
                L.mark_granule("G", "p")
            assert commits == [1]               # one commit for the whole unit

    def test_tx_nested(self, tmp_path):
        with Lake(tmp_path) as L:
            seen = []
            L.db.set_trace_callback(lambda s: seen.append(s.strip()))
            with L.tx():
                with L.tx():
                    L.put("F1", "obs", "sm", np.array([20240101]),
                          np.array([0.5]))
            assert seen.count("COMMIT") == 1
            assert seen.count("BEGIN IMMEDIATE") == 1

    def test_snapshot_isolation(self, tmp_path):
        with Lake(tmp_path) as L:
            L.put("F1", "obs", "sm", np.array([20240101]), np.array([0.1]))
            with L.tx():
                L.put("F1", "obs", "sm", np.array([20240102]), np.array([0.2]))
                with Lake(tmp_path, ro=True) as R:
                    d, _ = R.get("F1", "obs", "sm")
                    assert d.tolist() == [20240101]   # pre-tx snapshot
            with Lake(tmp_path, ro=True) as R:
                assert R.get("F1", "obs", "sm")[0].tolist() \
                    == [20240101, 20240102]

    def test_read_only_no_files(self, tmp_path):
        with Lake(tmp_path) as L:
            L.put("F1", "obs", "sm", np.array([20240101]), np.array([0.5]))
        before = {p.name for p in tmp_path.iterdir()}
        with Lake(tmp_path, ro=True) as R:
            assert R.get("F1", "obs", "sm") is not None
        assert {p.name for p in tmp_path.iterdir()} == before

    def test_missing_ro(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Lake(tmp_path / "nowhere", ro=True)

    def test_one_file(self, tmp_path):
        with Lake(tmp_path) as L:
            L.field("F1", 1.0, 2.0)
            L.put("F1", "obs", "sm", np.array([20240101]), np.array([0.5]))
            L.mark_granule("G", "p")
            L.put_vec("F1", "aef", np.ones(4, np.float32))
        assert [p.name for p in tmp_path.iterdir()] == ["lake.database"]


class TestMigrate:
    def _legacy(self, root):
        import zarr
        g = zarr.open_group(root / "F1.zarr", mode="w")
        g.attrs["soil"] = {"clay": 30}
        for grp, var, d, v in [
            ("forcing", "tmin", [20240101, 20240102], [4.0, 5.0]),
            ("obs", "sm", [20240102], [0.4]),
            ("emb", "aef", [20240000],
             np.arange(4, dtype=np.float32).reshape(1, 4)),
        ]:
            node = g.require_group(f"{grp}/{var}")
            node.create_array("dates", data=np.array(d, np.int32))
            node.create_array("values", data=np.asarray(v, np.float32))
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
            assert np.array_equal(v, np.array([4.0, 5.0], np.float32))
            assert np.allclose(L.get("F1", "obs", "sm")[1], [0.4])
            assert L.vec("F1", "aef").tolist() == [0, 1, 2, 3]
            assert L.meta("F1")["soil"] == {"clay": 30}
            assert L.has_granule("G1")
            assert L.fields()[0]["lat"] == 48.0

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
