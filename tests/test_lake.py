"""Lake store + catalog — roundtrip, merge-dedupe, resumability."""

import numpy as np

from nasa_farm.lake import Catalog, get, put
from nasa_farm.lake.store import attrs, meta


class TestStore:
    def test_roundtrip(self, tmp_path):
        put(tmp_path, "F1", "forcing", "T2M_MIN",
            np.array([20240101, 20240102]), np.array([4.0, 5.0]))
        d, v = get(tmp_path, "F1", "forcing", "T2M_MIN")
        assert d.tolist() == [20240101, 20240102]
        assert np.allclose(v, [4.0, 5.0])

    def test_put_merges_and_dedupes(self, tmp_path):
        put(tmp_path, "F1", "obs", "sm",
            np.array([20240102, 20240101]), np.array([0.5, 0.4]))
        put(tmp_path, "F1", "obs", "sm",
            np.array([20240102, 20240103]), np.array([0.9, 0.6]))
        d, v = get(tmp_path, "F1", "obs", "sm")
        assert d.tolist() == [20240101, 20240102, 20240103]
        assert v[1] in (0.5, 0.9)  # first-wins on dup, order restored

    def test_missing_returns_none(self, tmp_path):
        assert get(tmp_path, "nope", "obs", "sm") is None

    def test_attrs(self, tmp_path):
        attrs(tmp_path, "F1", soil={"clay": 30})
        assert meta(tmp_path, "F1")["soil"] == {"clay": 30}
        assert meta(tmp_path, "nope") == {}


class TestCatalog:
    def test_granule_dedupe(self, tmp_path):
        with Catalog(tmp_path / "c.db") as c:
            assert not c.has_granule("G1")
            c.mark_granule("G1", "smap")
            assert c.has_granule("G1")
            c.mark_granule("G1", "smap")  # idempotent

    def test_series_coverage(self, tmp_path):
        with Catalog(tmp_path / "c.db") as c:
            c.field("F1", 48.0, 1.5, "parcelle")
            c.set_series("F1", "obs", "sm", 20240101, 20241231, 366, "smap")
            rows = c.series("F1")
            assert rows[0][2] == "sm" and rows[0][5] == 366
            assert c.series("F2") == []
