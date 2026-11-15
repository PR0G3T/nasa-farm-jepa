"""CSB acquisition — pure-function tests (no network).

`fields`/`sequence` are thin adapters over DuckDB/Icechunk; everything
testable offline lives here: CDL mapping, ring selection, centroids,
stratification, GeoJSON emission."""

from nasa_farm.acquire.csb import (Boundary, CDL, _centroid, _ring, _sql,
                                   _stratify, feature, label)
from nasa_farm.acquire.base import Bounds


def _square(x: float, y: float, side: float) -> list[list[float]]:
    return [[x, y], [x + side, y], [x + side, y + side],
            [x, y + side], [x, y]]


class TestCentroid:
    def test_square(self):
        ring = [tuple(p) for p in _square(0, 0, 2)]
        assert _centroid(ring) == (1.0, 1.0)

    def test_direction_invariant(self):
        ring = [tuple(p) for p in _square(0, 0, 2)]
        assert _centroid(ring) == _centroid(ring[::-1])


class TestRing:
    def test_polygon_outer(self):
        geometry = {"type": "Polygon", "coordinates": [_square(1, 2, 3)]}
        assert _ring(geometry)[0] == (1.0, 2.0)

    def test_multipolygon_largest(self):
        geometry = {"type": "MultiPolygon",
                    "coordinates": [[_square(0, 0, 1)], [_square(10, 10, 4)]]}
        assert _ring(geometry)[1] == (14.0, 10.0)

    def test_open_ring_closed(self):
        geometry = {"type": "Polygon",
                    "coordinates": [_square(0, 0, 1)[:-1]]}
        ring = _ring(geometry)
        assert ring[0] == ring[-1]

    def test_rejects_other_geometry(self):
        import pytest
        with pytest.raises(ValueError, match="unsupported"):
            _ring({"type": "Point", "coordinates": [0, 0]})


class TestStratify:
    def _rows(self, spec: dict[int, int]) -> list[tuple]:
        rows = []
        for code, count in spec.items():
            rows += [(f"id{code}_{i}", code, float(100 - i), 0, 0, "")
                     for i in range(count)]
        return rows

    def test_quota_per_kind_then_fill(self):
        rows = self._rows({1: 30, 5: 30, 36: 10, 24: 5})
        chosen = _stratify(rows, 20, 5)
        kinds = [CDL[r[1]][1] for r in chosen]
        assert len(chosen) == 20
        for kind in ("maize", "soybean", "alfalfa", "wheat"):
            assert kinds.count(kind) <= 5
        assert "wheat" in kinds  # rare kind is represented

    def test_unmapped_only_backfills(self):
        rows = self._rows({1: 3, 44: 30})
        chosen = _stratify(rows, 10, 5)
        kinds = [CDL[r[1]][1] for r in chosen]
        assert kinds.count("maize") == 3
        assert kinds.count(None) == 7


class TestSql:
    def test_cultivated_filter(self):
        assert 'BETWEEN 1 AND 61' in _sql(None, None, 10)

    def test_county_and_bounds(self):
        sql = _sql(Bounds(-93.8, 41.8, -93.4, 42.2), "Story", 40)
        assert "administrative_area_level_2 = 'Story'" in sql
        assert "bbox.xmax" in sql and "LIMIT 40" in sql


class TestFeature:
    def test_emits_app_schema(self):
        ring = [tuple(p) for p in _square(-93.6, 42.0, 0.01)]
        boundary = Boundary(
            id="abc", name="Story 1", crop="Corn", kind="maize", ha=50.0,
            center=_centroid(ring), ring=ring, x=1.0, y=2.0)
        f = feature(boundary, [1, 5, 1])
        assert f["geometry"]["type"] == "Polygon"
        assert f["properties"]["sequence"] == ["Corn", "Soybeans", "Corn"]
        assert f["properties"]["kind"] == "maize"

    def test_label_fallback(self):
        assert label(1) == "Corn"
        assert label(999) == "Other"
