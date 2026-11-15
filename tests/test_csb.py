"""CSB acquisition — pure-function tests (no network).

`fields`/`sequence` are thin adapters over pyarrow/Icechunk; everything
testable offline lives here: WKB decoding, ring selection, centroids,
stratification, GeoJSON emission."""

import struct

import pytest

from nasa_farm.acquire.csb import (CDL, Boundary, _centroid, _ring,
                                   _shoelace, _stratify, _wkb, feature,
                                   label)


def _square(x: float, y: float, side: float) -> list[tuple[float, float]]:
    return [(x, y), (x + side, y), (x + side, y + side),
            (x, y + side), (x, y)]


def _wkb_polygon(*rings: list) -> bytes:
    out = bytearray(b"\x01" + struct.pack("<I", 3) + struct.pack("<I", len(rings)))
    for ring in rings:
        out += struct.pack("<I", len(ring))
        for x, y in ring:
            out += struct.pack("<dd", x, y)
    return bytes(out)


def _wkb_multi(*polygons: bytes) -> bytes:
    out = bytearray(b"\x01" + struct.pack("<I", 6) + struct.pack("<I", len(polygons)))
    for polygon in polygons:
        out += polygon
    return bytes(out)


class TestWkb:
    def test_polygon_outer_ring_only(self):
        outer = _square(0, 0, 4)
        hole = _square(1, 1, 1)
        rings = _wkb(_wkb_polygon(outer, hole))
        assert rings == [outer]

    def test_multipolygon_all_parts(self):
        big = _wkb_polygon(_square(10, 10, 4))
        small = _wkb_polygon(_square(0, 0, 1))
        rings = _wkb(_wkb_multi(small, big))
        assert len(rings) == 2
        assert rings[1][1] == (14.0, 10.0)

    def test_big_endian(self):
        ring = _square(0, 0, 2)
        blob = bytearray(b"\x00" + struct.pack(">I", 3) + struct.pack(">I", 1)
                         + struct.pack(">I", len(ring)))
        for x, y in ring:
            blob += struct.pack(">dd", x, y)
        assert _wkb(bytes(blob)) == [ring]

    def test_rejects_non_polygon(self):
        point = b"\x01" + struct.pack("<I", 1) + struct.pack("<dd", 1, 2)
        with pytest.raises(ValueError, match="unsupported WKB"):
            _wkb(point)


class TestRing:
    def test_picks_largest(self):
        blob = _wkb_multi(_wkb_polygon(_square(0, 0, 1)),
                          _wkb_polygon(_square(10, 10, 4)))
        assert _ring(blob)[1] == (14.0, 10.0)

    def test_closes_open_ring(self):
        ring = _ring(_wkb_polygon(_square(0, 0, 1)[:-1]))
        assert ring[0] == ring[-1]


class TestCentroid:
    def test_square(self):
        ring = _square(0, 0, 2) + [(0, 0)]
        assert _centroid(ring) == (1.0, 1.0)

    def test_direction_invariant(self):
        ring = _square(0, 0, 2) + [(0, 0)]
        assert _centroid(ring) == _centroid(ring[::-1])


class TestStratify:
    def _rows(self, spec: dict[int, int]) -> list[tuple]:
        rows = []
        for code, count in spec.items():
            rows += [(f"id{code}_{i}", code, float(100 - i), [])
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


class TestFeature:
    def test_emits_app_schema(self):
        ring = _square(-93.6, 42.0, 0.01)
        boundary = Boundary(
            id="abc", name="Story 1", crop="Corn", kind="maize", ha=50.0,
            center=_centroid(ring + [ring[0]]), ring=ring + [ring[0]],
            x=1.0, y=2.0)
        f = feature(boundary, [1, 5, 1])
        assert f["geometry"]["type"] == "Polygon"
        assert f["properties"]["sequence"] == ["Corn", "Soybeans", "Corn"]
        assert f["properties"]["kind"] == "maize"

    def test_label_fallback(self):
        assert label(1) == "Corn"
        assert label(999) == "Other"


class TestShoelace:
    def test_signed_area(self):
        assert _shoelace(_square(0, 0, 3) + [(0, 0)]) == 18.0
