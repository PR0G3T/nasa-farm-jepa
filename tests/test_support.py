"""Footprint-aware sampling — kernel convexity, representativeness."""

import numpy as numpy
import pytest
import xarray as xarray

from nasa_farm.acquire.support import coverage, kernel, sample


def _grid(values, lats=(48.0, 49.0, 50.0), lons=(1.0, 2.0, 3.0)):
    return xarray.Dataset(
        {"v": (("lat", "lon"), numpy.asarray(values, float))},
        coords={"lat": list(lats), "lon": list(lons)})


class TestKernel:
    def test_weights_are_convex_cutoff(self):
        d = numpy.array([0.0, 5.0, 10.0, 30.0])
        w = kernel(d, 20.0)
        assert w[0] == pytest.approx(1.0)
        assert (w[1] > 0) and (w[2] > 0) and w[3] == 0.0
        assert w[1] > w[2]                          # monotone decay


class TestSample:
    def test_zero_radius_is_nearest(self):
        ds = _grid(numpy.arange(9.0).reshape(3, 3))
        assert sample(ds, "v", 48.1, 2.1) == pytest.approx(1.0)

    def test_constant_field_exact(self):
        ds = _grid(numpy.full((3, 3), 4.2))
        assert sample(ds, "v", 48.9, 2.2, radius_km=200.0) == \
            pytest.approx(4.2)

    def test_linear_field_returns_center(self):
        """Symmetric kernel over a linear field = value at the point."""
        lats, lons = numpy.linspace(40.0, 42.0, 21), numpy.linspace(0.0, 2.0, 21)
        gl, gn = numpy.meshgrid(lats, lons, indexing="ij")
        v = 2.0 * gl + 3.0 * gn - 80.0
        ds = _grid(v, lats, lons)
        latitude, longitude = 41.0, 1.0
        assert sample(ds, "v", latitude, longitude, radius_km=60.0) == \
            pytest.approx(2.0 * latitude + 3.0 * longitude - 80.0, relative=1e-9)

    def test_within_footprint_extremes(self):
        v = numpy.arange(9.0).reshape(3, 3)
        ds = _grid(v)
        output = sample(ds, "v", 49.0, 2.0, radius_km=250.0)
        assert v.min() <= output <= v.max()

    def test_nan_cells_dropped(self):
        v = numpy.ones((3, 3))
        v[0, 0] = numpy.nan
        ds = _grid(v)
        assert sample(ds, "v", 48.05, 1.05, radius_km=300.0) == \
            pytest.approx(1.0)

    def test_confidence_zeroes_cells(self):
        v = numpy.arange(9.0).reshape(3, 3)
        confidence = numpy.zeros((3, 3))
        confidence[2, 2] = 1.0
        ds = _grid(v)
        assert sample(ds, "v", 48.5, 1.5, radius_km=400.0,
                      confidence=confidence) == pytest.approx(8.0)

    def test_all_masked_is_nan(self):
        ds = _grid(numpy.ones((3, 3)))
        assert numpy.isnan(sample(ds, "v", 49.0, 2.0, radius_km=300.0,
                               confidence=numpy.zeros((3, 3))))


class TestCoverage:
    def test_full_valid_is_one(self):
        ds = _grid(numpy.ones((3, 3)))
        assert coverage(ds, "v", 49.0, 2.0, 300.0) == pytest.approx(1.0)

    def test_half_nan_is_partial(self):
        v = numpy.ones((2, 2))
        v[:, 0] = numpy.nan
        ds = _grid(v, lats=(49.0, 49.01), lons=(2.0, 2.01))
        c = coverage(ds, "v", 49.0, 2.0, 1000.0)
        assert 0.0 < c < 1.0

    def test_confidence_scales(self):
        ds = _grid(numpy.ones((3, 3)))
        c = coverage(ds, "v", 49.0, 2.0, 300.0,
                     confidence=numpy.full((3, 3), 0.5))
        assert c == pytest.approx(0.5)

    def test_outside_is_zero(self):
        ds = _grid(numpy.ones((3, 3)))
        assert coverage(ds, "v", 0.0, 0.0, 50.0) == 0.0
