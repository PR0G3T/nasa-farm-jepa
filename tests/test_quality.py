"""QC flags → confidence maps — per-product decoding."""

import numpy as np
import pytest
import xarray as xr

from nasa_farm.acquire.qc import FLAG_QC, SMAP_QC, _smap_conf, _strict_conf


class TestSmapConf:
    def test_recommended(self):
        """0 and 8 are recommended quality per the product spec."""
        assert _smap_conf(np.array([[0.0, 8.0]])).tolist() == [[1.0, 1.0]]

    def test_attempted_but_uncertain(self):
        """bit0 set, bit1 clear → retrieved, uncertain → 0.4."""
        assert _smap_conf(np.array([[1.0, 9.0]])).tolist() == [[0.4, 0.4]]

    def test_skipped_is_zero(self):
        """bit1 set → retrieval skipped → unusable."""
        assert _smap_conf(np.array([[2.0, 3.0, 7.0]])).tolist() == \
            [[0.0, 0.0, 0.0]]

    def test_nodata_is_zero(self):
        assert _smap_conf(np.array([[-9999.0, 300.0]])).tolist() == \
            [[0.0, 0.0]]


class TestStrictConf:
    def test_zero_clean(self):
        assert _strict_conf(np.array([[0.0, 1.0]])).tolist() == \
            [[1.0, 0.4]]

    def test_nan_is_zero(self):
        assert _strict_conf(np.array([[np.nan]])).tolist() == [[0.0]]


class TestSpec:
    def test_resolves_first_candidate(self):
        ds = xr.Dataset({"retrieval_qual_flag": (("y", "x"),
                                                 np.zeros((2, 2)))})
        assert SMAP_QC.mask(ds).tolist() == [[1.0, 1.0], [1.0, 1.0]]

    def test_absent_band_is_none(self):
        ds = xr.Dataset({"v": (("y", "x"), np.ones((2, 2)))})
        assert SMAP_QC.mask(ds) is None
        assert FLAG_QC.mask(ds) is None
