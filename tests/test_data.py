"""Lake→tensor contract — weighted targets, embeddings, soil profiles."""

import numpy as np
import pytest

from nasa_farm.lake import Lake
from nasa_farm.optimize.data import (
    emb, soil, soil_sample, targets, to_tensors,
)


class TestTargets:
    def test_weights_loaded(self, tmp_path):
        d = np.array([20240101, 20240106], np.int32)
        with Lake(tmp_path) as L:
            L.put("F1", "obs", "sm", d, np.array([0.3, 0.4], np.float32))
            L.put("F1", "obs", "sm_w", d, np.array([1.0, 0.5], np.float32))
        t = targets(tmp_path, "F1")
        assert t["sm"][2].tolist() == [1.0, 0.5]

    def test_missing_weights_none(self, tmp_path):
        d = np.array([20240101], np.int32)
        with Lake(tmp_path) as L:
            L.put("F1", "obs", "et", d, np.array([2.5], np.float32))
        t = targets(tmp_path, "F1")
        assert t["et"][2] is None

    def test_proxy_not_a_target(self, tmp_path):
        d = np.array([20240101], np.int32)
        with Lake(tmp_path) as L:
            L.put("F1", "obs", "smg", d, np.array([0.5], np.float32))
        assert "smg" not in targets(tmp_path, "F1")

    def test_to_tensors(self, tmp_path):
        d = np.array([20240101], np.int32)
        with Lake(tmp_path) as L:
            L.put("F1", "obs", "ndvi", d, np.array([0.6], np.float32))
            L.put("F1", "obs", "ndvi_w", d, np.array([0.8], np.float32))
        out = to_tensors(targets(tmp_path, "F1"))
        dd, vv, ww = out["ndvi"]
        assert float(vv[0]) == pytest.approx(0.6)
        assert float(ww[0]) == pytest.approx(0.8)


class TestEmb:
    def test_roundtrip(self, tmp_path):
        v = np.arange(64, dtype=np.float32).reshape(1, 64)
        with Lake(tmp_path) as L:
            L.put_vec("F1", "aef", v)
        e = emb(tmp_path, "F1")
        assert e.shape == (64,) and e[63] == pytest.approx(63.0)

    def test_absent_is_none(self, tmp_path):
        Lake(tmp_path).close()
        assert emb(tmp_path, "F1") is None


class TestSoil:
    def _d(self):
        import dataclasses
        from nasa_farm.state.soil import SoilProfile
        return dataclasses.asdict(
            SoilProfile.preset("silt_loam")).copy()

    def test_dict_roundtrip(self):
        s = soil(self._d())
        assert s is not None and s.n_layers == 2
        assert float(s.clay[0, 0]) == pytest.approx(20.0)

    def test_none_passthrough(self):
        assert soil(None) is None
        assert soil_sample(None, self._d(), 4) is None

    def test_sample_ensemble(self):
        lo, hi = self._d(), self._d()
        hi["clay"] = (np.asarray(hi["clay"]) + 10.0).tolist()
        s = soil_sample(lo, hi, 8, seed=0)
        assert s.batch == 8
        assert (s.clay >= 20.0).all() and (s.clay <= 30.0).all()
