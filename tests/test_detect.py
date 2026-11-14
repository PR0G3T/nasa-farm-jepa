"""Action inference — SM2RAIN irrigation, coherence sowing."""

import pytest
import torch

from nasa_farm.constants import DTYPE
from nasa_farm.observe.detect import infer_actions, infer_irrigation


class TestIrrigation:
    def test_dry_no_rain_is_zero(self):
        sm = torch.full((60,), 0.25, dtype=DTYPE)
        p = torch.zeros(60, dtype=DTYPE)
        output = infer_irrigation(sm, p)
        assert float(output.max()) == pytest.approx(0.0, abs=1.0)

    def test_step_without_rain_detected(self):
        """A 0.1 jump over 200 mm of soil ≈ 20 mm unexplained input."""
        sm = torch.full((60,), 0.25, dtype=DTYPE)
        sm[20:] = 0.35
        p = torch.zeros(60, dtype=DTYPE)
        output = infer_irrigation(sm, p, depth_mm=200.0)
        assert 18.0 < float(output[20]) < 25.0
        assert float(output[:20].sum()) < 1.0

    def test_rain_covering_rise_is_zero(self):
        """Rain that already explains the wetting leaves no irrigation."""
        sm = torch.full((60,), 0.25, dtype=DTYPE)
        sm[20:] = 0.30
        p = torch.zeros(60, dtype=DTYPE)
        p[20] = 40.0
        output = infer_irrigation(sm, p, depth_mm=200.0)
        assert float(output[20]) < 1.0

    def test_small_residuals_clipped(self):
        """Sub-mm balance noise must not become phantom events."""
        g = torch.Generator().manual_seed(0)
        sm = 0.25 + 0.001 * torch.randn(60, generator=g, dtype=DTYPE)
        p = torch.zeros(60, dtype=DTYPE)
        output = infer_irrigation(sm, p, min_mm=1.0)
        assert float(output.max()) < 1.0

    def test_monotone_in_storage_gain(self):
        """Bigger unexplained wetting steps → bigger detected volumes."""
        p = torch.zeros(60, dtype=DTYPE)
        sm_a = torch.full((60,), 0.25, dtype=DTYPE)
        sm_a[30:] += 0.05
        sm_b = torch.full((60,), 0.25, dtype=DTYPE)
        sm_b[30:] += 0.15
        a = infer_irrigation(sm_a, p)
        b = infer_irrigation(sm_b, p)
        assert float(b.sum()) > float(a.sum())


class TestActions:
    def _ndvi(self):
        t = torch.arange(120, dtype=DTYPE)
        ndvi = torch.full((120,), 0.15, dtype=DTYPE)
        ndvi[40:80] += torch.linspace(0, 0.6, 40)
        ndvi[80:] = torch.linspace(0.75, 0.2, 40)
        return t, ndvi

    def test_irrigation_filled(self):
        t, ndvi = self._ndvi()
        sm = torch.full((120,), 0.2, dtype=DTYPE)
        sm[50:] = 0.3
        p = torch.zeros(120, dtype=DTYPE)
        a = infer_actions(ndvi, t, crop_idx=2, sm=sm, precip=p)
        assert float(a.irrigation[0, 50]) > 10.0
        assert float(a.irrigation[0, 60]) < 1.0   # single event, not a trend

    def test_coherence_refines_sowing(self):
        t, ndvi = self._ndvi()
        coh = torch.full((120,), 0.8, dtype=DTYPE)
        coh[45] = 0.2                           # disturbance dip near green-up
        base = infer_actions(ndvi, t, crop_idx=2)
        reference = infer_actions(ndvi, t, crop_idx=2, coh=coh)
        d_base = int((base.sow[0] == 2).nonzero()[0])
        d_ref = int((reference.sow[0] == 2).nonzero()[0])
        assert d_ref == 45 and d_ref != d_base

    def test_no_signals_stays_zero(self):
        t, ndvi = self._ndvi()
        a = infer_actions(ndvi, t, crop_idx=2)
        assert float(a.irrigation.sum()) == 0.0
        assert float(a.till.sum()) == 0.0
