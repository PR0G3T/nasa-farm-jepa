"""N and C pool invariants — mass-conservative transfers, legume logic."""

import pytest
import torch

from nasa_farm.constants import DTYPE
from nasa_farm.dynamics.biogeochemistry import (
    K_LEACH, K_MIN, carbon_step, nitrogen_step, nitrogen_stress,
)


def t(v):
    return torch.tensor([float(v)], dtype=DTYPE)


class TestNitrogen:
    def test_mineralization_moves_mass(self):
        n_min, n_org = t(20.0), t(1000.0)
        mn2, no2, f = nitrogen_step(n_min, n_org, t(0.5), t(0.0), t(500.0),
                                    t(0.0), t(0.0), t(0.0), t(0.0))
        assert float(f["mineralized"]) == pytest.approx(1000.0 * K_MIN * 0.5)
        assert float(n_min + n_org) == pytest.approx(float(mn2 + no2))

    def test_no_uptake_beyond_available(self):
        mn2, _, f = nitrogen_step(t(10.0), t(0.0), t(0.0), t(0.0), t(500.0),
                                  t(50.0), t(0.0), t(0.0), t(0.0))
        assert float(f["uptake"]) <= 10.0
        assert float(mn2) >= 0.0

    def test_fixation_legume_only(self):
        _, _, f_leg = nitrogen_step(t(5.0), t(0.0), t(0.0), t(0.0), t(500.0),
                                    t(10.0), t(1.0), t(0.8), t(0.0))
        _, _, f_cer = nitrogen_step(t(5.0), t(0.0), t(0.0), t(0.0), t(500.0),
                                    t(10.0), t(0.0), t(0.8), t(0.0))
        assert float(f_leg["fixed"]) > 0.0
        assert float(f_cer["fixed"]) == 0.0

    def test_fixation_downregulates_when_n_sufficient(self):
        args = (t(10.0), t(1.0), t(0.8), t(0.0))
        _, _, f_low = nitrogen_step(t(0.0), t(0.0), t(0.0), t(0.0), t(500.0), *args)
        _, _, f_hi = nitrogen_step(t(200.0), t(0.0), t(0.0), t(0.0), t(500.0), *args)
        assert float(f_low["fixed"]) > float(f_hi["fixed"]) > -1e-12
        assert float(f_hi["fixed"]) == pytest.approx(0.0)

    def test_leaching_requires_drainage(self):
        def run(drain):
            return nitrogen_step(t(100.0), t(0.0), t(0.5), drain, t(500.0),
                                 t(0.0), t(0.0), t(0.0), t(0.0))[2]["leached"]
        assert float(run(t(0.0))) == 0.0
        assert 0.0 < float(run(t(50.0))) <= 100.0 * K_LEACH

    def test_fertilizer_adds_mineral_n(self):
        mn2, _, _ = nitrogen_step(t(10.0), t(0.0), t(0.0), t(0.0), t(500.0),
                                  t(0.0), t(0.0), t(0.0), t(80.0))
        assert float(mn2) == pytest.approx(90.0)

    def test_nitrogen_stress_michaelis(self):
        assert float(nitrogen_stress(t(0.0))) == 0.0
        assert float(nitrogen_stress(t(25.0))) == pytest.approx(0.5)
        assert float(nitrogen_stress(t(1e6))) > 0.999


class TestCarbon:
    def test_litter_decay_feeds_soc(self):
        litter, soc = t(1000.0), t(20_000.0)
        l2, s2, f = carbon_step(litter, soc, t(0.5), t(0.0))
        assert float(f["litter_decay"]) == pytest.approx(1000.0 * 3e-3 * 0.5)
        assert float(l2) < float(litter)
        # Δsoc = humified − soc_decay
        assert float(s2 - soc) == pytest.approx(
            float(f["humified"] - f["soc_decay"]))

    def test_residue_input_enters_litter(self):
        l2, _, _ = carbon_step(t(0.0), t(0.0), t(0.0), t(500.0))
        assert float(l2) == pytest.approx(500.0)

    def test_pools_nonnegative(self):
        l2, s2, _ = carbon_step(t(0.0), t(0.0), t(1.0), t(0.0))
        assert float(l2) >= 0.0 and float(s2) >= 0.0
