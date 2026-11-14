"""Free latent layer — identity init, SIGReg statistics, JEPA loss."""

import pytest
import torch

from nasa_farm.constants import DTYPE
from nasa_farm.dynamics.latent import Dynamics, ObservationEncoder, embedding_loss, regularization
from nasa_farm.dynamics.residual import N_FEATURES


class TestLatentDyn:
    def test_zero_init_is_identity(self):
        dynamics = Dynamics(4)
        z = torch.randn(8, 4, dtype=DTYPE)
        feats = torch.randn(8, N_FEATURES, dtype=DTYPE)
        assert torch.equal(dynamics(feats, z, None)[0], z)

    def test_output_shape(self):
        dynamics = Dynamics(6)
        output, zh = dynamics(torch.randn(3, N_FEATURES, dtype=DTYPE),
                      torch.randn(3, 6, dtype=DTYPE))
        assert output.shape == (3, 6) and zh.shape == (3, 64)


class TestObsEncoder:
    def test_shape(self):
        encoder = ObservationEncoder(observed_dimension=5, z_dim=4)
        assert encoder(torch.randn(7, 5, dtype=DTYPE)).shape == (7, 4)


class TestSIGReg:
    def test_gaussian_scores_low(self):
        z = torch.randn(4096, 8, dtype=DTYPE,
                        generator=torch.Generator().manual_seed(0))
        assert float(regularization(z)) < 0.02

    def test_non_gaussian_scores_higher(self):
        g = torch.Generator().manual_seed(0)
        zg = torch.randn(4096, 8, dtype=DTYPE, generator=g)
        zu = torch.rand(4096, 8, dtype=DTYPE, generator=g)
        assert float(regularization(zu)) > float(regularization(zg))

    def test_degenerate_worst(self):
        z = torch.ones(256, 8, dtype=DTYPE)
        assert float(regularization(z)) > float(
            regularization(torch.randn(256, 8, dtype=DTYPE,
                               generator=torch.Generator().manual_seed(0))))

    def test_deterministic(self):
        z = torch.randn(64, 8, dtype=DTYPE,
                        generator=torch.Generator().manual_seed(1))
        assert float(regularization(z, seed=5)) == pytest.approx(
            float(regularization(z, seed=5)))


class TestJEPALoss:
    def test_perfect_prediction_leaves_only_regularizer(self):
        z = torch.randn(64, 8, dtype=DTYPE,
                        generator=torch.Generator().manual_seed(0))
        l = embedding_loss(z, z.clone(), sig_weight=1.0)
        assert float(l) == pytest.approx(float(regularization(z)), relative=1e-9)

    def test_prediction_error_dominates(self):
        g = torch.Generator().manual_seed(0)
        z_observed = torch.randn(64, 8, dtype=DTYPE, generator=g)
        z_pred = z_observed + 1.0
        assert float(embedding_loss(z_observed, z_pred)) > 0.5

    def test_gradients_flow(self):
        g = torch.Generator().manual_seed(0)
        z_observed = torch.randn(32, 8, dtype=DTYPE, generator=g)
        z_pred = torch.randn(32, 8, dtype=DTYPE, generator=g,
                             ).requires_grad_(True)
        embedding_loss(z_observed, z_pred).backward()
        assert z_pred.gradient is not None and torch.isfinite(z_pred.gradient).all()
