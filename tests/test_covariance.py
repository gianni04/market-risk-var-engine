"""Tests du module marketrisk.covariance."""

from __future__ import annotations

import numpy as np

from marketrisk import covariance as COV


class TestLedoitWolfShrinkage:
    def test_shrunk_covariance_is_positive_definite(self, synthetic_returns) -> None:
        shrunk, delta = COV.ledoit_wolf_shrinkage(synthetic_returns)
        assert COV.is_positive_definite(shrunk)
        assert 0.0 <= delta <= 1.0

    def test_sklearn_cross_check_is_positive_definite(self, synthetic_returns) -> None:
        shrunk, delta = COV.ledoit_wolf_sklearn(synthetic_returns)
        assert COV.is_positive_definite(shrunk)
        assert 0.0 <= delta <= 1.0

    def test_shrinkage_reduces_condition_number(self, synthetic_returns) -> None:
        sample = COV.sample_covariance(synthetic_returns)
        shrunk, delta = COV.ledoit_wolf_shrinkage(synthetic_returns)
        assert delta > 0.0
        cond_sample = np.linalg.cond(sample)
        cond_shrunk = np.linalg.cond(shrunk)
        assert cond_shrunk <= cond_sample


class TestNearestPSD:
    def test_repairs_non_psd_matrix(self) -> None:
        broken = np.array(
            [
                [1.0, 0.99, 0.99],
                [0.99, 1.0, -0.99],
                [0.99, -0.99, 1.0],
            ]
        )
        assert not COV.is_positive_definite(broken)
        fixed = COV.nearest_psd(broken)
        assert COV.is_positive_definite(fixed)

    def test_already_psd_matrix_is_almost_unchanged(self, synthetic_returns) -> None:
        cov = COV.sample_covariance(synthetic_returns)
        fixed = COV.nearest_psd(cov, epsilon=1e-12)
        np.testing.assert_allclose(cov, fixed, atol=1e-8)

    def test_symmetry_preserved(self) -> None:
        rng = np.random.default_rng(0)
        m = rng.normal(size=(6, 6))
        m = m + m.T
        fixed = COV.nearest_psd(m)
        np.testing.assert_allclose(fixed, fixed.T, atol=1e-10)


class TestEwmaCovariance:
    def test_output_is_positive_definite(self, synthetic_returns) -> None:
        ewma_cov = COV.ewma_covariance(synthetic_returns, lam=0.94)
        assert COV.is_positive_definite(ewma_cov)

    def test_output_shape(self, synthetic_returns) -> None:
        ewma_cov = COV.ewma_covariance(synthetic_returns, lam=0.94)
        n_assets = synthetic_returns.shape[1]
        assert ewma_cov.shape == (n_assets, n_assets)
