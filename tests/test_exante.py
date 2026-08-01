"""Tests du module marketrisk.exante."""

from __future__ import annotations

import numpy as np

from marketrisk import covariance as COV
from marketrisk import exante as EX


class TestEulerDecomposition:
    """La somme des VaR de composante doit égaler exactement la VaR totale."""

    def test_component_var_sums_to_total(self, synthetic_returns, portfolio_weights) -> None:
        cov = COV.sample_covariance(synthetic_returns)
        result = EX.component_var(portfolio_weights, cov, confidence=0.99)
        assert np.isclose(result.component_var.sum(), result.portfolio_var, rtol=1e-10)

    def test_pct_contributions_sum_to_one(self, synthetic_returns, portfolio_weights) -> None:
        cov = COV.sample_covariance(synthetic_returns)
        result = EX.component_var(portfolio_weights, cov, confidence=0.99)
        assert np.isclose(result.pct_contribution.sum(), 1.0, atol=1e-10)

    def test_holds_across_confidence_levels(self, synthetic_returns, portfolio_weights) -> None:
        cov = COV.sample_covariance(synthetic_returns)
        for confidence in (0.90, 0.95, 0.99, 0.999):
            result = EX.component_var(portfolio_weights, cov, confidence=confidence)
            assert np.isclose(result.component_var.sum(), result.portfolio_var, rtol=1e-9)

    def test_holds_for_random_weights(self, synthetic_returns) -> None:
        cov = COV.sample_covariance(synthetic_returns)
        rng = np.random.default_rng(0)
        for _ in range(10):
            w = rng.dirichlet(np.ones(synthetic_returns.shape[1]))
            result = EX.component_var(w, cov, confidence=0.975)
            assert np.isclose(result.component_var.sum(), result.portfolio_var, rtol=1e-9)


class TestTrackingErrorAndBeta:
    def test_zero_active_weights_gives_zero_te(self, synthetic_returns) -> None:
        cov = COV.sample_covariance(synthetic_returns)
        w = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        te = EX.tracking_error_ex_ante(w, w, cov)
        assert te == 0.0

    def test_beta_of_asset_vs_itself_is_one(self, synthetic_returns) -> None:
        r = synthetic_returns["Actions_EU"]
        assert np.isclose(EX.beta(r, r), 1.0)

    def test_blume_adjustment_shrinks_toward_one(self) -> None:
        raw = 1.6
        adjusted = EX.beta_blume_adjusted(raw)
        assert 1.0 < adjusted < raw

        raw_low = 0.3
        adjusted_low = EX.beta_blume_adjusted(raw_low)
        assert raw_low < adjusted_low < 1.0

    def test_information_ratio_zero_for_identical_series(self, synthetic_returns) -> None:
        r = synthetic_returns["Actions_EU"]
        assert EX.information_ratio(r, r) == 0.0
