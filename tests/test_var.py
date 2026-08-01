"""Tests du module marketrisk.var."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from marketrisk import var as V


class TestParametricVarAnalytical:
    """La VaR paramétrique gaussienne doit correspondre exactement à la théorie."""

    def test_matches_normal_quantile_formula(self, gaussian_returns: np.ndarray) -> None:
        confidence = 0.99
        mu = gaussian_returns.mean()
        sigma = gaussian_returns.std(ddof=1)
        z = stats.norm.ppf(1 - confidence)
        expected = -(mu + z * sigma)

        computed = V.parametric_var_gaussian(gaussian_returns, confidence=confidence)
        assert computed == pytest.approx(expected, rel=1e-6, abs=1e-8)

    def test_close_to_true_population_var(self) -> None:
        # Rendements gaussiens à moyenne/vol connues : la VaR estimée doit
        # être proche de la vraie VaR théorique de la population.
        rng = np.random.default_rng(7)
        mu_true, sigma_true = 0.0005, 0.015
        r = rng.normal(mu_true, sigma_true, 500_000)
        true_var = -(mu_true + stats.norm.ppf(0.01) * sigma_true)
        computed = V.parametric_var_gaussian(r, confidence=0.99)
        assert abs(computed - true_var) < 5e-4


class TestMonteCarloConvergence:
    """La VaR Monte Carlo doit converger vers la VaR paramétrique sous normalité."""

    def test_mc_var_converges_to_parametric_single_asset(self) -> None:
        import pandas as pd

        rng = np.random.default_rng(11)
        n = 4000
        r = rng.normal(0.0003, 0.012, n)
        returns_df = pd.DataFrame({"asset": r})
        weights = np.array([1.0])

        param_var = V.parametric_var_gaussian(r, confidence=0.99)
        mc_result = V.monte_carlo_var(
            returns_df, weights, confidence=0.99, n_sims=200_000, dist="normal", seed=99
        )
        # Tolérance relative généreuse : erreur d'échantillonnage Monte Carlo.
        assert abs(mc_result.var - param_var) / param_var < 0.05


class TestMonotonicity:
    """La VaR doit croître avec le niveau de confiance."""

    def test_var_increases_with_confidence(self, synthetic_returns) -> None:
        r = synthetic_returns["Actions_EU"]
        levels = [0.90, 0.95, 0.99, 0.995, 0.999]
        historical = [V.historical_var(r, c) for c in levels]
        parametric = [V.parametric_var_gaussian(r, c) for c in levels]
        cornish_fisher = [V.cornish_fisher_var(r, c) for c in levels]

        assert all(historical[i] <= historical[i + 1] for i in range(len(historical) - 1))
        assert all(parametric[i] <= parametric[i + 1] for i in range(len(parametric) - 1))
        assert all(cornish_fisher[i] <= cornish_fisher[i + 1] for i in range(len(cornish_fisher) - 1))


class TestExpectedShortfallCoherence:
    """ES >= VaR au même niveau de confiance, pour toutes les méthodes."""

    def test_historical_es_ge_var(self, synthetic_returns) -> None:
        for col in synthetic_returns.columns:
            r = synthetic_returns[col]
            for c in (0.95, 0.99, 0.999):
                assert V.historical_es(r, c) >= V.historical_var(r, c) - 1e-12

    def test_parametric_es_ge_var(self, synthetic_returns) -> None:
        for col in synthetic_returns.columns:
            r = synthetic_returns[col]
            for c in (0.95, 0.99, 0.999):
                assert V.parametric_es_gaussian(r, c) >= V.parametric_var_gaussian(r, c) - 1e-12

    def test_cornish_fisher_es_ge_var(self, synthetic_returns) -> None:
        for col in synthetic_returns.columns:
            r = synthetic_returns[col]
            for c in (0.95, 0.99, 0.999):
                assert V.cornish_fisher_es(r, c) >= V.cornish_fisher_var(r, c) - 1e-12

    def test_monte_carlo_es_ge_var(self, synthetic_returns, portfolio_weights) -> None:
        result = V.monte_carlo_var(
            synthetic_returns, portfolio_weights, confidence=0.99, n_sims=20_000, seed=5
        )
        assert result.es >= result.var - 1e-12


class TestHistoricalVarWeighted:
    def test_weighted_var_is_positive_and_finite(self, synthetic_returns) -> None:
        r = synthetic_returns["Credit_IG"]
        var = V.historical_var_weighted(r, confidence=0.99, decay=0.98)
        assert var > 0
        assert np.isfinite(var)

    def test_weighted_var_reacts_more_to_recent_vol(self) -> None:
        # Série avec un régime calme suivi d'un régime volatil récent : la
        # VaR pondérée doit être plus élevée que la VaR historique simple,
        # qui traite également le passé calme et le présent volatil.
        rng = np.random.default_rng(3)
        calm = rng.normal(0, 0.005, 1000)
        volatile = rng.normal(0, 0.03, 200)
        r = np.concatenate([calm, volatile])

        var_simple = V.historical_var(r, confidence=0.95)
        var_weighted = V.historical_var_weighted(r, confidence=0.95, decay=0.94)
        assert var_weighted > var_simple


class TestTimeScalingDiagnostic:
    def test_sqrt_time_scaling_formula(self) -> None:
        assert V.sqrt_time_scaling(1.0, 4) == pytest.approx(2.0)
        assert V.sqrt_time_scaling(2.0, 25) == pytest.approx(10.0)

    def test_diagnostic_returns_expected_columns(self, synthetic_returns) -> None:
        r = synthetic_returns["Actions_EU"]
        df = V.time_scaling_diagnostic(r, confidence=0.99, horizons=(1, 5, 10))
        assert list(df.columns) == [
            "horizon",
            "VaR_sqrt_time",
            "VaR_historique_h_jours",
            "ecart_relatif_pct",
        ]
        assert len(df) == 3
        assert (df["VaR_sqrt_time"] > 0).all()
