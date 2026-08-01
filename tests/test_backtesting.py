"""Tests du module marketrisk.backtesting."""

from __future__ import annotations

import numpy as np
from scipy import stats

from marketrisk import backtesting as BT


class TestKupiecPOF:
    def test_does_not_reject_well_calibrated_model(self) -> None:
        rng = np.random.default_rng(0)
        sigma = 0.01
        confidence = 0.99
        n = 2000
        r = rng.normal(0, sigma, n)
        var_correct = -stats.norm.ppf(1 - confidence) * sigma

        exceptions = BT.var_exceptions(r, var_correct)
        result = BT.kupiec_pof_test(exceptions, confidence=confidence, significance_level=0.05)

        assert result.reject_h0 is False
        assert result.p_value > 0.05

    def test_rejects_deliberately_miscalibrated_model(self) -> None:
        # VaR délibérément trop faible (sous-estime le risque) -> bien plus
        # d'exceptions que le taux attendu -> Kupiec doit rejeter H0.
        rng = np.random.default_rng(0)
        sigma = 0.01
        confidence = 0.99
        n = 2000
        r = rng.normal(0, sigma, n)
        var_too_low = (-stats.norm.ppf(1 - confidence) * sigma) * 0.3

        exceptions = BT.var_exceptions(r, var_too_low)
        result = BT.kupiec_pof_test(exceptions, confidence=confidence, significance_level=0.05)

        assert result.reject_h0 is True
        assert result.p_value < 0.05

    def test_rejects_overly_conservative_model(self) -> None:
        # VaR délibérément beaucoup trop élevée -> quasiment aucune
        # exception -> le taux observé s'écarte aussi du taux attendu et
        # Kupiec doit rejeter (mauvaise couverture, même "trop prudente").
        rng = np.random.default_rng(1)
        sigma = 0.01
        confidence = 0.99
        n = 3000
        r = rng.normal(0, sigma, n)
        var_too_high = (-stats.norm.ppf(1 - confidence) * sigma) * 4.0

        exceptions = BT.var_exceptions(r, var_too_high)
        result = BT.kupiec_pof_test(exceptions, confidence=confidence, significance_level=0.05)

        assert result.reject_h0 is True


class TestChristoffersenIndependence:
    def test_detects_clustered_exceptions(self) -> None:
        exceptions = np.zeros(250, dtype=bool)
        exceptions[100:112] = True  # 12 exceptions consécutives = fort clustering
        result = BT.christoffersen_independence_test(exceptions, significance_level=0.05)
        assert result.reject_h0 is True

    def test_does_not_reject_scattered_exceptions(self) -> None:
        rng = np.random.default_rng(42)
        # Exceptions dispersées aléatoirement au taux attendu (1%) sur un
        # grand échantillon : ne doit pas rejeter l'indépendance.
        exceptions = rng.random(5000) < 0.01
        result = BT.christoffersen_independence_test(exceptions, significance_level=0.01)
        assert result.reject_h0 is False


class TestTrafficLight:
    def test_green_zone_for_few_exceptions(self) -> None:
        result = BT.traffic_light_test(n_exceptions=2, n_obs=250, confidence=0.99)
        assert result.zone == "verte"
        assert result.multiplier_addon == 0.0

    def test_red_zone_for_many_exceptions(self) -> None:
        result = BT.traffic_light_test(n_exceptions=15, n_obs=250, confidence=0.99)
        assert result.zone == "rouge"
        assert result.multiplier_addon == 1.0

    def test_yellow_zone_intermediate(self) -> None:
        result = BT.traffic_light_test(n_exceptions=6, n_obs=250, confidence=0.99)
        assert result.zone == "jaune"
        assert 0.0 < result.multiplier_addon < 1.0


class TestBerkowitz:
    def test_does_not_reject_well_specified_model(self) -> None:
        rng = np.random.default_rng(5)
        n = 2000
        sigma = np.full(n, 0.01)
        mu = np.zeros(n)
        r = rng.normal(0, sigma)
        result = BT.berkowitz_test(r, mu, sigma, significance_level=0.05)
        assert result.reject_h0 is False

    def test_rejects_understated_volatility_forecast(self) -> None:
        rng = np.random.default_rng(6)
        n = 2000
        true_sigma = 0.02
        forecast_sigma = np.full(n, 0.005)  # vol prévue trop faible
        mu = np.zeros(n)
        r = rng.normal(0, true_sigma, n)
        result = BT.berkowitz_test(r, mu, forecast_sigma, significance_level=0.05)
        assert result.reject_h0 is True


class TestBacktestVarOrchestrator:
    def test_returns_all_expected_keys(self, synthetic_returns) -> None:
        r = synthetic_returns["Actions_EU"].iloc[-250:]
        var_est = 0.02
        result = BT.backtest_var(r, var_est, confidence=0.99)
        assert set(result.keys()) == {
            "exceptions",
            "kupiec",
            "christoffersen_independence",
            "christoffersen_joint",
            "traffic_light",
        }
        assert result["traffic_light"].n_obs == 250
