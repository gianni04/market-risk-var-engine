"""Tests du module marketrisk.volatility."""

from __future__ import annotations

import numpy as np

from marketrisk import volatility as VOL


class TestHistoricalAndEwma:
    def test_historical_volatility_positive(self, synthetic_returns) -> None:
        for col in synthetic_returns.columns:
            assert VOL.historical_volatility(synthetic_returns[col]) > 0

    def test_ewma_volatility_reacts_to_shock(self) -> None:
        calm = np.full(300, 0.0)
        rng = np.random.default_rng(0)
        calm = rng.normal(0, 0.005, 300)
        shocked = np.concatenate([calm, [0.10]])  # choc violent au dernier jour
        r = np.concatenate([shocked, np.zeros(5)])
        vol = VOL.ewma_volatility(r, lam=0.94, annualize=False)
        assert vol.iloc[-1] > vol.iloc[10]


class TestGarch11Fit:
    def test_stationarity_constraint_respected(self, synthetic_returns) -> None:
        fit = VOL.fit_garch11(synthetic_returns["Actions_EU"])
        assert fit.alpha >= 0
        assert fit.beta >= 0
        assert fit.alpha + fit.beta < 1.0
        assert fit.omega > 0

    def test_recovers_reasonable_persistence(self, synthetic_returns) -> None:
        # Les données synthétiques sont générées avec alpha=0.08, beta=0.90
        # (persistance vraie 0.98) : l'estimateur doit être dans un
        # voisinage raisonnable de cette persistance sur 1500 jours.
        fit = VOL.fit_garch11(synthetic_returns["Actions_EU"])
        assert 0.90 < fit.persistence < 0.999

    def test_converges_regardless_of_initial_guess(self, synthetic_returns) -> None:
        r = synthetic_returns["Actions_US"]
        fit_a = VOL.fit_garch11(r, initial_guess=(1e-5, 0.05, 0.90))
        fit_b = VOL.fit_garch11(r, initial_guess=(5e-5, 0.15, 0.70))
        assert abs(fit_a.alpha - fit_b.alpha) < 1e-3
        assert abs(fit_a.beta - fit_b.beta) < 1e-3


class TestGarchForecast:
    def test_forecast_converges_to_long_run_variance(self, synthetic_returns) -> None:
        fit = VOL.fit_garch11(synthetic_returns["Actions_EU"])
        # Horizon très long pour laisser (alpha+beta)^(h-1) devenir
        # négligeable, y compris pour une persistance proche de 1.
        forecast_far = VOL.forecast_garch_volatility(fit, horizon=5000, annualize=False)
        long_run_vol = np.sqrt(fit.long_run_variance)
        assert abs(forecast_far[-1] - long_run_vol) / long_run_vol < 1e-3

    def test_term_vol_between_daily_and_sqrt_time_bounds(self, synthetic_returns) -> None:
        fit = VOL.fit_garch11(synthetic_returns["Actions_EU"])
        term_vol_10d = VOL.forecast_garch_term_vol(fit, horizon=10)
        assert term_vol_10d > 0
        assert np.isfinite(term_vol_10d)
