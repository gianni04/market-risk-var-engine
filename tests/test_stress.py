"""Tests du module marketrisk.stress."""

from __future__ import annotations

import numpy as np
import pandas as pd

from marketrisk import covariance as COV
from marketrisk import stress as ST


class TestHistoricalScenarios:
    def test_all_scenarios_produce_a_loss(self, portfolio_weights, synthetic_returns) -> None:
        w = pd.Series(dict(zip(synthetic_returns.columns, portfolio_weights)))
        df = ST.run_all_historical_scenarios(w)
        assert (df["pnl_pct"] < 0).all()
        assert len(df) == len(ST.HISTORICAL_SCENARIOS)

    def test_unknown_scenario_raises(self, portfolio_weights, synthetic_returns) -> None:
        w = pd.Series(dict(zip(synthetic_returns.columns, portfolio_weights)))
        try:
            ST.apply_historical_scenario(w, "scenario_inexistant")
            assert False, "devrait lever KeyError"
        except KeyError:
            pass


class TestHypotheticalScenario:
    def test_equity_only_shock_hits_only_equities(self, portfolio_weights, synthetic_returns) -> None:
        w = pd.Series(dict(zip(synthetic_returns.columns, portfolio_weights)))
        shocks = ST.HypotheticalShocks(equity_shock=-0.20)
        result = ST.hypothetical_scenario(w, shocks)
        assert result["Or"] == 0.0
        assert result["Actions_EU"] < 0
        assert result["__total__"] < 0


class TestCorrelationStress:
    def test_forcing_correlation_to_one_increases_volatility(self, synthetic_returns, portfolio_weights) -> None:
        cov = COV.sample_covariance(synthetic_returns)
        result = ST.stress_correlation_shock(portfolio_weights, cov, forced_correlation=1.0)
        assert result["vol_stress"] >= result["vol_base"]
        assert result["increase_pct"] >= 0.0


class TestReverseStress:
    def test_achieves_target_loss(self, synthetic_returns, portfolio_weights) -> None:
        cov = COV.sample_covariance(synthetic_returns, annualize=True)
        target = 0.15
        result = ST.reverse_stress_test(portfolio_weights, cov, target_loss=target)
        assert abs(result.achieved_loss - target) < 1e-6
        assert result.converged
        assert result.mahalanobis_distance > 0

    def test_larger_target_loss_gives_larger_distance(self, synthetic_returns, portfolio_weights) -> None:
        cov = COV.sample_covariance(synthetic_returns, annualize=True)
        small = ST.reverse_stress_test(portfolio_weights, cov, target_loss=0.05)
        large = ST.reverse_stress_test(portfolio_weights, cov, target_loss=0.20)
        assert large.mahalanobis_distance > small.mahalanobis_distance
