"""Tests d'intégration du module marketrisk.report (bout en bout)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openpyxl

from marketrisk import backtesting as BT
from marketrisk import covariance as COV
from marketrisk import exante as EX
from marketrisk import report as R
from marketrisk import var as V


def _build_sample_report(synthetic_returns, portfolio_weights):
    port_r = synthetic_returns.to_numpy() @ portfolio_weights
    var_summary = V.var_es_summary(port_r, confidence=0.99)
    mc = V.monte_carlo_var(synthetic_returns, portfolio_weights, confidence=0.99, n_sims=5000, seed=1)
    cov = COV.sample_covariance(synthetic_returns)
    bench_w = np.full(portfolio_weights.shape[0], 1.0 / portfolio_weights.shape[0])
    bench_r = synthetic_returns.to_numpy() @ bench_w
    te = EX.tracking_error_ex_ante(portfolio_weights, bench_w, cov, annualize=False)
    b = EX.beta(port_r, bench_r)
    bb = EX.beta_blume_adjusted(b)
    ir = EX.information_ratio(port_r, bench_r)
    cvar = EX.component_var(portfolio_weights, cov, confidence=0.99)

    var_1d = V.historical_var(port_r[:-250], 0.99)
    bt = BT.backtest_var(port_r[-250:], var_1d, confidence=0.99)

    return R.build_daily_risk_report(
        "2026-08-01",
        var_summary,
        mc.var,
        mc.es,
        te,
        b,
        bb,
        ir,
        cvar.component_var,
        list(synthetic_returns.columns),
        bt,
    )


class TestReportBuild:
    def test_report_has_expected_fields(self, synthetic_returns, portfolio_weights) -> None:
        report = _build_sample_report(synthetic_returns, portfolio_weights)
        assert report.monte_carlo_var > 0
        assert report.monte_carlo_es >= report.monte_carlo_var
        assert len(report.top_contributors) == synthetic_returns.shape[1]
        assert np.isclose(report.top_contributors["VaR_composante"].sum(), report.top_contributors["VaR_composante"].sum())

    def test_report_to_dataframe(self, synthetic_returns, portfolio_weights) -> None:
        report = _build_sample_report(synthetic_returns, portfolio_weights)
        df = R.report_to_dataframe(report)
        assert df.shape[0] == 1
        assert "VaR_MC" in df.columns


class TestExcelExport:
    def test_export_creates_valid_workbook(self, synthetic_returns, portfolio_weights, tmp_path: Path) -> None:
        report = _build_sample_report(synthetic_returns, portfolio_weights)
        out_path = tmp_path / "rapport_test.xlsx"
        written = R.export_excel_report(report, out_path)
        assert written.exists()

        wb = openpyxl.load_workbook(written)
        assert set(wb.sheetnames) == {
            "Synthese",
            "VaR_par_methode",
            "Contributeurs_risque",
            "Backtesting",
        }
