"""Construction du rapport de risque quotidien et export Excel formaté."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class DailyRiskReport:
    """Rapport de risque quotidien consolidé."""

    report_date: str
    var_summary: pd.DataFrame
    monte_carlo_var: float
    monte_carlo_es: float
    tracking_error_ex_ante: float
    beta: float
    beta_blume: float
    information_ratio: float
    top_contributors: pd.DataFrame
    backtest_summary: pd.DataFrame


def build_daily_risk_report(
    report_date: str,
    var_summary: pd.DataFrame,
    monte_carlo_var: float,
    monte_carlo_es: float,
    tracking_error_ex_ante: float,
    beta: float,
    beta_blume: float,
    information_ratio: float,
    component_var: np.ndarray,
    asset_names: list[str],
    backtest_results: dict,
) -> DailyRiskReport:
    """Assemble un rapport de risque quotidien à partir des sorties du package."""
    contrib_df = pd.DataFrame(
        {"Actif": asset_names, "VaR_composante": component_var}
    ).sort_values("VaR_composante", ascending=False).reset_index(drop=True)
    contrib_df["Contribution_pct"] = 100.0 * contrib_df["VaR_composante"] / contrib_df["VaR_composante"].sum()

    backtest_rows = []
    for key in ("kupiec", "christoffersen_independence", "christoffersen_joint"):
        test = backtest_results[key]
        backtest_rows.append(
            {
                "Test": test.name,
                "Statistique_LR": test.statistic,
                "p_value": test.p_value,
                "Rejet_H0": test.reject_h0,
                "Verdict": "MODELE REJETE" if test.reject_h0 else "Non rejeté",
            }
        )
    tl = backtest_results["traffic_light"]
    backtest_rows.append(
        {
            "Test": "Traffic Light Bâle",
            "Statistique_LR": tl.n_exceptions,
            "p_value": tl.cumulative_probability_pct / 100.0,
            "Rejet_H0": tl.zone == "rouge",
            "Verdict": f"Zone {tl.zone} (multiplicateur {tl.total_multiplier:.2f})",
        }
    )
    backtest_df = pd.DataFrame(backtest_rows)

    return DailyRiskReport(
        report_date=report_date,
        var_summary=var_summary,
        monte_carlo_var=monte_carlo_var,
        monte_carlo_es=monte_carlo_es,
        tracking_error_ex_ante=tracking_error_ex_ante,
        beta=beta,
        beta_blume=beta_blume,
        information_ratio=information_ratio,
        top_contributors=contrib_df,
        backtest_summary=backtest_df,
    )


def export_excel_report(report: DailyRiskReport, path: str | Path) -> Path:
    """Exporte un :class:`DailyRiskReport` en classeur Excel mis en forme."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        workbook = writer.book

        header_fmt = workbook.add_format(
            {"bold": True, "bg_color": "#1F4E78", "font_color": "white", "border": 1}
        )
        pct_fmt = workbook.add_format({"num_format": "0.00%"})
        title_fmt = workbook.add_format({"bold": True, "font_size": 14})
        reject_fmt = workbook.add_format({"bg_color": "#F8CBAD"})
        ok_fmt = workbook.add_format({"bg_color": "#C6EFCE"})

        summary_sheet = workbook.add_worksheet("Synthese")
        writer.sheets["Synthese"] = summary_sheet
        summary_sheet.write(0, 0, f"Rapport de risque de marché — {report.report_date}", title_fmt)
        summary_rows = [
            ("VaR Monte Carlo (portefeuille)", report.monte_carlo_var),
            ("ES Monte Carlo (portefeuille)", report.monte_carlo_es),
            ("Tracking Error ex-ante (annualisée)", report.tracking_error_ex_ante),
            ("Beta (brut)", report.beta),
            ("Beta ajusté (Blume)", report.beta_blume),
            ("Ratio d'information", report.information_ratio),
        ]
        summary_sheet.write(2, 0, "Indicateur", header_fmt)
        summary_sheet.write(2, 1, "Valeur", header_fmt)
        for i, (label, value) in enumerate(summary_rows, start=3):
            summary_sheet.write(i, 0, label)
            fmt = pct_fmt if "Beta" not in label and "information" not in label.lower() else None
            summary_sheet.write(i, 1, value, fmt)
        summary_sheet.set_column(0, 0, 38)
        summary_sheet.set_column(1, 1, 16)
        summary_sheet.freeze_panes(3, 0)

        report.var_summary.to_excel(writer, sheet_name="VaR_par_methode", startrow=1)
        var_sheet = writer.sheets["VaR_par_methode"]
        var_sheet.write(0, 0, "VaR et Expected Shortfall par méthode (niveau de confiance indiqué dans le script)", title_fmt)
        for col_idx, col_name in enumerate([""] + list(report.var_summary.columns)):
            var_sheet.write(2, col_idx, col_name if col_name else "Méthode", header_fmt)
        var_sheet.set_column(0, 0, 28)
        var_sheet.set_column(1, 2, 14, pct_fmt)
        var_sheet.freeze_panes(3, 1)

        report.top_contributors.to_excel(writer, sheet_name="Contributeurs_risque", index=False, startrow=1)
        contrib_sheet = writer.sheets["Contributeurs_risque"]
        contrib_sheet.write(0, 0, "Contribution au risque par position (allocation d'Euler)", title_fmt)
        for col_idx, col_name in enumerate(report.top_contributors.columns):
            contrib_sheet.write(2, col_idx, col_name, header_fmt)
        contrib_sheet.set_column(0, 0, 22)
        contrib_sheet.set_column(1, 1, 16, pct_fmt)
        contrib_sheet.set_column(2, 2, 18)
        contrib_sheet.freeze_panes(3, 0)

        report.backtest_summary.to_excel(writer, sheet_name="Backtesting", index=False, startrow=1)
        bt_sheet = writer.sheets["Backtesting"]
        bt_sheet.write(0, 0, "Résultats des tests de backtesting réglementaire", title_fmt)
        for col_idx, col_name in enumerate(report.backtest_summary.columns):
            bt_sheet.write(2, col_idx, col_name, header_fmt)
        n_rows = len(report.backtest_summary)
        reject_col = report.backtest_summary.columns.get_loc("Rejet_H0")
        for row_offset in range(n_rows):
            row = 3 + row_offset
            is_reject = bool(report.backtest_summary.iloc[row_offset]["Rejet_H0"])
            fmt = reject_fmt if is_reject else ok_fmt
            bt_sheet.write(row, reject_col, is_reject, fmt)
        bt_sheet.set_column(0, 0, 40)
        bt_sheet.set_column(1, 4, 16)
        bt_sheet.freeze_panes(3, 0)

    return out_path


def report_to_dataframe(report: DailyRiskReport) -> pd.DataFrame:
    """Convertit la synthèse du rapport en un DataFrame plat (pour affichage console/CSV)."""
    return pd.DataFrame(
        [
            {
                "date": report.report_date,
                "VaR_MC": report.monte_carlo_var,
                "ES_MC": report.monte_carlo_es,
                "TE_ex_ante": report.tracking_error_ex_ante,
                "beta": report.beta,
                "beta_blume": report.beta_blume,
                "information_ratio": report.information_ratio,
            }
        ]
    )
