"""Exemple 3 : stress tests — scénarios historiques, hypothétique, corrélation, reverse.

Rejoue les scénarios de crise historiques (2008, COVID, choc de taux 2022) et
un scénario hypothétique multi-facteurs sur un portefeuille multi-actifs,
applique un stress de corrélation (corrélations forcées à 1 en crise
systémique) et calcule un reverse stress test (choc le plus plausible pour
une perte cible de -15%).

Exécution : ``python examples/03_stress_tests.py`` (aucune dépendance
réseau, aucun argument requis).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marketrisk.covariance import sample_covariance
from marketrisk.data import generate_synthetic_returns
from marketrisk.stress import (
    HypotheticalShocks,
    hypothetical_scenario,
    reverse_stress_test,
    run_all_historical_scenarios,
    stress_correlation_shock,
)

IMG_DIR = Path(__file__).resolve().parents[1] / "docs" / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)

ASSET_NAMES = ["Actions_EU", "Actions_US", "Obligations_Souv", "Credit_IG", "Or"]
PORTFOLIO_WEIGHTS = np.array([0.30, 0.30, 0.20, 0.15, 0.05])
TARGET_LOSS = 0.15


def main() -> None:
    weights = dict(zip(ASSET_NAMES, PORTFOLIO_WEIGHTS))

    returns = generate_synthetic_returns(n_days=1500, seed=42)
    cov_daily = sample_covariance(returns, annualize=False)

    print("=" * 78)
    print("EXEMPLE 3 — Stress tests (historiques, hypothétique, corrélation, reverse)")
    print("=" * 78)
    print(f"Portefeuille : {weights}")
    print()

    # -----------------------------------------------------------------
    # 1) Scénarios historiques rejoués.
    # -----------------------------------------------------------------
    hist_table = run_all_historical_scenarios(weights)
    print("Scénarios historiques rejoués (P&L en % de la valeur du portefeuille) :\n")
    print(hist_table.to_string(index=False, float_format=lambda x: f"{x:+.2%}"))
    print()

    # -----------------------------------------------------------------
    # 2) Scénario hypothétique paramétrique.
    # -----------------------------------------------------------------
    hypo_shocks = HypotheticalShocks(
        equity_shock=-0.25, rate_shock_bp=150.0, credit_spread_shock_bp=120.0, fx_shock=0.08
    )
    hypo_pnl = hypothetical_scenario(weights, hypo_shocks)
    hypo_total = hypo_pnl.pop("__total__")
    print(
        "Scénario hypothétique (actions -25%, taux +150bp, spread crédit +120bp, "
        "USD +8%) :\n"
    )
    for asset, pnl in hypo_pnl.items():
        print(f"  - {asset:20s} {pnl:+.2%}")
    print(f"  {'TOTAL':22s} {hypo_total:+.2%}")
    print()

    # -----------------------------------------------------------------
    # 3) Stress de corrélation (corrélations forcées à 1).
    # -----------------------------------------------------------------
    corr_stress = stress_correlation_shock(PORTFOLIO_WEIGHTS, cov_daily, forced_correlation=1.0)
    vol_base_annual = corr_stress["vol_base"] * np.sqrt(252)
    vol_stress_annual = corr_stress["vol_stress"] * np.sqrt(252)
    print("Stress de corrélation (toutes les corrélations forcées à 1, crise systémique) :\n")
    print(f"  Volatilité de base (annualisée)   : {vol_base_annual:.2%}")
    print(f"  Volatilité sous stress (annuali.) : {vol_stress_annual:.2%}")
    print(f"  Hausse relative de la volatilité  : {corr_stress['increase_pct']:+.1f}%")
    print()

    # -----------------------------------------------------------------
    # 4) Reverse stress test.
    # -----------------------------------------------------------------
    reverse = reverse_stress_test(PORTFOLIO_WEIGHTS, cov_daily, target_loss=TARGET_LOSS)
    print(f"Reverse stress test (perte cible {TARGET_LOSS:.0%}) :\n")
    print(f"  Convergence de l'optimisation : {reverse.converged}")
    print(f"  Perte effectivement atteinte  : {reverse.achieved_loss:+.2%}")
    print(f"  Distance de Mahalanobis       : {reverse.mahalanobis_distance:.4f}")
    print("  Choc le plus plausible par actif :")
    for asset, shock in zip(ASSET_NAMES, reverse.shock_vector):
        print(f"    - {asset:20s} {shock:+.2%}")
    print()

    # -----------------------------------------------------------------
    # Graphique 1 : P&L par scénario, barres horizontales.
    # -----------------------------------------------------------------
    scenario_names = list(hist_table["scenario"]) + ["Hypothétique (actions/taux/crédit/FX)"]
    scenario_pnls = list(hist_table["pnl_pct"]) + [hypo_total]
    order = np.argsort(scenario_pnls)
    scenario_names = [scenario_names[i] for i in order]
    scenario_pnls = [scenario_pnls[i] for i in order]
    colors = ["#C00000" if p < 0 else "#548235" for p in scenario_pnls]

    fig, ax = plt.subplots(figsize=(9.5, 5))
    y_pos = np.arange(len(scenario_names))
    ax.barh(y_pos, [p * 100 for p in scenario_pnls], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(scenario_names, fontsize=9)
    for i, p in enumerate(scenario_pnls):
        ax.text(p * 100 + (0.3 if p >= 0 else -0.3), i, f"{p:+.1%}", va="center",
                ha="left" if p >= 0 else "right", fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("P&L du portefeuille par scénario de stress")
    ax.set_xlabel("P&L (% de la valeur du portefeuille)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out1 = IMG_DIR / "03_stress_scenarios.png"
    fig.savefig(out1, dpi=130)
    plt.close(fig)
    print(f"Graphique sauvegardé : {out1}")

    # -----------------------------------------------------------------
    # Graphique 2 : waterfall de décomposition du scénario hypothétique.
    # -----------------------------------------------------------------
    labels = list(hypo_pnl.keys()) + ["TOTAL"]
    values = list(hypo_pnl.values()) + [hypo_total]
    cum = 0.0
    starts = []
    for v in values[:-1]:
        starts.append(cum)
        cum += v
    starts.append(0.0)  # la barre TOTAL part de zéro

    fig2, ax2 = plt.subplots(figsize=(9.5, 5))
    bar_colors = ["#C00000" if v < 0 else "#548235" for v in values[:-1]] + ["#1F4E78"]
    ax2.bar(labels, [v * 100 for v in values], bottom=[s * 100 for s in starts], color=bar_colors)
    for i, v in enumerate(values):
        top = (starts[i] + v) * 100
        ax2.text(i, top + (0.15 if v >= 0 else -0.35), f"{v:+.2%}", ha="center", fontsize=8)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_title("Décomposition du P&L — scénario hypothétique (waterfall par actif)")
    ax2.set_ylabel("Contribution au P&L (%)")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax2.grid(axis="y", alpha=0.25)
    fig2.tight_layout()
    out2 = IMG_DIR / "03_stress_waterfall.png"
    fig2.savefig(out2, dpi=130)
    plt.close(fig2)
    print(f"Graphique sauvegardé : {out2}")


if __name__ == "__main__":
    main()
