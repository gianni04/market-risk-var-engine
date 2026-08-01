"""Exemple 2 : backtest réglementaire de la VaR sur 250 jours.

Calibre une VaR historique glissante (fenêtre de 500 jours, recalculée
chaque jour) sur un portefeuille multi-actifs, puis backteste les 250
derniers jours (un an réglementaire) avec les tests de Kupiec, de
Christoffersen (indépendance + couverture conditionnelle jointe) et le
Traffic Light de Bâle.

Exécution : ``python examples/02_var_backtest.py``
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

from marketrisk.backtesting import backtest_var
from marketrisk.data import generate_synthetic_returns
from marketrisk.var import historical_var

IMG_DIR = Path(__file__).resolve().parents[1] / "docs" / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)

CONFIDENCE = 0.99
CALIBRATION_WINDOW = 500
BACKTEST_WINDOW = 250
PORTFOLIO_WEIGHTS = np.array([0.30, 0.30, 0.20, 0.15, 0.05])


def rolling_historical_var(portfolio_returns: pd.Series, window: int, confidence: float) -> pd.Series:
    """VaR historique glissante : recalibrée chaque jour sur les ``window`` jours précédents.

    Reproduit la pratique de backtesting réglementaire : la VaR du jour t
    n'utilise que l'information disponible jusqu'à t-1 (pas de fuite de
    données du futur).
    """
    var_series = pd.Series(index=portfolio_returns.index, dtype=float)
    for i in range(window, len(portfolio_returns)):
        past = portfolio_returns.iloc[i - window : i]
        var_series.iloc[i] = historical_var(past, confidence)
    return var_series


def main() -> None:
    n_days = CALIBRATION_WINDOW + BACKTEST_WINDOW + 50
    returns = generate_synthetic_returns(n_days=n_days, seed=42)
    portfolio_returns = pd.Series(returns.to_numpy() @ PORTFOLIO_WEIGHTS, index=returns.index, name="portefeuille")

    var_series = rolling_historical_var(portfolio_returns, CALIBRATION_WINDOW, CONFIDENCE)

    backtest_returns = portfolio_returns.iloc[-BACKTEST_WINDOW:]
    backtest_var_series = var_series.iloc[-BACKTEST_WINDOW:]

    print("=" * 78)
    print("EXEMPLE 2 — Backtest réglementaire de la VaR (250 jours)")
    print("=" * 78)
    print(f"Fenêtre de calibration : {CALIBRATION_WINDOW} jours glissants")
    print(f"Fenêtre de backtest    : {BACKTEST_WINDOW} jours (1 an réglementaire)")
    print(f"Niveau de confiance    : {CONFIDENCE:.0%}")
    print()

    results = backtest_var(backtest_returns, backtest_var_series, confidence=CONFIDENCE)

    exceptions = results["exceptions"]
    n_exceptions = int(exceptions.sum())

    rows = []
    kupiec = results["kupiec"]
    rows.append(("Kupiec POF (couverture inconditionnelle)", kupiec.statistic, kupiec.p_value, kupiec.reject_h0))
    ind = results["christoffersen_independence"]
    rows.append(("Christoffersen (indépendance)", ind.statistic, ind.p_value, ind.reject_h0))
    joint = results["christoffersen_joint"]
    rows.append(("Christoffersen (couverture conditionnelle jointe)", joint.statistic, joint.p_value, joint.reject_h0))

    table = pd.DataFrame(rows, columns=["Test", "Statistique LR", "p-value", "Rejet H0 (5%)"])
    print(f"Nombre d'exceptions observées : {n_exceptions} / {BACKTEST_WINDOW} "
          f"(taux attendu {1 - CONFIDENCE:.1%}, taux observé {n_exceptions / BACKTEST_WINDOW:.2%})\n")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    tl = results["traffic_light"]
    print(
        f"Traffic Light Bâle : zone {tl.zone.upper()} "
        f"({tl.n_exceptions} exceptions / {tl.n_obs} obs, "
        f"probabilité cumulée {tl.cumulative_probability_pct:.2f}%) "
        f"-> multiplicateur {tl.base_multiplier:.2f} + {tl.multiplier_addon:.2f} = {tl.total_multiplier:.2f}"
    )
    print()

    # -----------------------------------------------------------------
    # Graphique : rendements du backtest, VaR glissante, exceptions.
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(backtest_returns.index, backtest_returns.values, lw=0.7, color="#1F4E78", label="Rendement quotidien réalisé")
    ax.plot(backtest_var_series.index, -backtest_var_series.values, color="#C00000", lw=1.3, label=f"-VaR historique glissante ({CONFIDENCE:.0%})")
    exc_mask = exceptions
    ax.scatter(
        backtest_returns.index[exc_mask],
        backtest_returns.values[exc_mask],
        color="#C00000",
        s=32,
        zorder=5,
        marker="x",
        label=f"Exceptions ({n_exceptions})",
    )
    ax.set_title(f"Backtest VaR {CONFIDENCE:.0%} sur {BACKTEST_WINDOW} jours — zone Bâle : {tl.zone}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rendement quotidien")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path = IMG_DIR / "02_var_backtest_exceptions.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Graphique sauvegardé : {out_path}")


if __name__ == "__main__":
    main()
