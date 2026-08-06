"""Comparaison des 4 méthodes de VaR sur un portefeuille multi-actifs."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marketrisk.data import generate_synthetic_returns
from marketrisk.var import historical_var, monte_carlo_var, var_es_summary

IMG_DIR = Path(__file__).resolve().parents[1] / "docs" / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)

CONFIDENCE = 0.99
PORTFOLIO_WEIGHTS = np.array([0.30, 0.30, 0.20, 0.15, 0.05])


def main() -> None:
    returns = generate_synthetic_returns(n_days=1500, seed=42)
    portfolio_returns = pd.Series(returns.to_numpy() @ PORTFOLIO_WEIGHTS, index=returns.index, name="portefeuille")

    print("=" * 78)
    print("EXEMPLE 1 — Comparaison des méthodes de VaR sur un portefeuille multi-actifs")
    print("=" * 78)
    print(f"Actifs : {list(returns.columns)}")
    print(f"Poids  : {PORTFOLIO_WEIGHTS.tolist()}")
    print(f"Historique : {returns.index[0].date()} -> {returns.index[-1].date()} ({len(returns)} jours)")
    print()

    summary = var_es_summary(portfolio_returns, confidence=CONFIDENCE)

    mc_result = monte_carlo_var(
        returns, PORTFOLIO_WEIGHTS, confidence=CONFIDENCE, n_sims=100_000, dist="student_t", dof=6, seed=7
    )
    summary.loc["Monte Carlo (Student-t, Cholesky)"] = [mc_result.var, mc_result.es]

    print(f"VaR et ES à {CONFIDENCE:.1%} de confiance (portefeuille, 1 jour) :\n")
    print(summary.to_string(float_format=lambda x: f"{x:.4%}"))
    print()

    var_hist = historical_var(portfolio_returns, CONFIDENCE)
    exceptions_mask = (-portfolio_returns) > var_hist
    n_exceptions = int(exceptions_mask.sum())
    exception_rate = n_exceptions / len(portfolio_returns)
    print(
        f"VaR historique {CONFIDENCE:.0%} = {var_hist:.4%} -> {n_exceptions} exceptions sur "
        f"{len(portfolio_returns)} jours (taux observé {exception_rate:.3%}, taux attendu {1 - CONFIDENCE:.3%})"
    )
    print()

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(portfolio_returns.index, portfolio_returns.values, lw=0.6, color="#1F4E78", label="Rendement quotidien du portefeuille")
    ax.axhline(-var_hist, color="#C00000", lw=1.4, ls="--", label=f"VaR historique {CONFIDENCE:.0%} (-{var_hist:.2%})")
    ax.scatter(
        portfolio_returns.index[exceptions_mask],
        portfolio_returns.values[exceptions_mask],
        color="#C00000",
        s=22,
        zorder=5,
        label=f"Exceptions ({n_exceptions})",
    )
    ax.set_title("Rendements du portefeuille et VaR historique à 99% — dépassements marqués")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rendement quotidien")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out1 = IMG_DIR / "01_var_returns_exceptions.png"
    fig.savefig(out1, dpi=130)
    plt.close(fig)
    print(f"Graphique sauvegardé : {out1}")

    methods = summary.index.tolist()
    vars_pct = (summary["VaR"] * 100).values
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    colors = ["#1F4E78", "#2E75B6", "#9DC3E6", "#BF8F00", "#548235"]
    ax2.bar(methods, vars_pct, color=colors[: len(methods)])
    for i, v in enumerate(vars_pct):
        ax2.text(i, v + 0.03, f"{v:.2f}%", ha="center", fontsize=9)
    ax2.set_title(f"VaR à {CONFIDENCE:.0%} par méthode — portefeuille multi-actifs")
    ax2.set_ylabel("VaR (% de la valeur du portefeuille)")
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=20, ha="right", fontsize=8)
    ax2.grid(axis="y", alpha=0.25)
    fig2.tight_layout()
    out2 = IMG_DIR / "01_var_methodes_comparaison.png"
    fig2.savefig(out2, dpi=130)
    plt.close(fig2)
    print(f"Graphique sauvegardé : {out2}")

    print()
    print("Résumé chiffré (arrondi) :")
    for method in summary.index:
        print(f"  - {method:38s} VaR={summary.loc[method, 'VaR']:.4%}  ES={summary.loc[method, 'ES']:.4%}")


if __name__ == "__main__":
    main()
