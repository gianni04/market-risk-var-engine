"""Exemple 4 : risque ex-ante relatif — tracking error, beta, ratio d'information, contributions.

Compare un portefeuille multi-actifs à un benchmark de référence : tracking
error ex-ante (covariance estimée sur une fenêtre passée) confrontée à la
tracking error ex-post (réalisée sur la fenêtre suivante, hors échantillon
de calibration — pas de tautologie), beta et beta ajusté de Blume, ratio
d'information, décomposition marginal/component VaR par position
(allocation d'Euler) et décomposition du risque actif en risque factoriel
(facteur de marché unique) vs risque spécifique.

Exécution : ``python examples/04_ex_ante_risk.py`` (aucune dépendance
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
from marketrisk.exante import (
    beta,
    beta_blume_adjusted,
    component_var,
    factor_specific_decomposition,
    information_ratio,
    tracking_error_ex_ante,
    tracking_error_ex_post,
)

IMG_DIR = Path(__file__).resolve().parents[1] / "docs" / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)

ASSET_NAMES = ["Actions_EU", "Actions_US", "Obligations_Souv", "Credit_IG", "Or"]
PORTFOLIO_WEIGHTS = np.array([0.30, 0.30, 0.20, 0.15, 0.05])
BENCHMARK_WEIGHTS = np.array([0.25, 0.25, 0.25, 0.15, 0.10])
CONFIDENCE = 0.99
TE_WINDOW = 250  # fenêtre de calibration / de réalisation (~1 an réglementaire)
TE_STEP = 21  # pas de recalibration (~mensuel) pour la série glissante


def format_weights(weights: np.ndarray) -> pd.Series:
    """Formate un vecteur de poids en Series pandas lisible (pourcentages)."""
    return pd.Series(weights, index=ASSET_NAMES, name="poids").map(lambda w: f"{w:.1%}")


def main() -> None:
    returns = generate_synthetic_returns(n_days=1500, seed=42)

    portfolio_returns = pd.Series(returns.to_numpy() @ PORTFOLIO_WEIGHTS, index=returns.index, name="portefeuille")
    benchmark_returns = pd.Series(returns.to_numpy() @ BENCHMARK_WEIGHTS, index=returns.index, name="benchmark")

    print("=" * 78)
    print("EXEMPLE 4 — Risque ex-ante relatif (Tracking Error, beta, contributions)")
    print("=" * 78)
    print("Portefeuille :")
    print(format_weights(PORTFOLIO_WEIGHTS).to_string())
    print("\nBenchmark :")
    print(format_weights(BENCHMARK_WEIGHTS).to_string())
    print()

    # -----------------------------------------------------------------
    # 1) Tracking error ex-ante (fenêtre passée) vs ex-post (fenêtre suivante).
    #
    # La covariance ex-ante est estimée sur les TE_WINDOW jours PRÉCÉDANT
    # l'instant t ; la TE ex-post est réalisée sur les TE_WINDOW jours
    # SUIVANT t. Les deux quantités portent donc sur des périodes disjointes
    # : leur écart mesure l'erreur de prévision du modèle de risque, et non
    # une tautologie (ce que donnerait un calcul sur le même échantillon).
    # -----------------------------------------------------------------
    calib = returns.iloc[0:TE_WINDOW]
    realized = returns.iloc[TE_WINDOW : 2 * TE_WINDOW]
    cov_calib = sample_covariance(calib, annualize=False)
    te_ante_headline = tracking_error_ex_ante(PORTFOLIO_WEIGHTS, BENCHMARK_WEIGHTS, cov_calib, annualize=True)
    te_post_headline = tracking_error_ex_post(
        portfolio_returns.loc[realized.index], benchmark_returns.loc[realized.index], annualize=True
    )
    ratio_headline = te_post_headline / te_ante_headline if te_ante_headline > 0 else np.nan
    print("Tracking Error (annualisée), prévision hors échantillon :\n")
    print(f"  Calibration ex-ante  : {calib.index[0].date()} -> {calib.index[-1].date()} ({TE_WINDOW} jours)")
    print(f"  Réalisation ex-post  : {realized.index[0].date()} -> {realized.index[-1].date()} ({TE_WINDOW} jours)")
    print(f"  TE ex-ante prévue    : {te_ante_headline:.2%}")
    print(f"  TE ex-post réalisée  : {te_post_headline:.2%}")
    print(f"  Ratio réalisé/prévu  : {ratio_headline:.2f}x")
    if ratio_headline > 1.1:
        print("  -> le modèle a sous-estimé le risque relatif réalisé sur cette période.")
    elif ratio_headline < 0.9:
        print("  -> le modèle a surestimé le risque relatif réalisé sur cette période.")
    else:
        print("  -> prévision et réalisation restent proches sur cette période.")
    print()

    # Série glissante : TE ex-ante prévue vs TE ex-post réalisée, recalibrée
    # tous les TE_STEP jours, sur l'ensemble de l'échantillon disponible.
    dates, te_ante_series, te_post_series = [], [], []
    for i in range(TE_WINDOW, len(returns) - TE_WINDOW + 1, TE_STEP):
        cov_i = sample_covariance(returns.iloc[i - TE_WINDOW : i], annualize=False)
        te_a = tracking_error_ex_ante(PORTFOLIO_WEIGHTS, BENCHMARK_WEIGHTS, cov_i, annualize=True)
        te_p = tracking_error_ex_post(
            portfolio_returns.iloc[i : i + TE_WINDOW], benchmark_returns.iloc[i : i + TE_WINDOW], annualize=True
        )
        dates.append(returns.index[i])
        te_ante_series.append(te_a)
        te_post_series.append(te_p)
    te_series_df = pd.DataFrame({"date": dates, "te_ante": te_ante_series, "te_post": te_post_series})
    mean_ratio = (te_series_df["te_post"] / te_series_df["te_ante"]).mean()
    print(
        f"Série glissante ({len(te_series_df)} points, pas {TE_STEP}j) : "
        f"ratio moyen réalisé/prévu = {mean_ratio:.2f}x"
    )
    print()

    # -----------------------------------------------------------------
    # 2) Beta, beta de Blume, ratio d'information (calculés sur tout l'échantillon).
    # -----------------------------------------------------------------
    raw_beta = beta(portfolio_returns, benchmark_returns)
    adj_beta = beta_blume_adjusted(raw_beta)
    ir = information_ratio(portfolio_returns, benchmark_returns)
    print("Beta et ratio d'information (échantillon complet) :\n")
    print(f"  Beta brut (régression MCO)     : {raw_beta:.3f}")
    print(f"  Beta ajusté (Blume, 1975)      : {adj_beta:.3f}")
    print(f"  Ratio d'information (annualisé): {ir:.3f}")
    print()

    # -----------------------------------------------------------------
    # 3) Décomposition marginal / component VaR (Euler).
    #
    # La VaR à 1 jour est la mesure de référence (convention de place).
    # Une version "annualisée" par la règle racine du temps est indiquée à
    # titre illustratif uniquement : elle suppose des rendements i.i.d. sans
    # autocorrélation ni changement de régime de volatilité sur un horizon
    # d'un an, hypothèse fragile en pratique (cf. section Limites du README).
    # -----------------------------------------------------------------
    cov_full = sample_covariance(returns, annualize=False)
    contrib = component_var(PORTFOLIO_WEIGHTS, cov_full, confidence=CONFIDENCE, annualize=False)
    contrib_table = pd.DataFrame(
        {
            "actif": ASSET_NAMES,
            "poids": PORTFOLIO_WEIGHTS,
            "VaR_marginale_1j": contrib.marginal_var,
            "VaR_composante_1j": contrib.component_var,
            "contribution_pct": contrib.pct_contribution * 100,
        }
    )
    var_annualized_illustrative = contrib.portfolio_var * np.sqrt(252)
    print(
        f"Décomposition de la VaR paramétrique {CONFIDENCE:.0%} par position (Euler), "
        f"VaR totale à 1 jour = {contrib.portfolio_var:.2%} :\n"
    )
    print(contrib_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(
        f"\n  Pour mémoire, VaR annualisée par la règle racine du temps "
        f"(VaR_1j x sqrt(252), hypothèse i.i.d. — indicative uniquement) : "
        f"{var_annualized_illustrative:.2%}"
    )
    print()

    # -----------------------------------------------------------------
    # 4) Décomposition facteur / spécifique du risque actif.
    # -----------------------------------------------------------------
    active_weights = PORTFOLIO_WEIGHTS - BENCHMARK_WEIGHTS
    factor_returns = benchmark_returns.to_numpy()
    factor_var_daily = float(np.var(factor_returns, ddof=1))

    asset_returns = returns.to_numpy()
    loadings = np.empty(len(ASSET_NAMES))
    specific_var = np.empty(len(ASSET_NAMES))
    for i in range(len(ASSET_NAMES)):
        cov_i = np.cov(asset_returns[:, i], factor_returns, ddof=1)[0, 1]
        loadings[i] = cov_i / factor_var_daily
        residual = asset_returns[:, i] - loadings[i] * factor_returns
        specific_var[i] = float(np.var(residual, ddof=1))

    decomposition = factor_specific_decomposition(active_weights, loadings.reshape(-1, 1), np.array([[factor_var_daily]]), specific_var)
    print("Décomposition du risque actif (modèle à 1 facteur = benchmark) :\n")
    print(f"  Variance active totale (annualisée) : {decomposition.total_variance * 252:.6f}")
    print(f"  Risque factoriel                    : {decomposition.factor_pct:.1%}")
    print(f"  Risque spécifique                   : {decomposition.specific_pct:.1%}")
    print()

    # -----------------------------------------------------------------
    # Graphique 1 : contributions au risque par position (barres).
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#C00000" if c < 0 else "#1F4E78" for c in contrib.pct_contribution]
    ax.bar(ASSET_NAMES, contrib.pct_contribution * 100, color=colors)
    for i, v in enumerate(contrib.pct_contribution * 100):
        ax.text(i, v + (0.5 if v >= 0 else -1.0), f"{v:.1f}%", ha="center", fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"Contribution au risque par position (VaR 1 jour {CONFIDENCE:.0%}, allocation d'Euler)")
    ax.set_ylabel("Contribution à la VaR totale (%)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out1 = IMG_DIR / "04_contributions_risque.png"
    fig.savefig(out1, dpi=130)
    plt.close(fig)
    print(f"Graphique sauvegardé : {out1}")

    # -----------------------------------------------------------------
    # Graphique 2 : décomposition facteur vs spécifique (camembert).
    # -----------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(6.5, 5.5))
    ax2.pie(
        [decomposition.factor_pct, decomposition.specific_pct],
        labels=["Risque factoriel (marché)", "Risque spécifique"],
        colors=["#1F4E78", "#BF8F00"],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1},
    )
    ax2.set_title("Décomposition du risque actif (Tracking Error) — facteur vs spécifique")
    fig2.tight_layout()
    out2 = IMG_DIR / "04_te_decomposition.png"
    fig2.savefig(out2, dpi=130)
    plt.close(fig2)
    print(f"Graphique sauvegardé : {out2}")

    # -----------------------------------------------------------------
    # Graphique 3 : TE ex-ante prévue vs TE ex-post réalisée (série glissante).
    # -----------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(10.5, 5.5))
    ax3.plot(te_series_df["date"], te_series_df["te_ante"] * 100, color="#1F4E78", lw=1.6, label="TE ex-ante prévue (fenêtre passée)")
    ax3.plot(te_series_df["date"], te_series_df["te_post"] * 100, color="#C00000", lw=1.6, label="TE ex-post réalisée (fenêtre suivante)")
    ax3.set_title("Tracking Error ex-ante prévue vs ex-post réalisée (fenêtres glissantes, 250j)")
    ax3.set_xlabel("Date de recalibration")
    ax3.set_ylabel("Tracking Error annualisée (%)")
    ax3.legend(loc="upper left", fontsize=9)
    ax3.grid(alpha=0.25)
    fig3.tight_layout()
    out3 = IMG_DIR / "04_te_ante_vs_post.png"
    fig3.savefig(out3, dpi=130)
    plt.close(fig3)
    print(f"Graphique sauvegardé : {out3}")


if __name__ == "__main__":
    main()
