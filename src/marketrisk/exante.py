"""Risque ex-ante relatif : tracking error, beta, ratio d'information, contribution au risque.

Références : Grinold, R. & Kahn, R. *Active Portfolio Management*, 2e éd.,
McGraw-Hill, 1999 ; Blume, M. (1975), « Betas and Their Regression
Tendencies », *Journal of Finance* ; Menchero, J. (2010), « Risk
Attribution and Portfolio Performance Attribution », pour la décomposition
d'Euler de la VaR/vol.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS_PER_YEAR = 252


def tracking_error_ex_ante(
    weights_portfolio: np.ndarray,
    weights_benchmark: np.ndarray,
    cov_matrix: np.ndarray,
    annualize: bool = True,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Tracking Error ex-ante à partir des poids actifs et de la covariance.

    .. math::
        TE_{ex-ante} = \\sqrt{(w_p - w_b)' \\Sigma (w_p - w_b)}

    Mesure prospective du risque relatif, calculée sur les poids courants et
    une matrice de covariance estimée (échantillon, EWMA ou shrinkée — cf.
    :mod:`marketrisk.covariance`), sans attendre de réaliser l'historique de
    performance.

    Parameters
    ----------
    weights_portfolio : np.ndarray
        Poids du portefeuille.
    weights_benchmark : np.ndarray
        Poids du benchmark (même univers d'actifs).
    cov_matrix : np.ndarray
        Matrice de covariance des rendements (journalière, sauf si
        ``annualize=False``).
    annualize : bool
        Annualise la covariance journalière avant le calcul si ``True``.
    periods_per_year : int
        Périodes par an pour l'annualisation.

    Returns
    -------
    float
        Tracking error ex-ante (écart-type de l'écart de performance).
    """
    active_weights = np.asarray(weights_portfolio, dtype=float) - np.asarray(weights_benchmark, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float) * periods_per_year if annualize else np.asarray(cov_matrix, dtype=float)
    variance = active_weights @ cov @ active_weights
    return float(np.sqrt(max(variance, 0.0)))


def tracking_error_ex_post(
    portfolio_returns: pd.Series | np.ndarray,
    benchmark_returns: pd.Series | np.ndarray,
    annualize: bool = True,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Tracking Error ex-post (réalisée), écart-type des écarts de performance.

    .. math::
        TE_{ex-post} = \\text{std}(r_p - r_b)

    Parameters
    ----------
    portfolio_returns : array-like
        Rendements réalisés du portefeuille.
    benchmark_returns : array-like
        Rendements réalisés du benchmark.
    annualize : bool
        Annualise si ``True``.
    periods_per_year : int
        Périodes par an.

    Returns
    -------
    float
        Tracking error ex-post.
    """
    active = np.asarray(portfolio_returns, dtype=float) - np.asarray(benchmark_returns, dtype=float)
    factor = np.sqrt(periods_per_year) if annualize else 1.0
    return float(active.std(ddof=1) * factor)


def beta(
    portfolio_returns: pd.Series | np.ndarray,
    benchmark_returns: pd.Series | np.ndarray,
) -> float:
    """Beta du portefeuille par rapport au benchmark (régression MCO).

    .. math::
        \\beta = \\frac{Cov(r_p, r_b)}{Var(r_b)}

    Parameters
    ----------
    portfolio_returns : array-like
        Rendements du portefeuille.
    benchmark_returns : array-like
        Rendements du benchmark.

    Returns
    -------
    float
        Beta estimé.
    """
    p = np.asarray(portfolio_returns, dtype=float)
    b = np.asarray(benchmark_returns, dtype=float)
    cov = np.cov(p, b, ddof=1)
    return float(cov[0, 1] / cov[1, 1])


def beta_blume_adjusted(raw_beta: float) -> float:
    """Beta ajusté selon la règle empirique de Blume (1975).

    Corrige le biais de régression vers la moyenne observé empiriquement
    sur les betas historiques (les betas extrêmes tendent à se rapprocher
    de 1 dans le futur) :

    .. math::
        \\beta_{adj} = \\frac{1}{3} \\times 1 + \\frac{2}{3} \\times \\beta_{brut}

    Parameters
    ----------
    raw_beta : float
        Beta historique brut (régression MCO).

    Returns
    -------
    float
        Beta ajusté, plus proche de 1 que le beta brut.
    """
    return (1.0 / 3.0) * 1.0 + (2.0 / 3.0) * raw_beta


def information_ratio(
    portfolio_returns: pd.Series | np.ndarray,
    benchmark_returns: pd.Series | np.ndarray,
    annualize: bool = True,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Ratio d'information : rendement actif moyen / tracking error ex-post.

    .. math::
        IR = \\frac{\\overline{r_p - r_b}}{\\text{std}(r_p - r_b)}

    Parameters
    ----------
    portfolio_returns : array-like
        Rendements du portefeuille.
    benchmark_returns : array-like
        Rendements du benchmark.
    annualize : bool
        Annualise le numérateur et le dénominateur si ``True``.
    periods_per_year : int
        Périodes par an.

    Returns
    -------
    float
        Ratio d'information annualisé (ou non).
    """
    active = np.asarray(portfolio_returns, dtype=float) - np.asarray(benchmark_returns, dtype=float)
    mean_active = active.mean()
    std_active = active.std(ddof=1)
    if std_active == 0:
        return 0.0
    if annualize:
        mean_active *= periods_per_year
        std_active *= np.sqrt(periods_per_year)
    return float(mean_active / std_active)


# ---------------------------------------------------------------------------
# Contribution au risque : marginal VaR, component VaR, allocation d'Euler
# ---------------------------------------------------------------------------


@dataclass
class RiskContributionResult:
    """Décomposition du risque de portefeuille par position (allocation d'Euler).

    Attributes
    ----------
    portfolio_std : float
        Écart-type total du portefeuille.
    portfolio_var : float
        VaR totale du portefeuille (positive).
    marginal_var : np.ndarray
        VaR marginale par position (dérivée de la VaR par rapport au poids).
    component_var : np.ndarray
        VaR de composante par position (``weight * marginal_var``) ; leur
        somme égale exactement la VaR totale (propriété d'Euler pour une
        fonction homogène de degré 1).
    pct_contribution : np.ndarray
        Contribution en % de la VaR totale par position.
    """

    portfolio_std: float
    portfolio_var: float
    marginal_var: np.ndarray
    component_var: np.ndarray
    pct_contribution: np.ndarray


def component_var(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
    confidence: float = 0.99,
    annualize: bool = False,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> RiskContributionResult:
    """Décomposition de la VaR paramétrique gaussienne par contribution (Euler).

    La VaR paramétrique gaussienne d'un portefeuille est une fonction
    homogène de degré 1 des poids :

    .. math::
        VaR_p = z_\\alpha \\sqrt{w' \\Sigma w} = z_\\alpha \\sigma_p

    Le théorème d'Euler pour les fonctions homogènes de degré 1 permet de
    décomposer exactement la VaR totale en contributions par position :

    .. math::
        VaR_p = \\sum_i w_i \\underbrace{\\frac{\\partial VaR_p}{\\partial w_i}}
        _{\\text{VaR marginale}_i}, \\qquad
        \\frac{\\partial VaR_p}{\\partial w_i} = z_\\alpha
        \\frac{(\\Sigma w)_i}{\\sigma_p}

    La **VaR de composante** de la position ``i`` est
    :math:`CVaR_i = w_i \\times \\text{VaR marginale}_i`, et
    :math:`\\sum_i CVaR_i = VaR_p` exactement.

    Parameters
    ----------
    weights : np.ndarray
        Poids du portefeuille (peuvent être négatifs pour des positions
        courtes).
    cov_matrix : np.ndarray
        Matrice de covariance des rendements.
    confidence : float
        Niveau de confiance de la VaR (hypothèse gaussienne).
    annualize : bool
        Annualise la covariance journalière avant le calcul si ``True``.
    periods_per_year : int
        Périodes par an.

    Returns
    -------
    RiskContributionResult
        VaR totale, VaR marginale et VaR de composante par position.
    """
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float) * periods_per_year if annualize else np.asarray(cov_matrix, dtype=float)

    portfolio_variance = w @ cov @ w
    portfolio_std = float(np.sqrt(max(portfolio_variance, 1e-18)))

    z = float(-stats.norm.ppf(1.0 - confidence))  # positif car quantile de queue gauche négatif
    portfolio_var = z * portfolio_std

    sigma_w = cov @ w
    marginal_var = z * sigma_w / portfolio_std
    comp_var = w * marginal_var
    pct = comp_var / portfolio_var if portfolio_var != 0 else np.zeros_like(comp_var)

    return RiskContributionResult(
        portfolio_std=portfolio_std,
        portfolio_var=float(portfolio_var),
        marginal_var=marginal_var,
        component_var=comp_var,
        pct_contribution=pct,
    )


@dataclass
class FactorRiskDecomposition:
    """Décomposition du risque actif en risque factoriel et risque spécifique.

    Attributes
    ----------
    total_variance : float
        Variance active totale.
    factor_variance : float
        Variance expliquée par les facteurs communs.
    specific_variance : float
        Variance résiduelle (idiosyncratique).
    factor_pct : float
        Part du risque factoriel en % du risque actif total.
    specific_pct : float
        Part du risque spécifique en % du risque actif total.
    """

    total_variance: float
    factor_variance: float
    specific_variance: float
    factor_pct: float
    specific_pct: float


def factor_specific_decomposition(
    active_weights: np.ndarray,
    factor_loadings: np.ndarray,
    factor_cov: np.ndarray,
    specific_variance: np.ndarray,
) -> FactorRiskDecomposition:
    """Décompose le risque actif en composante factorielle et spécifique.

    Sous un modèle multi-facteurs :math:`r_i = \\sum_k B_{ik} f_k + \\epsilon_i`
    avec facteurs indépendants des résidus, la variance active se décompose
    exactement :

    .. math::
        \\text{Var}(w_a' r) = \\underbrace{w_a' B \\Omega_F B' w_a}
        _{\\text{risque factoriel}} +
        \\underbrace{w_a' D w_a}_{\\text{risque spécifique}}

    où ``B`` est la matrice de loadings factoriels (n_assets x n_factors),
    :math:`\\Omega_F` la covariance des facteurs et ``D`` la matrice
    diagonale des variances spécifiques.

    Parameters
    ----------
    active_weights : np.ndarray
        Poids actifs (portefeuille - benchmark), taille n_assets.
    factor_loadings : np.ndarray
        Matrice de loadings (n_assets, n_factors).
    factor_cov : np.ndarray
        Matrice de covariance des facteurs (n_factors, n_factors).
    specific_variance : np.ndarray
        Variances spécifiques par actif, taille n_assets.

    Returns
    -------
    FactorRiskDecomposition
        Répartition du risque actif entre facteurs communs et risque
        spécifique.
    """
    w = np.asarray(active_weights, dtype=float)
    b_mat = np.asarray(factor_loadings, dtype=float)
    omega_f = np.asarray(factor_cov, dtype=float)
    d_specific = np.asarray(specific_variance, dtype=float)

    factor_exposure = b_mat.T @ w
    factor_variance = float(factor_exposure @ omega_f @ factor_exposure)
    spec_variance = float(np.sum((w**2) * d_specific))
    total_variance = factor_variance + spec_variance

    factor_pct = factor_variance / total_variance if total_variance > 0 else 0.0
    specific_pct = spec_variance / total_variance if total_variance > 0 else 0.0

    return FactorRiskDecomposition(
        total_variance=total_variance,
        factor_variance=factor_variance,
        specific_variance=spec_variance,
        factor_pct=factor_pct,
        specific_pct=specific_pct,
    )
