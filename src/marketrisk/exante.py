"""Risque ex-ante relatif : tracking error, beta, ratio d'information, contribution au risque."""

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
    """Tracking Error ex-ante à partir des poids actifs et de la covariance."""
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
    """Tracking Error ex-post (réalisée), écart-type des écarts de performance."""
    active = np.asarray(portfolio_returns, dtype=float) - np.asarray(benchmark_returns, dtype=float)
    factor = np.sqrt(periods_per_year) if annualize else 1.0
    return float(active.std(ddof=1) * factor)


def beta(
    portfolio_returns: pd.Series | np.ndarray,
    benchmark_returns: pd.Series | np.ndarray,
) -> float:
    """Beta du portefeuille par rapport au benchmark (régression MCO)."""
    p = np.asarray(portfolio_returns, dtype=float)
    b = np.asarray(benchmark_returns, dtype=float)
    cov = np.cov(p, b, ddof=1)
    return float(cov[0, 1] / cov[1, 1])


def beta_blume_adjusted(raw_beta: float) -> float:
    """Beta ajusté selon la règle empirique de Blume (1975)."""
    return (1.0 / 3.0) * 1.0 + (2.0 / 3.0) * raw_beta


def information_ratio(
    portfolio_returns: pd.Series | np.ndarray,
    benchmark_returns: pd.Series | np.ndarray,
    annualize: bool = True,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Ratio d'information : rendement actif moyen / tracking error ex-post."""
    active = np.asarray(portfolio_returns, dtype=float) - np.asarray(benchmark_returns, dtype=float)
    mean_active = active.mean()
    std_active = active.std(ddof=1)
    if std_active == 0:
        return 0.0
    if annualize:
        mean_active *= periods_per_year
        std_active *= np.sqrt(periods_per_year)
    return float(mean_active / std_active)


@dataclass
class RiskContributionResult:
    """Décomposition du risque de portefeuille par position (allocation d'Euler)."""

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
    """Décomposition de la VaR paramétrique gaussienne par contribution (Euler)."""
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float) * periods_per_year if annualize else np.asarray(cov_matrix, dtype=float)

    portfolio_variance = w @ cov @ w
    portfolio_std = float(np.sqrt(max(portfolio_variance, 1e-18)))

    z = float(-stats.norm.ppf(1.0 - confidence))
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
    """Décomposition du risque actif en risque factoriel et risque spécifique."""

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
    """Décompose le risque actif en composante factorielle et spécifique."""
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
