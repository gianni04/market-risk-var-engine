"""Estimateurs de matrices de covariance : échantillon, EWMA, shrinkage Ledoit-Wolf."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

TRADING_DAYS_PER_YEAR = 252


def sample_covariance(returns: pd.DataFrame, annualize: bool = False, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> np.ndarray:
    """Matrice de covariance empirique (estimateur sans biais, ``ddof=1``)."""
    cov = returns.cov().to_numpy()
    return cov * periods_per_year if annualize else cov


def ewma_covariance(
    returns: pd.DataFrame,
    lam: float = 0.94,
    annualize: bool = False,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> np.ndarray:
    """Matrice de covariance conditionnelle EWMA (RiskMetrics, lambda=0.94)."""
    r = returns.to_numpy()
    n, k = r.shape
    sigma = np.cov(r, rowvar=False, ddof=1)
    for t in range(1, n):
        outer = np.outer(r[t - 1], r[t - 1])
        sigma = lam * sigma + (1 - lam) * outer
    return sigma * periods_per_year if annualize else sigma


def ledoit_wolf_shrinkage(returns: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Covariance shrinkée de Ledoit-Wolf (implémentation manuelle, formule 2004)."""
    x = returns.to_numpy()
    t, n = x.shape
    x_centered = x - x.mean(axis=0, keepdims=True)

    sample = (x_centered.T @ x_centered) / t

    var = np.diag(sample).copy()
    std = np.sqrt(var)
    corr = sample / np.outer(std, std)
    off_diag_sum = corr.sum() - np.trace(corr)
    n_off_diag = n * (n - 1)
    avg_corr = off_diag_sum / n_off_diag if n_off_diag > 0 else 0.0

    target = avg_corr * np.outer(std, std)
    np.fill_diagonal(target, var)

    pi_mat = np.zeros((n, n))
    for tt in range(t):
        outer_tt = np.outer(x_centered[tt], x_centered[tt])
        pi_mat += (outer_tt - sample) ** 2
    pi_mat /= t
    pi_hat = pi_mat.sum()

    rho_hat = np.trace(pi_mat)

    gamma_hat = np.sum((target - sample) ** 2)

    if gamma_hat <= 1e-16:
        delta = 0.0
    else:
        kappa_hat = (pi_hat - rho_hat) / gamma_hat
        delta = max(0.0, min(1.0, kappa_hat / t))

    shrunk = delta * target + (1 - delta) * sample
    return shrunk, float(delta)


def ledoit_wolf_sklearn(returns: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Covariance shrinkée de Ledoit-Wolf via `sklearn.covariance.LedoitWolf`."""
    model = LedoitWolf().fit(returns.to_numpy())
    return model.covariance_, float(model.shrinkage_)


def nearest_psd(matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Projette une matrice symétrique sur le cône des matrices définies positives."""
    sym = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals_clipped = np.clip(eigvals, epsilon, None)
    psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    return (psd + psd.T) / 2.0


def is_positive_definite(matrix: np.ndarray, tol: float = 1e-10) -> bool:
    """Vérifie qu'une matrice symétrique est définie positive."""
    eigvals = np.linalg.eigvalsh((matrix + matrix.T) / 2.0)
    return bool(np.all(eigvals > tol))
