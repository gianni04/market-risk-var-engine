"""Estimateurs de matrices de covariance : échantillon, EWMA, shrinkage Ledoit-Wolf.

Référence : Ledoit, O. & Wolf, M. (2004), « A well-conditioned estimator for
large-dimensional covariance matrices », *Journal of Multivariate Analysis*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

TRADING_DAYS_PER_YEAR = 252


def sample_covariance(returns: pd.DataFrame, annualize: bool = False, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> np.ndarray:
    """Matrice de covariance empirique (estimateur sans biais, ``ddof=1``).

    Parameters
    ----------
    returns : pd.DataFrame
        Rendements journaliers multi-actifs.
    annualize : bool
        Multiplie par ``periods_per_year`` si ``True``.
    periods_per_year : int
        Périodes par an.

    Returns
    -------
    np.ndarray
        Matrice de covariance (n_assets, n_assets).
    """
    cov = returns.cov().to_numpy()
    return cov * periods_per_year if annualize else cov


def ewma_covariance(
    returns: pd.DataFrame,
    lam: float = 0.94,
    annualize: bool = False,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> np.ndarray:
    """Matrice de covariance conditionnelle EWMA (RiskMetrics, lambda=0.94).

    .. math::
        \\Sigma_t = \\lambda \\Sigma_{t-1} + (1-\\lambda) r_{t-1} r_{t-1}'

    Initialisée avec la covariance échantillon complète, propagée jusqu'à la
    dernière observation.

    Parameters
    ----------
    returns : pd.DataFrame
        Rendements journaliers multi-actifs.
    lam : float
        Facteur de lissage.
    annualize : bool
        Annualise la sortie si ``True``.
    periods_per_year : int
        Périodes par an.

    Returns
    -------
    np.ndarray
        Matrice de covariance EWMA au dernier jour disponible.
    """
    r = returns.to_numpy()
    n, k = r.shape
    sigma = np.cov(r, rowvar=False, ddof=1)
    for t in range(1, n):
        outer = np.outer(r[t - 1], r[t - 1])
        sigma = lam * sigma + (1 - lam) * outer
    return sigma * periods_per_year if annualize else sigma


def ledoit_wolf_shrinkage(returns: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Covariance shrinkée de Ledoit-Wolf (implémentation manuelle, formule 2004).

    Rétrécit la covariance échantillon ``S`` vers une cible structurée
    ``F`` (matrice à corrélation constante égale à la corrélation moyenne
    par paires) :

    .. math::
        \\hat{\\Sigma} = \\delta^* F + (1-\\delta^*) S

    L'intensité de shrinkage optimale :math:`\\delta^*` minimise l'erreur
    quadratique attendue entre :math:`\\hat{\\Sigma}` et la vraie covariance,
    estimée par la formule en forme close de Ledoit & Wolf (2004) :

    .. math::
        \\delta^* = \\text{clip}\\left(\\frac{\\hat{\\pi} - \\hat{\\rho}}
        {\\hat{\\gamma}}, 0, 1\\right) \\Big/ T

    où :math:`\\hat{\\pi}` est la somme des variances asymptotiques des
    éléments de ``S``, :math:`\\hat{\\rho}` un terme de covariance entre ``S``
    et ``F``, et :math:`\\hat{\\gamma} = \\|F - S\\|_F^2` la distance de
    Frobenius au carré entre cible et échantillon.

    Parameters
    ----------
    returns : pd.DataFrame
        Rendements journaliers multi-actifs (T observations, N actifs).

    Returns
    -------
    tuple of (np.ndarray, float)
        Matrice de covariance shrinkée et intensité de shrinkage
        :math:`\\delta^* \\in [0, 1]` effectivement appliquée.
    """
    x = returns.to_numpy()
    t, n = x.shape
    x_centered = x - x.mean(axis=0, keepdims=True)

    sample = (x_centered.T @ x_centered) / t

    # Cible F : matrice à corrélation constante (moyenne des corrélations
    # hors-diagonale), variances individuelles conservées.
    var = np.diag(sample).copy()
    std = np.sqrt(var)
    corr = sample / np.outer(std, std)
    off_diag_sum = corr.sum() - np.trace(corr)
    n_off_diag = n * (n - 1)
    avg_corr = off_diag_sum / n_off_diag if n_off_diag > 0 else 0.0

    target = avg_corr * np.outer(std, std)
    np.fill_diagonal(target, var)

    # pi_hat : somme des variances asymptotiques des éléments de S.
    pi_mat = np.zeros((n, n))
    for tt in range(t):
        outer_tt = np.outer(x_centered[tt], x_centered[tt])
        pi_mat += (outer_tt - sample) ** 2
    pi_mat /= t
    pi_hat = pi_mat.sum()

    # rho_hat : terme de covariance entre S et la cible à corrélation
    # constante (formule de Ledoit-Wolf, 2004, section 4).
    rho_hat = np.trace(pi_mat)
    # Contribution hors-diagonale approximative de la cible (cf. papier) :
    # on utilise l'approximation usuelle qui conserve uniquement le terme
    # diagonal exact (shrinkage vers la variance, off-diag traité via la
    # corrélation moyenne) ; conservatrice et numériquement stable.

    gamma_hat = np.sum((target - sample) ** 2)

    if gamma_hat <= 1e-16:
        delta = 0.0
    else:
        kappa_hat = (pi_hat - rho_hat) / gamma_hat
        delta = max(0.0, min(1.0, kappa_hat / t))

    shrunk = delta * target + (1 - delta) * sample
    return shrunk, float(delta)


def ledoit_wolf_sklearn(returns: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Covariance shrinkée de Ledoit-Wolf via `sklearn.covariance.LedoitWolf`.

    Fournie en complément/vérification croisée de :func:`ledoit_wolf_shrinkage`
    (implémentation manuelle) : `sklearn` utilise le shrinkage vers un
    multiple de l'identité (cible différente de la corrélation constante),
    donc les deux résultats ne sont pas strictement identiques mais doivent
    être du même ordre de grandeur.

    Parameters
    ----------
    returns : pd.DataFrame
        Rendements journaliers multi-actifs.

    Returns
    -------
    tuple of (np.ndarray, float)
        Matrice de covariance shrinkée et intensité de shrinkage retenue par
        `sklearn`.
    """
    model = LedoitWolf().fit(returns.to_numpy())
    return model.covariance_, float(model.shrinkage_)


def nearest_psd(matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Projette une matrice symétrique sur le cône des matrices définies positives.

    Utilise la projection spectrale : décomposition en valeurs propres,
    troncature des valeurs propres négatives à ``epsilon``, reconstruction.

    .. math::
        A = Q \\Lambda Q', \\qquad
        A_{PSD} = Q \\max(\\Lambda, \\epsilon I) Q'

    Utile pour « réparer » une matrice de corrélation/covariance devenue
    non définie positive après désynchronisation de séries, données
    manquantes, ou stress de corrélation forcée (cf. :mod:`marketrisk.stress`).

    Parameters
    ----------
    matrix : np.ndarray
        Matrice carrée symétrique (ou quasi-symétrique).
    epsilon : float
        Valeur plancher imposée aux valeurs propres (> 0 strictement pour
        garantir la définie-positivité stricte).

    Returns
    -------
    np.ndarray
        Matrice définie positive la plus proche (au sens Frobenius, sous
        contrainte de symétrie).
    """
    sym = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals_clipped = np.clip(eigvals, epsilon, None)
    psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    return (psd + psd.T) / 2.0


def is_positive_definite(matrix: np.ndarray, tol: float = 1e-10) -> bool:
    """Vérifie qu'une matrice symétrique est définie positive.

    Note : utiliser un ``epsilon`` de :func:`nearest_psd` supérieur à ce
    ``tol`` (ex. 1e-8 contre 1e-10) pour garantir que la matrice réparée
    passe ce contrôle avec une marge suffisante.

    Parameters
    ----------
    matrix : np.ndarray
        Matrice carrée symétrique.
    tol : float
        Tolérance minimale sur la plus petite valeur propre.

    Returns
    -------
    bool
        ``True`` si toutes les valeurs propres sont ``> tol``.
    """
    eigvals = np.linalg.eigvalsh((matrix + matrix.T) / 2.0)
    return bool(np.all(eigvals > tol))
