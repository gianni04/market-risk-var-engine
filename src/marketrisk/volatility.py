"""Estimation et prévision de la volatilité : historique, EWMA, GARCH(1,1).

Le GARCH(1,1) est implémenté « à la main » par maximum de vraisemblance
(scipy.optimize), sans dépendre du package `arch` (non disponible dans cet
environnement). Référence : Bollerslev, T. (1986), « Generalized
Autoregressive Conditional Heteroskedasticity », *Journal of Econometrics*.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS_PER_YEAR = 252


def historical_volatility(
    returns: pd.Series | np.ndarray,
    window: int | None = None,
    annualize: bool = True,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float | pd.Series:
    """Volatilité historique (écart-type empirique), ponctuelle ou glissante.

    .. math::
        \\hat{\\sigma} = \\sqrt{\\frac{1}{n-1}\\sum_{t=1}^n (r_t - \\bar{r})^2}

    Parameters
    ----------
    returns : array-like
        Rendements journaliers.
    window : int, optional
        Si fourni, calcule une volatilité glissante sur cette fenêtre
        (retourne une ``pd.Series``) ; sinon un scalaire sur toute la série.
    annualize : bool
        Multiplie par ``sqrt(periods_per_year)`` si ``True``.
    periods_per_year : int
        Nombre de périodes par an pour l'annualisation.

    Returns
    -------
    float or pd.Series
        Volatilité (annualisée ou non).
    """
    factor = np.sqrt(periods_per_year) if annualize else 1.0
    if window is None:
        r = np.asarray(returns, dtype=float)
        return float(r.std(ddof=1) * factor)
    s = pd.Series(returns).astype(float)
    return s.rolling(window=window).std(ddof=1) * factor


def ewma_volatility(
    returns: pd.Series | np.ndarray,
    lam: float = 0.94,
    annualize: bool = True,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Volatilité conditionnelle EWMA (RiskMetrics, lambda=0.94 par défaut).

    .. math::
        \\sigma_t^2 = \\lambda \\sigma_{t-1}^2 + (1-\\lambda) r_{t-1}^2

    Initialisée avec la variance échantillon complète. Référence :
    J.P. Morgan/Reuters, *RiskMetrics — Technical Document*, 4e éd., 1996.

    Parameters
    ----------
    returns : array-like
        Rendements journaliers, ordonnés chronologiquement.
    lam : float
        Facteur de lissage ``lambda`` (0.94 = standard RiskMetrics
        quotidien ; 0.97 = standard mensuel).
    annualize : bool
        Annualise la sortie si ``True``.
    periods_per_year : int
        Périodes par an.

    Returns
    -------
    pd.Series
        Volatilité conditionnelle EWMA, même index que ``returns``.
    """
    r = pd.Series(returns).astype(float)
    n = len(r)
    sigma2 = np.empty(n)
    sigma2[0] = r.var(ddof=1)
    r_vals = r.to_numpy()
    for t in range(1, n):
        sigma2[t] = lam * sigma2[t - 1] + (1 - lam) * r_vals[t - 1] ** 2
    factor = np.sqrt(periods_per_year) if annualize else 1.0
    return pd.Series(np.sqrt(sigma2) * factor, index=r.index, name="ewma_vol")


@dataclass
class GarchResult:
    """Résultat de l'estimation d'un GARCH(1,1) par maximum de vraisemblance.

    Attributes
    ----------
    omega, alpha, beta : float
        Paramètres estimés du GARCH(1,1).
    sigma2 : np.ndarray
        Trajectoire de variance conditionnelle en échantillon.
    log_likelihood : float
        Log-vraisemblance au maximum.
    converged : bool
        Indicateur de convergence de l'optimiseur.
    n_iterations : int
        Nombre d'itérations de l'optimiseur.
    """

    omega: float
    alpha: float
    beta: float
    sigma2: np.ndarray
    log_likelihood: float
    converged: bool
    n_iterations: int

    @property
    def long_run_variance(self) -> float:
        """Variance inconditionnelle de long terme : ``omega / (1 - alpha - beta)``."""
        return self.omega / (1.0 - self.alpha - self.beta)

    @property
    def persistence(self) -> float:
        """Persistance du choc de volatilité : ``alpha + beta``."""
        return self.alpha + self.beta


def _garch11_sigma2(returns: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    """Reconstruit la trajectoire de variance conditionnelle GARCH(1,1)."""
    n = returns.shape[0]
    sigma2 = np.empty(n)
    sigma2[0] = omega / max(1.0 - alpha - beta, 1e-8)
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
    return sigma2


def _garch11_neg_log_likelihood(params: np.ndarray, returns: np.ndarray) -> float:
    """Log-vraisemblance négative gaussienne d'un GARCH(1,1) (à minimiser)."""
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999999:
        return 1e10
    sigma2 = _garch11_sigma2(returns, omega, alpha, beta)
    sigma2 = np.maximum(sigma2, 1e-12)
    ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + returns**2 / sigma2)
    if not np.isfinite(ll):
        return 1e10
    return -ll


def fit_garch11(
    returns: pd.Series | np.ndarray,
    demean: bool = True,
    initial_guess: tuple[float, float, float] | None = None,
) -> GarchResult:
    """Estime un GARCH(1,1) par maximum de vraisemblance (scipy.optimize).

    Modèle :

    .. math::
        r_t = \\mu + \\varepsilon_t, \\quad \\varepsilon_t = \\sigma_t z_t,
        \\quad z_t \\sim N(0,1)

        \\sigma_t^2 = \\omega + \\alpha \\varepsilon_{t-1}^2 + \\beta \\sigma_{t-1}^2

    Optimisation sous contraintes de stationnarité et positivité
    (:math:`\\omega > 0`, :math:`\\alpha, \\beta \\ge 0`,
    :math:`\\alpha + \\beta < 1`) via `scipy.optimize.minimize` (SLSQP).

    Parameters
    ----------
    returns : array-like
        Rendements journaliers.
    demean : bool
        Si ``True``, retire la moyenne échantillon avant estimation
        (pratique standard pour isoler la dynamique de variance).
    initial_guess : tuple of float, optional
        Point de départ ``(omega, alpha, beta)``. Par défaut calibré sur la
        variance échantillon.

    Returns
    -------
    GarchResult
        Paramètres estimés, trajectoire de variance et diagnostics.
    """
    r = np.asarray(returns, dtype=float)
    eps = r - r.mean() if demean else r.copy()
    sample_var = eps.var(ddof=1)

    # SLSQP est mal conditionné lorsque les paramètres ont des échelles très
    # différentes (omega ~ 1e-5 vs alpha/beta ~ O(0.1-1)) : le pas de
    # différences finies par défaut fait alors quasi-stationner le
    # gradient sur omega et l'optimiseur déclare une convergence prématurée.
    # On normalise les rendements par leur écart-type avant estimation
    # (omega_scaled ~ O(0.01-1)), puis on annule la mise à l'échelle sur
    # omega en sortie : omega = omega_scaled * scale**2 (alpha, beta sont
    # invariants par changement d'échelle linéaire des rendements).
    scale = np.sqrt(sample_var) if sample_var > 0 else 1.0
    eps_scaled = eps / scale

    if initial_guess is None:
        alpha0, beta0 = 0.08, 0.85
        omega0_scaled = 1.0 - alpha0 - beta0
        initial_guess_scaled = (omega0_scaled, alpha0, beta0)
    else:
        omega0, alpha0, beta0 = initial_guess
        initial_guess_scaled = (omega0 / (scale**2), alpha0, beta0)

    bounds = [(1e-8, 50.0), (1e-6, 0.4), (1e-6, 0.995)]
    constraints = [{"type": "ineq", "fun": lambda p: 0.999 - p[1] - p[2]}]

    with warnings.catch_warnings():
        # SLSQP peut ponctuellement dépasser une borne pendant un pas de
        # différences finies avant d'être re-projeté : avertissement
        # interne à scipy, sans incidence sur la convergence.
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")
        result = minimize(
            _garch11_neg_log_likelihood,
            x0=np.array(initial_guess_scaled),
            args=(eps_scaled,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )

    omega_scaled, alpha, beta = result.x
    omega = float(omega_scaled * scale**2)
    sigma2 = _garch11_sigma2(eps, omega, alpha, beta)
    log_likelihood = -_garch11_neg_log_likelihood(np.array([omega, alpha, beta]), eps)
    return GarchResult(
        omega=omega,
        alpha=float(alpha),
        beta=float(beta),
        sigma2=sigma2,
        log_likelihood=float(log_likelihood),
        converged=bool(result.success),
        n_iterations=int(result.nit),
    )


def forecast_garch_volatility(
    fit: GarchResult,
    horizon: int = 10,
    annualize: bool = True,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> np.ndarray:
    """Prévision de la variance conditionnelle GARCH(1,1) à ``horizon`` jours.

    Utilise la propriété de retour à la moyenne du GARCH(1,1) :

    .. math::
        E_t[\\sigma_{t+h}^2] = \\sigma_{LR}^2 +
        (\\alpha+\\beta)^{h-1}\\left(\\sigma_{t+1}^2 - \\sigma_{LR}^2\\right)

    où :math:`\\sigma_{LR}^2 = \\omega/(1-\\alpha-\\beta)` est la variance de
    long terme.

    Parameters
    ----------
    fit : GarchResult
        Résultat d'estimation de :func:`fit_garch11`.
    horizon : int
        Nombre de jours à prévoir (1 à horizon inclus).
    annualize : bool
        Annualise les volatilités prévues si ``True``.
    periods_per_year : int
        Périodes par an.

    Returns
    -------
    np.ndarray
        Volatilités prévues (écart-type, pas variance), taille ``horizon``,
        pour t+1, ..., t+horizon.
    """
    long_run_var = fit.long_run_variance
    persistence = fit.persistence
    sigma2_t1 = fit.sigma2[-1]  # dernière variance en échantillon = base pour h=1... voir note ci-dessous

    # sigma2_{t+1} est déjà connu (dernière obs disponible sert de "présent")
    forecasts_var = np.empty(horizon)
    for h in range(1, horizon + 1):
        forecasts_var[h - 1] = long_run_var + (persistence ** (h - 1)) * (sigma2_t1 - long_run_var)

    factor = np.sqrt(periods_per_year) if annualize else 1.0
    return np.sqrt(forecasts_var) * factor


def forecast_garch_term_vol(fit: GarchResult, horizon: int) -> float:
    """Volatilité cumulée (« term volatility ») sur ``horizon`` jours.

    Somme des variances conditionnelles prévues jour par jour puis racine
    carrée, plus précis que la simple règle racine du temps appliquée à la
    volatilité du jour courant lorsque le GARCH est loin de son régime de
    long terme.

    .. math::
        \\sigma_{[t, t+h]} = \\sqrt{\\sum_{k=1}^{h} E_t[\\sigma_{t+k}^2]}

    Parameters
    ----------
    fit : GarchResult
        Résultat d'estimation de :func:`fit_garch11`.
    horizon : int
        Horizon en jours.

    Returns
    -------
    float
        Volatilité cumulée non-annualisée sur l'horizon.
    """
    daily_vols = forecast_garch_volatility(fit, horizon, annualize=False)
    return float(np.sqrt(np.sum(daily_vols**2)))
