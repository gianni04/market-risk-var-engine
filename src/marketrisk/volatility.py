"""Estimation et prévision de la volatilité : historique, EWMA, GARCH(1,1)."""

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
    """Volatilité historique (écart-type empirique), ponctuelle ou glissante."""
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
    """Volatilité conditionnelle EWMA (RiskMetrics, lambda=0.94 par défaut)."""
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
    """Résultat de l'estimation d'un GARCH(1,1) par maximum de vraisemblance."""

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
    """Estime un GARCH(1,1) par maximum de vraisemblance (scipy.optimize)."""
    r = np.asarray(returns, dtype=float)
    eps = r - r.mean() if demean else r.copy()
    sample_var = eps.var(ddof=1)

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
    """Prévision de la variance conditionnelle GARCH(1,1) à ``horizon`` jours."""
    long_run_var = fit.long_run_variance
    persistence = fit.persistence
    sigma2_t1 = fit.sigma2[-1]

    forecasts_var = np.empty(horizon)
    for h in range(1, horizon + 1):
        forecasts_var[h - 1] = long_run_var + (persistence ** (h - 1)) * (sigma2_t1 - long_run_var)

    factor = np.sqrt(periods_per_year) if annualize else 1.0
    return np.sqrt(forecasts_var) * factor


def forecast_garch_term_vol(fit: GarchResult, horizon: int) -> float:
    """Volatilité cumulée (« term volatility ») sur ``horizon`` jours."""
    daily_vols = forecast_garch_volatility(fit, horizon, annualize=False)
    return float(np.sqrt(np.sum(daily_vols**2)))
