"""Backtesting réglementaire de la VaR : Kupiec, Christoffersen, Traffic Light, Berkowitz."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


def var_exceptions(losses: pd.Series | np.ndarray, var_forecast: pd.Series | np.ndarray | float) -> np.ndarray:
    """Identifie les jours de dépassement (« exceptions ») de la VaR."""
    r = np.asarray(losses, dtype=float)
    v = np.asarray(var_forecast, dtype=float) if not np.isscalar(var_forecast) else var_forecast
    return (-r) > v


@dataclass
class TestResult:
    """Résultat générique d'un test statistique de backtesting."""

    name: str
    statistic: float
    p_value: float
    degrees_of_freedom: int
    reject_h0: bool
    significance_level: float
    details: dict = field(default_factory=dict)


def kupiec_pof_test(
    exceptions: np.ndarray,
    confidence: float = 0.99,
    significance_level: float = 0.05,
) -> TestResult:
    """Test de proportion de défaillances de Kupiec (POF, Kupiec 1995)."""
    x = int(np.sum(exceptions))
    n = len(exceptions)
    p = 1.0 - confidence
    p_hat = x / n if n > 0 else 0.0

    def _log_lik(prob: float, x: int, n: int) -> float:
        if prob <= 0.0:
            prob = 1e-12
        if prob >= 1.0:
            prob = 1 - 1e-12
        return (n - x) * np.log(1 - prob) + x * np.log(prob)

    ll_null = _log_lik(p, x, n)
    ll_alt = _log_lik(p_hat, x, n)
    lr_stat = -2.0 * (ll_null - ll_alt)
    lr_stat = max(lr_stat, 0.0)
    p_value = float(1.0 - stats.chi2.cdf(lr_stat, df=1))

    return TestResult(
        name="Kupiec POF (couverture inconditionnelle)",
        statistic=float(lr_stat),
        p_value=p_value,
        degrees_of_freedom=1,
        reject_h0=p_value < significance_level,
        significance_level=significance_level,
        details={"n_exceptions": x, "n_obs": n, "expected_rate": p, "observed_rate": p_hat},
    )


def christoffersen_independence_test(
    exceptions: np.ndarray,
    significance_level: float = 0.05,
) -> TestResult:
    """Test d'indépendance de Christoffersen (1998)."""
    ind = np.asarray(exceptions, dtype=int)
    n00 = n01 = n10 = n11 = 0
    for t in range(1, len(ind)):
        prev, curr = ind[t - 1], ind[t]
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1

    n0_total = n00 + n01
    n1_total = n10 + n11
    pi = (n01 + n11) / max(n0_total + n1_total, 1)
    pi01 = n01 / n0_total if n0_total > 0 else 0.0
    pi11 = n11 / n1_total if n1_total > 0 else 0.0

    def _safe_log(prob: float) -> float:
        prob = min(max(prob, 1e-12), 1 - 1e-12)
        return np.log(prob)

    ll_null = (n00 + n10) * _safe_log(1 - pi) + (n01 + n11) * _safe_log(pi)
    ll_alt = (
        n00 * _safe_log(1 - pi01)
        + n01 * _safe_log(pi01)
        + n10 * _safe_log(1 - pi11)
        + n11 * _safe_log(pi11)
    )
    lr_stat = max(-2.0 * (ll_null - ll_alt), 0.0)
    p_value = float(1.0 - stats.chi2.cdf(lr_stat, df=1))

    return TestResult(
        name="Christoffersen (indépendance des exceptions)",
        statistic=float(lr_stat),
        p_value=p_value,
        degrees_of_freedom=1,
        reject_h0=p_value < significance_level,
        significance_level=significance_level,
        details={"n00": n00, "n01": n01, "n10": n10, "n11": n11, "pi01": pi01, "pi11": pi11},
    )


def christoffersen_joint_test(
    exceptions: np.ndarray,
    confidence: float = 0.99,
    significance_level: float = 0.05,
) -> TestResult:
    """Test conjoint de couverture conditionnelle (Christoffersen, 1998)."""
    uc = kupiec_pof_test(exceptions, confidence, significance_level)
    ind = christoffersen_independence_test(exceptions, significance_level)
    lr_stat = uc.statistic + ind.statistic
    p_value = float(1.0 - stats.chi2.cdf(lr_stat, df=2))

    return TestResult(
        name="Christoffersen (couverture conditionnelle jointe)",
        statistic=float(lr_stat),
        p_value=p_value,
        degrees_of_freedom=2,
        reject_h0=p_value < significance_level,
        significance_level=significance_level,
        details={"LR_uc": uc.statistic, "LR_ind": ind.statistic},
    )


_BASEL_TABLE_250_99 = {
    0: ("verte", 0.00),
    1: ("verte", 0.00),
    2: ("verte", 0.00),
    3: ("verte", 0.00),
    4: ("verte", 0.00),
    5: ("jaune", 0.40),
    6: ("jaune", 0.50),
    7: ("jaune", 0.65),
    8: ("jaune", 0.75),
    9: ("jaune", 0.85),
}


@dataclass
class TrafficLightResult:
    """Résultat du test Traffic Light de Bâle."""

    zone: str
    n_exceptions: int
    n_obs: int
    cumulative_probability_pct: float
    multiplier_addon: float
    base_multiplier: float
    total_multiplier: float


def traffic_light_test(
    n_exceptions: int,
    n_obs: int = 250,
    confidence: float = 0.99,
    base_multiplier: float = 3.0,
) -> TrafficLightResult:
    """Classification Traffic Light de Bâle du nombre d'exceptions de VaR."""
    p = 1.0 - confidence
    cum_prob = float(stats.binom.cdf(n_exceptions, n_obs, p)) * 100.0

    if n_obs == 250 and abs(confidence - 0.99) < 1e-9:
        if n_exceptions in _BASEL_TABLE_250_99:
            zone, addon = _BASEL_TABLE_250_99[n_exceptions]
        elif n_exceptions >= 10:
            zone, addon = "rouge", 1.00
        else:
            zone, addon = "verte", 0.00
    else:
        if cum_prob < 95.0:
            zone, addon = "verte", 0.00
        elif cum_prob < 99.99:
            zone = "jaune"
            frac = (cum_prob - 95.0) / (99.99 - 95.0)
            addon = float(np.clip(0.40 + frac * (0.85 - 0.40), 0.40, 0.85))
        else:
            zone, addon = "rouge", 1.00

    return TrafficLightResult(
        zone=zone,
        n_exceptions=n_exceptions,
        n_obs=n_obs,
        cumulative_probability_pct=cum_prob,
        multiplier_addon=addon,
        base_multiplier=base_multiplier,
        total_multiplier=base_multiplier + addon,
    )


def berkowitz_test(
    returns: np.ndarray,
    forecast_mean: np.ndarray,
    forecast_std: np.ndarray,
    significance_level: float = 0.05,
) -> TestResult:
    """Test de Berkowitz (2001) sur la densité complète des prévisions de risque."""
    r = np.asarray(returns, dtype=float)
    mu = np.asarray(forecast_mean, dtype=float)
    sigma = np.asarray(forecast_std, dtype=float)
    z = (r - mu) / sigma

    z_lag = z[:-1]
    z_curr = z[1:]
    n = z_curr.shape[0]

    x = np.column_stack([np.ones(n), z_lag])
    beta_hat, *_ = np.linalg.lstsq(x, z_curr, rcond=None)
    resid_free = z_curr - x @ beta_hat
    sigma2_free = np.sum(resid_free**2) / n
    ll_free = -0.5 * n * (np.log(2 * np.pi) + np.log(sigma2_free) + 1)

    ll_constrained = -0.5 * np.sum(np.log(2 * np.pi) + z_curr**2)

    lr_stat = max(-2.0 * (ll_constrained - ll_free), 0.0)
    p_value = float(1.0 - stats.chi2.cdf(lr_stat, df=3))

    return TestResult(
        name="Berkowitz (densité, PIT + AR(1))",
        statistic=float(lr_stat),
        p_value=p_value,
        degrees_of_freedom=3,
        reject_h0=p_value < significance_level,
        significance_level=significance_level,
        details={
            "mean_z": float(z.mean()),
            "std_z": float(z.std(ddof=1)),
            "ar1_coef": float(beta_hat[1]),
        },
    )


def backtest_var(
    returns: pd.Series | np.ndarray,
    var_forecast: pd.Series | np.ndarray | float,
    confidence: float = 0.99,
    significance_level: float = 0.05,
) -> dict:
    """Exécute la suite complète de tests de backtesting réglementaire de VaR."""
    exceptions = var_exceptions(returns, var_forecast)
    kupiec = kupiec_pof_test(exceptions, confidence, significance_level)
    christoffersen_ind = christoffersen_independence_test(exceptions, significance_level)
    christoffersen_joint = christoffersen_joint_test(exceptions, confidence, significance_level)
    traffic_light = traffic_light_test(int(exceptions.sum()), n_obs=len(exceptions), confidence=confidence)

    return {
        "exceptions": exceptions,
        "kupiec": kupiec,
        "christoffersen_independence": christoffersen_ind,
        "christoffersen_joint": christoffersen_joint,
        "traffic_light": traffic_light,
    }
