"""Value-at-Risk et Expected Shortfall : méthodes historique, paramétrique, Cornish-Fisher et Monte Carlo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def historical_var(returns: pd.Series | np.ndarray, confidence: float = 0.99) -> float:
    """VaR par simulation historique non paramétrique (quantile empirique)."""
    r = np.asarray(returns, dtype=float)
    alpha = 1.0 - confidence
    return float(-np.quantile(r, alpha))


def historical_var_weighted(
    returns: pd.Series | np.ndarray,
    confidence: float = 0.99,
    decay: float = 0.98,
) -> float:
    """VaR historique pondérée exponentiellement (BRW, Boudoukh-Richardson-Whitelaw, 1998)."""
    r = np.asarray(returns, dtype=float)
    n = r.shape[0]
    i = np.arange(1, n + 1)
    weights = (1 - decay) * decay ** (n - i)
    weights = weights / weights.sum()

    order = np.argsort(r)
    r_sorted = r[order]
    w_sorted = weights[order]
    cum_weights = np.cumsum(w_sorted)

    alpha = 1.0 - confidence
    idx = int(np.searchsorted(cum_weights, alpha))
    idx = min(idx, n - 1)
    return float(-r_sorted[idx])


def historical_es(returns: pd.Series | np.ndarray, confidence: float = 0.99) -> float:
    """Expected Shortfall (CVaR) historique : moyenne des pertes au-delà de la VaR."""
    r = np.asarray(returns, dtype=float)
    var = historical_var(r, confidence)
    tail = r[r <= -var]
    if tail.size == 0:
        tail = np.array([r.min()])
    return float(-tail.mean())


def parametric_var_gaussian(
    returns: pd.Series | np.ndarray,
    confidence: float = 0.99,
    horizon: int = 1,
) -> float:
    """VaR paramétrique sous hypothèse de normalité des rendements."""
    r = np.asarray(returns, dtype=float)
    mu, sigma = r.mean(), r.std(ddof=1)
    z = stats.norm.ppf(1.0 - confidence)
    return float(-(mu * horizon + z * sigma * np.sqrt(horizon)))


def parametric_es_gaussian(
    returns: pd.Series | np.ndarray,
    confidence: float = 0.99,
    horizon: int = 1,
) -> float:
    """Expected Shortfall paramétrique gaussien (formule fermée)."""
    r = np.asarray(returns, dtype=float)
    mu, sigma = r.mean(), r.std(ddof=1)
    alpha_tail = 1.0 - confidence
    z = stats.norm.ppf(alpha_tail)
    es_1d = -mu + sigma * stats.norm.pdf(z) / alpha_tail
    es_h = -mu * horizon + (es_1d + mu) * np.sqrt(horizon)
    return float(es_h)


def cornish_fisher_quantile(z: float, skew: float, excess_kurtosis: float) -> float:
    """Quantile ajusté de Cornish-Fisher à partir du quantile normal."""
    return (
        z
        + (z**2 - 1) * skew / 6
        + (z**3 - 3 * z) * excess_kurtosis / 24
        - (2 * z**3 - 5 * z) * skew**2 / 36
    )


def cornish_fisher_var(
    returns: pd.Series | np.ndarray,
    confidence: float = 0.99,
    horizon: int = 1,
) -> float:
    """VaR de Cornish-Fisher : correction du quantile normal par skew/kurtosis."""
    r = np.asarray(returns, dtype=float)
    mu, sigma = r.mean(), r.std(ddof=1)
    skew = stats.skew(r)
    excess_kurt = stats.kurtosis(r, fisher=True)
    z = stats.norm.ppf(1.0 - confidence)
    z_cf = cornish_fisher_quantile(z, skew, excess_kurt)
    return float(-(mu * horizon + z_cf * sigma * np.sqrt(horizon)))


def cornish_fisher_es(
    returns: pd.Series | np.ndarray,
    confidence: float = 0.99,
    horizon: int = 1,
    n_integration_points: int = 20_000,
) -> float:
    """Expected Shortfall Cornish-Fisher, par intégration numérique de la queue."""
    r = np.asarray(returns, dtype=float)
    mu, sigma = r.mean(), r.std(ddof=1)
    skew = stats.skew(r)
    excess_kurt = stats.kurtosis(r, fisher=True)
    alpha_tail = 1.0 - confidence

    u = np.linspace(1e-6, alpha_tail, n_integration_points)
    z_u = stats.norm.ppf(u)
    z_cf_u = cornish_fisher_quantile(z_u, skew, excess_kurt)
    mean_tail_z = z_cf_u.mean()

    es_1d = -(mu + sigma * mean_tail_z)
    es_h = -mu * horizon + (es_1d + mu) * np.sqrt(horizon)
    return float(es_h)


@dataclass
class MonteCarloVaRResult:
    """Résultat d'une simulation Monte Carlo de VaR/ES portefeuille."""

    var: float
    es: float
    simulated_portfolio_returns: np.ndarray


def monte_carlo_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float = 0.99,
    horizon: int = 1,
    n_sims: int = 50_000,
    dist: str = "student_t",
    dof: float = 6.0,
    seed: int | None = None,
) -> MonteCarloVaRResult:
    """VaR/ES Monte Carlo par simulation de scénarios corrélés (Cholesky)."""
    w = np.asarray(weights, dtype=float)
    if w.shape[0] != returns.shape[1]:
        raise ValueError("weights doit avoir autant d'éléments que de colonnes dans returns.")

    mu = returns.mean().to_numpy()
    cov = returns.cov().to_numpy()
    n_assets = w.shape[0]

    cov_reg = cov + np.eye(n_assets) * 1e-12
    chol = np.linalg.cholesky(cov_reg)

    rng = np.random.default_rng(seed)
    if dist == "normal":
        z = rng.standard_normal((n_sims, n_assets))
    elif dist == "student_t":
        g = rng.standard_normal((n_sims, n_assets))
        u = rng.chisquare(dof, size=n_sims) / dof
        z = g / np.sqrt(u)[:, None]
        z /= np.sqrt(dof / (dof - 2))
    else:
        raise ValueError("dist doit être 'normal' ou 'student_t'.")

    shocks = z @ chol.T
    sim_returns_1d = mu[None, :] + shocks
    port_returns_1d = sim_returns_1d @ w

    mu_p = mu @ w
    port_returns_h = mu_p * horizon + (port_returns_1d - mu_p) * np.sqrt(horizon)

    var = historical_var(port_returns_h, confidence)
    es = historical_es(port_returns_h, confidence)
    return MonteCarloVaRResult(var=var, es=es, simulated_portfolio_returns=port_returns_h)


def sqrt_time_scaling(var_1day: float, horizon: int) -> float:
    """Applique la règle racine du temps à une VaR 1 jour."""
    return float(var_1day * np.sqrt(horizon))


def time_scaling_diagnostic(
    returns: pd.Series | np.ndarray,
    confidence: float = 0.99,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
) -> pd.DataFrame:
    """Compare la VaR racine-du-temps à la VaR historique réelle à h jours."""
    r = pd.Series(np.asarray(returns, dtype=float))
    var_1d = historical_var(r, confidence)

    rows = []
    for h in horizons:
        var_sqrt = sqrt_time_scaling(var_1d, h)
        if h == 1:
            var_hist_h = var_1d
        else:
            rolling_h = r.rolling(window=h).sum().dropna()
            var_hist_h = historical_var(rolling_h, confidence)
        ecart_pct = 100.0 * (var_sqrt - var_hist_h) / var_hist_h if var_hist_h != 0 else np.nan
        rows.append(
            {
                "horizon": h,
                "VaR_sqrt_time": var_sqrt,
                "VaR_historique_h_jours": var_hist_h,
                "ecart_relatif_pct": ecart_pct,
            }
        )
    return pd.DataFrame(rows)


def var_es_summary(
    returns: pd.Series | np.ndarray,
    confidence: float = 0.99,
    ewma_decay: float = 0.94,
) -> pd.DataFrame:
    """Calcule VaR et ES par les quatre méthodes univariées sur une série."""
    r = np.asarray(returns, dtype=float)
    rows = {
        "Historique": (historical_var(r, confidence), historical_es(r, confidence)),
        "Historique pondérée (BRW)": (
            historical_var_weighted(r, confidence, decay=ewma_decay),
            historical_es(r, confidence),
        ),
        "Paramétrique gaussienne": (
            parametric_var_gaussian(r, confidence),
            parametric_es_gaussian(r, confidence),
        ),
        "Cornish-Fisher": (
            cornish_fisher_var(r, confidence),
            cornish_fisher_es(r, confidence),
        ),
    }
    df = pd.DataFrame(rows, index=["VaR", "ES"]).T
    return df
