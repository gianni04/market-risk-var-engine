"""Génération et chargement de données de marché."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ASSET_NAMES: tuple[str, ...] = (
    "Actions_EU",
    "Actions_US",
    "Obligations_Souv",
    "Credit_IG",
    "Or",
)

_DEFAULT_CORRELATION = np.array(
    [
        [1.00, 0.75, -0.20, 0.45, 0.05],
        [0.75, 1.00, -0.15, 0.40, 0.00],
        [-0.20, -0.15, 1.00, 0.55, 0.10],
        [0.45, 0.40, 0.55, 1.00, -0.05],
        [0.05, 0.00, 0.10, -0.05, 1.00],
    ]
)

_DEFAULT_ANNUAL_DRIFT = np.array([0.06, 0.07, 0.01, 0.03, 0.02])

_DEFAULT_ANNUAL_VOL = np.array([0.18, 0.20, 0.06, 0.09, 0.15])

TRADING_DAYS_PER_YEAR = 252

_PACKAGE_DIR = Path(__file__).resolve().parent
_EMBEDDED_CSV_PATH = _PACKAGE_DIR.parent.parent / "data" / "sample_returns.csv"


def _simulate_correlated_t_shocks(
    n_days: int,
    correlation: np.ndarray,
    dof: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simule des chocs multivariés Student-t standardisés et corrélés."""
    if dof <= 2:
        raise ValueError("dof doit être strictement supérieur à 2 pour une variance finie.")
    n_assets = correlation.shape[0]
    chol = np.linalg.cholesky(correlation)
    g = rng.standard_normal((n_days, n_assets)) @ chol.T
    u = rng.chisquare(dof, size=n_days) / dof
    t_shocks = g / np.sqrt(u)[:, None]
    t_shocks /= np.sqrt(dof / (dof - 2))
    return t_shocks


def _garch11_path(
    z_shocks: np.ndarray,
    omega: float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Génère un chemin de rendements centrés selon un GARCH(1,1) donné."""
    n_days = z_shocks.shape[0]
    long_run_var = omega / (1.0 - alpha - beta)
    sigma2 = np.empty(n_days)
    r = np.empty(n_days)
    sigma2[0] = long_run_var
    r[0] = np.sqrt(sigma2[0]) * z_shocks[0]
    for t in range(1, n_days):
        sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
        r[t] = np.sqrt(sigma2[t]) * z_shocks[t]
    return r


def generate_synthetic_returns(
    n_days: int = 1500,
    asset_names: tuple[str, ...] | None = None,
    correlation: np.ndarray | None = None,
    annual_drift: np.ndarray | None = None,
    annual_vol: np.ndarray | None = None,
    garch_alpha: float = 0.08,
    garch_beta: float = 0.90,
    student_t_dof: float = 6.0,
    start_date: str = "2019-01-02",
    seed: int | None = 42,
) -> pd.DataFrame:
    """Génère des rendements quotidiens synthétiques multi-actifs réalistes."""
    if garch_alpha + garch_beta >= 1.0:
        raise ValueError(
            "garch_alpha + garch_beta doit être < 1 pour un GARCH(1,1) stationnaire "
            f"(reçu {garch_alpha + garch_beta:.4f})."
        )
    names = list(asset_names) if asset_names is not None else list(DEFAULT_ASSET_NAMES)
    n_assets = len(names)

    corr = _DEFAULT_CORRELATION if correlation is None else np.asarray(correlation, dtype=float)
    drift = _DEFAULT_ANNUAL_DRIFT if annual_drift is None else np.asarray(annual_drift, dtype=float)
    vol = _DEFAULT_ANNUAL_VOL if annual_vol is None else np.asarray(annual_vol, dtype=float)

    if corr.shape != (n_assets, n_assets):
        raise ValueError(f"correlation doit être de forme ({n_assets}, {n_assets}).")
    if drift.shape[0] != n_assets or vol.shape[0] != n_assets:
        raise ValueError("annual_drift et annual_vol doivent avoir n_assets éléments.")

    rng = np.random.default_rng(seed)
    z = _simulate_correlated_t_shocks(n_days, corr, student_t_dof, rng)

    daily_drift = drift / TRADING_DAYS_PER_YEAR
    daily_var_target = (vol**2) / TRADING_DAYS_PER_YEAR

    returns = np.empty((n_days, n_assets))
    for i in range(n_assets):
        omega_i = daily_var_target[i] * (1.0 - garch_alpha - garch_beta)
        centered = _garch11_path(z[:, i], omega_i, garch_alpha, garch_beta)
        returns[:, i] = daily_drift[i] + centered

    dates = pd.bdate_range(start=start_date, periods=n_days)
    return pd.DataFrame(returns, index=dates, columns=names)


def returns_to_prices(returns: pd.DataFrame, start_price: float = 100.0) -> pd.DataFrame:
    """Convertit une trajectoire de rendements arithmétiques en indices de prix."""
    growth = (1.0 + returns).cumprod()
    prices = start_price * growth
    first_row = pd.DataFrame(
        [[start_price] * returns.shape[1]],
        columns=returns.columns,
        index=[returns.index[0] - pd.tseries.offsets.BDay(1)],
    )
    return pd.concat([first_row, prices])


def load_yfinance_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
) -> pd.DataFrame | None:
    """Tente de télécharger des prix de clôture ajustés via `yfinance`."""
    try:
        import yfinance as yf  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
        if raw is None or len(raw) == 0:
            return None
        prices = raw["Close"] if "Close" in raw else raw
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(tickers[0] if tickers else "asset")
        prices = prices.dropna(how="all")
        if prices.empty:
            return None
        returns = prices.pct_change().dropna(how="all")
        return returns
    except Exception:
        return None


def load_embedded_csv(path: str | Path | None = None) -> pd.DataFrame:
    """Charge le CSV de rendements embarqué dans le dépôt (secours ultime)."""
    csv_path = Path(path) if path is not None else _EMBEDDED_CSV_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV embarqué introuvable ({csv_path}). "
            "Utilisez generate_synthetic_returns() pour produire des données."
        )
    return pd.read_csv(csv_path, index_col=0, parse_dates=True)


def load_market_data(
    tickers: list[str] | None = None,
    start: str = "2019-01-02",
    end: str | None = None,
    n_days: int = 1500,
    seed: int | None = 42,
    use_live_data: bool = False,
) -> pd.DataFrame:
    """Point d'entrée unique pour obtenir des rendements de marché."""
    if use_live_data and tickers:
        live = load_yfinance_prices(tickers, start=start, end=end)
        if live is not None and not live.empty:
            return live
        warnings.warn(
            "Téléchargement yfinance indisponible ou échoué : repli sur le "
            "générateur synthétique.",
            RuntimeWarning,
            stacklevel=2,
        )
    return generate_synthetic_returns(n_days=n_days, start_date=start, seed=seed)


def save_embedded_csv(returns: pd.DataFrame, path: str | Path | None = None) -> Path:
    """Sauvegarde un DataFrame de rendements comme CSV embarqué de secours."""
    csv_path = Path(path) if path is not None else _EMBEDDED_CSV_PATH
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    returns.to_csv(csv_path)
    return csv_path
