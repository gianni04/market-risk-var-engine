"""Génération et chargement de données de marché.

Ce module fournit :

1. Un générateur synthétique multi-actifs réaliste combinant :
   - une structure de corrélation entre classes d'actifs (copule gaussienne
     appliquée à des chocs Student-t, cf. :func:`_simulate_correlated_t_shocks`) ;
   - un processus GARCH(1,1) par actif pour reproduire le *clustering* de
     volatilité (Engle, 1982 ; Bollerslev, 1986) ;
   - des innovations Student-t pour les queues épaisses observées sur les
     marchés (Mandelbrot, 1963).
2. Un loader optionnel `yfinance` (réseau) avec repli automatique et
   silencieux sur le générateur synthétique ou sur un CSV embarqué si
   `yfinance` n'est pas installé, si le réseau est indisponible, ou si le
   téléchargement échoue pour toute autre raison.

Tout le reste du package (VaR, backtesting, stress tests...) ne dépend que
des DataFrames de rendements produits ici : aucune fonction du package ne
requiert d'accès réseau pour s'exécuter.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Univers d'actifs par défaut : un portefeuille diversifié multi-classes tel
# qu'on le rencontre dans un fonds diversifié / mandat institutionnel.
# ---------------------------------------------------------------------------

DEFAULT_ASSET_NAMES: tuple[str, ...] = (
    "Actions_EU",
    "Actions_US",
    "Obligations_Souv",
    "Credit_IG",
    "Or",
)

# Matrice de corrélation cible (illustrative, ordres de grandeur observés
# sur longue période entre grandes classes d'actifs). Vérifiée définie
# positive (valeurs propres > 0).
_DEFAULT_CORRELATION = np.array(
    [
        [1.00, 0.75, -0.20, 0.45, 0.05],
        [0.75, 1.00, -0.15, 0.40, 0.00],
        [-0.20, -0.15, 1.00, 0.55, 0.10],
        [0.45, 0.40, 0.55, 1.00, -0.05],
        [0.05, 0.00, 0.10, -0.05, 1.00],
    ]
)

# Dérive annualisée cible par actif (rendement espéré, ordre de grandeur).
_DEFAULT_ANNUAL_DRIFT = np.array([0.06, 0.07, 0.01, 0.03, 0.02])

# Volatilité annualisée de long terme cible par actif (utilisée pour calibrer
# le omega du GARCH de sorte que la variance inconditionnelle corresponde).
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
    """Simule des chocs multivariés Student-t standardisés et corrélés.

    Utilise la représentation classique de la loi t multivariée comme un
    mélange normal-chi2 :

    .. math::
        X = \\frac{G}{\\sqrt{U / \\nu}}, \\quad G \\sim N(0, R),\\; U \\sim \\chi^2_\\nu

    La matrice de corrélation de Pearson de X converge vers ``R`` (pour
    :math:`\\nu > 2`), et chaque composante est ensuite redivisée par son
    écart-type théorique :math:`\\sqrt{\\nu/(\\nu-2)}` afin d'obtenir des
    marginales de variance unitaire, directement utilisables comme chocs
    ``z_t`` dans une équation GARCH.

    Parameters
    ----------
    n_days : int
        Nombre de jours à simuler.
    correlation : np.ndarray
        Matrice de corrélation (n_assets, n_assets), définie positive.
    dof : float
        Degrés de liberté de la loi de Student (``dof > 2`` requis pour une
        variance finie).
    rng : np.random.Generator
        Générateur aléatoire NumPy.

    Returns
    -------
    np.ndarray
        Tableau (n_days, n_assets) de chocs standardisés (variance unitaire,
        corrélation ~ ``correlation``, queues épaisses).
    """
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
    """Génère un chemin de rendements centrés selon un GARCH(1,1) donné.

    .. math::
        \\sigma_t^2 = \\omega + \\alpha r_{t-1}^2 + \\beta \\sigma_{t-1}^2, \\qquad
        r_t = \\sigma_t z_t

    Parameters
    ----------
    z_shocks : np.ndarray
        Chocs standardisés (moyenne 0, variance 1), de taille (n_days,).
    omega, alpha, beta : float
        Paramètres GARCH(1,1) avec ``alpha + beta < 1`` (stationnarité).

    Returns
    -------
    np.ndarray
        Rendements centrés simulés, taille (n_days,).
    """
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
    """Génère des rendements quotidiens synthétiques multi-actifs réalistes.

    Combine trois ingrédients standard en modélisation de risque de marché :

    1. **Corrélation entre actifs** : chocs simulés via une copule gaussienne
       appliquée à un mélange normal/chi2 (loi t multivariée), garantissant
       une structure de dépendance stable proche de ``correlation``.
    2. **Clustering de volatilité** : chaque actif suit son propre processus
       GARCH(1,1) (Bollerslev, 1986), calibré pour que sa variance
       inconditionnelle corresponde à ``annual_vol``.
    3. **Queues épaisses** : les chocs standardisés suivent une loi de
       Student à ``student_t_dof`` degrés de liberté plutôt qu'une loi
       normale, conformément aux faits stylisés des rendements financiers.

    Parameters
    ----------
    n_days : int
        Nombre de jours de bourse à simuler.
    asset_names : tuple of str, optional
        Noms des actifs. Par défaut, l'univers 5 classes d'actifs du module.
    correlation : np.ndarray, optional
        Matrice de corrélation cible (définie positive). Par défaut la
        matrice illustrative du module.
    annual_drift : np.ndarray, optional
        Rendement moyen annualisé par actif.
    annual_vol : np.ndarray, optional
        Volatilité annualisée de long terme par actif (calibre omega du
        GARCH via ``omega = annual_vol**2 / 252 * (1 - alpha - beta)``).
    garch_alpha, garch_beta : float
        Paramètres GARCH(1,1) communs à tous les actifs (réaction aux chocs
        récents et persistance). Doit vérifier ``alpha + beta < 1``.
    student_t_dof : float
        Degrés de liberté des innovations Student-t (queues épaisses si
        petit, proche de la normale si grand).
    start_date : str
        Date de départ (jours ouvrés, fréquence 'B').
    seed : int, optional
        Graine du générateur aléatoire pour reproductibilité.

    Returns
    -------
    pd.DataFrame
        Rendements quotidiens (log-rendements simples au sens rendement
        arithmétique journalier), index DatetimeIndex, une colonne par actif.

    Raises
    ------
    ValueError
        Si ``garch_alpha + garch_beta >= 1`` (non-stationnarité) ou si les
        dimensions des paramètres ne correspondent pas au nombre d'actifs.
    """
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
    """Convertit une trajectoire de rendements arithmétiques en indices de prix.

    .. math::
        P_t = P_0 \\prod_{s=1}^{t} (1 + r_s)

    Parameters
    ----------
    returns : pd.DataFrame
        Rendements quotidiens (une colonne par actif).
    start_price : float
        Valeur de base de l'indice de prix à t=0.

    Returns
    -------
    pd.DataFrame
        Indices de prix, même index/colonnes que ``returns``, avec une ligne
        supplémentaire au temps t=0.
    """
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
    """Tente de télécharger des prix de clôture ajustés via `yfinance`.

    Repli automatique et silencieux : retourne ``None`` (jamais d'exception)
    si `yfinance` n'est pas installé, si l'appel réseau échoue, ou si les
    données renvoyées sont vides/inexploitables. Conçu pour être appelé
    depuis :func:`load_market_data` sans jamais casser un pipeline offline.

    Parameters
    ----------
    tickers : list of str
        Tickers boursiers (ex. ``["^GSPC", "^STOXX50E"]``).
    start : str
        Date de début ``YYYY-MM-DD``.
    end : str, optional
        Date de fin ``YYYY-MM-DD``.

    Returns
    -------
    pd.DataFrame or None
        Rendements quotidiens si le téléchargement a réussi, sinon ``None``.
    """
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
        # Toute erreur réseau / DNS / API yfinance -> repli silencieux.
        return None


def load_embedded_csv(path: str | Path | None = None) -> pd.DataFrame:
    """Charge le CSV de rendements embarqué dans le dépôt (secours ultime).

    Ce fichier est généré une fois par :func:`generate_synthetic_returns`
    et versionné dans ``data/sample_returns.csv`` afin que le package puisse
    fonctionner même sans capacité de génération aléatoire cohérente entre
    versions de NumPy.

    Parameters
    ----------
    path : str or Path, optional
        Chemin du CSV. Par défaut, ``data/sample_returns.csv`` à la racine
        du dépôt.

    Returns
    -------
    pd.DataFrame
        Rendements quotidiens, index DatetimeIndex.

    Raises
    ------
    FileNotFoundError
        Si le fichier CSV embarqué est introuvable.
    """
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
    """Point d'entrée unique pour obtenir des rendements de marché.

    Stratégie de repli (« fallback chain ») :

    1. Si ``use_live_data=True`` et ``tickers`` est fourni, tente
       :func:`load_yfinance_prices` (réseau).
    2. Sinon, ou en cas d'échec du (1), génère des données synthétiques via
       :func:`generate_synthetic_returns`.

    Ce comportement garantit que tout script du dépôt s'exécute à l'identique
    avec ou sans connexion réseau.

    Parameters
    ----------
    tickers : list of str, optional
        Tickers à tenter de télécharger si ``use_live_data=True``.
    start : str
        Date de début.
    end : str, optional
        Date de fin (ignorée en mode synthétique, utilisée en mode réseau).
    n_days : int
        Nombre de jours pour le générateur synthétique.
    seed : int, optional
        Graine aléatoire pour le générateur synthétique.
    use_live_data : bool
        Si ``True``, tente d'abord un téléchargement réseau via `yfinance`.

    Returns
    -------
    pd.DataFrame
        Rendements quotidiens multi-actifs.
    """
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
    """Sauvegarde un DataFrame de rendements comme CSV embarqué de secours.

    Utilisé pour régénérer ``data/sample_returns.csv`` versionné dans le
    dépôt (voir :func:`load_embedded_csv`).

    Parameters
    ----------
    returns : pd.DataFrame
        Rendements à sauvegarder.
    path : str or Path, optional
        Chemin de destination. Par défaut ``data/sample_returns.csv``.

    Returns
    -------
    Path
        Chemin effectivement écrit.
    """
    csv_path = Path(path) if path is not None else _EMBEDDED_CSV_PATH
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    returns.to_csv(csv_path)
    return csv_path
