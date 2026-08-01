"""Value-at-Risk et Expected Shortfall : méthodes historique, paramétrique,
Cornish-Fisher et Monte Carlo.

Convention de signe : toutes les fonctions de ce module retournent la VaR et
l'Expected Shortfall comme des **nombres positifs** représentant l'ampleur
de la perte au niveau de confiance donné (convention standard en gestion des
risques : « la VaR à 99% est de 2,3% du portefeuille »).

Référence académique : Jorion, P. *Value at Risk: The New Benchmark for
Managing Financial Risk*, 3e éd., McGraw-Hill, 2007.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# VaR historique (simulation historique) et sa variante pondérée BRW
# ---------------------------------------------------------------------------


def historical_var(returns: pd.Series | np.ndarray, confidence: float = 0.99) -> float:
    """VaR par simulation historique non paramétrique (quantile empirique).

    .. math::
        VaR_\\alpha = -Q_{1-\\alpha}(R)

    où :math:`Q_{1-\\alpha}` est le quantile empirique des rendements au
    niveau ``1 - confidence``.

    Parameters
    ----------
    returns : array-like
        Série de rendements historiques (P&L relatif).
    confidence : float
        Niveau de confiance (ex. 0.99 pour une VaR à 99%).

    Returns
    -------
    float
        VaR positive (perte potentielle).
    """
    r = np.asarray(returns, dtype=float)
    alpha = 1.0 - confidence
    return float(-np.quantile(r, alpha))


def historical_var_weighted(
    returns: pd.Series | np.ndarray,
    confidence: float = 0.99,
    decay: float = 0.98,
) -> float:
    """VaR historique pondérée exponentiellement (BRW, Boudoukh-Richardson-Whitelaw, 1998).

    Chaque observation historique reçoit un poids décroissant avec son
    ancienneté :

    .. math::
        w_i = \\frac{(1-\\lambda)\\lambda^{n-i}}{1-\\lambda^n}, \\quad i=1,\\dots,n

    (``i=n`` = observation la plus récente). La VaR est le quantile de la
    distribution empirique **pondérée** (fonction de répartition cumulée en
    poids), ce qui permet à la simulation historique de réagir plus vite à
    un regain de volatilité récent, sans hypothèse de distribution.

    Parameters
    ----------
    returns : array-like
        Série de rendements ordonnée chronologiquement (le premier élément
        est le plus ancien).
    confidence : float
        Niveau de confiance.
    decay : float
        Facteur de décroissance ``lambda`` (proche de 1 = mémoire longue).

    Returns
    -------
    float
        VaR positive pondérée.
    """
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
    """Expected Shortfall (CVaR) historique : moyenne des pertes au-delà de la VaR.

    .. math::
        ES_\\alpha = -E[R \\mid R \\le -VaR_\\alpha]

    Parameters
    ----------
    returns : array-like
        Rendements historiques.
    confidence : float
        Niveau de confiance.

    Returns
    -------
    float
        ES positif, toujours ``>= VaR`` au même niveau de confiance.
    """
    r = np.asarray(returns, dtype=float)
    var = historical_var(r, confidence)
    tail = r[r <= -var]
    if tail.size == 0:
        tail = np.array([r.min()])
    return float(-tail.mean())


# ---------------------------------------------------------------------------
# VaR paramétrique gaussienne
# ---------------------------------------------------------------------------


def parametric_var_gaussian(
    returns: pd.Series | np.ndarray,
    confidence: float = 0.99,
    horizon: int = 1,
) -> float:
    """VaR paramétrique sous hypothèse de normalité des rendements.

    .. math::
        VaR_\\alpha(h) = -\\left(\\mu h + z_\\alpha \\sigma \\sqrt{h}\\right), \\quad
        z_\\alpha = \\Phi^{-1}(1-\\alpha)

    Parameters
    ----------
    returns : array-like
        Rendements journaliers.
    confidence : float
        Niveau de confiance.
    horizon : int
        Horizon en jours (utilise la règle racine du temps, cf.
        :func:`sqrt_time_scaling` et sa critique dans
        :func:`time_scaling_diagnostic`).

    Returns
    -------
    float
        VaR positive à l'horizon ``horizon``.
    """
    r = np.asarray(returns, dtype=float)
    mu, sigma = r.mean(), r.std(ddof=1)
    z = stats.norm.ppf(1.0 - confidence)
    return float(-(mu * horizon + z * sigma * np.sqrt(horizon)))


def parametric_es_gaussian(
    returns: pd.Series | np.ndarray,
    confidence: float = 0.99,
    horizon: int = 1,
) -> float:
    """Expected Shortfall paramétrique gaussien (formule fermée).

    .. math::
        ES_\\alpha = -\\mu + \\sigma \\frac{\\phi(z_\\alpha)}{1-\\alpha}

    avec :math:`\\phi` la densité normale standard et
    :math:`z_\\alpha = \\Phi^{-1}(1-\\alpha)`.

    Parameters
    ----------
    returns : array-like
        Rendements journaliers.
    confidence : float
        Niveau de confiance.
    horizon : int
        Horizon en jours (règle racine du temps).

    Returns
    -------
    float
        ES positif.
    """
    r = np.asarray(returns, dtype=float)
    mu, sigma = r.mean(), r.std(ddof=1)
    alpha_tail = 1.0 - confidence
    z = stats.norm.ppf(alpha_tail)
    es_1d = -mu + sigma * stats.norm.pdf(z) / alpha_tail
    # mise à l'échelle horizon : moyenne linéaire, écart-type en racine(h)
    es_h = -mu * horizon + (es_1d + mu) * np.sqrt(horizon)
    return float(es_h)


# ---------------------------------------------------------------------------
# VaR Cornish-Fisher (ajustement skew / kurtosis)
# ---------------------------------------------------------------------------


def cornish_fisher_quantile(z: float, skew: float, excess_kurtosis: float) -> float:
    """Quantile ajusté de Cornish-Fisher à partir du quantile normal.

    .. math::
        z_{CF} = z + \\frac{z^2-1}{6}S + \\frac{z^3-3z}{24}K
                  - \\frac{2z^3-5z}{36}S^2

    où ``S`` est le skewness et ``K`` le kurtosis en excès (kurtosis - 3).

    Parameters
    ----------
    z : float
        Quantile de la loi normale standard.
    skew : float
        Asymétrie (skewness) échantillon.
    excess_kurtosis : float
        Kurtosis en excès (aplatissement au-delà de 3).

    Returns
    -------
    float
        Quantile ajusté ``z_CF``.
    """
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
    """VaR de Cornish-Fisher : correction du quantile normal par skew/kurtosis.

    Étend la VaR paramétrique gaussienne en tenant compte de l'asymétrie et
    de l'aplatissement empiriques de la distribution des rendements, ce qui
    en fait une approximation « semi-paramétrique » plus fidèle aux queues
    épaisses observées sur les marchés (cf. Zangari, 1996).

    Parameters
    ----------
    returns : array-like
        Rendements journaliers.
    confidence : float
        Niveau de confiance.
    horizon : int
        Horizon en jours (règle racine du temps appliquée à sigma).

    Returns
    -------
    float
        VaR positive ajustée Cornish-Fisher.
    """
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
    """Expected Shortfall Cornish-Fisher, par intégration numérique de la queue.

    Il n'existe pas de formule fermée simple pour l'ES sous Cornish-Fisher ;
    on intègre numériquement la moyenne des quantiles ajustés en dessous du
    seuil ``1 - confidence`` :

    .. math::
        ES_\\alpha = -\\left(\\mu + \\sigma \\cdot \\frac{1}{\\alpha}
                     \\int_0^\\alpha z_{CF}(u)\\, du\\right)

    Parameters
    ----------
    returns : array-like
        Rendements journaliers.
    confidence : float
        Niveau de confiance.
    horizon : int
        Horizon en jours.
    n_integration_points : int
        Nombre de points de quadrature sur la queue ``(0, alpha)``.

    Returns
    -------
    float
        ES positif.
    """
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


# ---------------------------------------------------------------------------
# VaR Monte Carlo (Cholesky sur matrice de corrélation / covariance)
# ---------------------------------------------------------------------------


@dataclass
class MonteCarloVaRResult:
    """Résultat d'une simulation Monte Carlo de VaR/ES portefeuille.

    Attributes
    ----------
    var : float
        VaR positive au niveau de confiance demandé.
    es : float
        Expected Shortfall positif.
    simulated_portfolio_returns : np.ndarray
        Rendements simulés du portefeuille à l'horizon demandé.
    """

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
    """VaR/ES Monte Carlo par simulation de scénarios corrélés (Cholesky).

    Étapes :

    1. Estime la moyenne ``mu`` et la matrice de covariance ``Sigma`` des
       rendements journaliers.
    2. Décompose ``Sigma = L L'`` par factorisation de Cholesky.
    3. Simule des chocs iid (normaux ou Student-t standardisés), les
       corrèle via ``L`` : ``shocks = Z L'``.
    4. Calcule le P&L simulé du portefeuille ``w'`` et sa VaR/ES empirique à
       l'horizon ``h`` (mise à l'échelle racine du temps sur les chocs
       centrés, cf. :func:`sqrt_time_scaling`).

    Parameters
    ----------
    returns : pd.DataFrame
        Rendements journaliers multi-actifs (colonnes = actifs).
    weights : np.ndarray
        Poids du portefeuille (somme = 1 en général), taille n_assets.
    confidence : float
        Niveau de confiance.
    horizon : int
        Horizon de projection en jours.
    n_sims : int
        Nombre de scénarios Monte Carlo.
    dist : {"normal", "student_t"}
        Loi des chocs sous-jacents.
    dof : float
        Degrés de liberté si ``dist="student_t"``.
    seed : int, optional
        Graine aléatoire.

    Returns
    -------
    MonteCarloVaRResult
        VaR, ES et rendements simulés du portefeuille.

    Raises
    ------
    ValueError
        Si ``dist`` n'est pas reconnu ou si les dimensions ne correspondent
        pas.
    """
    w = np.asarray(weights, dtype=float)
    if w.shape[0] != returns.shape[1]:
        raise ValueError("weights doit avoir autant d'éléments que de colonnes dans returns.")

    mu = returns.mean().to_numpy()
    cov = returns.cov().to_numpy()
    n_assets = w.shape[0]

    # Régularisation légère pour garantir la factorisation de Cholesky même
    # si la covariance échantillon est quasi-singulière.
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
    # Horizon h : moyenne mise à l'échelle linéairement, écart autour de la
    # moyenne mis à l'échelle en racine(h) (accroissements iid).
    port_returns_h = mu_p * horizon + (port_returns_1d - mu_p) * np.sqrt(horizon)

    var = historical_var(port_returns_h, confidence)
    es = historical_es(port_returns_h, confidence)
    return MonteCarloVaRResult(var=var, es=es, simulated_portfolio_returns=port_returns_h)


# ---------------------------------------------------------------------------
# Horizons multiples : règle racine du temps et sa critique empirique
# ---------------------------------------------------------------------------


def sqrt_time_scaling(var_1day: float, horizon: int) -> float:
    """Applique la règle racine du temps à une VaR 1 jour.

    .. math::
        VaR(h) = VaR(1) \\times \\sqrt{h}

    Règle exacte uniquement si les rendements sont i.i.d. et gaussiens
    (absence d'autocorrélation, de clustering de volatilité et de queues
    épaisses). Voir :func:`time_scaling_diagnostic` pour une comparaison
    empirique à la VaR historique multi-jours réelle.

    Parameters
    ----------
    var_1day : float
        VaR à horizon 1 jour.
    horizon : int
        Horizon cible en jours.

    Returns
    -------
    float
        VaR mise à l'échelle.
    """
    return float(var_1day * np.sqrt(horizon))


def time_scaling_diagnostic(
    returns: pd.Series | np.ndarray,
    confidence: float = 0.99,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
) -> pd.DataFrame:
    """Compare la VaR racine-du-temps à la VaR historique réelle à h jours.

    Pour chaque horizon ``h``, calcule :

    - ``VaR_sqrt_time`` : VaR 1 jour historique multipliée par
      :math:`\\sqrt{h}` (règle théorique) ;
    - ``VaR_historique_h_jours`` : VaR historique calculée directement sur
      les rendements cumulés glissants sur ``h`` jours (fenêtres
      chevauchantes), qui capture l'autocorrélation et le clustering de
      volatilité réels des données.

    Un écart important entre les deux colonnes illustre la limite bien
    documentée de la règle racine du temps (Comité de Bâle, FRTB 2019 ;
    Danielsson & Zigrand, 2006) : elle sous-estime typiquement le risque à
    horizon long en présence de clustering de volatilité et d'autocorrélation
    positive des rendements agrégés.

    Parameters
    ----------
    returns : array-like
        Rendements journaliers historiques.
    confidence : float
        Niveau de confiance.
    horizons : tuple of int
        Horizons à comparer (en jours).

    Returns
    -------
    pd.DataFrame
        Colonnes ``horizon``, ``VaR_sqrt_time``, ``VaR_historique_h_jours``,
        ``ecart_relatif_pct``.
    """
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


# ---------------------------------------------------------------------------
# Table de synthèse multi-méthodes
# ---------------------------------------------------------------------------


def var_es_summary(
    returns: pd.Series | np.ndarray,
    confidence: float = 0.99,
    ewma_decay: float = 0.94,
) -> pd.DataFrame:
    """Calcule VaR et ES par les quatre méthodes univariées sur une série.

    Regroupe :func:`historical_var`/:func:`historical_es`,
    :func:`historical_var_weighted`, :func:`parametric_var_gaussian`/
    :func:`parametric_es_gaussian` et :func:`cornish_fisher_var`/
    :func:`cornish_fisher_es` (la VaR Monte Carlo, intrinsèquement
    multivariée, est calculée séparément via :func:`monte_carlo_var`).

    Parameters
    ----------
    returns : array-like
        Rendements journaliers (portefeuille ou actif unique).
    confidence : float
        Niveau de confiance.
    ewma_decay : float
        Facteur de décroissance pour la VaR historique pondérée BRW.

    Returns
    -------
    pd.DataFrame
        Une ligne par méthode, colonnes ``VaR`` et ``ES``.
    """
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
