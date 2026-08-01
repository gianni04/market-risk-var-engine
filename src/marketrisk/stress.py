"""Stress testing : scénarios historiques, hypothétiques, stress de corrélation, reverse stress test.

Références : Comité de Bâle (2018), « Stress testing principles » ;
FRTB (2019) pour l'exigence de scénarios extrêmes hors modèle interne ;
Studer, G. (1997), « Maximum Loss for Measurement of Market Risk », pour le
reverse stress testing sous ellipsoïde de vraisemblance.

Les chocs des scénarios historiques ci-dessous sont des **approximations
pédagogiques** des ordres de grandeur observés lors des crises citées
(ampleur de la baisse des actions, écartement des spreads de crédit,
mouvements de taux et de l'or comme valeur refuge). Ils ne prétendent pas
reproduire des données de marché exactes et doivent être recalibrés avec des
séries réelles avant tout usage en production.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from marketrisk.covariance import nearest_psd

# ---------------------------------------------------------------------------
# Scénarios historiques rejoués (chocs par classe d'actifs, en dur, documentés)
# ---------------------------------------------------------------------------

# Chocs approximatifs (rendement sur la période de crise) par classe d'actif
# de l'univers par défaut du module marketrisk.data. Ordres de grandeur
# pédagogiques inspirés des mouvements largement documentés dans la presse
# financière et la littérature académique sur ces épisodes ; NE PAS utiliser
# tels quels pour un stress test réglementaire réel sans recalibration sur
# séries de marché vérifiées.
HISTORICAL_SCENARIOS: dict[str, dict[str, float]] = {
    "Crise_financiere_2008": {
        # Pic-à-creux approximatif sept. 2008 - mars 2009 (actions monde),
        # écartement massif des spreads IG, fuite vers les obligations
        # souveraines "core" et vers l'or.
        "Actions_EU": -0.46,
        "Actions_US": -0.42,
        "Obligations_Souv": 0.09,
        "Credit_IG": -0.22,
        "Or": 0.05,
    },
    "COVID_Mars_2020": {
        # Krach éclair de fév-mars 2020 : chute violente et rapide des
        # actions, écartement du crédit, détente des taux souverains
        # "core" puis dislocation temporaire de l'or (liquidité).
        "Actions_EU": -0.35,
        "Actions_US": -0.34,
        "Obligations_Souv": 0.04,
        "Credit_IG": -0.12,
        "Or": -0.03,
    },
    "Choc_taux_2022": {
        # Resserrement monétaire mondial 2022 : forte baisse simultanée des
        # obligations et des actions (corrélation actions-obligations
        # devenue positive), crédit sous pression, or globalement stable.
        "Actions_EU": -0.14,
        "Actions_US": -0.19,
        "Obligations_Souv": -0.16,
        "Credit_IG": -0.14,
        "Or": -0.01,
    },
}


def apply_historical_scenario(
    weights: pd.Series | dict[str, float],
    scenario_name: str,
    scenarios: dict[str, dict[str, float]] | None = None,
) -> float:
    """Applique un scénario historique rejoué au portefeuille et retourne le P&L.

    .. math::
        \\Delta V = \\sum_i w_i \\times \\text{choc}_i

    Parameters
    ----------
    weights : pd.Series or dict
        Poids du portefeuille par classe d'actif (doivent couvrir les clés
        du scénario).
    scenario_name : str
        Nom du scénario dans ``scenarios`` (par défaut
        :data:`HISTORICAL_SCENARIOS`).
    scenarios : dict, optional
        Dictionnaire de scénarios à utiliser ; par défaut
        :data:`HISTORICAL_SCENARIOS`.

    Returns
    -------
    float
        Variation de valeur du portefeuille (négative = perte), en fraction
        de la valeur du portefeuille.

    Raises
    ------
    KeyError
        Si ``scenario_name`` est absent de ``scenarios``.
    """
    scenarios = scenarios if scenarios is not None else HISTORICAL_SCENARIOS
    if scenario_name not in scenarios:
        raise KeyError(f"Scénario '{scenario_name}' inconnu. Disponibles : {list(scenarios)}")
    shocks = scenarios[scenario_name]
    w = dict(weights)
    return float(sum(w.get(asset, 0.0) * shock for asset, shock in shocks.items()))


def run_all_historical_scenarios(
    weights: pd.Series | dict[str, float],
    scenarios: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Rejoue tous les scénarios historiques et retourne un tableau de P&L.

    Parameters
    ----------
    weights : pd.Series or dict
        Poids du portefeuille par classe d'actif.
    scenarios : dict, optional
        Scénarios à utiliser (par défaut :data:`HISTORICAL_SCENARIOS`).

    Returns
    -------
    pd.DataFrame
        Colonnes ``scenario``, ``pnl_pct``, triées de la perte la plus
        sévère à la moins sévère.
    """
    scenarios = scenarios if scenarios is not None else HISTORICAL_SCENARIOS
    rows = [
        {"scenario": name, "pnl_pct": apply_historical_scenario(weights, name, scenarios)}
        for name in scenarios
    ]
    return pd.DataFrame(rows).sort_values("pnl_pct").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Scénarios hypothétiques paramétriques
# ---------------------------------------------------------------------------

# Sensibilités par défaut par classe d'actif : duration (en années, exposée
# au risque de taux), spread duration (exposée au risque de crédit) et
# exposition FX nette (fraction de la position exposée à l'EUR/USD).
DEFAULT_ASSET_SENSITIVITIES: dict[str, dict[str, float]] = {
    "Actions_EU": {"equity_beta": 1.0, "rate_duration": 0.0, "credit_spread_duration": 0.0, "fx_usd_exposure": 0.0},
    "Actions_US": {"equity_beta": 1.0, "rate_duration": 0.0, "credit_spread_duration": 0.0, "fx_usd_exposure": 1.0},
    "Obligations_Souv": {"equity_beta": 0.0, "rate_duration": 7.0, "credit_spread_duration": 0.0, "fx_usd_exposure": 0.0},
    "Credit_IG": {"equity_beta": 0.0, "rate_duration": 5.0, "credit_spread_duration": 5.0, "fx_usd_exposure": 0.0},
    "Or": {"equity_beta": 0.0, "rate_duration": 0.0, "credit_spread_duration": 0.0, "fx_usd_exposure": 0.3},
}


@dataclass
class HypotheticalShocks:
    """Paramètres d'un scénario hypothétique multi-facteurs.

    Attributes
    ----------
    equity_shock : float
        Choc actions relatif (ex. -0.20 pour -20%).
    rate_shock_bp : float
        Choc de taux en points de base (ex. +200 pour +200bp).
    credit_spread_shock_bp : float
        Choc de spread de crédit en points de base (ex. +150 pour +150bp).
    fx_shock : float
        Choc de change EUR/USD relatif (ex. +0.10 pour +10% USD).
    """

    equity_shock: float = 0.0
    rate_shock_bp: float = 0.0
    credit_spread_shock_bp: float = 0.0
    fx_shock: float = 0.0


def hypothetical_scenario(
    weights: pd.Series | dict[str, float],
    shocks: HypotheticalShocks,
    sensitivities: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    """Applique un scénario hypothétique paramétrique via des sensibilités factorielles.

    Pour chaque actif ``i``, le P&L relatif est approximé par une somme de
    chocs factoriels pondérés par ses sensibilités :

    .. math::
        \\Delta V_i = \\beta_i \\cdot s_{equity} - D_i \\cdot \\Delta y
                     - SD_i \\cdot \\Delta spread + fx_i \\cdot s_{fx}

    où :math:`D_i` est la duration de taux, :math:`SD_i` la duration de
    spread (approximation classique duration x variation de taux, au
    premier ordre, cf. Fabozzi, *Bond Markets, Analysis and Strategies*),
    et :math:`\\Delta y, \\Delta spread` sont exprimés en décimal (bp / 10000).

    Parameters
    ----------
    weights : pd.Series or dict
        Poids du portefeuille par actif.
    shocks : HypotheticalShocks
        Amplitude des chocs par facteur.
    sensitivities : dict, optional
        Sensibilités par actif (par défaut
        :data:`DEFAULT_ASSET_SENSITIVITIES`).

    Returns
    -------
    dict of str to float
        P&L relatif par actif et total (clé ``"__total__"``).
    """
    sens = sensitivities if sensitivities is not None else DEFAULT_ASSET_SENSITIVITIES
    w = dict(weights)
    dy = shocks.rate_shock_bp / 10_000.0
    d_spread = shocks.credit_spread_shock_bp / 10_000.0

    pnl_by_asset: dict[str, float] = {}
    total = 0.0
    for asset, weight in w.items():
        s = sens.get(asset, {"equity_beta": 0.0, "rate_duration": 0.0, "credit_spread_duration": 0.0, "fx_usd_exposure": 0.0})
        asset_pnl = (
            s["equity_beta"] * shocks.equity_shock
            - s["rate_duration"] * dy
            - s["credit_spread_duration"] * d_spread
            + s["fx_usd_exposure"] * shocks.fx_shock
        )
        pnl_by_asset[asset] = weight * asset_pnl
        total += weight * asset_pnl

    pnl_by_asset["__total__"] = total
    return pnl_by_asset


# ---------------------------------------------------------------------------
# Stress de corrélation
# ---------------------------------------------------------------------------


def stress_correlation_shock(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
    forced_correlation: float = 1.0,
) -> dict[str, float]:
    """Recalcule le risque du portefeuille avec des corrélations forcées (crise).

    En période de stress systémique, les corrélations entre actifs risqués
    tendent vers 1 (perte de la diversification, cf. littérature sur la
    « corrélation en régime de crise »). Ce scénario reconstruit la matrice
    de covariance en conservant les variances individuelles (volatilités
    inchangées) mais en forçant toutes les corrélations hors-diagonale à
    ``forced_correlation`` :

    .. math::
        \\Sigma^{stress}_{ij} = \\rho^{stress} \\sigma_i \\sigma_j
        \\; (i \\ne j), \\qquad \\Sigma^{stress}_{ii} = \\sigma_i^2

    La matrice résultante est reprojetée sur le cône défini positif via
    :func:`marketrisk.covariance.nearest_psd` si nécessaire.

    Parameters
    ----------
    weights : np.ndarray
        Poids du portefeuille.
    cov_matrix : np.ndarray
        Matrice de covariance de base.
    forced_correlation : float
        Corrélation imposée entre toutes les paires d'actifs (``1.0`` =
        stress maximal).

    Returns
    -------
    dict of str to float
        ``vol_base``, ``vol_stress``, ``increase_pct`` (variation relative
        de la volatilité du portefeuille sous stress de corrélation).
    """
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float)
    std = np.sqrt(np.diag(cov))

    corr_stress = np.full(cov.shape, forced_correlation)
    np.fill_diagonal(corr_stress, 1.0)
    cov_stress = corr_stress * np.outer(std, std)
    cov_stress = nearest_psd(cov_stress)

    vol_base = float(np.sqrt(max(w @ cov @ w, 0.0)))
    vol_stress = float(np.sqrt(max(w @ cov_stress @ w, 0.0)))
    increase_pct = 100.0 * (vol_stress - vol_base) / vol_base if vol_base > 0 else np.nan

    return {"vol_base": vol_base, "vol_stress": vol_stress, "increase_pct": increase_pct}


# ---------------------------------------------------------------------------
# Reverse stress testing
# ---------------------------------------------------------------------------


@dataclass
class ReverseStressResult:
    """Résultat d'un reverse stress test.

    Attributes
    ----------
    shock_vector : np.ndarray
        Vecteur de chocs factoriels le plus probable produisant la perte
        cible.
    mahalanobis_distance : float
        Distance de Mahalanobis du choc trouvé (mesure de « vraisemblance » :
        plus elle est faible, plus le scénario est plausible sous
        l'hypothèse elliptique/gaussienne).
    achieved_loss : float
        Perte de portefeuille effectivement obtenue avec ce choc (doit être
        proche de ``target_loss``).
    converged : bool
        Indicateur de convergence de l'optimisation.
    """

    shock_vector: np.ndarray
    mahalanobis_distance: float
    achieved_loss: float
    converged: bool


def reverse_stress_test(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
    target_loss: float,
) -> ReverseStressResult:
    """Reverse stress test : trouve le choc factoriel le plus plausible pour une perte cible.

    Cherche le vecteur de chocs ``x`` qui minimise la distance de
    Mahalanobis (i.e. maximise la vraisemblance sous une hypothèse
    elliptique/gaussienne de covariance ``Sigma``) sous la contrainte que la
    perte de portefeuille atteigne exactement ``target_loss`` :

    .. math::
        \\min_x \\; x' \\Sigma^{-1} x \\quad \\text{s.c.} \\quad w'x = -L^{target}

    Ce problème d'optimisation quadratique sous contrainte linéaire admet
    une solution analytique par multiplicateur de Lagrange :

    .. math::
        x^* = -L^{target} \\frac{\\Sigma w}{w' \\Sigma w}

    Le résultat est vérifié numériquement par `scipy.optimize.minimize`
    (SLSQP) afin d'illustrer explicitement la démarche d'optimisation sous
    contrainte (cf. Studer, 1997, « Maximum Loss »).

    Parameters
    ----------
    weights : np.ndarray
        Poids du portefeuille.
    cov_matrix : np.ndarray
        Matrice de covariance des facteurs/actifs.
    target_loss : float
        Perte cible (positive = perte, ex. 0.10 pour -10%).

    Returns
    -------
    ReverseStressResult
        Choc le plus plausible, distance de Mahalanobis et perte atteinte.
    """
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float)
    cov_inv = np.linalg.inv(cov)
    n = w.shape[0]

    def mahalanobis_sq(x: np.ndarray) -> float:
        return float(x @ cov_inv @ x)

    constraints = [{"type": "eq", "fun": lambda x: w @ x + target_loss}]
    x0 = -target_loss * (cov @ w) / (w @ cov @ w)

    result = minimize(
        mahalanobis_sq,
        x0=x0,
        method="SLSQP",
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-12},
    )

    x_star = result.x
    achieved_loss = float(-w @ x_star)
    distance = float(np.sqrt(max(mahalanobis_sq(x_star), 0.0)))

    return ReverseStressResult(
        shock_vector=x_star,
        mahalanobis_distance=distance,
        achieved_loss=achieved_loss,
        converged=bool(result.success),
    )
