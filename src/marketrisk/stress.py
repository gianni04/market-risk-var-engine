"""Stress testing : scénarios historiques, hypothétiques, stress de corrélation, reverse stress test."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from marketrisk.covariance import nearest_psd


HISTORICAL_SCENARIOS: dict[str, dict[str, float]] = {
    "Crise_financiere_2008": {
        "Actions_EU": -0.46,
        "Actions_US": -0.42,
        "Obligations_Souv": 0.09,
        "Credit_IG": -0.22,
        "Or": 0.05,
    },
    "COVID_Mars_2020": {
        "Actions_EU": -0.35,
        "Actions_US": -0.34,
        "Obligations_Souv": 0.04,
        "Credit_IG": -0.12,
        "Or": -0.03,
    },
    "Choc_taux_2022": {
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
    """Applique un scénario historique rejoué au portefeuille et retourne le P&L."""
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
    """Rejoue tous les scénarios historiques et retourne un tableau de P&L."""
    scenarios = scenarios if scenarios is not None else HISTORICAL_SCENARIOS
    rows = [
        {"scenario": name, "pnl_pct": apply_historical_scenario(weights, name, scenarios)}
        for name in scenarios
    ]
    return pd.DataFrame(rows).sort_values("pnl_pct").reset_index(drop=True)


DEFAULT_ASSET_SENSITIVITIES: dict[str, dict[str, float]] = {
    "Actions_EU": {"equity_beta": 1.0, "rate_duration": 0.0, "credit_spread_duration": 0.0, "fx_usd_exposure": 0.0},
    "Actions_US": {"equity_beta": 1.0, "rate_duration": 0.0, "credit_spread_duration": 0.0, "fx_usd_exposure": 1.0},
    "Obligations_Souv": {"equity_beta": 0.0, "rate_duration": 7.0, "credit_spread_duration": 0.0, "fx_usd_exposure": 0.0},
    "Credit_IG": {"equity_beta": 0.0, "rate_duration": 5.0, "credit_spread_duration": 5.0, "fx_usd_exposure": 0.0},
    "Or": {"equity_beta": 0.0, "rate_duration": 0.0, "credit_spread_duration": 0.0, "fx_usd_exposure": 0.3},
}


@dataclass
class HypotheticalShocks:
    """Paramètres d'un scénario hypothétique multi-facteurs."""

    equity_shock: float = 0.0
    rate_shock_bp: float = 0.0
    credit_spread_shock_bp: float = 0.0
    fx_shock: float = 0.0


def hypothetical_scenario(
    weights: pd.Series | dict[str, float],
    shocks: HypotheticalShocks,
    sensitivities: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    """Applique un scénario hypothétique paramétrique via des sensibilités factorielles."""
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


def stress_correlation_shock(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
    forced_correlation: float = 1.0,
) -> dict[str, float]:
    """Recalcule le risque du portefeuille avec des corrélations forcées (crise)."""
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


@dataclass
class ReverseStressResult:
    """Résultat d'un reverse stress test."""

    shock_vector: np.ndarray
    mahalanobis_distance: float
    achieved_loss: float
    converged: bool


def reverse_stress_test(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
    target_loss: float,
) -> ReverseStressResult:
    """Reverse stress test : trouve le choc factoriel le plus plausible pour une perte cible."""
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
