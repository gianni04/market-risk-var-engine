"""Fixtures partagées pour la suite de tests marketrisk."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marketrisk.data import generate_synthetic_returns


@pytest.fixture(scope="session")
def synthetic_returns() -> pd.DataFrame:
    """Rendements synthétiques multi-actifs (5 actifs, 1500 jours, seed fixe)."""
    return generate_synthetic_returns(n_days=1500, seed=42)


@pytest.fixture(scope="session")
def portfolio_weights() -> np.ndarray:
    """Poids de portefeuille illustratifs, cohérents avec l'univers 5 actifs."""
    return np.array([0.30, 0.30, 0.20, 0.15, 0.05])


@pytest.fixture(scope="session")
def gaussian_returns() -> np.ndarray:
    """Rendements strictement gaussiens i.i.d. pour les tests de convergence analytique."""
    rng = np.random.default_rng(123)
    return rng.normal(loc=0.0002, scale=0.01, size=200_000)
