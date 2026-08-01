"""marketrisk : moteur de risque de marché (VaR, ES, backtesting, stress tests, ex-ante).

Package pédagogique et opérationnel reproduisant les briques standard d'une
fonction Risque de Marché en société de gestion : mesure de la VaR/ES par
plusieurs méthodes, backtesting réglementaire, stress testing, risque relatif
ex-ante (tracking error, contribution au risque) et estimation de matrices de
covariance robustes.

Tout le package fonctionne hors ligne : les données de marché peuvent être
générées de façon synthétique (mouvement brownien corrélé + GARCH(1,1) +
innovations Student-t) sans dépendance réseau.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = [
    "data",
    "var",
    "volatility",
    "backtesting",
    "stress",
    "exante",
    "covariance",
    "report",
]
