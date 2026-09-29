# Market Risk VaR Engine

Python package for the usual market risk reports of an asset manager: VaR and
Expected Shortfall with five methods, VaR backtesting, stress tests and
ex-ante tracking error.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Tests](https://img.shields.io/badge/tests-65%20passed-brightgreen)

**Data:** the examples run on synthetic returns (multi-asset GARCH(1,1) with a
Student-t copula), so the numbers below show how the methods behave, not the
risk of a real fund.

## What it covers

| Area | Methods |
|---|---|
| VaR / ES | historical, age-weighted historical (BRW), parametric normal, Cornish-Fisher, Monte Carlo (Student-t, Cholesky) |
| Backtesting | Kupiec POF, Christoffersen independence and conditional coverage, Basel traffic light |
| Stress tests | historical replay, hypothetical shocks, correlation stress, reverse stress test (Mahalanobis distance) |
| Ex-ante risk | tracking error, beta and Blume-adjusted beta, information ratio, Euler VaR contributions, Ledoit-Wolf covariance |

## Run

```bash
pip install -e ".[dev]"
python examples/01_var_comparison.py
python examples/02_var_backtest.py
python examples/03_stress_tests.py
python examples/04_ex_ante_risk.py
pytest tests -q
```

## Results

Portfolio: 30% EU equities, 30% US equities, 20% government bonds, 15% IG
credit, 5% gold. 1,500 days.

**VaR and ES, one day, 99%**

| Method | VaR | ES |
|---|---|---|
| Historical | 1.96% | 2.38% |
| Age-weighted historical (BRW) | 1.28% | 2.38% |
| Parametric normal | 1.81% | 2.08% |
| Cornish-Fisher | 2.15% | 3.09% |
| Monte Carlo Student-t | 2.00% | 2.52% |

Cornish-Fisher gives the highest numbers because it adjusts for skewness and
fat tails; the normal model gives the lowest ES.

![VaR methods](docs/img/01_var_methodes_comparaison.png)

**Backtest over 250 days** (500-day rolling window): 4 exceptions (1.6% vs
1% expected). Kupiec p = 0.38 and Christoffersen p = 0.72, so the model is not
rejected. Basel traffic light: green zone.

**Stress tests:** 2008 replay -27.7%, COVID March 2020 -21.9%, 2022 rate shock
-15.3%. Setting all correlations to 1 raises portfolio volatility from 12.5% to
16.0%. The most likely shock that causes a 15% loss is mainly on equities
(EU -20.7%, US -26.3%).

**Ex-ante tracking error:** forecast 2.13% vs 1.68% realised over the next 250
days. Over 48 rolling windows, realised TE is on average 1.17x the forecast.
Equities make up 94% of the one-day VaR (Euler decomposition).

## Limitations

- Synthetic data; the historical scenarios are approximations of real crises.
- Linear positions only (no option gamma or vega).
- The parametric VaR and the Euler decomposition assume normal returns.

## References

Jorion, *Value at Risk* (2007); Kupiec (1995); Christoffersen (1998); Basel
Committee, *Supervisory framework for backtesting* (1996); Ledoit and Wolf
(2004); Boudoukh, Richardson and Whitelaw (1998).
