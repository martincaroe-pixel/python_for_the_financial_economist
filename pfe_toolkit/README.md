# pfe_toolkit

Tested helper functions for *Python for the Financial Economist* (CBS 2026), built lecture by lecture
along Meucci's chain: pricing → risk drivers → invariants → forecast → repricing → risk → optimisation.

```python
import sys; sys.path.insert(0, r"<path to this repo>")
from pfe_toolkit import pricing, linalg, portfolio, simulation, invariants
portfolio.min_variance_weights(linalg.cov_from_vol_corr(vols, corr))
```

| Lecture | Module | Functions |
|---|---|---|
| 1 Pricing | `pricing` | `zcb_price`, `zcb_yield`, `interpolate_yield`, `cash_flow_price`, `price_duration_convexity`, `bullet_bond_price`, `vasicek_zcb_price`, `vasicek_yield`, `credit_zcb_price`, `forward_price`, `fx_forward`, `bsm_price`, `bsm_delta`, `implied_vol`, `gbm_q_drift` |
| 1 Linear algebra | `linalg` | `cov_from_vol_corr`, `corr_from_cov`, `equicorrelation`, `is_positive_definite`, `pca` |
| 1 Portfolios | `portfolio` | `portfolio_mean`, `portfolio_std`, `min_variance_weights`, `mean_variance_weights`, `risk_contributions` |
| 1 Simulation | `simulation` | `rng`, `simulate_mvn`, `simulate_gbm_terminal`, `simulate_gbm_paths`, `gbm_moments`, `fisher_ci` |
| 2 Risk drivers + invariants | `invariants` | `project_normal`, `lognormal_moments`, `ar1_cum_std`, `simulate_random_walk`, `invariance_checks`, `garch_variance`, `garch_fit`, `garch_forecast`, `interp_weights`, `lognormal_values`, `pnl_quantile` |

Tests (from the repo root): `python -m pytest pfe_toolkit/tests` (expected values = Exercise Notes).
