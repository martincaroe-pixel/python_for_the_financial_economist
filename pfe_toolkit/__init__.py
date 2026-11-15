"""pfe_toolkit – tested building blocks for Python for the Financial Economist (Meucci chain).

Modules
-------
pricing    : step 1 – zero-coupon/bullet bonds, duration/convexity, Vasicek, credit, forwards, Black–Scholes
linalg     : covariance matrices, definiteness, Cholesky/PCA helpers
portfolio  : portfolio moments, minimum-variance / mean-variance weights, risk contributions
simulation : multivariate normal and GBM simulation, log-normal moments, Fisher z confidence intervals
invariants : lecture 2 – projection to the horizon, linear vs log moments, AR(1)/square-root rule, invariance checks, GARCH(1,1), exponential-affine repricing
scenarios  : lecture 3 – flexible probabilities (rolling, exponential, crisp, kernel, ENS), weighted moments/quantiles, ECDF + DKW band, VAR(1) fit/forecast/simulation
"""
from . import pricing, linalg, portfolio, simulation, invariants, scenarios  # noqa: F401
__version__ = "0.3.0 (lectures 1–3)"
