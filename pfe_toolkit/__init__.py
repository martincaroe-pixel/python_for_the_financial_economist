"""pfe_toolkit – tested building blocks for Python for the Financial Economist (Meucci chain).

Modules
-------
pricing    : step 1 – zero-coupon/bullet bonds, duration/convexity, Vasicek, credit, forwards, Black–Scholes
linalg     : covariance matrices, definiteness, Cholesky/PCA helpers
portfolio  : portfolio moments, minimum-variance / mean-variance weights, risk contributions
simulation : multivariate normal and GBM simulation, log-normal moments, Fisher z confidence intervals
"""
from . import pricing, linalg, portfolio, simulation  # noqa: F401
__version__ = "0.1.0 (lecture 1)"
