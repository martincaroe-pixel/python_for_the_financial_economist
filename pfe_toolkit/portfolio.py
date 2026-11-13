"""Portfolio moments and closed-form optimal portfolios."""
import numpy as np


def portfolio_mean(w, mu) -> float:
    return float(np.asarray(w) @ np.asarray(mu))


def portfolio_std(w, cov) -> float:
    w = np.asarray(w, float)
    return float(np.sqrt(w @ cov @ w))


def min_variance_weights(cov: np.ndarray) -> np.ndarray:
    """w = Sigma^-1 1 / (1' Sigma^-1 1) (budget constraint, short sales allowed)."""
    ones = np.ones(cov.shape[0])
    x = np.linalg.solve(cov, ones)
    return x / (ones @ x)


def mean_variance_weights(mu: np.ndarray, cov: np.ndarray, risk_aversion: float, budget: bool = True) -> np.ndarray:
    """Maximise w'mu - lambda/2 w'Sigma w. budget=False: w = Sigma^-1 mu / lambda.
    budget=True (1'w = 1): w = Sigma^-1 mu/lambda - Sigma^-1 1 (1'Sigma^-1 mu - lambda) / (lambda 1'Sigma^-1 1)."""
    inv_mu = np.linalg.solve(cov, mu)
    if not budget:
        return inv_mu / risk_aversion
    ones = np.ones(len(mu))
    inv_one = np.linalg.solve(cov, ones)
    return inv_mu / risk_aversion - inv_one * (ones @ inv_mu - risk_aversion) / (risk_aversion * (ones @ inv_one))


def risk_contributions(w: np.ndarray, cov: np.ndarray, relative: bool = True) -> np.ndarray:
    """Euler risk contributions RC_i = w_i (Sigma w)_i / sigma_p; relative: divided by sigma_p (sum to 1)."""
    w = np.asarray(w, float)
    var = w @ cov @ w
    rc = w * (cov @ w) / np.sqrt(var)
    return rc / np.sqrt(var) if relative else rc
