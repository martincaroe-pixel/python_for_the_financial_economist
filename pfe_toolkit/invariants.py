"""Steps 2–5 of the Meucci chain (lecture 2): risk drivers, invariants, projection to the horizon, repricing.

All functions are pure numpy/scipy/pandas; numbers in the docstrings are from Exercise Notes 2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize


# ---------------------------------------------------------------- projection
def project_normal(mu, cov, horizon: float, dt: float):
    """i.i.d. normal invariants: mean and covariance of the sum over horizon/dt steps (square-root rule)."""
    n = horizon / dt
    return n * np.asarray(mu, float), n * np.asarray(cov, float)


def lognormal_moments(mu, cov):
    """Mean and covariance of R = exp(r) - 1 when r ~ N(mu, cov) (linear returns from log returns).

    Example: r ~ N(0.06, 0.15^2) -> E[R] = 0.0738, Sd[R] = 0.1620.
    """
    mu = np.atleast_1d(np.asarray(mu, float)); cov = np.atleast_2d(np.asarray(cov, float))
    m = np.exp(mu + 0.5 * np.diag(cov))
    return m - 1.0, np.outer(m, m) * (np.exp(cov) - 1.0)


def ar1_cum_std(sigma: float, alpha: float, n_steps: int) -> float:
    """Sd of the n-step log return when the deviation from trend is AR(1) with coefficient (1 - alpha).

    Problem 3: sigma = 0.05, alpha = 0.01, 120 months -> 0.338 (square-root rule would give 0.548).
    """
    k = np.arange(n_steps)
    return float(sigma * np.sqrt(np.sum((1.0 - alpha) ** (2 * k))))


def simulate_random_walk(x0, mu, cov, n_steps: int, n_sim: int, seed=None):
    """Paths of X_{t+1} = X_t + eps, eps ~ N(mu, cov). Returns array (n_sim, n_steps + 1, d)."""
    rng = np.random.default_rng(seed)
    mu = np.atleast_1d(mu); cov = np.atleast_2d(cov)
    eps = rng.multivariate_normal(mu, cov, size=(n_sim, n_steps))
    paths = np.concatenate([np.zeros((n_sim, 1, len(mu))), np.cumsum(eps, axis=1)], axis=1)
    return paths + np.asarray(x0, float)


# ---------------------------------------------------------------- invariance checks
def invariance_checks(series: pd.Series, lags: int = 20) -> dict:
    """Informal + formal checks on a candidate invariant (already differenced).

    Returns KS p-value (first vs second half), lag-1 autocorrelation, ACF up to `lags`,
    and the lag-1 autocorrelation of squared values (volatility clustering).
    """
    s = pd.Series(series).dropna()
    half = len(s) // 2
    return dict(ks_pvalue=float(stats.ks_2samp(s.iloc[:half], s.iloc[half:]).pvalue),
                rho1=float(s.autocorr(1)),
                acf=[float(s.autocorr(k)) for k in range(1, lags + 1)],
                rho1_squared=float((s ** 2).autocorr(1)),
                acf_band=1.96 / np.sqrt(len(s)))


# ---------------------------------------------------------------- GARCH(1,1)
def garch_variance(params, r):
    """h_t = omega + alpha*(r_{t-1}-mu)^2 + beta*h_{t-1}, h_0 = sample variance. params = (mu, omega, alpha, beta)."""
    mu, omega, alpha, beta = params
    r = np.asarray(r, float); eps = r - mu
    h = np.empty_like(r); h[0] = r.var()
    for t in range(1, len(r)):
        h[t] = omega + alpha * eps[t - 1] ** 2 + beta * h[t - 1]
    return h


def garch_fit(r, x0=None):
    """Gaussian ML estimate of GARCH(1,1). Scale returns to percent for a well-behaved optimisation."""
    r = np.asarray(r, float)

    def nll(p):
        h = garch_variance(p, r)
        return 0.5 * np.sum(np.log(h) + (r - p[0]) ** 2 / h)
    x0 = x0 if x0 is not None else [r.mean(), 0.05 * r.var(), 0.05, 0.9]
    res = minimize(nll, x0=x0, method="L-BFGS-B", bounds=[(None, None), (1e-8, None), (1e-6, 1), (1e-6, 1)])
    return res.x


def garch_forecast(params, h_next: float, n_ahead: int):
    """h_{t+k|t} = hbar + (alpha+beta)^(k-1) (h_{t+1} - hbar), k = 1..n_ahead."""
    _, omega, alpha, beta = params
    hbar = omega / (1.0 - alpha - beta)
    k = np.arange(1, n_ahead + 1)
    return hbar + (alpha + beta) ** (k - 1) * (h_next - hbar)


# ---------------------------------------------------------------- repricing (exponential-affine prices)
def interp_weights(tau: float, tenors) -> np.ndarray:
    """Linear-interpolation weights of tau on the key-rate grid `tenors`."""
    tenors = np.asarray(tenors, float); w = np.zeros(len(tenors))
    j = int(np.clip(np.searchsorted(tenors, tau), 1, len(tenors) - 1))
    t1, t2 = tenors[j - 1], tenors[j]
    w[j - 1], w[j] = (t2 - tau) / (t2 - t1), (tau - t1) / (t2 - t1)
    return w


def lognormal_values(b, mean_x, cov_x):
    """V = exp(b X) with X ~ N(mean_x, cov_x): mean vector and covariance of V (shifted log-normal PnL = V - v0)."""
    b = np.atleast_2d(b)
    m, S = b @ mean_x, b @ cov_x @ b.T
    ev = np.exp(m + 0.5 * np.diag(S))
    return ev, np.outer(ev, ev) * (np.exp(S) - 1.0), m, S


def pnl_quantile(b, mean_x, cov_x, v0, q: float = 0.05):
    """q-quantile of the PnL of each asset (exact, since each log-value is normal)."""
    _, _, m, S = lognormal_values(b, mean_x, cov_x)
    return np.exp(m + stats.norm.ppf(q) * np.sqrt(np.diag(S))) - np.asarray(v0, float)
