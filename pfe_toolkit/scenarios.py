"""Lecture 3 – multivariate forecasting (VAR(1)) and fully flexible probabilities.

Scenario probabilities p_t (sum to one) replace equal weights 1/T in every statistic.
All functions are pure numpy/scipy; numbers in the docstrings are from Exercise Notes 3.
Convention for VAR(1): rows = time, X_{t+1} = a + B X_t + eps  (B = b.T of the course notebook,
where the notebook writes x_{t+1} = a + x_t @ b).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


# ---------------------------------------------------------------- scenario probabilities
def _normalise(w):
    w = np.asarray(w, float)
    return w / w.sum()


def equal_probs(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def rolling_probs(n: int, window: int, last_index: int | None = None) -> np.ndarray:
    """p_t ∝ 1 for the `window` observations ending at last_index (default: the last one), else 0."""
    end = n if last_index is None else min(n, last_index + 1)
    w = np.zeros(n)
    w[max(0, end - window):end] = 1.0
    return _normalise(w)


def exponential_probs(time_points, target_time_point=None, half_life: float | None = None,
                      decay: float | None = None) -> np.ndarray:
    """p_t ∝ exp(-λ (T - t)), λ = ln 2 / half_life (or give decay = λ directly).

    Observations after the target time get the weight of the target (as in the course function).
    Example: half_life = 100 on t = 0..1000 -> p_1000 = 0.006914.
    """
    t = np.asarray(time_points, float)
    T = t.max() if target_time_point is None else target_time_point
    lam = decay if decay is not None else np.log(2) / half_life
    return _normalise(np.exp(-lam * np.clip(T - t, 0, np.inf)))


def half_life_from_decay(decay: float) -> float:
    """κ = ln 2 / λ.  λ = 0.0055 -> 126 days."""
    return np.log(2) / decay


def crisp_probs(state, lower=-np.inf, upper=np.inf) -> np.ndarray:
    """p_t ∝ 1 if lower <= y_t <= upper else 0 (e.g. VIX >= 20)."""
    y = np.asarray(state, float)
    return _normalise((y >= lower) & (y <= upper))


def kernel_probs(state, target: float, bandwidth: float) -> np.ndarray:
    """Gaussian kernel: p_t ∝ exp(-(y_t - target)^2 / (2 s^2))."""
    y = np.asarray(state, float)
    return _normalise(np.exp(-0.5 * ((y - target) / bandwidth) ** 2))


def effective_number_of_scenarios(probs) -> float:
    """Meucci (2012): ENS = exp(-Σ p ln p). Equal weights -> T; one scenario -> 1."""
    p = np.asarray(probs, float)
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


# ---------------------------------------------------------------- moments with probabilities
def mean(x, probs=None, axis=0):
    return np.average(x, weights=probs, axis=axis)


def var(x, probs=None, axis=0):
    """Σ p (x - E[x])^2 (axis-correct, unlike the course notebook's second np.average)."""
    x = np.asarray(x, float)
    m = np.expand_dims(mean(x, probs, axis), axis)
    return np.average((x - m) ** 2, weights=probs, axis=axis)


def std(x, probs=None, axis=0):
    """Example (Problem 3): x scenarios -> 2.9721."""
    return np.sqrt(var(x, probs, axis))


def cov_mat(x, probs=None):
    """Probability-weighted covariance matrix of the columns of x (= np.cov(x, aweights=p, rowvar=False, ddof=0))."""
    x = np.asarray(x, float)
    p = equal_probs(len(x)) if probs is None else _normalise(probs)
    d = x - p @ x
    return (d * p[:, None]).T @ d


def corr_mat(x, probs=None):
    c = cov_mat(x, probs)
    s = np.sqrt(np.diag(c))
    return c / np.outer(s, s)


def skew(x, probs=None, axis=0):
    """E[(X - μ)^3] / σ^3.  Example: -0.1186."""
    x = np.asarray(x, float)
    m = np.expand_dims(mean(x, probs, axis), axis)
    return np.average((x - m) ** 3, weights=probs, axis=axis) / std(x, probs, axis) ** 3


def kurt(x, probs=None, excess=True, axis=0):
    """E[(X - μ)^4] / σ^4 (- 3 if excess).  Example: -1.4798."""
    x = np.asarray(x, float)
    m = np.expand_dims(mean(x, probs, axis), axis)
    k = np.average((x - m) ** 4, weights=probs, axis=axis) / std(x, probs, axis) ** 4
    return k - 3.0 if excess else k


def weighted_quantile(x, q, probs=None) -> float:
    """Smallest x with cumulative probability >= q (1-D)."""
    x = np.asarray(x, float)
    p = equal_probs(len(x)) if probs is None else _normalise(probs)
    idx = np.argsort(x)
    cum = np.cumsum(p[idx])
    return float(x[idx][np.searchsorted(cum, q)])


# ---------------------------------------------------------------- empirical distribution
def ecdf(x_eval, data):
    """F_n(x) = (1/n) Σ 1{X_i <= x}."""
    d = np.sort(np.asarray(data, float))
    return np.searchsorted(d, x_eval, side='right') / len(d)


def dkw_epsilon(n: int, alpha: float = 0.05) -> float:
    """Dvoretzky–Kiefer–Wolfowitz band half-width sqrt(ln(2/α)/(2n)).  n = 25 -> 0.2716."""
    return float(np.sqrt(np.log(2.0 / alpha) / (2 * n)))


# ---------------------------------------------------------------- Student t
def mvt_cov(shape, nu: float):
    """Cov[X] = ν/(ν-2) Σ for X ~ t(μ, Σ, ν), ν > 2.  Σ = I, ν = 10 -> 1.25 I."""
    if nu <= 2:
        raise ValueError('covariance only exists for nu > 2')
    return nu / (nu - 2) * np.asarray(shape, float)


# ---------------------------------------------------------------- VAR(1)
def var1_fit(x):
    """OLS of X_{t+1} = a + B X_t + eps on a (T x K) array. Returns a (K,), B (K x K), residuals, Σ_eps (ddof=1 as np.cov)."""
    x = np.asarray(x, float)
    Z = np.hstack([np.ones((len(x) - 1, 1)), x[:-1]])
    beta = np.linalg.lstsq(Z, x[1:], rcond=None)[0]
    a, B = beta[0], beta[1:].T
    eps = x[1:] - Z @ beta
    return a, B, eps, np.cov(eps, rowvar=False)


def var1_mean(a, B, x0, h: int):
    """E[X_{t+h}] = (I - B^h)(I - B)^{-1} a + B^h x_t   (h steps AHEAD; h = 0 returns x_t)."""
    K = len(a)
    Bh = np.linalg.matrix_power(B, h)
    return (np.eye(K) - Bh) @ np.linalg.solve(np.eye(K) - B, a) + Bh @ np.asarray(x0, float)


def var1_cov(B, sigma_eps, h: int):
    """Cov[X_{t+h}] = Σ_{i=0}^{h-1} B^i Σ_eps (B^i)'."""
    out = np.zeros_like(np.asarray(sigma_eps, float))
    for i in range(h):
        Bi = np.linalg.matrix_power(B, i)
        out += Bi @ sigma_eps @ Bi.T
    return out


def var1_unconditional(a, B, sigma_eps):
    """μ = (I - B)^{-1} a,  vec Σ = (I - B⊗B)^{-1} vec Σ_eps (needs all |eig(B)| < 1)."""
    K = len(a)
    mu = np.linalg.solve(np.eye(K) - B, a)
    S = np.linalg.solve(np.eye(K * K) - np.kron(B, B), np.asarray(sigma_eps, float).reshape(-1)).reshape(K, K)
    return mu, S


def var1_simulate(a, B, x0, h: int, n_sim: int, sigma_eps=None, eps_hist=None, probs=None, rng=None):
    """Paths (n_sim, h+1, K). Normal errors (sigma_eps) or bootstrap of historical residuals (eps_hist) drawn with probs."""
    rng = np.random.default_rng(rng)
    K = len(a)
    out = np.empty((n_sim, h + 1, K))
    out[:, 0] = x0
    for t in range(1, h + 1):
        if eps_hist is None:
            e = rng.multivariate_normal(np.zeros(K), sigma_eps, size=n_sim)
        else:
            e = np.asarray(eps_hist)[rng.choice(len(eps_hist), size=n_sim, replace=True, p=probs)]
        out[:, t] = a + out[:, t - 1] @ B.T + e
    return out
