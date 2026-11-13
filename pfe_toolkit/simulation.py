"""Simulation helpers (use a numpy Generator for reproducibility)."""
import numpy as np
from scipy import stats


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def simulate_mvn(mu: np.ndarray, cov: np.ndarray, size: int, generator=None) -> np.ndarray:
    """N(mu, Sigma) draws via Cholesky: x = mu + z L'."""
    generator = generator or rng()
    L = np.linalg.cholesky(cov)
    z = generator.standard_normal((size, len(mu)))
    return np.asarray(mu) + z @ L.T


def simulate_gbm_terminal(v0, mu, cov, horizon: float, size: int, generator=None) -> np.ndarray:
    """Exact terminal values of correlated GBMs: ln V_T ~ N(ln V0 + (mu - diag(Sigma)/2) T, Sigma T)."""
    mu, cov = np.atleast_1d(mu).astype(float), np.atleast_2d(cov).astype(float)
    m = np.log(v0) + (mu - 0.5 * np.diag(cov)) * horizon
    return np.exp(simulate_mvn(m, cov * horizon, size, generator))


def simulate_gbm_paths(v0, mu, cov, horizon: float, steps: int, size: int, generator=None) -> np.ndarray:
    """Paths of correlated GBMs, shape (size, steps + 1, n_assets)."""
    mu, cov = np.atleast_1d(mu).astype(float), np.atleast_2d(cov).astype(float)
    dt = horizon / steps
    generator = generator or rng()
    incr = simulate_mvn((mu - 0.5 * np.diag(cov)) * dt, cov * dt, size * steps, generator).reshape(size, steps, len(mu))
    logv = np.concatenate([np.zeros((size, 1, len(mu))), np.cumsum(incr, axis=1)], axis=1)
    return np.asarray(v0) * np.exp(logv)


def gbm_moments(v0, mu, cov, horizon: float):
    """E[V_T] = V0 e^{mu T}; Cov[V_i, V_j] = E_i E_j (e^{Sigma_ij T} - 1)."""
    mean = np.asarray(v0, float) * np.exp(np.asarray(mu) * horizon)
    return mean, np.outer(mean, mean) * (np.exp(np.asarray(cov) * horizon) - 1)


def fisher_ci(rho_hat: float, n: int, level: float = 0.95):
    """Confidence interval for a correlation via Fisher z ~ N(artanh rho, 1/(n-3))."""
    z = np.arctanh(rho_hat)
    h = stats.norm.ppf(0.5 + level / 2) / np.sqrt(n - 3)
    return float(np.tanh(z - h)), float(np.tanh(z + h))
