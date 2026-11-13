"""Covariance-matrix helpers."""
import numpy as np


def cov_from_vol_corr(vols: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """Sigma = diag(v) C diag(v) = v v' ⊙ C."""
    vols = np.asarray(vols, float)
    return np.outer(vols, vols) * np.asarray(corr, float)


def corr_from_cov(cov: np.ndarray) -> np.ndarray:
    sd = np.sqrt(np.diag(cov))
    return cov / np.outer(sd, sd)


def equicorrelation(n: int, rho: float) -> np.ndarray:
    c = np.full((n, n), rho, dtype=float)
    np.fill_diagonal(c, 1.0)
    return c


def is_positive_definite(mat: np.ndarray, tol: float = 0.0) -> bool:
    """Symmetric and all eigenvalues > tol."""
    mat = np.asarray(mat, float)
    return bool(np.allclose(mat, mat.T) and np.all(np.linalg.eigvalsh(mat) > tol))


def pca(cov: np.ndarray):
    """Eigenvalues (descending), eigenvectors (columns) and explained variance share."""
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    return vals, vecs, vals / vals.sum()
