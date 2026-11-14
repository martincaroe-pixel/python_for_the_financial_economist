"""Run from the repo root: python -m pytest pfe_toolkit/tests  (numbers = Exercise Notes 2)."""
import os
import numpy as np
import pandas as pd
import pytest
from pfe_toolkit import invariants as inv

HERE = os.path.dirname(__file__)
WEEK2 = os.path.join(HERE, '..', '..', 'lectures_and_exercises', '2026', 'week_2')


def test_projection_and_lognormal_moments():
    m, v = inv.project_normal(0.06 / 12, 0.15 ** 2 / 12, 1.0, 1 / 12)
    assert m == pytest.approx(0.06) and v == pytest.approx(0.0225)
    mL, cL = inv.lognormal_moments(m, v)
    assert mL[0] == pytest.approx(0.07385, abs=1e-5)
    assert np.sqrt(cL[0, 0]) == pytest.approx(0.16199, abs=1e-5)


def test_lognormal_moments_by_simulation():
    mu = np.array([0.006, 0.008]) * 60; cov = np.array([[0.0020, 0.0012], [0.0012, 0.0030]]) * 60
    mL, cL = inv.lognormal_moments(mu, cov)
    sim = np.exp(np.random.default_rng(1).multivariate_normal(mu, cov, size=400_000)) - 1
    assert sim.mean(axis=0) == pytest.approx(mL, abs=3e-3)
    assert np.cov(sim.T) == pytest.approx(cL, abs=3e-3)


def test_square_root_rule_vs_mean_reversion():
    assert inv.ar1_cum_std(0.05, 0.0, 120) == pytest.approx(np.sqrt(120) * 0.05)
    assert inv.ar1_cum_std(0.05, 0.01, 120) == pytest.approx(0.33818, abs=1e-5)


def test_invariance_checks_random_walk():
    rng = np.random.default_rng(4)
    s = pd.Series(rng.normal(0, 0.01, 1500))
    out = inv.invariance_checks(s)
    assert out['ks_pvalue'] > 0.05 and abs(out['rho1']) < out['acf_band']


def test_garch_recovers_parameters():
    rng = np.random.default_rng(7)
    mu, omega, alpha, beta, n = 0.04, 0.02, 0.09, 0.89, 4000
    r, h = np.empty(n), omega / (1 - alpha - beta)
    for t in range(n):
        r[t] = mu + np.sqrt(h) * rng.standard_normal(); h = omega + alpha * (r[t] - mu) ** 2 + beta * h
    p = inv.garch_fit(r)
    assert p[2] + p[3] < 1 and 0.85 < p[2] + p[3] < 1.0
    f = inv.garch_forecast(p, 1.0, 500)
    assert f[-1] == pytest.approx(p[1] / (1 - p[2] - p[3]), rel=0.05)


@pytest.mark.skipif(not os.path.exists(os.path.join(WEEK2, 'init_values.xlsx')), reason='week 2 data files not found')
def test_pnl_example_exact_distribution():
    cov = pd.read_excel(os.path.join(WEEK2, 'covariance_matrix.xlsx'), index_col=0).values
    x0 = pd.read_excel(os.path.join(WEEK2, 'init_values.xlsx'), index_col=0).values.flatten()
    mu = np.r_[0.0, 0.07 / 52, 0.06 / 52, np.zeros(12)]
    ten = [1 / 12, 1, 3, 5, 7, 10]

    def load(tau):
        a = np.zeros((4, 15)); a[0, 1], a[0, 0] = 1, -1; a[1, 2] = 1
        a[2, 9:15], a[2, 0] = -tau * inv.interp_weights(tau, ten), -1; a[3, 3:9] = -tau * inv.interp_weights(tau, ten)
        return a
    v0 = np.exp(load(5) @ x0)
    ev, cv, _, _ = inv.lognormal_values(load(4), x0 + 52 * mu, 52 * cov)
    assert v0 == pytest.approx([0.9462, 1.0, 0.7848, 0.9046], abs=1e-4)
    assert ev - v0 == pytest.approx([0.0809, 0.0748, 0.0309, 0.0192], abs=1e-4)
    assert np.sqrt(np.diag(cv)) == pytest.approx([0.1605, 0.1682, 0.0606, 0.0220], abs=1e-4)
    assert inv.pnl_quantile(load(4), x0 + 52 * mu, 52 * cov, v0) == pytest.approx([-0.1602, -0.1779, -0.0648, -0.0166], abs=1e-4)
