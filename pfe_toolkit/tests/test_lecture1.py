"""Run from the repo root: python -m pytest pfe_toolkit/tests  (numbers = Exercise Notes 1)."""
import numpy as np
import pytest
from pfe_toolkit import pricing as pr, linalg as la, portfolio as pf, simulation as sm

C = np.array([[1, .2, .15], [.2, 1, .4], [.15, .4, 1]]); V = np.array([.1, .15, .2]); MU = np.array([.04, .06, .08])


def test_mvp_exercise():
    cov = la.cov_from_vol_corr(V, C)
    assert la.is_positive_definite(cov)
    w = np.ones(3) / 3
    assert pf.portfolio_mean(w, MU) == pytest.approx(0.06)
    assert pf.portfolio_std(w, cov) == pytest.approx(0.10980, abs=1e-5)
    wm = pf.min_variance_weights(cov)
    assert wm == pytest.approx([0.703846, 0.215385, 0.080769], abs=1e-6)
    assert pf.portfolio_std(wm, cov) == pytest.approx(0.089033, abs=1e-6)


def test_mean_variance_2021():
    mu, v = np.array([.02, .04, .08]), np.array([.075, .15, .3])
    cov0 = la.cov_from_vol_corr(v, np.eye(3))
    assert pf.mean_variance_weights(mu, cov0, 2.0, budget=False) == pytest.approx([1.777778, .888889, .444444], abs=1e-6)
    cov = la.cov_from_vol_corr(v, np.array([[1, .2, .1], [.2, 1, .4], [.1, .4, 1]]))
    w5 = pf.mean_variance_weights(mu, cov, 5.0)
    assert w5.sum() == pytest.approx(1.0)
    assert w5 == pytest.approx([0.664149, 0.21152, 0.124331], abs=1e-5)
    assert pf.risk_contributions(w5, cov) == pytest.approx([0.436782, 0.262993, 0.300226], abs=1e-5)


def test_pricing():
    assert pr.zcb_price(0.03, 10) == pytest.approx(0.740818, abs=1e-6)
    assert pr.cash_flow_price([6, 6, 106], [1, 2, 3], 0.03) == pytest.approx(108.35, abs=0.01)
    p, d, k = pr.price_duration_convexity([6, 6, 106], [1, 2, 3], 0.03)
    assert (d, k) == pytest.approx((2.8404, 8.3093), abs=1e-4)
    assert pr.bullet_bond_price(10, 6.0, lambda t: 0.04 * np.ones_like(t)) == pytest.approx(115.5016, abs=1e-3)
    assert pr.vasicek_yield(np.array([1, 30.]), 0.03, .36, .09, .0265) == pytest.approx([0.03952, 0.08211], abs=1e-5)
    assert pr.credit_zcb_price(0.03, 5, 0.02) == pytest.approx(0.778801, abs=1e-6)
    assert pr.forward_price(100, np.exp(-0.03)) == pytest.approx(103.0455, abs=1e-4)
    c = pr.bsm_price(1, 1, .02, .18, 1)
    assert c == pytest.approx(0.081340, abs=1e-6)
    assert pr.bsm_price(1, 1.1, .02, .16, 1) == pytest.approx(0.034335, abs=1e-6)
    assert pr.implied_vol(c, 1, 1, .02, 1) == pytest.approx(0.18, abs=1e-8)
    put = pr.bsm_price(1, 1, .02, .18, 1, call=False)
    assert c - put == pytest.approx(1 - np.exp(-.02), abs=1e-10)          # put-call parity
    assert pr.fx_forward(1.10, np.exp(-.03), np.exp(-.05)) == pytest.approx(1.10 * np.exp(.02))


def test_simulation():
    g = sm.rng(0)
    mu, sig = np.array([.03, .06]), np.array([[.075**2, .2*.075*.15], [.2*.075*.15, .15**2]])
    mean, cov = sm.gbm_moments(np.ones(2), mu, sig, 5.0)
    assert mean == pytest.approx([1.161834, 1.349859], abs=1e-6)
    v = sm.simulate_gbm_terminal(np.ones(2), mu, sig, 5.0, 400_000, g)
    assert v.mean(0) == pytest.approx(mean, rel=3e-3)
    assert np.cov(v, rowvar=False) == pytest.approx(cov, rel=3e-2)
    paths = sm.simulate_gbm_paths(np.ones(2), mu, sig, 1.0, 52, 20_000, g)
    assert paths.shape == (20_000, 53, 2) and paths[:, -1].mean(0) == pytest.approx(np.exp(mu), rel=5e-3)
    lo, hi = sm.fisher_ci(0.5, 25)
    assert lo < 0.5 < hi and (lo, hi) == pytest.approx((0.1307, 0.7475), abs=1e-3)
