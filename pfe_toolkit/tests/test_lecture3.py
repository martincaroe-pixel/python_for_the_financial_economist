"""Run from the repo root: python -m pytest pfe_toolkit/tests  (numbers = Exercise Notes 3)."""
import numpy as np
import pytest
from scipy import stats
from pfe_toolkit import scenarios as sc

X = np.array([0.38874223, 9.64297756, 1.27555492, 3.12537902, 2.4580506, 0.8053753, 6.13998831, 7.55690088, 3.97384118, 7.86858951])
Y = np.array([10.21360159, 13.08162811, 9.33243406, 1.90613543, 17.15583533, 17.87137586, 1.84765146, 1.8249457, 7.36965481, 0.20770526])
P = np.array([0.04650611, 0.06914496, 0.21835384, 0.09998765, 0.02251887, 0.00831124, 0.14610489, 0.10819035, 0.07328199, 0.20760009])


def test_moments_with_scenario_probabilities():
    assert sc.mean(X, P) == pytest.approx(4.977307, abs=1e-6)
    assert sc.var(X, P) == pytest.approx(8.833624, abs=1e-6)
    assert sc.std(X, P) == pytest.approx(2.972141, abs=1e-6)
    c = sc.cov_mat(np.column_stack([X, Y]), P)
    assert c[0, 1] == pytest.approx(-7.042614, abs=1e-5)
    assert sc.corr_mat(np.column_stack([X, Y]), P)[0, 1] == pytest.approx(-0.491689, abs=1e-6)
    assert sc.skew(X, P) == pytest.approx(-0.118631, abs=1e-6)
    assert sc.kurt(X, P) == pytest.approx(-1.479834, abs=1e-6)


def test_weighted_equals_numpy_aweights_and_axis():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(500, 3)); p = rng.random(500)
    assert np.allclose(sc.cov_mat(data, p), np.cov(data, aweights=p, rowvar=False, ddof=0))
    assert np.allclose(sc.std(data, p, axis=0), np.sqrt(np.diag(sc.cov_mat(data, p))))
    assert np.allclose(sc.std(data), data.std(axis=0))


def test_probability_schemes():
    t = np.linspace(0, 1000, 1001)
    p = sc.exponential_probs(t, 1000, half_life=100)
    assert p.sum() == pytest.approx(1) and p[-1] == pytest.approx(0.0069142, abs=1e-7)
    assert p[-101] / p[-1] == pytest.approx(0.5)
    assert sc.half_life_from_decay(0.0055) == pytest.approx(126.03, abs=0.01)
    r = sc.rolling_probs(1000, 750)
    assert r[:250].sum() == 0 and r[-1] == pytest.approx(1 / 750)
    vix = np.array([12, 25, 31, 18, 40.0])
    assert np.allclose(sc.crisp_probs(vix, 20), [0, 1 / 3, 1 / 3, 0, 1 / 3])
    k = sc.kernel_probs(vix, 35, 5)
    assert k.argmax() == 2 and k.sum() == pytest.approx(1)
    assert sc.effective_number_of_scenarios(sc.equal_probs(50)) == pytest.approx(50)
    assert sc.effective_number_of_scenarios(P) == pytest.approx(7.8042, abs=1e-4)


def test_edf_dkw_and_t():
    assert sc.dkw_epsilon(25) == pytest.approx(0.27162, abs=1e-5)
    assert sc.dkw_epsilon(5000) == pytest.approx(0.019206, abs=1e-6)
    assert np.allclose(sc.ecdf([0.5, 2.0], [1, 2, 3, 4]), [0, 0.5])
    assert np.allclose(sc.mvt_cov(np.eye(2), 10), 1.25 * np.eye(2))
    assert stats.t.expect(lambda u: 5 * u ** 2, args=(5,)) == pytest.approx(25 / 3)


def test_var1_formulas_against_recursion_and_simulation():
    rng = np.random.default_rng(1)
    B = np.array([[0.9, 0.05], [0.1, 0.8]]); a = np.array([0.002, 0.003])
    S = np.array([[1e-4, 5e-5], [5e-5, 2e-4]])
    x = np.zeros((4000, 2)); x[0] = [0.02, 0.03]
    for t in range(1, 4000):
        x[t] = a + B @ x[t - 1] + rng.multivariate_normal([0, 0], S)
    ah, Bh, eps, Sh = sc.var1_fit(x)
    assert np.allclose(Bh, B, atol=0.03)
    x0 = np.array([0.05, 0.01]); xr = x0.copy()
    for _ in range(12):
        xr = a + B @ xr
    assert np.allclose(sc.var1_mean(a, B, x0, 12), xr)          # 12 steps ahead, not 11
    sims = sc.var1_simulate(a, B, x0, 12, 20_000, sigma_eps=S, rng=2)
    assert np.allclose(sims[:, -1].mean(axis=0), xr, atol=5e-4)
    assert np.allclose(np.cov(sims[:, -1], rowvar=False), sc.var1_cov(B, S, 12), rtol=0.05)
    mu, Su = sc.var1_unconditional(a, B, S)
    assert np.allclose(sc.var1_mean(a, B, x0, 2000), mu)
    assert np.allclose(sc.var1_cov(B, S, 3000), Su)
    hist = sc.var1_simulate(a, B, x0, 3, 1000, eps_hist=eps, probs=sc.equal_probs(len(eps)), rng=3)
    assert hist.shape == (1000, 4, 2)
