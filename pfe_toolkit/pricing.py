"""Lecture 1 – pricing functions v0 (all rates continuously compounded)."""
from typing import Callable, Union
import numpy as np
from scipy import stats
from scipy.optimize import brentq

Array = Union[float, np.ndarray]


def zcb_price(zero_yield: Array, ttm: Array) -> Array:
    """Zero-coupon bond price V = exp(-y * tau)."""
    return np.exp(-np.asarray(zero_yield) * np.asarray(ttm))


def zcb_yield(price: Array, ttm: Array) -> Array:
    """Continuously compounded yield Y = -ln(V) / tau."""
    return -np.log(price) / ttm


def interpolate_yield(ttm: Array, tenors: np.ndarray, yields: np.ndarray) -> Array:
    """Linear interpolation of a zero curve (flat extrapolation), as in the 2024 exam."""
    return np.interp(ttm, tenors, yields)


def cash_flow_price(cash_flows: np.ndarray, times: np.ndarray, zero_yields: Array) -> float:
    """Price of fixed cash flows: sum CF_i * exp(-y_i * T_i)."""
    cash_flows, times = np.asarray(cash_flows, float), np.asarray(times, float)
    return float(np.sum(cash_flows * np.exp(-np.asarray(zero_yields) * times)))


def price_duration_convexity(cash_flows: np.ndarray, times: np.ndarray, zero_yields: Array):
    """(price, duration, convexity) of fixed cash flows; D = sum t*PV/P, K = sum t^2*PV/P."""
    times = np.asarray(times, float)
    pv = np.asarray(cash_flows, float) * np.exp(-np.asarray(zero_yields) * times)
    p = pv.sum()
    return float(p), float((pv * times).sum() / p), float((pv * times ** 2).sum() / p)


def bullet_bond_price(maturity: int, coupon: float, zero_curve: Callable[[np.ndarray], np.ndarray],
                      t: float = 0.0, principal: float = 100.0, freq: int = 1) -> float:
    """Bullet bond with payments at 1/freq, ..., maturity (years from issue); priced at time t with zero_curve(tau)."""
    times = np.arange(1, maturity * freq + 1) / freq
    cfs = np.full(times.shape, coupon / freq, dtype=float)
    cfs[-1] += principal
    keep = times > t + 1e-12
    tau = times[keep] - t
    return float(np.sum(cfs[keep] * np.exp(-zero_curve(tau) * tau)))


def vasicek_ab(tau: Array, kappa: float, theta: float, beta: float):
    """a(tau), b(tau) in V = exp(-a - b r) for dr = kappa(theta - r)dt + beta dW (Q)."""
    tau = np.asarray(tau, float)
    y_inf = theta - beta ** 2 / (2 * kappa ** 2)
    b = (1 - np.exp(-kappa * tau)) / kappa
    a = y_inf * (tau - b) + beta ** 2 / (4 * kappa) * b ** 2
    return a, b


def vasicek_zcb_price(tau: Array, r0: float, kappa: float, theta: float, beta: float) -> Array:
    a, b = vasicek_ab(tau, kappa, theta, beta)
    return np.exp(-a - b * r0)


def vasicek_yield(tau: Array, r0: float, kappa: float, theta: float, beta: float) -> Array:
    a, b = vasicek_ab(tau, kappa, theta, beta)
    return a / tau + b / tau * r0


def credit_zcb_price(zero_yield: Array, ttm: Array, intensity: float) -> Array:
    """Defaultable ZCB, exponential default (intensity lambda), zero recovery: exp(-(y + lambda) tau)."""
    return np.exp(-(np.asarray(zero_yield) + intensity) * np.asarray(ttm))


def forward_price(spot: Array, zcb: Array) -> Array:
    """No-arbitrage forward price F = V0 / ZCB(T) (no dividends)."""
    return np.asarray(spot) / np.asarray(zcb)


def fx_forward(spot_fx: Array, zcb_domestic: Array, zcb_foreign: Array) -> Array:
    """Covered interest parity. spot_fx = units of FOREIGN currency per 1 DOMESTIC (e.g. USD per EUR):
    F = FX0 * Z_dom / Z_for = FX0 * exp((y_for - y_dom) T)."""
    return np.asarray(spot_fx) * np.asarray(zcb_domestic) / np.asarray(zcb_foreign)


def bsm_price(spot: Array, strike: Array, rate: Array, vol: Array, ttm: Array, call: bool = True) -> Array:
    """Black–Scholes–Merton price of a European call (or put)."""
    spot, strike, rate, vol, ttm = map(np.asarray, (spot, strike, rate, vol, ttm))
    d1 = (np.log(spot / strike) + (rate + 0.5 * vol ** 2) * ttm) / (vol * np.sqrt(ttm))
    d2 = d1 - vol * np.sqrt(ttm)
    if call:
        return spot * stats.norm.cdf(d1) - strike * np.exp(-rate * ttm) * stats.norm.cdf(d2)
    return strike * np.exp(-rate * ttm) * stats.norm.cdf(-d2) - spot * stats.norm.cdf(-d1)


def bsm_delta(spot, strike, rate, vol, ttm, call: bool = True):
    d1 = (np.log(spot / strike) + (rate + 0.5 * vol ** 2) * ttm) / (vol * np.sqrt(ttm))
    return stats.norm.cdf(d1) if call else stats.norm.cdf(d1) - 1


def implied_vol(price: float, spot: float, strike: float, rate: float, ttm: float, call: bool = True) -> float:
    """Implied volatility by root search (brentq)."""
    return brentq(lambda s: bsm_price(spot, strike, rate, s, ttm, call) - price, 1e-6, 5.0)


def gbm_q_drift(mu: float, sigma: float, rate: float) -> float:
    """Market price of risk lambda = (mu - r) / sigma (Girsanov)."""
    return (mu - rate) / sigma
