import math

import numpy as np
import pytest

from timeseries_entropy import (plugin_entropy, unbiased_entropy,
                                level_corrections, integrated_autocorr_time)


def test_plugin_entropy_exact():
    assert plugin_entropy([0, 1]) == pytest.approx(1.0)
    assert plugin_entropy([3, 3, 3]) == pytest.approx(0.0)
    assert plugin_entropy([0, 0, 0, 1]) == pytest.approx(
        -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25)))


def test_unbiased_on_iid_categorical():
    p = np.array([0.5, 0.3, 0.2])
    true_h = -np.sum(p * np.log2(p))
    rng = np.random.default_rng(1)
    draw = lambda k: rng.choice(3, size=k, p=p)
    vals = np.array([unbiased_entropy(draw, n0=32, r=1.5, rng=rng)
                     for _ in range(4000)])
    se = vals.std(ddof=1) / math.sqrt(len(vals))
    assert abs(vals.mean() - true_h) < 5 * se
    assert se < 0.01


def test_unbiased_on_autocorrelated_chain():
    # Stationary two-state Markov chain with sticky transitions; the marginal
    # is uniform, so H = 1 bit despite strong autocorrelation.
    rng = np.random.default_rng(2)
    state = [int(rng.random() < 0.5)]

    def draw(k):
        out = np.empty(k, dtype=int)
        for i in range(k):
            if rng.random() < 0.1:
                state[0] = 1 - state[0]
            out[i] = state[0]
        return out

    vals = np.array([unbiased_entropy(draw, n0=64, r=1.5, rng=rng)
                     for _ in range(3000)])
    se = vals.std(ddof=1) / math.sqrt(len(vals))
    assert abs(vals.mean() - 1.0) < 5 * se
    assert se < 0.02


def test_autocorr_time_iid():
    rng = np.random.default_rng(6)
    tau = integrated_autocorr_time(rng.integers(0, 4, size=4096))
    assert 0.8 < tau < 1.5


def test_autocorr_time_sticky_markov():
    # Two-state chain flipping w.p. eps: rho_k = (1 - 2 eps)^k, so
    # tau = 1 + 2 rho / (1 - rho) = 19 at eps = 0.05.
    rng = np.random.default_rng(7)
    flips = rng.random(200000) < 0.05
    x = np.cumsum(flips) % 2
    tau = integrated_autocorr_time(x)
    assert 12 < tau < 28


def test_autocorr_time_degenerate():
    assert integrated_autocorr_time([3]) == 1.0
    assert integrated_autocorr_time([2, 2, 2, 2]) == 1.0


def test_level_corrections_decay():
    rng = np.random.default_rng(3)
    p = np.array([0.5, 0.3, 0.2])
    deltas = np.array([
        level_corrections(lambda k: rng.choice(3, size=k, p=p), n0=64, levels=5)
        for _ in range(200)])
    rms = np.sqrt((deltas ** 2).mean(axis=0))
    # E[Delta_m^2] should decay roughly like 2^(-2m); demand clear decay.
    assert rms[-1] < rms[0] / 4
