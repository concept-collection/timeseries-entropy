import math

import numpy as np

from timeseries_entropy import ConditionalChain, estimate_conditional_entropy
from timeseries_entropy.model import truncated_std_normal


def quantized_gaussian_entropy_bits(s):
    """Exact entropy of round(N(0, s^2)) by direct summation."""
    zmax = int(math.ceil(8 * s + 4))
    H = 0.0
    prev = 0.5 * (1 + math.erf((-zmax - 0.5) / (s * math.sqrt(2))))
    for z in range(-zmax, zmax + 1):
        cur = 0.5 * (1 + math.erf((z + 0.5) / (s * math.sqrt(2))))
        p = cur - prev
        prev = cur
        if p > 0:
            H -= p * math.log2(p)
    return H


def test_truncated_std_normal_bounds():
    rng = np.random.default_rng(0)
    lo = np.array([-1.0, 0.5, -np.inf, -8.0])
    hi = np.array([-0.5, 2.0, np.inf, -7.0])
    for _ in range(100):
        x = truncated_std_normal(lo, hi, rng)
        assert np.all(x >= lo) and np.all(x <= hi)


def test_gibbs_preserves_constraints():
    rng = np.random.default_rng(4)
    kernel = np.full(3, 1 / 3)
    chain = ConditionalChain(kernel, sigma=2.0, past=64, rng=rng)
    z0 = chain.z.copy()
    chain.draw(50)
    y = np.convolve(chain.x, chain.h, mode='valid')
    assert np.array_equal(np.floor(y + 0.5), z0)


def test_no_filter_matches_exact_entropy():
    # kernel [1]: z is iid round(N(0, sigma^2)), so the conditional entropy
    # equals the exact marginal entropy for any past length.
    sigma = 1.5
    exact = quantized_gaussian_entropy_bits(sigma)
    est = estimate_conditional_entropy(
        [1.0], sigma, past=8, pasts=12, reps=24, n0=64, seed=5)
    assert abs(est.mean - exact) < max(5 * est.se, 0.02)
