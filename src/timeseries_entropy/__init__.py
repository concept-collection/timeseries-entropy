"""Unbiased Monte-Carlo entropy estimation for quantized filtered Gaussian
time series: x iid N(0, sigma^2) -> y = h * x -> z = round(y)."""

import math
from dataclasses import dataclass

import numpy as np

from .estimator import plugin_entropy, unbiased_entropy, level_corrections
from .model import ConditionalChain
from . import kernels

__all__ = [
    'estimate_conditional_entropy', 'Estimate', 'ConditionalChain',
    'plugin_entropy', 'unbiased_entropy', 'level_corrections', 'kernels',
]


@dataclass
class Estimate:
    """mean +/- se (over independent pasts) of H(z_{M+1} | z_1..z_M), bits."""
    mean: float
    se: float
    per_past: np.ndarray


def estimate_conditional_entropy(kernel, sigma, past=None, pasts=24, reps=8,
                                 n0=128, r=1.5, thin=1, seed=None,
                                 progress=None):
    """Unbiased estimate of H(z_{M+1} | z_1..z_M) in bits.

    For each of `pasts` independent pasts, a stationary Gibbs chain of
    z_{M+1} draws feeds `reps` randomized-telescoping realizations (on
    consecutive segments of the chain); their average is one unbiased value
    per past. Returns the mean and standard error over pasts — valid because
    pasts are independent.

    past defaults to max(512, 4 * len(kernel)); the estimand decreases toward
    the entropy rate as it grows. progress, if given, is called as
    progress(i, values) after each past.
    """
    kernel = np.asarray(kernel, dtype=float)
    M = int(past) if past is not None else max(512, 4 * kernel.size)
    rng = np.random.default_rng(seed)
    values = []
    for i in range(pasts):
        chain = ConditionalChain(kernel, sigma, M, rng, thin)
        values.append(float(np.mean(
            [unbiased_entropy(chain.draw, n0, r, rng) for _ in range(reps)])))
        if progress is not None:
            progress(i, values)
    per_past = np.array(values)
    se = (float(per_past.std(ddof=1) / math.sqrt(len(per_past)))
          if len(per_past) > 1 else float('nan'))
    return Estimate(float(per_past.mean()), se, per_past)
