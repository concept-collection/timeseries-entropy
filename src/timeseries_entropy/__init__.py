"""Unbiased Monte-Carlo entropy estimation for quantized filtered Gaussian
time series: x iid N(0, sigma^2) -> y = h * x -> z = round(y)."""

import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _one_past(kernel, sigma, M, thin, n0, r, reps, seed_seq):
    rng = np.random.default_rng(seed_seq)
    chain = ConditionalChain(kernel, sigma, M, rng, thin)
    return float(np.mean(
        [unbiased_entropy(chain.draw, n0, r, rng) for _ in range(reps)]))


def estimate_conditional_entropy(kernel, sigma, past=None, pasts=24, reps=8,
                                 n0=128, r=1.5, thin=1, seed=None,
                                 progress=None, workers=None):
    """Unbiased estimate of H(z_{M+1} | z_1..z_M) in bits.

    For each of `pasts` independent pasts, a stationary Gibbs chain of
    z_{M+1} draws feeds `reps` randomized-telescoping realizations (on
    consecutive segments of the chain); their average is one unbiased value
    per past. Returns the mean and standard error over pasts — valid because
    pasts are independent.

    past defaults to max(512, 4 * len(kernel)); the estimand decreases toward
    the entropy rate as it grows. progress, if given, is called as
    progress(i, values) after each past finishes (completion order when
    parallel).

    Pasts run in parallel across `workers` processes (default: all cores).
    Each past gets its own spawned RNG stream, so a given seed yields the
    same result for any worker count.
    """
    kernel = np.asarray(kernel, dtype=float)
    M = int(past) if past is not None else max(512, 4 * kernel.size)
    seeds = np.random.SeedSequence(seed).spawn(pasts)
    if workers is None:
        workers = min(pasts, os.cpu_count() or 1)
    per_past = np.empty(pasts)
    values = []
    if workers <= 1:
        for i in range(pasts):
            per_past[i] = _one_past(kernel, sigma, M, thin, n0, r, reps,
                                    seeds[i])
            values.append(per_past[i])
            if progress is not None:
                progress(i, values)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_one_past, kernel, sigma, M, thin, n0, r, reps,
                            seeds[i]): i
                for i in range(pasts)}
            for done, fut in enumerate(as_completed(futures)):
                per_past[futures[fut]] = fut.result()
                values.append(per_past[futures[fut]])
                if progress is not None:
                    progress(done, values)
    se = (float(per_past.std(ddof=1) / math.sqrt(len(per_past)))
          if len(per_past) > 1 else float('nan'))
    return Estimate(float(per_past.mean()), se, per_past)
