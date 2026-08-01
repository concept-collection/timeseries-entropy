"""Unbiased Monte-Carlo entropy estimation for quantized filtered Gaussian
time series: x iid N(0, sigma^2) -> y = h * x -> z = round(y)."""

import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np

from .estimator import (plugin_entropy, unbiased_entropy, level_corrections,
                        integrated_autocorr_time)
from .model import ConditionalChain
from . import kernels

__all__ = [
    'estimate_conditional_entropy', 'Estimate', 'ConditionalChain',
    'plugin_entropy', 'unbiased_entropy', 'level_corrections',
    'integrated_autocorr_time', 'kernels',
]


@dataclass
class Estimate:
    """mean +/- se (over independent pasts) of H(z_{M+1} | z_1..z_M), bits.

    per_past holds one unbiased value per past; thin, reps, and tau record,
    per past, the resolved Gibbs sweeps per draw, the realizations averaged,
    and the probe's autocorrelation-time estimate (nan when thin was fixed).
    """
    mean: float
    se: float
    per_past: np.ndarray
    thin: np.ndarray
    reps: np.ndarray
    tau: np.ndarray


def _resolve_thin(chain, thin, probe, thin_cap):
    """Set chain.thin; returns (resolved thin, probe tau or nan)."""
    if thin != 'auto':
        chain.thin = int(thin)
        return int(thin), float('nan')
    tau = integrated_autocorr_time(chain.draw(probe))
    chain.thin = int(min(thin_cap, math.ceil(tau)))
    return chain.thin, tau


def _one_past(kernel, sigma, M, thin, n0, r, reps, probe, thin_cap, seed_seq):
    rng = np.random.default_rng(seed_seq)
    chain = ConditionalChain(kernel, sigma, M, rng, 1)
    t, tau = _resolve_thin(chain, thin, probe, thin_cap)
    n_reps = max(1, round(reps / t)) if thin == 'auto' else reps
    val = float(np.mean(
        [unbiased_entropy(chain.draw, n0, r, rng) for _ in range(n_reps)]))
    return val, t, n_reps, tau


def estimate_conditional_entropy(kernel, sigma, past=None, pasts=24, reps=8,
                                 n0=128, r=1.5, thin='auto', probe=512,
                                 thin_cap=64, seed=None, progress=None,
                                 workers=None):
    """Unbiased estimate of H(z_{M+1} | z_1..z_M) in bits.

    For each of `pasts` independent pasts, a stationary Gibbs chain of
    z_{M+1} draws feeds randomized-telescoping realizations (on consecutive
    segments of the chain); their average is one unbiased value per past.
    Returns the mean and standard error over pasts — valid because pasts are
    independent.

    thin='auto' (the default) matches the chain's thinning to its measured
    mixing: each past first draws `probe` samples at thin=1, estimates their
    integrated autocorrelation time tau, and thins by ceil(tau), capped at
    `thin_cap`. This keeps the level corrections Delta_m decaying fast
    enough for the r exponent — with slowly mixing chains (narrowband
    kernels x large sigma) and no thinning, E[Delta_m^2] can decay slower
    than 2^(-r m), which makes the estimator's variance infinite: still
    unbiased, but with rare enormous realizations and meaningless standard
    errors. Under 'auto', `reps` is a per-past budget at thin=1: the
    realization count becomes max(1, round(reps / thin)), so per-past cost
    stays roughly flat and slow cells trade realizations for better draws.
    Pass an integer thin to control both knobs explicitly.

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
    thins = np.empty(pasts, dtype=int)
    n_reps = np.empty(pasts, dtype=int)
    taus = np.empty(pasts)
    values = []

    def record(i, result):
        per_past[i], thins[i], n_reps[i], taus[i] = result
        values.append(per_past[i])

    if workers <= 1:
        for i in range(pasts):
            record(i, _one_past(kernel, sigma, M, thin, n0, r, reps, probe,
                                thin_cap, seeds[i]))
            if progress is not None:
                progress(i, values)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_one_past, kernel, sigma, M, thin, n0, r, reps,
                            probe, thin_cap, seeds[i]): i
                for i in range(pasts)}
            for done, fut in enumerate(as_completed(futures)):
                record(futures[fut], fut.result())
                if progress is not None:
                    progress(done, values)
    se = (float(per_past.std(ddof=1) / math.sqrt(len(per_past)))
          if len(per_past) > 1 else float('nan'))
    return Estimate(float(per_past.mean()), se, per_past, thins, n_reps, taus)
