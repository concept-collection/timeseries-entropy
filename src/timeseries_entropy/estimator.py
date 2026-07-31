"""Rhee-Glynn (randomized telescoping) unbiased entropy estimation.

Input: a stationary, ergodic sequence of discrete draws, possibly
autocorrelated, each with the target marginal p. Plug-in entropies of blocks
whose sizes double from level to level form a telescoping sum via the
antithetic correction

    Delta_m = h(B_m) - [h(B_m^1) + h(B_m^2)] / 2,

where B_m^1, B_m^2 are the two halves of B_m. Truncating the sum at a random
level N with P(N >= m) = 2^(-r m) and reweighting by the survival
probabilities gives an estimator whose expectation is exactly the entropy of
p, despite the bias of every finite-block plug-in estimate.

All entropies are in bits.
"""

import numpy as np


def plugin_entropy(samples):
    """Plug-in entropy (bits) of a block of discrete samples."""
    _, counts = np.unique(np.asarray(samples), return_counts=True)
    return _entropy(counts.astype(float), counts.sum())


def _entropy(counts, n):
    return float(np.log2(n) - (counts * np.log2(counts)).sum() / n)


def _dict_entropy(counts, n):
    return _entropy(np.fromiter(counts.values(), dtype=float), n)


def _telescope(draw, n0, levels):
    """h(B_0) and [Delta_1, ..., Delta_levels] over one growing block.

    Counts are merged upward (each sample is counted once), so the cost is
    linear in the n0 * 2**levels samples drawn.
    """
    counts = {}
    _count_into(counts, draw(n0))
    size = n0
    h0 = _dict_entropy(counts, size)
    h_prev = h0
    deltas = []
    for _ in range(levels):
        half = {}
        _count_into(half, draw(size))
        h2 = _dict_entropy(half, size)
        for v, c in half.items():
            counts[v] = counts.get(v, 0) + c
        size *= 2
        h_full = _dict_entropy(counts, size)
        deltas.append(h_full - 0.5 * (h_prev + h2))
        h_prev = h_full
    return h0, deltas


def _count_into(counts, seg):
    vals, cnts = np.unique(np.asarray(seg), return_counts=True)
    for v, c in zip(vals.tolist(), cnts.tolist()):
        counts[v] = counts.get(v, 0) + c


def unbiased_entropy(draw, n0=128, r=1.5, rng=None):
    """One randomized-telescoping realization of the marginal entropy (bits).

    draw(k) must return the next k consecutive samples of a stationary
    discrete chain; successive calls continue the chain. The realization
    consumes n0 * 2**N samples with P(N >= m) = 2^(-r m). Average many
    realizations (they may continue one chain back-to-back) to reduce
    variance; each has expectation exactly H(p).

    r trades expected work against variance: levels must decay like
    E[Delta_m^2] = O(2^(-r'm)) with r' > r > 1 for both to be finite.
    r = 1.5 suits the typical second-moment decay 2^(-2m); check with
    level_corrections when in doubt.
    """
    rng = np.random.default_rng() if rng is None else rng
    rho = 2.0 ** -r
    N = 0
    while rng.random() < rho:
        N += 1
    h0, deltas = _telescope(draw, n0, N)
    return h0 + sum(d * 2.0 ** (r * m) for m, d in enumerate(deltas, start=1))


def level_corrections(draw, n0=128, levels=6):
    """Deterministic pilot: [Delta_1, ..., Delta_levels] from one block.

    Draws n0 * 2**levels samples. Repeat over fresh chains, look at the decay
    of mean(Delta_m^2) with m, and pick r below the decay exponent.
    """
    _, deltas = _telescope(draw, n0, levels)
    return np.array(deltas)
