"""The process and its conditional sampler.

Model: x iid N(0, sigma^2) -> y = h * x (causal FIR, kernel length L)
-> z = round(y). A zero-phase or otherwise shifted application of the same
kernel gives the same law, so causal convolution loses no generality.

ConditionalChain targets H(z_{M+1} | z_1..z_M): it fixes an observed past
z_1..z_M and Gibbs-samples the latent x under the box constraints
y_t in [z_t - 1/2, z_t + 1/2), emitting one exact draw of z_{M+1} per step.
"""

import numpy as np
from scipy.special import ndtr, ndtri


def truncated_std_normal(lo, hi, rng):
    """Standard normal truncated to [lo, hi], elementwise, by inverse CDF.
    Mirrored into the lower tail so the CDF differences keep precision."""
    flip = lo > -hi  # midpoint above 0 (robust to (-inf, inf) intervals)
    a = np.where(flip, -hi, lo)
    b = np.where(flip, -lo, hi)
    fa = ndtr(a)
    fb = ndtr(b)
    u = fa + (fb - fa) * rng.random(a.shape)
    x = ndtri(np.clip(u, 1e-300, 1 - 1e-16))
    x = np.where(flip, -x, x)
    return np.clip(x, lo, hi)


class ConditionalChain:
    """Stationary chain of exact draws of z_{M+1} given a fixed past z_1..z_M.

    The constructor draws the past from the prior; the generating latents are
    themselves an exact draw from p(x | z), so the Gibbs chain starts in
    stationarity — no burn-in bias, only autocorrelation. draw(k) advances the
    chain k steps (thin sweeps each) and returns the k sampled z_{M+1} values,
    each marginally distributed exactly as z_{M+1} | z_1..z_M. thin may be
    reassigned between draws (e.g. probe at thin=1, then thin by the measured
    autocorrelation time); stationarity is unaffected.

    Each Gibbs conditional x_i | rest is N(0, sigma^2) truncated to the
    interval read off the <= L constraint boxes x_i appears in. Coordinates a
    multiple of L apart share no constraint, so each of the L "colors"
    updates as one vectorized block.
    """

    def __init__(self, kernel, sigma, past, rng=None, thin=1):
        h = np.asarray(kernel, dtype=float)
        if h.ndim != 1 or h.size == 0:
            raise ValueError('kernel must be a nonempty 1-D array')
        if sigma <= 0:
            raise ValueError('sigma must be positive')
        if past < 1:
            raise ValueError('past must be >= 1')
        self.h = h
        self.sigma = float(sigma)
        self.thin = int(thin)
        self.rng = np.random.default_rng() if rng is None else rng

        L = h.size
        M = int(past)
        self.L, self.M = L, M

        # The past, with its true latents as the (stationary) chain start.
        x = self.sigma * self.rng.standard_normal(M + L - 1)
        y = np.convolve(x, h, mode='valid')
        self.z = np.floor(y + 0.5)
        self.x = x

        # Boxes and y live in padded arrays so that every coordinate x_i sees
        # exactly L constraint rows (rows outside the data are unconstrained).
        P = L - 1
        self.ypad = np.zeros(M + 2 * P)
        self.lo = np.full(M + 2 * P, -np.inf)
        self.hi = np.full(M + 2 * P, np.inf)
        self.lo[P:P + M] = self.z - 0.5
        self.hi[P:P + M] = self.z + 0.5

        # Row i+j (padded) carries coefficient h[j] for coordinate i.
        self.classes = [np.arange(c0, M + L - 1, L) for c0 in range(L)]
        self.rowmats = [idx[:, None] + np.arange(L)[None, :]
                        for idx in self.classes]
        self.nonzero = h != 0
        # z_{M+1} = round(hr_head @ x[M:] + h[0] * x_free), x_free fresh.
        self.hr_head = h[::-1][:-1]

    def _sweep(self):
        h, x, sigma = self.h, self.x, self.sigma
        M, L = self.M, self.L
        P = L - 1
        self.ypad[P:P + M] = np.convolve(x, h, mode='valid')  # kill fp drift
        for idx, rows in zip(self.classes, self.rowmats):
            r = self.ypad[rows] - np.outer(x[idx], h)
            with np.errstate(divide='ignore', invalid='ignore'):
                b1 = (self.lo[rows] - r) / h[None, :]
                b2 = (self.hi[rows] - r) / h[None, :]
            xlo = np.where(h[None, :] > 0, b1, b2)
            xhi = np.where(h[None, :] > 0, b2, b1)
            xlo[:, ~self.nonzero] = -np.inf
            xhi[:, ~self.nonzero] = np.inf
            xlo = xlo.max(axis=1)
            xhi = xhi.min(axis=1)
            xnew = truncated_std_normal(xlo / sigma, xhi / sigma, self.rng) * sigma
            self.ypad[rows] += (xnew - x[idx])[:, None] * h[None, :]
            x[idx] = xnew

    def draw(self, k):
        """The next k samples of z_{M+1}, continuing the chain."""
        out = np.empty(k, dtype=np.int64)
        M = self.M
        for i in range(k):
            for _ in range(self.thin):
                self._sweep()
            c = float(self.hr_head @ self.x[M:]) if self.L > 1 else 0.0
            y_next = c + self.sigma * self.h[0] * self.rng.standard_normal()
            out[i] = int(np.floor(y_next + 0.5))
        return out
