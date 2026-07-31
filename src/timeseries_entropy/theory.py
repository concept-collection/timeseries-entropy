"""Analytic prediction of the entropy rate of z = round(h * x), x iid
N(0, sigma^2), from the Fourier modes H(f) = sum_j h_j e^{-2 pi i f j}.

Derivation, in three steps:

1. Szego / Kolmogorov. y = h * x is stationary Gaussian with power spectrum
   S(f) = sigma^2 |H(f)|^2. Its one-step prediction error variance from the
   exact past is the geometric mean of the spectrum,

       sigma_inf^2 = exp( int_0^1 ln S(f) df ),

   so in the high-resolution regime (sigma_inf >> 1 quantization step)

       H_rate ~ 1/2 log2(2 pi e sigma_inf^2).

   This fails in two ways tied to quantization: it ignores that only the
   *quantized* past is observed, and it diverges to -inf wherever H(f) has
   zeros or deep stopbands (the log integral blows up while the true entropy
   stays >= 0).

2. Quantization noise floor. Model roundoff as additive dither: z_t ~ y_t +
   u_t with u_t iid U(-1/2, 1/2), variance 1/12, independent of y. The best
   linear one-step prediction of w = y + u has error variance
   exp(int ln(S + 1/12)) (Szego again, on S_w = S + 1/12), and u_{t+1} being
   unpredictable splits off exactly, leaving for y_{t+1} itself

       s*^2 = exp( int_0^1 ln(S(f) + 1/12) df ) - 1/12.

   The 1/12 inside the log is what keeps the integral finite at spectral
   zeros; s*^2 >= sigma_inf^2 always (geometric means are superadditive),
   and s*^2 -> 0 as sigma -> 0.

3. Coarse-quantization saturation. Given the quantized past, y_{t+1} ~
   N(m, s*^2) with a conditional mean m that equidistributes modulo the
   quantization grid (valid when the marginal spread of y is >> 1 bin).
   The average discrete entropy of round(m + N(0, s^2)) over a uniform grid
   offset equals *exactly* the differential entropy of the Gaussian-uniform
   convolution (the standard dithered-quantization identity):

       G(s) = h( N(0, s^2) + U(-1/2, 1/2) )   [bits],

   which tends to 1/2 log2(2 pi e s^2) for s >> 1 and to 0 for s -> 0
   instead of diverging negative.

Final formula:

    H_rate ~ G( sqrt( exp( int_0^1 ln(sigma^2 |H(f)|^2 + 1/12) df ) - 1/12 ) )

Known approximations: the dither term is really deterministic roundoff, not
independent noise; w is not Gaussian (linear prediction is not optimal); and
for kernels whose conditional mean does not equidistribute mod 1 (e.g. the
identity kernel, where it is constant) G slightly overestimates at small s.
"""

import math

import numpy as np
from scipy.special import ndtr

TWO_PI_E = 2.0 * math.pi * math.e


def spectrum(kernel, n=1 << 18):
    """|H(f)|^2 on the rfft grid f = k/n, k = 0..n/2."""
    return np.abs(np.fft.rfft(np.asarray(kernel, dtype=float), n)) ** 2


def log_spectrum_mean(kernel, sigma, floor=1.0 / 12.0, n=1 << 18):
    """Mean over f in [0, 1) of ln(sigma^2 |H(f)|^2 + floor).

    Trapezoid on [0, 1/2] (the spectrum is symmetric). floor > 0 keeps the
    integrand finite at zeros of H.
    """
    logS = np.log(sigma * sigma * spectrum(kernel, n) + floor)
    return float((logS.sum() - 0.5 * (logS[0] + logS[-1])) / (logS.size - 1))


def sigma_infinity(kernel, sigma):
    """Szego one-step prediction error std of y = h * x from its exact past:
    the root geometric mean of the spectrum, sigma * |c_lead| * prod max(1, |b_k|)
    over the zeros b_k of the tap polynomial (exact, robust to zeros of H on
    the unit circle, where the log integral is still convergent)."""
    c = np.trim_zeros(np.asarray(kernel, dtype=float)[::-1], 'f')
    b = np.roots(c) if c.size > 1 else np.array([])
    return sigma * abs(c[0]) * float(np.prod(np.maximum(1.0, np.abs(b))))


def gauss_uniform_entropy(s):
    """G(s) = h(N(0, s^2) + U(-1/2, 1/2)) in bits — the exact average entropy
    of round(c + N(0, s^2)) over a uniform grid offset c."""
    s = float(s)
    if s <= 0:
        return 0.0
    if s < 1e-3:
        return _edge_constant() * s
    dv = min(s / 8.0, 0.01)
    v = np.arange(0.0, 0.5 + 8.0 * s + 1.0, dv)
    g = ndtr((v + 0.5) / s) - ndtr((v - 0.5) / s)
    term = np.where(g > 0, -g * np.log2(np.clip(g, 1e-300, None)), 0.0)
    return float(2.0 * dv * (term.sum() - 0.5 * term[0]))


_EDGE_C = None


def _edge_constant():
    """int_-inf^inf h2(Phi(t)) dt: small-s slope of G(s)."""
    global _EDGE_C
    if _EDGE_C is None:
        t = np.linspace(-12.0, 12.0, 20001)
        p = np.clip(ndtr(t), 1e-300, 1 - 1e-16)
        h2 = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        _EDGE_C = float(np.trapezoid(h2, t))
    return _EDGE_C


def predict_entropy_rate(kernel, sigma):
    """Both predictions of the entropy rate of z = round(h * x), in bits.

    Returns a dict:
      sigma_inf  Szego prediction error std from the exact past
      highres    1/2 log2(2 pi e sigma_inf^2) — valid when sigma_inf >~ 1
      s_star     prediction error std of y_next from the quantized past
      corrected  G(s_star) — stays valid at spectral zeros and coarse
                 quantization
    """
    s_inf = sigma_infinity(kernel, sigma)
    highres = (0.5 * math.log2(TWO_PI_E * s_inf * s_inf)
               if s_inf > 0 else -math.inf)
    gm_w = math.exp(log_spectrum_mean(kernel, sigma))
    s_star = math.sqrt(max(gm_w - 1.0 / 12.0, 0.0))
    return {
        'sigma_inf': s_inf,
        'highres': highres,
        's_star': s_star,
        'corrected': gauss_uniform_entropy(s_star),
    }
