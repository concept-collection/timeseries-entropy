import math

import numpy as np
import pytest

from timeseries_entropy import kernels
from timeseries_entropy.theory import (
    gauss_uniform_entropy, log_spectrum_mean, predict_entropy_rate,
    sigma_infinity)


def test_sigma_infinity_closed_forms():
    # MA(W): all tap-polynomial zeros on the unit circle -> sigma / W.
    assert sigma_infinity(kernels.moving_average(8), 4.0) == pytest.approx(0.5)
    # first difference: 1 - w, zero at w = 1 -> sigma.
    assert sigma_infinity(kernels.first_difference(), 8.0) == pytest.approx(8.0)
    assert sigma_infinity(kernels.identity(), 2.0) == pytest.approx(2.0)


def test_log_spectrum_mean_matches_szego_when_floor_negligible():
    # With a floor far below the spectrum minimum, the FFT integral must
    # agree with the exact roots-based geometric mean.
    h = np.array([1.0, -0.5])  # min |H|^2 = 0.25, floor 1e-12 negligible
    gm = math.exp(log_spectrum_mean(h, 1.0, floor=1e-12))
    assert gm == pytest.approx(sigma_infinity(h, 1.0) ** 2, rel=1e-6)


def test_gauss_uniform_entropy_limits():
    # Large s: h(N + U) -> 1/2 log2(2 pi e (s^2 + 1/12)) (Gaussian limit).
    for s in [4.0, 16.0]:
        expect = 0.5 * math.log2(2 * math.pi * math.e * (s * s + 1 / 12))
        assert gauss_uniform_entropy(s) == pytest.approx(expect, abs=2e-3)
    # Small s: -> 0 linearly, never negative.
    assert gauss_uniform_entropy(0.0) == 0.0
    assert 0 < gauss_uniform_entropy(0.01) < 0.05


def test_gauss_uniform_entropy_is_offset_averaged_quantized_entropy():
    # G(s) = E_c H(round(c + N(0, s^2))), c ~ U(0, 1)  (dither identity).
    s = 0.7
    cs = (np.arange(200) + 0.5) / 200
    zs = np.arange(-12, 13)
    edges = (zs[None, :] - 0.5 - cs[:, None]) / s
    from scipy.special import ndtr
    p = np.diff(ndtr(np.concatenate([edges, edges[:, -1:] + 1 / s], axis=1)))
    p = np.clip(p, 1e-300, None)
    mean_h = float((-p * np.log2(p)).sum(axis=1).mean())
    assert gauss_uniform_entropy(s) == pytest.approx(mean_h, abs=1e-3)


def test_predict_identity_kernel_high_resolution():
    # For h = [1], z is iid round(N(0, sigma^2)); at sigma >> 1 both
    # predictions must approach the exact marginal entropy.
    sigma = 16.0
    zmax = int(8 * sigma + 4)
    z = np.arange(-zmax, zmax + 1)
    from scipy.special import ndtr
    p = ndtr((z + 0.5) / sigma) - ndtr((z - 0.5) / sigma)
    p = p[p > 0]
    exact = float(-(p * np.log2(p)).sum())
    pred = predict_entropy_rate(kernels.identity(), sigma)
    assert pred['corrected'] == pytest.approx(exact, abs=2e-3)
    assert pred['highres'] == pytest.approx(exact, abs=2e-3)


def test_corrected_prediction_stays_finite_at_spectral_zeros():
    pred = predict_entropy_rate(kernels.windowed_sinc_lowpass(0.1, 101), 8.0)
    assert pred['highres'] < 0          # Szego diverges toward -inf
    assert 0 < pred['corrected'] < 16   # corrected saturates sensibly
