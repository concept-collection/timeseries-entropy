"""Kernel constructors matching timeseries-compressibility's filters."""

import numpy as np


def identity():
    return np.array([1.0])


def moving_average(width):
    return np.full(width, 1.0 / width)


def first_difference():
    return np.array([1.0, -1.0])


def windowed_sinc_lowpass(fc, taps):
    """Hamming-windowed sinc, cutoff fc in cycles/sample, unit DC gain."""
    n = taps | 1
    mid = (n - 1) / 2
    i = np.arange(n)
    t = i - mid
    sinc = np.where(t == 0, 2 * fc,
                    np.sin(2 * np.pi * fc * t) / (np.pi * np.where(t == 0, 1, t)))
    w = 0.54 - 0.46 * np.cos(2 * np.pi * i / (n - 1))
    h = sinc * w
    return h / h.sum()


def windowed_sinc_bandpass(f_lo, f_hi, taps):
    """Difference of two windowed-sinc lowpasses; edges in cycles/sample."""
    return windowed_sinc_lowpass(f_hi, taps) - windowed_sinc_lowpass(f_lo, taps)
