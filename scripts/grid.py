"""The parameter grid whose estimates are cached on the `estimates` branch.

Each cell is a plain dict of CLI-equivalent parameters; `cell_id` names it and
`kernel_from_params` rebuilds its kernel. Params are stored verbatim in every
record on the data branch, so aggregation never needs this file — the grid can
grow or change without invalidating what is already cached.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from timeseries_entropy import kernels  # noqa: E402

RATE = 30000.0
TAPS = 101

SIGMAS = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

FILTERS = [
    {'filter': 'none'},
    {'filter': 'moving-average', 'width': 8},
    {'filter': 'lowpass', 'high': 3000.0, 'rate': RATE, 'taps': TAPS},
    {'filter': 'bandpass', 'low': 300.0, 'high': 6000.0, 'rate': RATE,
     'taps': TAPS},
]


def grid():
    """The full list of cells: one representative per filter x every sigma."""
    return [dict(f, sigma=s) for f in FILTERS for s in SIGMAS]


def kernel_from_params(p):
    """Kernel array for a params dict — mirrors the CLI's --filter handling."""
    f = p['filter']
    if f == 'none':
        return kernels.identity()
    if f == 'first-difference':
        return kernels.first_difference()
    if f == 'moving-average':
        return kernels.moving_average(int(p['width']))
    if f == 'lowpass':
        return kernels.windowed_sinc_lowpass(p['high'] / p['rate'],
                                             int(p['taps']))
    if f == 'bandpass':
        return kernels.windowed_sinc_bandpass(p['low'] / p['rate'],
                                              p['high'] / p['rate'],
                                              int(p['taps']))
    raise ValueError(f'unknown filter {f!r}')


def cell_id(p):
    """Stable, filename-safe name for a cell, e.g. 'bp300-6000_s8'."""
    f = p['filter']
    if f == 'none':
        tag = 'none'
    elif f == 'first-difference':
        tag = 'diff'
    elif f == 'moving-average':
        tag = f"ma{_num(p['width'])}"
    elif f == 'lowpass':
        tag = f"lp{_num(p['high'])}"
    elif f == 'bandpass':
        tag = f"bp{_num(p['low'])}-{_num(p['high'])}"
    else:
        raise ValueError(f'unknown filter {f!r}')
    return f"{tag}_s{_num(p['sigma'])}"


def _num(v):
    """4.0 -> '4', 0.5 -> '0.5'."""
    return str(int(v)) if float(v) == int(v) else str(v)
