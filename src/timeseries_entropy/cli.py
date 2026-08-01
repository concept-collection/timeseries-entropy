"""Command-line interface: timeseries-entropy [options]."""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import (estimate_conditional_entropy, level_corrections, kernels,
               _resolve_thin)
from .model import ConditionalChain
from .theory import predict_entropy_rate


def _thin_arg(s):
    return s if s == 'auto' else int(s)


def design_kernel(args):
    if args.filter == 'none':
        return kernels.identity()
    if args.filter == 'moving-average':
        return kernels.moving_average(args.width)
    if args.filter == 'lowpass':
        if args.high is None:
            raise SystemExit('lowpass needs --high')
        return kernels.windowed_sinc_lowpass(args.high / args.rate, args.taps)
    if args.filter == 'bandpass':
        if args.low is None or args.high is None:
            raise SystemExit('bandpass needs --low and --high')
        return kernels.windowed_sinc_bandpass(
            args.low / args.rate, args.high / args.rate, args.taps)
    if args.filter == 'first-difference':
        return kernels.first_difference()
    raise ValueError(args.filter)


def main():
    ap = argparse.ArgumentParser(
        prog='timeseries-entropy',
        description='Unbiased Monte-Carlo estimate of H(z_next | M past '
                    'samples), in bits, for x iid N(0, sigma^2) -> h * x '
                    '-> round.')
    ap.add_argument('--sigma', type=float, required=True,
                    help='input std, in quantization steps')
    ap.add_argument('--filter', required=True,
                    choices=['none', 'moving-average', 'lowpass', 'bandpass',
                             'first-difference'])
    ap.add_argument('--low', type=float, help='bandpass low edge, Hz')
    ap.add_argument('--high', type=float,
                    help='lowpass cutoff / bandpass high edge, Hz')
    ap.add_argument('--taps', type=int, default=101,
                    help='windowed-sinc kernel length')
    ap.add_argument('--width', type=int, default=8, help='moving-average width')
    ap.add_argument('--rate', type=float, default=30000, help='sample rate, Hz')
    ap.add_argument('--past', type=int,
                    help='conditioning window M (default max(512, 4*L))')
    ap.add_argument('--pasts', type=int, default=24,
                    help='independent pasts to average')
    ap.add_argument('--reps', type=int, default=8,
                    help='per-past realization budget at thin=1')
    ap.add_argument('--n0', type=int, default=128, help='base block size')
    ap.add_argument('--r', type=float, default=1.5,
                    help='truncation exponent: P(N >= m) = 2^(-r m)')
    ap.add_argument('--thin', type=_thin_arg, default='auto',
                    help="Gibbs sweeps per emitted sample, or 'auto' "
                         '(default) to match the measured mixing per past')
    ap.add_argument('--workers', type=int,
                    help='parallel processes over pasts (default: all cores)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pilot', type=int, metavar='LEVELS',
                    help='instead of estimating, print RMS Delta_m over the '
                         'pasts for m = 1..LEVELS, to help choose --r')
    args = ap.parse_args()

    kernel = design_kernel(args)
    L = len(kernel)
    M = args.past if args.past is not None else max(512, 4 * L)
    print(f'model: sigma={args.sigma} filter={args.filter} L={L} M={M}')
    pred = predict_entropy_rate(kernel, args.sigma)
    print(f'predicted rate: {pred["corrected"]:.4f} bits '
          f'(quantization-corrected; s*={pred["s_star"]:.4g})   '
          f'high-res Szego: {pred["highres"]:.4f} '
          f'(sigma_inf={pred["sigma_inf"]:.4g})')

    if args.pilot is not None:
        run_pilot(kernel, args, M)
        return

    print(f'{args.pasts} pasts x {args.reps} reps, n0={args.n0} r={args.r} '
          f'thin={args.thin}')

    def progress(i, values):
        mean = float(np.mean(values))
        se = (float(np.std(values, ddof=1) / np.sqrt(len(values)))
              if len(values) > 1 else float('nan'))
        print(f'  past {i + 1:3d}/{args.pasts}: H = {values[-1]:.4f}   '
              f'running mean {mean:.4f} +/- {se:.4f}')

    est = estimate_conditional_entropy(
        kernel, args.sigma, past=M, pasts=args.pasts, reps=args.reps,
        n0=args.n0, r=args.r, thin=args.thin, seed=args.seed,
        progress=progress, workers=args.workers)
    if args.thin == 'auto':
        print(f'resolved thin {est.thin.min()}-{est.thin.max()} '
              f'(tau {est.tau.min():.1f}-{est.tau.max():.1f}), '
              f'{est.reps.min()}-{est.reps.max()} reps/past')
    ratio = f'   (ratio vs int16: {16 / est.mean:.3f}x)' if est.mean > 0 else ''
    print(f'\nH(z_next | {M} past samples) = {est.mean:.4f} +/- {est.se:.4f} '
          f'bits/sample{ratio}')
    print('note: an upper bound on the entropy rate that tightens as --past '
          'grows.')


def _one_pilot(kernel, sigma, M, thin, n0, levels, seed_seq):
    rng = np.random.default_rng(seed_seq)
    chain = ConditionalChain(kernel, sigma, M, rng, 1)
    t, _ = _resolve_thin(chain, thin, probe=512, thin_cap=64)
    return t, level_corrections(chain.draw, n0, levels)


def run_pilot(kernel, args, M):
    seeds = np.random.SeedSequence(args.seed).spawn(args.pasts)
    workers = args.workers or min(args.pasts, os.cpu_count() or 1)
    print(f'pilot: {args.pasts} pasts, levels 1..{args.pilot}, n0={args.n0}, '
          f'thin={args.thin}')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(
            _one_pilot,
            *zip(*[(kernel, args.sigma, M, args.thin, args.n0, args.pilot, s)
                   for s in seeds])))
    thins = np.array([t for t, _ in results])
    deltas = np.array([d for _, d in results])
    if args.thin == 'auto':
        print(f'  resolved thin {thins.min()}-{thins.max()}')
    rms = np.sqrt((deltas ** 2).mean(axis=0))
    for m in range(args.pilot):
        note = ''
        if m > 0 and rms[m] > 0:
            note = f'   decay exponent {np.log2(rms[m - 1] / rms[m]) * 2:.2f}'
        print(f'  m={m + 1}: rms Delta = {rms[m]:.5f}{note}')
    print('choose r safely below the E[Delta^2] decay exponent (and > 1); '
          'r=1.5 suits decay near 2.')


if __name__ == '__main__':
    main()
