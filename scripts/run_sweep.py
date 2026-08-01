"""Run the cached-estimate grid and fold the results into the data branch.

    python scripts/run_sweep.py --data-dir data

Appends one line per cell to <data-dir>/runs.jsonl and rebuilds
<data-dir>/estimates.json from the whole log. Each dispatch draws fresh,
independent pasts (the seed advances with the number of runs already recorded
for that cell), so the pooled mean tightens run after run.

Driven by .github/workflows/estimates.yml, but works standalone: point it at
any directory and run it as often as you like.
"""

import argparse
import json
import math
import os
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import grid  # first: also puts src/ on sys.path for a bare checkout
from timeseries_entropy import estimate_conditional_entropy
from timeseries_entropy.theory import predict_entropy_rate

RUNS = 'runs.jsonl'
SUMMARY = 'estimates.json'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', required=True,
                    help='checkout of the estimates branch')
    ap.add_argument('--pasts', type=int, default=8,
                    help='independent pasts added per cell per run')
    ap.add_argument('--reps', type=int, default=8,
                    help='randomized realizations averaged per past')
    ap.add_argument('--n0', type=int, default=128)
    ap.add_argument('--r', type=float, default=1.5)
    ap.add_argument('--thin', type=int, default=1)
    ap.add_argument('--workers', type=int)
    ap.add_argument('--only', help='substring of cell id, for local testing')
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    records = read_records(data_dir / RUNS)
    runs_so_far = {}
    for rec in records:
        runs_so_far[rec['cell']] = runs_so_far.get(rec['cell'], 0) + 1

    cells = [c for c in grid.grid()
             if args.only is None or args.only in grid.cell_id(c)]
    print(f'{len(cells)} cells x {args.pasts} pasts x {args.reps} reps '
          f'({len(records)} records already cached)')

    new = []
    t_all = time.time()
    for i, params in enumerate(cells):
        cid = grid.cell_id(params)
        run_index = runs_so_far.get(cid, 0)
        kernel = grid.kernel_from_params(params)
        M = max(512, 4 * kernel.size)
        seed = [zlib.crc32(cid.encode()), run_index]
        t0 = time.time()
        est = estimate_conditional_entropy(
            kernel, params['sigma'], past=M, pasts=args.pasts, reps=args.reps,
            n0=args.n0, r=args.r, thin=args.thin, seed=seed,
            workers=args.workers)
        v = np.asarray(est.per_past, dtype=float)
        rec = {
            'cell': cid,
            'params': params,
            'M': M,
            'L': int(kernel.size),
            'pasts': int(v.size),
            'reps': args.reps,
            'n0': args.n0,
            'r': args.r,
            'thin': args.thin,
            'sum': float(v.sum()),
            'sumsq': float((v ** 2).sum()),
            'seed': seed,
            'run_index': run_index,
            'utc': utc_now(),
            'code_sha': os.environ.get('GITHUB_SHA'),
            'workflow_run': os.environ.get('GITHUB_RUN_ID'),
        }
        new.append(rec)
        print(f'  [{i + 1:2d}/{len(cells)}] {cid:<16} '
              f'H = {est.mean:.4f} +/- {est.se:.4f}   '
              f'({time.time() - t0:.0f}s)', flush=True)

    append_records(data_dir / RUNS, new)
    summary = summarize(records + new, args)
    write_json(data_dir / SUMMARY, summary)
    write_readme(data_dir / 'README.md', summary)
    print(f'\n{len(new)} records appended in {time.time() - t_all:.0f}s; '
          f'{len(summary["cells"])} cells summarized')
    step_summary(summary)


def read_records(path):
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def append_records(path, records):
    with path.open('a') as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + '\n')


def summarize(records, args):
    """Pool every recorded per-past value, per cell, into a mean and an SE."""
    cells = {}
    for rec in records:
        c = cells.setdefault(rec['cell'], {
            'id': rec['cell'], 'n_pasts': 0, 'n_runs': 0,
            '_sum': 0.0, '_sumsq': 0.0})
        c['params'] = rec['params']
        c['M'] = rec['M']
        c['L'] = rec['L']
        c['n_pasts'] += rec['pasts']
        c['n_runs'] += 1
        c['_sum'] += rec['sum']
        c['_sumsq'] += rec['sumsq']
        c['last_utc'] = max(rec['utc'], c.get('last_utc', ''))

    out = []
    for c in sorted(cells.values(), key=lambda c: c['id']):
        n, s, ss = c['n_pasts'], c.pop('_sum'), c.pop('_sumsq')
        mean = s / n
        # Sample variance of the pooled per-past values; the pasts are
        # independent across runs as well as within one, so the SE of their
        # mean is the honest error bar on the whole cache.
        var = max((ss - s * s / n) / (n - 1), 0.0) if n > 1 else float('nan')
        pred = predict_entropy_rate(grid.kernel_from_params(c['params']),
                                    c['params']['sigma'])
        c.update({
            'mean': mean,
            'se': math.sqrt(var / n) if n > 1 else None,
            'sd': math.sqrt(var) if n > 1 else None,
            'predicted': pred['corrected'],
            'highres_szego': pred['highres'] if math.isfinite(pred['highres'])
            else None,
        })
        out.append(c)

    return {
        'generated': utc_now(),
        'estimand': 'H(z_{M+1} | z_1..z_M) in bits/sample, for '
                    'x ~ iid N(0, sigma^2) -> y = h * x -> z = round(y)',
        'total_runs': max((c['n_runs'] for c in out), default=0),
        'settings': {'pasts_per_run': args.pasts, 'reps': args.reps,
                     'n0': args.n0, 'r': args.r, 'thin': args.thin},
        'cells': out,
    }


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def utc_now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def rows(summary):
    for c in summary['cells']:
        se = '' if c['se'] is None else f"{c['se']:.4f}"
        yield (c['id'], f"{c['params']['sigma']:g}", str(c['M']),
               f"{c['mean']:.4f}", se, str(c['n_pasts']),
               f"{c['predicted']:.4f}")


HEAD = ('cell', 'sigma', 'M', 'mean', 'se', 'pasts', 'predicted')


def table(summary):
    body = [HEAD, tuple('-' * 3 for _ in HEAD)] + list(rows(summary))
    return '\n'.join('| ' + ' | '.join(r) + ' |' for r in body)


def write_readme(path, summary):
    path.write_text(f"""\
# Cached entropy estimates

Machine-generated by [`.github/workflows/estimates.yml`][wf] on `main` — do not
edit by hand. Every dispatch runs the whole grid again with fresh, independent
pasts, appends one record per cell to `runs.jsonl`, and rebuilds
`estimates.json` from the full log, so the means tighten run after run.

- **`runs.jsonl`** — append-only, one JSON object per (cell, dispatch): the
  cell's parameters, how many pasts it contributed, their sum and sum of
  squares, the seed, and the code commit that produced them.
- **`estimates.json`** — pooled `mean` and `se` per cell over every past ever
  drawn for it, alongside the analytic `predicted` rate for comparison.

The estimand is H(z_(M+1) | z_1..z_M) in bits/sample — an upper bound on the
entropy rate of z that tightens as the conditioning window M grows.

Fetch it from a browser or a script (raw.githubusercontent.com sends
`access-control-allow-origin: *`):

    https://raw.githubusercontent.com/concept-collection/timeseries-entropy/estimates/estimates.json

## Current state ({summary['total_runs']} runs, generated {summary['generated']})

{table(summary)}

[wf]: https://github.com/concept-collection/timeseries-entropy/blob/main/.github/workflows/estimates.yml
""")


def step_summary(summary):
    path = os.environ.get('GITHUB_STEP_SUMMARY')
    if not path:
        return
    with open(path, 'a') as f:
        f.write(f"## Cached estimates after {summary['total_runs']} runs\n\n"
                f'{table(summary)}\n')


if __name__ == '__main__':
    main()
