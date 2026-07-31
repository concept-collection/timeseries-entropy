# timeseries-entropy

Unbiased Monte-Carlo estimation of the entropy of a quantized filtered
Gaussian time series:

    x iid N(0, sigma^2)  ->  y = h * x  ->  z = round(y)

The estimand is the conditional entropy H(z_{M+1} | z_1..z_M) in bits, which
decreases to the entropy rate of z — the true lossless compression limit in
bits/sample — as the past window M grows beyond the memory of the process.

Companion to [timeseries-compressibility](https://github.com/concept-collection/timeseries-compressibility),
which carries a hand-synced TypeScript port of this estimator in `src/entropy/`
(run in the browser from a worker) — when changing the algorithm here, change
it there too.

## Method

1. **Stationary conditional sampling.** Draw x from the prior and push it
   through the pipeline to get a past z_1..z_M. The generating x is an exact
   draw from p(x | z), so a Gibbs chain started there is already in
   stationarity — no burn-in bias. Each Gibbs conditional is a box-truncated
   normal; after each sweep the free tail latent is drawn fresh, emitting one
   exact sample of z_{M+1}. The samples form a stationary, autocorrelated
   discrete chain.
2. **Unbiased entropy of the chain's marginal.** Plug-in entropies of blocks
   whose sizes double per level are combined by Rhee–Glynn randomized
   telescoping with antithetic half-block corrections
   Delta_m = h(B_m) - [h(B_m^1) + h(B_m^2)]/2, truncated at a random level N
   with P(N >= m) = 2^(-r m) and reweighted. The expectation is exactly
   H(z_{M+1} | that past) despite the plug-in bias at every finite block size
   and despite the autocorrelation, which affects only the variance.
3. **Average over independent pasts** to get H(z_{M+1} | z_1..z_M) with a
   valid standard error. The only remaining approximation to the entropy rate
   is the finite window M.

## Install

    pip install -e .

Requires numpy and scipy.

## Usage

```python
from timeseries_entropy import estimate_conditional_entropy, kernels

est = estimate_conditional_entropy(kernels.moving_average(8), sigma=4.0)
print(est.mean, est.se)   # bits/sample, over independent pasts
```

Lower level: `ConditionalChain(kernel, sigma, past, rng).draw(k)` yields the
stationary chain of z_{M+1} samples, and `unbiased_entropy(draw, n0, r, rng)`
is one randomized-telescoping realization for any stationary discrete chain.

## CLI

    timeseries-entropy --sigma 4 --filter moving-average --width 8
    timeseries-entropy --sigma 8 --filter lowpass --high 3000 --rate 30000
    timeseries-entropy --sigma 2 --filter none --pasts 8

Filters match the web app: `none`, `moving-average`, `lowpass`, `bandpass`,
`first-difference`.

## Tuning

- `r` (default 1.5) sets the truncation tail P(N >= m) = 2^(-r m). Finite
  expected work needs r > 1; finite variance needs E[Delta_m^2] to decay
  faster than 2^(-r m). Run `--pilot 6` to see the RMS Delta_m decay before
  trusting a value.
- `n0` (default 128) is the base block size; `thin` inserts extra Gibbs
  sweeps per emitted sample to cut autocorrelation for slowly mixing
  (narrowband, large-sigma) settings.
- `--past` sets M; increase it until the estimate stops moving to approach
  the rate.
