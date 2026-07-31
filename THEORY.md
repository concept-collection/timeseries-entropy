# Predicting the entropy rate of a quantized filtered Gaussian series

The model is

$$
x_t \overset{\text{iid}}{\sim} \mathcal N(0,\sigma^2)
\;\longrightarrow\;
y_t = \sum_j h_j\, x_{t-j}
\;\longrightarrow\;
z_t = \mathrm{round}(y_t),
$$

with unit quantization step. The quantity of interest is the entropy rate

$$
\bar H \;=\; \lim_{M\to\infty} H\!\left(z_{M+1}\mid z_1,\dots,z_M\right)
\qquad\text{[bits/sample]},
$$

the true lossless compression limit of $z$, which the Monte-Carlo estimator
in this package approaches from above as the window $M$ grows. This note
derives the analytic prediction implemented in
[`theory.py`](src/timeseries_entropy/theory.py),

$$
\boxed{\;\bar H \;\approx\; G(s_*),
\qquad
s_*^2 \;=\; \exp\!\left(\int_0^1 \ln\!\big(\sigma^2\lvert H(f)\rvert^2 + \tfrac1{12}\big)\, df\right) \;-\; \frac1{12},\;}
$$

where $H(f) = \sum_j h_j e^{-2\pi i f j}$ are the Fourier modes of the
kernel and $G(s) = h\big(\mathcal N(0,s^2) + \mathcal U(-\tfrac12,\tfrac12)\big)$
is the differential entropy (in bits) of a Gaussian convolved with a unit
uniform,

$$
G(s) \;=\; -\int_{-\infty}^{\infty} g_s(v)\,\log_2 g_s(v)\; dv,
\qquad
g_s(v) \;=\; \Phi\!\left(\frac{v + \tfrac12}{s}\right) - \Phi\!\left(\frac{v - \tfrac12}{s}\right),
$$

with $\Phi$ the standard normal CDF. The formula is built in three steps,
each repairing a failure of the previous one.

## Step 1 — Szegő–Kolmogorov: prediction from the exact past

$y$ is stationary Gaussian with power spectral density
$S(f) = \sigma^2 \lvert H(f)\rvert^2$, $f\in[0,1)$. Kolmogorov's form of
Szegő's theorem says the one-step prediction error variance from the
infinite (exact) past is the *geometric mean* of the spectrum:

$$
\sigma_\infty^2
= \exp\!\left(\int_0^1 \ln S(f)\, df\right)
= \sigma^2 \exp\!\left(\int_0^1 \ln \lvert H(f)\rvert^2\, df\right).
$$

Since a Gaussian process's entropy rate is the entropy of its innovation,

$$
\bar h(y) = \tfrac12\log_2\!\big(2\pi e\, \sigma_\infty^2\big),
$$

and in the high-resolution regime ($\sigma_\infty \gg$ 1 bin) the usual
approximation $H(\mathrm{round}(Y)) \approx h(Y) - \log_2\Delta$ with
$\Delta = 1$ gives the naive prediction

$$
\bar H \;\approx\; \tfrac12\log_2\!\big(2\pi e\,\sigma_\infty^2\big).
$$

**Closed form for FIR kernels.** Writing the tap polynomial
$P(w) = \sum_j h_j w^j = c \prod_k (w - b_k)$, Jensen's formula gives the
geometric mean of $\lvert w - b\rvert$ over the unit circle as
$\max(1, \lvert b\rvert)$, so

$$
\sigma_\infty = \sigma\, \lvert c\rvert \prod_k \max\big(1, \lvert b_k\rvert\big).
$$

Examples: the moving average of width $W$ has all zeros on the unit circle
and leading coefficient $1/W$, so $\sigma_\infty = \sigma/W$; the first
difference $h = (1,-1)$ has $\int_0^1 \ln(4\sin^2\pi f)\,df = 0$, so
$\sigma_\infty = \sigma$; the identity kernel has $\sigma_\infty = \sigma$.

**Two failures.** (i) Wherever $\lvert H(f)\rvert \approx 0$ — the stopband
of a lowpass or bandpass filter — the log integral dives toward $-\infty$
and the formula predicts *negative* entropy ($-4.8$ bits for the
$f_c = 0.1$ lowpass at $\sigma = 8$), while the truth is $\ge 0$. (ii) The
predictor only sees the *quantized* past, which carries strictly less
information than the exact past.

## Step 2 — Quantized past: the $1/12$ noise floor

Model roundoff as additive dither: $z_t = y_t + u_t$ with
$u_t \overset{\text{iid}}{\sim} \mathcal U(-\tfrac12,\tfrac12)$,
independent of $y$ (Bennett's approximation; exact under subtractive
dither). The observed process $w = y + u$ then has spectrum

$$
S_w(f) = S(f) + \tfrac1{12}.
$$

Kolmogorov's theorem is a statement about *linear* prediction and needs no
Gaussianity, so the one-step linear prediction error of $w$ from its past
is $\exp \int_0^1 \ln S_w$. Because $u_{t+1}$ is independent of both
$y_{t+1}$ and the past of $w$, its variance splits off exactly:

$$
\mathrm{Var}\big(w_{t+1}\mid w_{\le t}\big)
= \mathrm{Var}\big(y_{t+1}\mid w_{\le t}\big) + \tfrac1{12}
\quad\Longrightarrow\quad
s_*^2 = \exp\!\left(\int_0^1 \ln\!\big(S(f) + \tfrac1{12}\big) df\right) - \frac1{12}.
$$

This $s_*$ is the effective uncertainty of the next sample given the
quantized past. Three properties worth noting:

- **Regularization.** The $\tfrac1{12}$ inside the logarithm is exactly
  what keeps the integral finite at zeros of $H(f)$ — the fix for failure
  (i) falls out of modeling failure (ii).
- **Ordering.** $s_*^2 \ge \sigma_\infty^2$ always, because geometric means
  are superadditive: quantizing the past can only increase the entropy
  rate. The gap is a real, measurable effect even with no stopband (for the
  first difference at $\sigma = 2$ it is $+0.10$ bits, confirmed by Monte
  Carlo).
- **Limits.** $s_*^2 \to \sigma_\infty^2$ when $S \gg \tfrac1{12}$
  everywhere, and $s_* \to 0$ as $\sigma \to 0$ (for the identity kernel,
  $s_* = \sigma$ *exactly*).

## Step 3 — Quantized next sample: Gaussian ⊛ uniform entropy

Given the quantized past, $y_{t+1} \approx \mathcal N(m, s_*^2)$ with a
conditional mean $m$ that varies from past to past. When the marginal
spread of $y$ covers many bins, $m \bmod 1$ equidistributes, so

$$
\bar H \;\approx\; \mathbb E_{c\sim\mathcal U(0,1)}\,
H\!\big(\mathrm{round}(c + \mathcal N(0, s_*^2))\big).
$$

This average has a closed form — the standard dithered-quantization
identity. For any $X$ with density, $\mathrm{round}(X + c) = k$ iff
$X \in [k - c - \tfrac12,\, k - c + \tfrac12)$, an event of probability
$g(k - c)$ where $g(v) = F_X(v + \tfrac12) - F_X(v - \tfrac12)$ is exactly
the density of $X + U$, $U \sim \mathcal U(-\tfrac12, \tfrac12)$. The
intervals $\{k - c : c \in (0,1)\}$ tile the line, so

$$
\mathbb E_c\, H\big(\mathrm{round}(X + c)\big)
= -\int_0^1 \sum_k g(k - c) \log_2 g(k - c)\, dc
= -\int_{-\infty}^{\infty} g \log_2 g
= h(X + U).
$$

For Gaussian $X$ define

$$
G(s) = h\big(\mathcal N(0,s^2) + U\big)
= -\int_{-\infty}^{\infty} g_s\log_2 g_s\,dv,
\qquad
g_s(v) = \Phi\!\left(\frac{v + \tfrac12}{s}\right) - \Phi\!\left(\frac{v - \tfrac12}{s}\right).
$$

Its limits are exactly the right ones:

$$
G(s) \;\to\; \tfrac12\log_2\!\big(2\pi e\,(s^2 + \tfrac1{12})\big)
\quad (s \gg 1),
\qquad
G(s) \;\sim\; C\,s \;\to\; 0
\quad (s \to 0),
$$

with $C = \int_{-\infty}^{\infty} h_2(\Phi(t))\,dt \approx 2.6061$
($h_2$ the binary entropy). So $G$ reproduces the high-resolution formula
when quantization is fine and saturates to $0$ — instead of diverging to
$-\infty$ — when the conditional distribution concentrates inside one bin.

## Validity and failure modes

Monte-Carlo validation with this package's estimator (24+ independent
pasts; `--thin 4` for the slowly mixing narrowband cases):

| filter | $\sigma$ | $s_*$ | $G(s_*)$ | naive Szegő | Monte Carlo $\pm$ se |
|---|---|---|---|---|---|
| none | 0.5 | 0.50 | 1.2544 | 1.047 | 1.2380 ± 0.0069 |
| none | 2 | 2.00 | 3.0620 | 3.047 | 3.0491 ± 0.0124 |
| none | 8 | 8.00 | 5.0480 | 5.047 | 5.0009 ± 0.0372 |
| first-diff | 0.5 | 0.60 | 1.4579 | 1.047 | 1.4579 ± 0.0053 |
| first-diff | 2 | 2.13 | 3.1511 | 3.047 | 3.1709 ± 0.0192 |
| first-diff | 8 | 8.14 | 5.0731 | 5.047 | 5.1004 ± 0.0489 |
| MA(8) | 1 | 0.23 | 0.6100 | −0.953 | 0.4962 ± 0.0220 |
| MA(8) | 2 | 0.38 | 0.9825 | 0.047 | 0.9881 ± 0.0092 |
| MA(8) | 4 | 0.65 | 1.5529 | 1.047 | 1.5641 ± 0.0191 |
| MA(8) | 32 | 4.18 | 4.1125 | 4.047 | 4.1420 ± 0.0429 |
| lowpass $f_c$=0.1 | 8 | 0.50 | 1.2632 | −4.756 | 1.2232 ± 0.0190 (M=512), 1.2732 ± 0.0422 (M=1024) |
| lowpass $f_c$=0.1 | 64 | 0.89 | 1.9513 | −1.756 | 1.8258 ± 0.0796 |
| bandpass 0.01–0.2 | 8 | 1.03 | 2.1476 | −1.783 | 2.1362 ± 0.0217 |

The approximations, and where they bite:

1. **Dither independence** (Step 2) requires the marginal spread
   $\sigma\lVert h\rVert_2$ to be at least about one bin. First-difference
   at $\sigma = 0.5$ (spread 0.71 bins) still agrees to within its se;
   MA(8) at $\sigma = 1$ (spread 0.35 bins) is overpredicted by
   $\approx 0.11$ bits — when the whole signal lives inside one bin,
   roundoff is deterministic, not dither-like, and the true rate is lower.
2. **Equidistribution of the conditional mean** (Step 3) fails for kernels
   with no memory: the identity kernel pins $m = 0$, and the exact answer
   is the *centered* quantized-Gaussian entropy, below $G(\sigma)$ by
   $\approx 0.013$ bits at $\sigma = 0.5$ (and exponentially little for
   $\sigma \gtrsim 1$). Any kernel with real memory washes this out.
3. **Linear prediction / Gaussianity of $w$** (Step 2): $w$ is not
   Gaussian, and linear prediction of it is not optimal, so $s_*$ errs
   slightly high; the effect is within the Monte-Carlo error bars above.
4. **Near-singular spectra** (lowpass/bandpass) have long memory; the
   Monte-Carlo column is an upper bound that decreases in $M$, and the
   Gibbs sampler mixes slowly (hence `--thin`). The $M = 1024$ lowpass run
   agrees with the prediction to well within its error bar.

## Numerical notes

- $\sigma_\infty$ is computed exactly from the roots of the tap polynomial
  (robust to zeros of $H$ *on* the unit circle, where the log integral is
  still convergent — an integrable singularity).
- The Step-2 integral uses the trapezoid rule on the rfft grid over
  $[0, \tfrac12]$ with $n = 2^{18}$ points; the $\tfrac1{12}$ floor makes
  the integrand smooth and strictly positive, so no special handling of
  spectral zeros is needed.
- $G(s)$ is integrated on a grid of spacing $\min(s/8, 0.01)$; below
  $s = 10^{-3}$ the linear asymptote $C s$ is used.
