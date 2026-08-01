# Cached entropy estimates

Cache reset on 2026-08-01: estimates recorded before then were drawn without
auto-thinning, whose heavy-tailed outliers on slow-mixing cells polluted the
pooled means (one telescoping realization on `bp300-6000_s32` returned 305
bits). The [estimates workflow][wf] repopulates this branch on its next run;
see the main README's Tuning section for the details.

[wf]: https://github.com/concept-collection/timeseries-entropy/blob/main/.github/workflows/estimates.yml
