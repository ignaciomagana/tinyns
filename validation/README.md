# tinyns validation harness

This directory contains a small validation harness for `tinyns`. It is intended
for scientific reliability checks, not speed benchmarking.

The analytic targets check whether repeated nested-sampling runs produce
reasonable evidence estimates and posterior moments. The stress targets, such as
banana and eggbox likelihoods, are qualitative probes that can reveal obvious
sampler failures, mixing problems, or suspicious diagnostics.

Run multiple seeds whenever possible. A single run can look good or bad by
chance, and the reported `logzerr` is only a lightweight nested-sampling
diagnostic; it should not be blindly trusted from one run.

Suggested workflow:

```bash
python validation/run_validation.py --output validation_results.json
python validation/summarize_validation.py validation_results.json
```

Every run uses the default sampler (live-cov rwalk in jitted blocks). The
harness exposes the sampler's own knobs: `--walks` (default: the sampler
default, `max(25, 6 * ndim, ndim**2 // 6)`), `--replacement-chains` and `--block-size`.

Recommended smoke-to-medium validation command:

```bash
python validation/run_validation.py \
  --targets gaussian2d correlated_gaussian2d ring2d banana2d eggbox2d \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --nlive 200 \
  --dlogz 0.1 \
  --output validation_fulltargets_nlive200_dlogz01.json

python validation/summarize_validation.py \
  validation_fulltargets_nlive200_dlogz01.json
```

Batched replacement-chain validation example:

```bash
python validation/run_validation.py \
  --targets gaussian2d correlated_gaussian2d \
  --replacement-chains 16 \
  --seeds 0 1 2 3 4 \
  --nlive 200 \
  --dlogz 0.1 \
  --output validation_chains16.json
```


## Interpreting validation summaries

The validation harness is meant to detect reliability problems across repeated
seeds. A single `success=True` run does not guarantee calibrated evidence
estimates.

Useful warning signs:

- repeated `|z| > 2` or any `|z| > 3` on analytic targets
- coverage much lower than expected
- high final live-point weight fraction
- high maximum posterior weight fraction
- low posterior weight entropy
- large posterior mean or covariance errors

The recommendation column is heuristic and should be treated as a debugging aid,
not a formal statistical test.

`ring2d` is a qualitative annulus target. It is useful for checking whether constrained-replacement samplers can move around curved shell-like likelihood regions. If no analytic evidence is provided, use posterior diagnostics, insertion-rank behavior, and repeated-run stability rather than z-scores.


## Insertion-rank diagnostics

Insertion ranks test whether new constrained replacements are distributed like draws from the constrained prior. For normalized ranks, the expected mean is about 0.5 and the expected standard deviation is about 1/sqrt(12).

Large systematic deviations can indicate biased or poorly mixed constrained replacement samplers, even when a run reports `success=True`.

Insertion-rank mean errors should be interpreted with their sampling uncertainty. A small absolute rank-mean error can be significant when thousands of insertion ranks are recorded. Validation summaries therefore report insertion-rank z-scores.

Qualitative targets such as `ring2d` may not have analytic evidence references. For these targets, inspect posterior summaries, insertion-rank behavior, replacement failures, and seed-to-seed stability.

## Comparing sampler settings

Use `validation/compare_validation.py` to compare settings such as
`walks=25` and `walks=50`.

A setting is not better merely because it performs more accepted local moves.
Prefer settings that improve coverage, reduce large evidence z-scores, and keep
likelihood-call cost reasonable.

To compare two validation runs, for example `--walks 25` versus
`--walks 50`:

```bash
python validation/compare_validation.py \
  validation_walks25.json validation_walks50.json \
  --labels w25 w50
```

The comparison table reports changes in coverage, absolute evidence error,
large-z outlier fraction, maximum z-score, and likelihood-call cost. The verdict
column is heuristic and should be treated as a debugging aid.
