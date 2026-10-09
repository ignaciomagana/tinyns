# Head-to-head results

tinyns v1 against tinyns 0.x, BlackJAX NSS, JAXNS, dynesty, UltraNest and
Nautilus on the 24 standard targets of `bench/targets.py`, measured in
October 2026. `bench/README.md` describes the harness, the targets and the
columns. Every number below comes from `bench/summarize.py`:

```bash
python bench/summarize.py results_gpu.jsonl results_cpu.jsonl results_probe.jsonl \
    --sha tinyns_v1=afa58dd --reference-nlive 500,1000 --out report.md
```

The record files are not in the repository.

The tinyns v1 rows were measured at afa58dd, one merge before b67b458 (which
lowers the JAX floor to 0.4.31 and removes three optimization barriers). The
two differ only in roundoff: the quick validation gate passes 7 of 7 cases on
both, with the same printed digits. The rows were not rerun.

## What ran

| sampler | version | settings |
|---|---|---|
| `tinyns_v1` | v1 at afa58dd | defaults: `num_delete = nlive // 10`, `walks = max(25, 6 d, d^2 // 6)`, `dlogz = 0.1`; nlive 500 and 1000 (the default) on every target, 2000 and 4000 on the mixtures |
| `tinyns_v02` | `release/0.x` at 2f67111 (v0.2.5 behaviour) | defaults, cluster swap on; nlive 500, `dlogz = 0.1` |
| `blackjax_nss` | blackjax 1.7.1 | nlive 500, `num_delete = 50`, `num_inner_steps = max(5, 2 d)`, `dlogz = 0.1` |
| `jaxns` | jaxns 3.0.0 | all defaults: `30 d` chains, `5 d` slices, stop at `dlogZ = log1p(1e-3)` |
| `jaxns:nlive` | jaxns 3.0.0 | `root_allocation_degree = 500`, else defaults |
| `dynesty` | dynesty 3.1.0 | nlive 500, `bound='multi'`, `sample='auto'`, `dlogz = 0.1` |
| `dynesty:rwalk100` | dynesty 3.1.0 | nlive 500, `sample='rwalk'`, `walks=100`, `dlogz = 0.1` |
| `ultranest` | ultranest 4.5.2 | MLFriends, `min_num_live_points = 500`, its own `dlogz = 0.5` and `frac_remain = 0.01`; d <= 10 |
| `ultranest:slice` | ultranest 4.5.2 | the same with `SliceSampler(nsteps = 2 d)`; d > 10 |
| `nautilus` | nautilus-sampler 1.0.6 | defaults: `n_live = 2000`, `f_live = 0.01`, `n_eff = 10000`, no pool |
| `nautilus:discard` | nautilus-sampler 1.0.6 | the same with `discard_exploration=True` |

- **Seeds.** 20 per target, 40 for the mixtures (`sepM_d10` and `connW_d32`
  have 20). The CPU samplers have 10 at d >= 32.
- **Precision.** x64 everywhere.
- **Hardware.** The GPU samplers (tinyns, BlackJAX, JAXNS) ran on one NVIDIA
  H100, one run at a time. The CPU samplers (dynesty, UltraNest, Nautilus) ran
  on 4 cores of a Xeon E5-2695 v3 each and call the jitted JAX likelihood from
  Python. **Wall times on the GPU and on the CPU are not like for like**: they
  are the times a user of each package would see on that hardware, with a
  likelihood that costs microseconds.
- **Timeouts.** 1 h per run on the GPU, 4 h on a CPU. A timed-out run is
  counted, not dropped.
- **Times.** Every seed ran in a fresh process with a persistent JAX
  compilation cache. `wall s` is the median over seeds, so it includes
  loading the compiled programs from the cache (2.3 s for tinyns v1) but not
  compiling them. `run s` is the time of the compiled run (tinyns and
  BlackJAX time their compilation apart; JAXNS does not). `first s` is the
  first seed of the cell, which compiled everything: a cold start.
- **Likelihood calls** are what each sampler counts. tinyns counts every
  evaluation, the out-of-cube lanes of its vmapped chains included; BlackJAX
  and JAXNS do not count the lanes that idle inside a vmap.
- **Accurate** is defined in `bench/README.md`: at least 90% of the runs
  finished, the logZ bias is within 3 standard errors or 0.1 nats, and on a
  mixture at most 10% of the seeds lost a mode and the logit bias of the
  mode weight is within 3 standard errors or 0.1.
- **Fairness probe.** The rows labelled `[key=value]`, `dynesty` at nlive
  1600 and `ultranest:slice` at d = 10 are a small probe of settings that
  suit a competitor better than the sweep's (5 or 10 seeds each). It is not
  a full rerun; see Known gaps.

## Reading

**Where tinyns v1 stands at the competitors' settings (nlive 500 and its
default 1000).**

- It is accurate on 18 of 24 targets at nlive 500 and on 22 of 24 at 1000.
  The failures at 500 are `rosen_d10` (logZ low by 0.22 ± 0.06), `sepW_d32`
  and `connW_d32` (the minor mode lost in 25% and 30% of the seeds) and the
  three `sepM` targets. At 1000 only `sepM_d18` and `sepM_d32` fail.
- Its error bars are honest: sd(logZ) / logzerr is between 0.69 and 1.32 in
  all 72 cells (nlive 2000 and 4000 included).
- No run of tinyns v1 timed out or raised an error (2240 runs).
- **Time.** A compiled run takes 1.1 to 20 s on the H100 and a first run 9 to
  31 s. BlackJAX NSS takes 9 to 77 s compiled, JAXNS 16 to 360 s with its
  programs cached (7 s to 19 min at its default chain count), and tinyns 0.x
  6 to 510 s. The CPU samplers take from 6 s (dynesty, 2-D) to more than 4 h
  on 4 cores; that is a different machine and a likelihood called from
  Python, so it measures a workflow and not an algorithm.
- **Likelihood calls.** tinyns v1 is not the sampler with the fewest calls.
  - Nautilus needs 1.7 to 24 times fewer calls than tinyns at nlive 500, and
    reaches an rms 2 to 28 times smaller with them. Where it is accurate it
    is by far the best choice for an expensive likelihood. It timed out on
    `gauss_d64`, lost the minor mode on every `sepM` target and on both
    mixtures at d = 32, and its default (which keeps the exploration
    points) has a logZ bias of up to -0.22 that its own precision resolves.
  - dynesty's default and UltraNest's MLFriends need 1.5 to 13 times fewer
    calls than tinyns at d <= 10 on the unimodal targets.
  - BlackJAX NSS needs 0.9 to 1.9 times the calls of tinyns and 4 to 8
    times its compiled time; JAXNS with 500 chains needs 2.2 to 6.6 times
    the calls.
  - tinyns v1 spends 1.0 to 1.9 times the calls of tinyns 0.x (it counts the
    out-of-cube lanes of its vmapped chains) and is 4 to 45 times faster
    compiled.
- **High dimensions.** On `gauss_d64` tinyns v1 is unbiased (+0.21 ± 0.13 at
  nlive 500, -0.04 ± 0.12 at 1000), as is JAXNS (+0.06 ± 0.10 with 2.8
  times the calls and 19 times the time). BlackJAX NSS is high by 3.3,
  dynesty by 7.4 (12.1 with `rwalk`), UltraNest's slice sampler by 1.6 and
  tinyns 0.x by 0.9, at these settings (see Known gaps: their chains are
  shorter than tinyns's at d >= 30).
- **Mode weights.** On the well-separated and the touching mixtures up to
  d = 18 the logit of the 6% mode's weight scatters by 0.08 to 0.47 at nlive
  500 and by 0.06 to 0.08 at 1000, and one seed in all loses the mode
  (`connW_d18` at 500). At d = 10 and 18 BlackJAX NSS, JAXNS and dynesty
  scatter by 0.4 to 1.2 and lose the mode in up to 30% of the seeds; at
  d = 4 they are closer (0.08 to 0.45). Nautilus (0.03 to 0.22) and
  UltraNest's MLFriends at d <= 10 (0.07 to 0.09) are as tight as tinyns or
  tighter; MLFriends pays for it with 6
  to 11 times the calls at d = 10 and timeouts (5 of 40 on `sepW_d10`, all
  40 on `mix3_d10`).

**`sepM`: the narrow minor mode.** Its 6% mode has 0.03 of the main mode's
volume.

- At nlive 500 and 1000 tinyns v1 is accurate on 1 of the 6 `sepM` cells
  (`sepM_d10` at 1000: logit sd 0.04, the mode lost in 1 seed of 20). On the
  other five it loses the mode in 20 to 70% of the seeds.
- No competitor is accurate on any `sepM` target at the settings of this
  sweep: BlackJAX NSS, dynesty and UltraNest's slice sampler lose the mode
  in 30 to 70% of the seeds, Nautilus in 95 to 100%, and UltraNest's
  MLFriends timed out on 12 of 20 seeds of `sepM_d10`. tinyns 0.x and JAXNS have not run them
  yet. In the probe, Nautilus with `n_live=8000` is accurate on `sepM_d10`
  (5 seeds, logit sd 0.07, 5.6e5 calls).
- With more live points tinyns recovers the mode once it holds about
  `5 * ndim` of them (the rule in the README's Limitations; the mode holds
  `0.06 * nlive`):
  - `sepM_d10` works from nlive 1000: 2.4e6 calls and 2.0 s compiled. At
    2000 (4.9e6 calls, 2.1 s) no seed loses the mode.
  - `sepM_d18` works at nlive 2000: logit sd 0.03, no seed of 40 lost,
    1.5e7 calls and 4.1 s compiled. That is 4 times the calls of nlive 500
    and 0.8 s more time.
  - `sepM_d32` still fails at 1000 (lost in 55% of the seeds) and at 2000
    (28%; its mode holds 120 live points against `5 * ndim` = 160). It works
    at nlive 4000: logit sd 0.08, the mode lost in 1 seed of 40, 9.3e7 calls
    and 10.3 s compiled. That is 8 times the calls of nlive 500 and 3.8 s
    more time.
  - At nlive 4000 all 12 mixtures are accurate, with a logit sd of 0.02 to
    0.13; at 2000 all but `sepM_d32` are.
- The same holds for the mixtures at d = 32: `sepW_d32` and `connW_d32`
  lose the mode in 25 to 30% of the seeds at 500, 3 to 5% at 1000 and none
  at 2000 (logit sd 0.04 and 0.23; 4.6e7 calls, 8.2 s compiled) or 4000.
- The cost of `nlive` is in calls, not in time: the calls double with
  `nlive` and the compiled time on the GPU grows by less than 25% per
  doubling.

## Known gaps

The comparison is not like for like in these respects. The first four are
reruns that were deferred; the probe says which way each would move.

- **Nautilus without a pool, at its default `n_live`.** Its wall times
  (1 min to more than 4 h) are for 4 cores with its networks trained one
  after the other; with `pool` it trains them in parallel. Its lost modes
  are at `n_live=2000`: at 8000 it resolves `sepM_d10`.
- **UltraNest at d = 10 ran MLFriends**, which timed out on `rosen_d10` and
  `mix3_d10` and on 12 of 20 seeds of `sepM_d10`. Its slice sampler, which
  the sweep used only above d = 10, finishes all of them in about 4 min (10
  seeds): accurate on `rosen_d10`, `sepW_d10`, `connW_d10` and `mix3_d10`,
  with a logit sd of 0.4 to 0.7 against 0.07 for MLFriends.
- **Chain lengths at d >= 30 are not matched.** tinyns walks
  `max(25, 6 d, d^2 / 6)` steps: 192 at d = 32 and 682 at d = 64, against
  100 for `dynesty:rwalk100` and `2 d` slice steps for BlackJAX and
  UltraNest. With `walks=192` dynesty's `rwalk` is unbiased on `gauss_d32`
  (-0.05 ± 0.13, 5 seeds, 9.3e6 calls); with 384 it is still high by 1.8 on
  `gauss_d64`. The high-d biases of the other samplers above are therefore
  biases at their documented settings, not limits of the methods.
- **nlive 1000, 2000 and 4000 are tinyns only.** No competitor ran above
  nlive 500 (but for dynesty at 1600 on `gauss_d32`, still high by 0.36,
  and Nautilus's own 2000). The headline ratios use tinyns at 500 and 1000
  and correct for the difference with the costs at equal rms.
- **Missing cells.** tinyns 0.x, `jaxns` and `jaxns:nlive` have not run
  `connW`, `sepM` and `mix3` (and `jaxns:nlive` not `sepW_d32`).
- **Seeds.** 20 or 40 per cell (10 for the CPU samplers at d >= 32, 5 or 10
  in the probe): a logit sd has a relative error of 11 to 16%, and a lost
  fraction of 5% is one or two seeds.
- **One likelihood cost.** Every target costs microseconds. The ranking by
  wall time would change for a likelihood that fills the device by itself,
  where the count of calls is what matters.

## Validation gate

`python bench/validate.py --tier full` at afa58dd on the H100 (x64, 64 runs
per case, key 0). All 10 cases pass.

| case | nlive | walks | seeds | dlogZ ± se | z | scat/err [3 se] | logit bias ± se | logit sd | lost | modes | unres | ncall | wall s | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gauss_d2 | 1000 | 25 | 64 | -0.014 ± 0.009 | -1.6 | 0.96 [0.74, 1.26] | - | - | - | 1 (64/64 ok) | 0/64 | 2.11e+05 | 12 | PASS |
| gauss_d10 | 1000 | 60 | 64 | +0.001 ± 0.024 | +0.0 | 1.09 [0.83, 1.42] | - | - | - | 1 (64/64 ok) | 0/64 | 1.95e+06 | 14 | PASS |
| gauss_d32 | 1000 | 192 | 64 | +0.047 ± 0.035 | +1.3 | 0.92 [0.70, 1.20] | - | - | - | 1 (64/64 ok) | 0/64 | 1.83e+07 | 26 | PASS |
| rosen_d4 | 1000 | 100 | 64 | +0.010 ± 0.016 | +0.6 | 1.10 [0.84, 1.43] | - | - | - | 1 (64/64 ok) | 0/64 | 1.66e+06 | 12 | PASS |
| rosen_d10 | 1000 | 250 | 64 | -0.011 ± 0.026 | -0.4 | 1.03 [0.79, 1.35] | - | - | - | 1-2 (64/64 ok) | 5/64 | 1.07e+07 | 17 | PASS |
| funnel_d10 | 1000 | 60 | 64 | +0.011 ± 0.021 | +0.5 | 0.90 [0.69, 1.17] | - | - | - | 1 (64/64 ok) | 0/64 | 3.46e+06 | 12 | PASS |
| loggamma_d10 | 1000 | 60 | 64 | -0.009 ± 0.014 | -0.6 | 0.76 [0.58, 1.00] | +0.008 ± 0.005 | 0.038 | 0.00 | 4 (64/64 ok) | 0/64 | 1.42e+06 | 12 | PASS |
| eggbox_d2 | 1000 | 25 | 64 | +0.011 ± 0.012 | +1.0 | 1.17 [0.89, 1.52] | - | - | - | 15-18 (64/64 ok) | 30/64 | 2.28e+05 | 10 | PASS |
| sepW_d10 | 2000 | 60 | 64 | +0.026 ± 0.019 | +1.4 | 1.08 [0.83, 1.41] | -0.004 ± 0.005 | 0.039 | 0.00 | 2 (64/64 ok) | 0/64 | 4.8e+06 | 16 | PASS |
| sepW_d18 | 4000 | 108 | 64 | +0.005 ± 0.016 | +0.3 | 0.98 [0.75, 1.28] | -0.000 ± 0.003 | 0.021 | 0.00 | 2 (64/64 ok) | 0/64 | 3.02e+07 | 39 | PASS |

## tinyns v1 on the mixtures, by nlive

The competitors ran these targets at nlive 500 (Nautilus at its own 2000).
The rows at nlive 1000, 2000 and 4000 are tinyns only: they show where a
target starts to work and what it costs, and they are not a comparison.

| target | nlive | dlogZ ± se | logit bias ± se | logit sd | lost | ncall | run s | first s | accurate |
|---|---|---|---|---|---|---|---|---|---|
| sepW_d4 | 500 | -0.018 ± 0.028 | -0.01 ± 0.01 | 0.08 | 0.00 | 2.2e5 | 1.2 | 10 | yes |
| sepW_d4 | 1000 | +0.012 ± 0.017 | -0.01 ± 0.01 | 0.08 | 0.00 | 4.3e5 | 1.3 | 11 | yes |
| sepW_d4 | 2000 | +0.013 ± 0.013 | -0.00 ± 0.01 | 0.06 | 0.00 | 8.6e5 | 1.3 | 11 | yes |
| sepW_d4 | 4000 | +0.003 ± 0.009 | -0.00 ± 0.01 | 0.04 | 0.00 | 1.7e6 | 1.3 | 11 | yes |
| sepW_d10 | 500 | -0.026 ± 0.038 | -0.02 ± 0.02 | 0.14 | 0.00 | 1.2e6 | 1.9 | 10 | yes |
| sepW_d10 | 1000 | -0.016 ± 0.031 | -0.01 ± 0.01 | 0.06 | 0.00 | 2.4e6 | 2.0 | 10 | yes |
| sepW_d10 | 2000 | -0.040 ± 0.019 | -0.00 ± 0.00 | 0.03 | 0.00 | 4.8e6 | 2.1 | 11 | yes |
| sepW_d10 | 4000 | -0.005 ± 0.018 | -0.00 ± 0.00 | 0.02 | 0.00 | 9.6e6 | 2.4 | 12 | yes |
| sepW_d18 | 500 | +0.051 ± 0.048 | -0.14 ± 0.06 | 0.40 | 0.00 | 3.8e6 | 3.5 | 14 | yes |
| sepW_d18 | 1000 | -0.003 ± 0.039 | -0.01 ± 0.01 | 0.06 | 0.00 | 7.5e6 | 3.6 | 14 | yes |
| sepW_d18 | 2000 | +0.027 ± 0.031 | -0.00 ± 0.01 | 0.04 | 0.00 | 1.5e7 | 3.9 | 14 | yes |
| sepW_d18 | 4000 | +0.013 ± 0.018 | -0.00 ± 0.00 | 0.02 | 0.00 | 3.0e7 | 4.4 | 15 | yes |
| sepW_d32 | 500 | +0.110 ± 0.066 | -0.19 ± 0.14 | 0.79 | 0.25 | 1.2e7 | 7.4 | 16 | no: M |
| sepW_d32 | 1000 | -0.062 ± 0.055 | -0.01 ± 0.03 | 0.19 | 0.03 | 2.3e7 | 8.3 | 18 | yes |
| sepW_d32 | 2000 | +0.068 ± 0.031 | +0.00 ± 0.01 | 0.04 | 0.00 | 4.6e7 | 8.2 | 19 | yes |
| sepW_d32 | 4000 | -0.016 ± 0.035 | +0.00 ± 0.00 | 0.02 | 0.00 | 9.3e7 | 9.1 | 19 | yes |
| connW_d4 | 500 | +0.004 ± 0.022 | -0.04 ± 0.01 | 0.09 | 0.00 | 2.2e5 | 1.2 | 10 | yes |
| connW_d4 | 1000 | -0.007 ± 0.016 | -0.00 ± 0.01 | 0.06 | 0.00 | 4.3e5 | 1.3 | 10 | yes |
| connW_d4 | 2000 | -0.007 ± 0.013 | -0.02 ± 0.01 | 0.05 | 0.00 | 8.6e5 | 1.2 | 10 | yes |
| connW_d4 | 4000 | +0.008 ± 0.009 | -0.01 ± 0.01 | 0.04 | 0.00 | 1.7e6 | 1.3 | 10 | yes |
| connW_d10 | 500 | -0.068 ± 0.047 | -0.03 ± 0.02 | 0.13 | 0.00 | 1.2e6 | 1.9 | 10 | yes |
| connW_d10 | 1000 | -0.018 ± 0.025 | -0.01 ± 0.01 | 0.06 | 0.00 | 2.4e6 | 1.9 | 10 | yes |
| connW_d10 | 2000 | -0.022 ± 0.026 | -0.00 ± 0.01 | 0.04 | 0.00 | 4.8e6 | 2.0 | 10 | yes |
| connW_d10 | 4000 | +0.012 ± 0.017 | -0.00 ± 0.00 | 0.02 | 0.00 | 9.6e6 | 2.3 | 11 | yes |
| connW_d18 | 500 | -0.040 ± 0.050 | -0.12 ± 0.07 | 0.47 | 0.03 | 3.8e6 | 3.6 | 12 | yes |
| connW_d18 | 1000 | +0.022 ± 0.043 | -0.01 ± 0.01 | 0.07 | 0.00 | 7.5e6 | 3.6 | 12 | yes |
| connW_d18 | 2000 | -0.031 ± 0.027 | -0.01 ± 0.01 | 0.03 | 0.00 | 1.5e7 | 3.6 | 12 | yes |
| connW_d18 | 4000 | +0.017 ± 0.021 | +0.00 ± 0.00 | 0.02 | 0.00 | 3.0e7 | 4.0 | 13 | yes |
| connW_d32 | 500 | +0.232 ± 0.102 | -0.11 ± 0.32 | 1.19 | 0.30 | 1.2e7 | 7.9 | 16 | no: M |
| connW_d32 | 1000 | +0.110 ± 0.079 | -0.11 ± 0.15 | 0.65 | 0.05 | 2.3e7 | 8.7 | 18 | yes |
| connW_d32 | 2000 | +0.018 ± 0.050 | -0.01 ± 0.05 | 0.23 | 0.00 | 4.6e7 | 8.2 | 17 | yes |
| connW_d32 | 4000 | +0.016 ± 0.035 | +0.01 ± 0.03 | 0.13 | 0.00 | 9.3e7 | 8.3 | 17 | yes |
| sepM_d10 | 500 | +0.002 ± 0.058 | -0.06 ± 0.26 | 0.74 | 0.60 | 1.2e6 | 1.9 | 10 | no: M |
| sepM_d10 | 1000 | -0.020 ± 0.049 | +0.01 ± 0.01 | 0.04 | 0.05 | 2.4e6 | 2.0 | 10 | yes |
| sepM_d10 | 2000 | +0.016 ± 0.030 | +0.01 ± 0.01 | 0.03 | 0.00 | 4.9e6 | 2.1 | 10 | yes |
| sepM_d10 | 4000 | +0.018 ± 0.020 | -0.00 ± 0.00 | 0.02 | 0.00 | 9.7e6 | 2.4 | 11 | yes |
| sepM_d18 | 500 | +0.028 ± 0.075 | +0.02 ± 0.30 | 1.08 | 0.68 | 3.8e6 | 3.3 | 12 | no: M |
| sepM_d18 | 1000 | -0.023 ± 0.048 | -0.16 ± 0.07 | 0.40 | 0.20 | 7.5e6 | 4.0 | 12 | no: M |
| sepM_d18 | 2000 | -0.014 ± 0.033 | -0.01 ± 0.00 | 0.03 | 0.00 | 1.5e7 | 4.1 | 13 | yes |
| sepM_d18 | 4000 | -0.009 ± 0.020 | -0.01 ± 0.00 | 0.02 | 0.00 | 3.0e7 | 4.4 | 13 | yes |
| sepM_d32 | 500 | +0.148 ± 0.094 | +0.51 ± 0.43 | 1.48 | 0.70 | 1.2e7 | 6.5 | 16 | no: M |
| sepM_d32 | 1000 | -0.032 ± 0.046 | +0.26 ± 0.20 | 0.85 | 0.55 | 2.3e7 | 8.1 | 18 | no: M |
| sepM_d32 | 2000 | -0.002 ± 0.037 | -0.01 ± 0.06 | 0.34 | 0.28 | 4.6e7 | 9.2 | 17 | no: M |
| sepM_d32 | 4000 | -0.003 ± 0.032 | -0.01 ± 0.01 | 0.08 | 0.03 | 9.3e7 | 10.3 | 19 | yes |
| mix3_d10 | 500 | +0.062 ± 0.035 | -0.02 ± 0.01 | 0.06 | 0.00 | 1.2e6 | 2.0 | 10 | yes |
| mix3_d10 | 1000 | -0.011 ± 0.026 | -0.02 ± 0.01 | 0.05 | 0.00 | 2.4e6 | 2.1 | 11 | yes |
| mix3_d10 | 2000 | -0.013 ± 0.019 | -0.00 ± 0.00 | 0.03 | 0.00 | 4.8e6 | 2.3 | 11 | yes |
| mix3_d10 | 4000 | -0.001 ± 0.013 | +0.00 ± 0.00 | 0.02 | 0.00 | 9.6e6 | 2.6 | 11 | yes |

## Headline

Only the `tinyns_v1` cells at nlive 500, 1000 enter this section.

`tinyns_v1` per target family: its accurate cells, and its rank by cost at equal logZ rms among the samplers that are accurate on a common target.

| family | accurate | rank by calls | rank by wall time |
|---|---|---|---|
| gauss | 10/10 | 6 of 12 | 2 of 12 |
| rosen | 3/4 | 6 of 11 | 2 of 11 |
| funnel | 2/2 | 5 of 10 | 2 of 10 |
| loggamma | 6/6 | 6 of 10 | 2 of 10 |
| eggbox | 2/2 | 5 of 10 | 1 of 10 |
| sepW | 7/8 | 6 of 11 | 2 of 11 |
| connW | 7/8 | 5 of 8 | 3 of 8 |
| sepM | 1/6 | 2 of 2 | 1 of 2 |
| mix3 | 2/2 | 3 of 4 | 2 of 4 |

Competitor / `tinyns_v1`, geometric mean over the common accurate targets. Above 1: `tinyns_v1` needs fewer calls or less time, or has the smaller rms or logit sd. `at equal rms` is the cost ratio times the squared rms ratio (cost taken to go as 1 / rms^2). Wall times compare across hardware labels only as measured.

| family | competitor | hw | common / ran | calls | wall time | rms | calls at equal rms | wall time at equal rms | logit sd |
|---|---|---|---|---|---|---|---|---|---|
| gauss | blackjax_nss | GPU H100 | 6/10 | 1.02 | 3.02 | 1.28 | 1.67 | 4.92 | - |
| gauss | dynesty | CPU 4c | 4/10 | 0.0908 | 3.88 | 1.2 | 0.131 | 5.59 | - |
| gauss | dynesty:rwalk100 | CPU 4c | 6/10 | 1.23 | 34.4 | 1.53 | 2.88 | 80.6 | - |
| gauss | dynesty:rwalk100 [walks=192] | CPU 4c | 2/2 | 0.72 | 197 | 0.706 | 0.359 | 98.5 | - |
| gauss | dynesty:rwalk100 [walks=384] | CPU 4c | 0/2 | - | - | - | - | - | - |
| gauss | jaxns | GPU H100 | 10/10 | 2.43 | 12.6 | 1.51 | 5.55 | 28.9 | - |
| gauss | jaxns:nlive | GPU H100 | 10/10 | 2.98 | 12.8 | 1.2 | 4.31 | 18.6 | - |
| gauss | nautilus | CPU 4c | 6/10 | 0.101 | 74.9 | 0.195 | 0.00387 | 2.86 | - |
| gauss | nautilus:discard | CPU 4c | 8/10 | 0.0865 | 131 | 0.0411 | 0.000146 | 0.221 | - |
| gauss | tinyns_v02 | GPU H100 | 8/10 | 0.635 | 6.89 | 1.32 | 1.11 | 12 | - |
| gauss | ultranest | CPU 4c | 4/4 | 0.0547 | 7.01 | 1.34 | 0.0986 | 12.6 | - |
| gauss | ultranest:slice | CPU 4c | 4/6 | 1.01 | 166 | 1.43 | 2.05 | 337 | - |
| rosen | blackjax_nss | GPU H100 | 3/4 | 0.767 | 2.54 | 1.59 | 1.94 | 6.44 | - |
| rosen | dynesty | CPU 4c | 3/4 | 0.192 | 5.03 | 1.5 | 0.434 | 11.4 | - |
| rosen | dynesty:rwalk100 | CPU 4c | 3/4 | 1.38 | 16.7 | 1.44 | 2.84 | 34.4 | - |
| rosen | jaxns | GPU H100 | 3/4 | 0.635 | 2.9 | 3.33 | 7.04 | 32.2 | - |
| rosen | jaxns:nlive | GPU H100 | 3/4 | 2.64 | 6.93 | 1.53 | 6.2 | 16.3 | - |
| rosen | nautilus | CPU 4c | 3/4 | 0.153 | 65 | 0.11 | 0.00184 | 0.783 | - |
| rosen | nautilus:discard | CPU 4c | 3/4 | 0.193 | 68.9 | 0.149 | 0.00431 | 1.54 | - |
| rosen | tinyns_v02 | GPU H100 | 2/4 | 0.66 | 2.21 | 1.22 | 0.977 | 3.27 | - |
| rosen | ultranest | CPU 4c | 2/4 | 0.0834 | 3.89 | 0.848 | 0.06 | 2.79 | - |
| rosen | ultranest:slice | CPU 4c | 1/2 | 0.869 | 61.9 | 2.14 | 3.98 | 284 | - |
| funnel | blackjax_nss | GPU H100 | 2/2 | 0.825 | 2.49 | 1.38 | 1.56 | 4.71 | - |
| funnel | dynesty | CPU 4c | 2/2 | 0.353 | 25.4 | 1.65 | 0.966 | 69.6 | - |
| funnel | dynesty:rwalk100 | CPU 4c | 2/2 | 1.19 | 74.1 | 1.2 | 1.69 | 106 | - |
| funnel | jaxns | GPU H100 | 2/2 | 1.82 | 8.01 | 1.68 | 5.16 | 22.7 | - |
| funnel | jaxns:nlive | GPU H100 | 2/2 | 3.04 | 11.7 | 1.03 | 3.19 | 12.3 | - |
| funnel | nautilus | CPU 4c | 2/2 | 0.069 | 167 | 0.247 | 0.00419 | 10.2 | - |
| funnel | nautilus:discard | CPU 4c | 2/2 | 0.074 | 171 | 0.0431 | 0.000138 | 0.318 | - |
| funnel | tinyns_v02 | GPU H100 | 2/2 | 0.69 | 7.24 | 1.29 | 1.14 | 12 | - |
| funnel | ultranest | CPU 4c | 2/2 | 0.1 | 14.1 | 1.18 | 0.14 | 19.6 | - |
| loggamma | blackjax_nss | GPU H100 | 2/6 | 0.832 | 2.55 | 1.17 | 1.14 | 3.5 | 2.85 |
| loggamma | dynesty | CPU 4c | 4/6 | 0.262 | 4.28 | 1.54 | 0.621 | 10.1 | 2.55 |
| loggamma | dynesty:rwalk100 | CPU 4c | 4/6 | 1.19 | 12.6 | 1.2 | 1.71 | 18.1 | 2.84 |
| loggamma | jaxns | GPU H100 | 6/6 | 1.23 | 5.85 | 1.48 | 2.71 | 12.9 | 5.14 |
| loggamma | jaxns:nlive | GPU H100 | 6/6 | 2.15 | 7.37 | 1.15 | 2.86 | 9.8 | 3.49 |
| loggamma | nautilus | CPU 4c | 6/6 | 0.101 | 184 | 0.149 | 0.00223 | 4.06 | 0.167 |
| loggamma | nautilus:discard | CPU 4c | 6/6 | 0.122 | 198 | 0.0571 | 0.000398 | 0.644 | 0.239 |
| loggamma | tinyns_v02 | GPU H100 | 4/6 | 0.633 | 4.14 | 0.899 | 0.512 | 3.35 | 1.9 |
| loggamma | ultranest | CPU 4c | 4/4 | 0.373 | 17.1 | 1.12 | 0.468 | 21.5 | 0.854 |
| loggamma | ultranest:slice | CPU 4c | 0/2 | - | - | - | - | - | - |
| eggbox | blackjax_nss | GPU H100 | 2/2 | 1.35 | 2.6 | 1.3 | 2.28 | 4.37 | - |
| eggbox | dynesty | CPU 4c | 2/2 | 0.467 | 14.2 | 0.813 | 0.309 | 9.41 | - |
| eggbox | dynesty:rwalk100 | CPU 4c | 2/2 | 1.96 | 10.9 | 0.871 | 1.49 | 8.26 | - |
| eggbox | jaxns | GPU H100 | 2/2 | 0.353 | 1.77 | 3.32 | 3.89 | 19.6 | - |
| eggbox | jaxns:nlive | GPU H100 | 2/2 | 2.96 | 5.83 | 1.16 | 3.98 | 7.85 | - |
| eggbox | nautilus | CPU 4c | 2/2 | 0.424 | 1.41e+03 | 0.162 | 0.0111 | 37.1 | - |
| eggbox | nautilus:discard | CPU 4c | 2/2 | 0.546 | 1.55e+03 | 0.101 | 0.00561 | 16 | - |
| eggbox | tinyns_v02 | GPU H100 | 2/2 | 0.604 | 2.22 | 1.44 | 1.25 | 4.61 | - |
| eggbox | ultranest | CPU 4c | 2/2 | 0.136 | 5.64 | 0.86 | 0.1 | 4.17 | - |
| sepW | blackjax_nss | GPU H100 | 4/8 | 1.15 | 3.01 | 1.1 | 1.39 | 3.64 | 6.11 |
| sepW | dynesty | CPU 4c | 4/8 | 2.39 | 93.4 | 1.14 | 3.12 | 122 | 3.17 |
| sepW | dynesty:rwalk100 | CPU 4c | 4/8 | 1.67 | 41.7 | 1.18 | 2.32 | 58 | 6.22 |
| sepW | jaxns | GPU H100 | 2/8 | 1.18 | 2.89 | 2.93 | 10.1 | 24.8 | 5.99 |
| sepW | jaxns:nlive | GPU H100 | 4/6 | 4.5 | 10.3 | 2.16 | 21 | 48 | 5.57 |
| sepW | nautilus | CPU 4c | 4/8 | 0.127 | 108 | 0.122 | 0.00189 | 1.6 | 0.433 |
| sepW | nautilus:discard | CPU 4c | 6/8 | 0.105 | 197 | 0.0373 | 0.000147 | 0.275 | 0.482 |
| sepW | tinyns_v02 | GPU H100 | 6/8 | 0.578 | 7.33 | 1.15 | 0.764 | 9.68 | 1.51 |
| sepW | ultranest | CPU 4c | 2/4 | 0.142 | 9.42 | 1.17 | 0.193 | 12.8 | 1.13 |
| sepW | ultranest:slice | CPU 4c | 2/6 | 0.955 | 53.9 | 1.02 | 0.99 | 55.9 | 4.35 |
| connW | blackjax_nss | GPU H100 | 2/8 | 1.16 | 2.92 | 1.6 | 2.97 | 7.48 | 2.93 |
| connW | dynesty | CPU 4c | 2/8 | 0.201 | 8.91 | 1.54 | 0.475 | 21 | 1.2 |
| connW | dynesty:rwalk100 | CPU 4c | 4/8 | 1.66 | 42.3 | 1.25 | 2.6 | 66 | 4.17 |
| connW | nautilus | CPU 4c | 4/8 | 0.128 | 116 | 0.0747 | 0.000712 | 0.644 | 0.438 |
| connW | nautilus:discard | CPU 4c | 4/8 | 0.148 | 119 | 0.0515 | 0.000392 | 0.316 | 0.513 |
| connW | ultranest | CPU 4c | 4/4 | 0.679 | 39 | 1.2 | 0.973 | 55.9 | 0.952 |
| connW | ultranest:slice | CPU 4c | 2/6 | 0.925 | 54.5 | 1.54 | 2.19 | 129 | 8.79 |
| sepM | blackjax_nss | GPU H100 | 0/6 | - | - | - | - | - | - |
| sepM | dynesty | CPU 4c | 0/6 | - | - | - | - | - | - |
| sepM | dynesty:rwalk100 | CPU 4c | 0/6 | - | - | - | - | - | - |
| sepM | nautilus | CPU 4c | 0/6 | - | - | - | - | - | - |
| sepM | nautilus:discard | CPU 4c | 0/6 | - | - | - | - | - | - |
| sepM | nautilus:discard [n_live=8000] | CPU 4c | 1/2 | 0.229 | 950 | 0.0464 | 0.000494 | 2.05 | 1.48 |
| sepM | ultranest | CPU 4c | 0/2 | - | - | - | - | - | - |
| sepM | ultranest:slice | CPU 4c | 0/6 | - | - | - | - | - | - |
| mix3 | blackjax_nss | GPU H100 | 0/2 | - | - | - | - | - | - |
| mix3 | dynesty | CPU 4c | 0/2 | - | - | - | - | - | - |
| mix3 | dynesty:rwalk100 | CPU 4c | 0/2 | - | - | - | - | - | - |
| mix3 | nautilus | CPU 4c | 2/2 | 0.0878 | 342 | 0.304 | 0.00814 | 31.7 | 0.348 |
| mix3 | nautilus:discard | CPU 4c | 2/2 | 0.0976 | 351 | 0.0439 | 0.000188 | 0.677 | 0.589 |
| mix3 | ultranest | CPU 4c | 0/2 | - | - | - | - | - | - |
| mix3 | ultranest:slice | CPU 4c | 2/2 | 1.07 | 59.7 | 1.47 | 2.31 | 129 | 9.8 |

## Samplers

| sampler | version | commit | hw | cells | runs | timeouts | errors |
|---|---|---|---|---|---|---|---|
| blackjax_nss | blackjax 1.7.1 | 620ca72 | GPU H100 | 24 | 680 | 0 | 0 |
| dynesty | dynesty 3.1.0 | 620ca72 | CPU 4c | 25 | 595 | 1 | 0 |
| dynesty:rwalk100 | dynesty 3.1.0 | 620ca72 | CPU 4c | 24 | 590 | 0 | 0 |
| dynesty:rwalk100 [walks=192] | dynesty 3.1.0 | 620ca72 | CPU 4c | 1 | 5 | 0 | 0 |
| dynesty:rwalk100 [walks=384] | dynesty 3.1.0 | 620ca72 | CPU 4c | 1 | 5 | 0 | 0 |
| jaxns | jaxns 3.0.0 | 620ca72 | GPU H100 | 16 | 400 | 0 | 0 |
| jaxns:nlive | jaxns 3.0.0 | 620ca72 | GPU H100 | 15 | 360 | 0 | 0 |
| nautilus | nautilus-sampler 1.0.6 | 620ca72 | CPU 4c | 24 | 590 | 10 | 0 |
| nautilus:discard | nautilus-sampler 1.0.6 | 620ca72 | CPU 4c | 24 | 590 | 10 | 0 |
| nautilus:discard [n_live=8000] | nautilus-sampler 1.0.6 | 620ca72 | CPU 4c | 1 | 5 | 0 | 0 |
| tinyns_v02 | tinyns 0.2.5 | 620ca72 | GPU H100 | 16 | 400 | 0 | 0 |
| tinyns_v1 | tinyns 1.0.0.dev0 | afa58dd | GPU H100 | 72 | 2240 | 0 | 0 |
| ultranest | ultranest 4.5.2 | 620ca72 | CPU 4c | 14 | 375 | 78 | 0 |
| ultranest:slice | ultranest 4.5.2 | 620ca72 | CPU 4c | 15 | 260 | 0 | 0 |

## Accurate cells per sampler and target family

| sampler | gauss | rosen | funnel | loggamma | eggbox | sepW | connW | sepM | mix3 |
|---|---|---|---|---|---|---|---|---|---|
| blackjax_nss | 3/5 | 2/2 | 1/1 | 1/3 | 1/1 | 2/4 | 1/4 | 0/3 | 0/1 |
| dynesty (nlive 500) | 2/5 | 2/2 | 1/1 | 2/3 | 1/1 | 2/4 | 1/4 | 0/3 | 0/1 |
| dynesty (nlive 1600) | 0/1 | - | - | - | - | - | - | - | - |
| dynesty:rwalk100 | 3/5 | 2/2 | 1/1 | 2/3 | 1/1 | 2/4 | 2/4 | 0/3 | 0/1 |
| dynesty:rwalk100 [walks=192] | 1/1 | - | - | - | - | - | - | - | - |
| dynesty:rwalk100 [walks=384] | 0/1 | - | - | - | - | - | - | - | - |
| jaxns | 5/5 | 2/2 | 1/1 | 3/3 | 1/1 | 1/4 | - | - | - |
| jaxns:nlive | 5/5 | 2/2 | 1/1 | 3/3 | 1/1 | 2/3 | - | - | - |
| nautilus | 3/5 | 2/2 | 1/1 | 3/3 | 1/1 | 2/4 | 2/4 | 0/3 | 1/1 |
| nautilus:discard | 4/5 | 2/2 | 1/1 | 3/3 | 1/1 | 3/4 | 2/4 | 0/3 | 1/1 |
| nautilus:discard [n_live=8000] | - | - | - | - | - | - | - | 1/1 | - |
| tinyns_v02 | 4/5 | 1/2 | 1/1 | 2/3 | 1/1 | 3/4 | - | - | - |
| tinyns_v1 (nlive 500) | 5/5 | 1/2 | 1/1 | 3/3 | 1/1 | 3/4 | 3/4 | 0/3 | 1/1 |
| tinyns_v1 (nlive 1000) | 5/5 | 2/2 | 1/1 | 3/3 | 1/1 | 4/4 | 4/4 | 1/3 | 1/1 |
| tinyns_v1 (nlive 2000) | - | - | - | - | - | 4/4 | 4/4 | 2/3 | 1/1 |
| tinyns_v1 (nlive 4000) | - | - | - | - | - | 4/4 | 4/4 | 3/3 | 1/1 |
| ultranest | 2/2 | 1/2 | 1/1 | 2/2 | 1/1 | 1/2 | 2/2 | 0/1 | 0/1 |
| ultranest:slice | 2/3 | 1/1 | - | 0/1 | - | 1/3 | 1/3 | 0/3 | 1/1 |

## Cells

Hardware (wall times compare only within one label):
- `CPU 4c`: no GPU, Intel(R) Xeon(R) CPU E5-2695 v3 @ 2.30GHz, 4 threads visible, Slurm RM
- `GPU H100`: NVIDIA H100 80GB HBM3, Intel(R) Xeon(R) Platinum 8468, 20 threads visible, no Slurm

| target | sampler | hw | nlive | ok | dlogZ | rms | sd | logzerr | scat/err | in 3s | logit bias | logit sd [95% CI] | lost | ncall | wall s | compile s | run s | first s | acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gauss_d2 | blackjax_nss | GPU H100 | 500 | 20/20 | +0.012 ± 0.023 | 0.103 | 0.104 | 0.107 | 0.97 | 1.00 | - | - | - | 1.24e+05 | 10 | 1.1 | 9.0 | 17 | yes |
| gauss_d2 | dynesty | CPU 4c | 500 | 20/20 | -0.007 ± 0.016 | 0.070 | 0.071 | 0.132 | 0.54 | 1.00 | - | - | - | 2.21e+04 | 9 | - | - | - | yes |
| gauss_d2 | dynesty:rwalk100 | CPU 4c | 500 | 20/20 | -0.028 ± 0.022 | 0.102 | 0.100 | 0.132 | 0.76 | 1.00 | - | - | - | 2.8e+05 | 40 | - | - | - | yes |
| gauss_d2 | jaxns | GPU H100 | 60 | 20/20 | -0.105 ± 0.083 | 0.377 | 0.371 | 0.302 | 1.23 | 1.00 | - | - | - | 4.79e+04 | 7 | - | - | 30 | yes |
| gauss_d2 | jaxns:nlive | GPU H100 | 500 | 20/20 | -0.011 ± 0.019 | 0.082 | 0.084 | 0.106 | 0.79 | 1.00 | - | - | - | 4.35e+05 | 22 | - | - | 76 | yes |
| gauss_d2 | nautilus | CPU 4c | 2000 | 20/20 | -0.001 ± 0.002 | 0.008 | 0.008 | - | - | - | - | - | - | 3.34e+04 | 62 | - | - | - | yes |
| gauss_d2 | nautilus:discard | CPU 4c | 2000 | 20/20 | +0.001 ± 0.002 | 0.008 | 0.008 | - | - | - | - | - | - | 4.49e+04 | 69 | - | - | - | yes |
| gauss_d2 | tinyns_v02 | GPU H100 | 500 | 20/20 | +0.014 ± 0.026 | 0.113 | 0.115 | 0.104 | 1.10 | 1.00 | - | - | - | 9.07e+04 | 8 | 2.1 | 6.0 | 11 | yes |
| gauss_d2 | tinyns_v1 | GPU H100 | 500 | 20/20 | -0.014 ± 0.019 | 0.083 | 0.084 | 0.106 | 0.79 | 1.00 | - | - | - | 1.06e+05 | 3 | 2.3 | 1.1 | 10 | yes |
| gauss_d2 | tinyns_v1 | GPU H100 | 1000 | 20/20 | -0.001 ± 0.014 | 0.060 | 0.062 | 0.075 | 0.82 | 1.00 | - | - | - | 2.11e+05 | 3 | 2.3 | 1.1 | 10 | yes |
| gauss_d2 | ultranest | CPU 4c | >=500 | 20/20 | -0.005 ± 0.017 | 0.076 | 0.078 | 0.156 | 0.50 | 1.00 | - | - | - | 8.09e+03 | 16 | - | - | - | yes |
| gauss_d8 | blackjax_nss | GPU H100 | 500 | 20/20 | +0.047 ± 0.042 | 0.191 | 0.189 | 0.214 | 0.88 | 1.00 | - | - | - | 1.04e+06 | 11 | 1.1 | 10.3 | 18 | yes |
| gauss_d8 | dynesty | CPU 4c | 500 | 20/20 | +0.050 ± 0.052 | 0.233 | 0.233 | 0.251 | 0.93 | 1.00 | - | - | - | 5.02e+04 | 23 | - | - | - | yes |
| gauss_d8 | dynesty:rwalk100 | CPU 4c | 500 | 20/20 | -0.063 ± 0.076 | 0.336 | 0.339 | 0.251 | 1.35 | 1.00 | - | - | - | 1.24e+06 | 175 | - | - | - | yes |
| gauss_d8 | jaxns | GPU H100 | 240 | 20/20 | +0.045 ± 0.077 | 0.337 | 0.342 | 0.301 | 1.14 | 0.95 | - | - | - | 1.53e+06 | 22 | - | - | 56 | yes |
| gauss_d8 | jaxns:nlive | GPU H100 | 500 | 20/20 | -0.007 ± 0.047 | 0.206 | 0.211 | 0.209 | 1.01 | 1.00 | - | - | - | 3.2e+06 | 39 | - | - | 81 | yes |
| gauss_d8 | nautilus | CPU 4c | 2000 | 20/20 | -0.033 ± 0.001 | 0.034 | 0.006 | - | - | - | - | - | - | 8.75e+04 | 345 | - | - | - | yes |
| gauss_d8 | nautilus:discard | CPU 4c | 2000 | 20/20 | +0.000 ± 0.002 | 0.008 | 0.008 | - | - | - | - | - | - | 1.01e+05 | 347 | - | - | - | yes |
| gauss_d8 | tinyns_v02 | GPU H100 | 500 | 20/20 | -0.010 ± 0.054 | 0.234 | 0.240 | 0.209 | 1.15 | 0.95 | - | - | - | 5.78e+05 | 18 | 2.1 | 15.6 | 20 | yes |
| gauss_d8 | tinyns_v1 | GPU H100 | 500 | 20/20 | +0.046 ± 0.035 | 0.160 | 0.157 | 0.220 | 0.71 | 1.00 | - | - | - | 6.36e+05 | 4 | 2.3 | 1.4 | 11 | yes |
| gauss_d8 | tinyns_v1 | GPU H100 | 1000 | 20/20 | -0.050 ± 0.034 | 0.157 | 0.153 | 0.156 | 0.98 | 1.00 | - | - | - | 1.28e+06 | 4 | 2.4 | 1.5 | 11 | yes |
| gauss_d8 | ultranest | CPU 4c | >=500 | 20/20 | +0.009 ± 0.061 | 0.266 | 0.272 | 0.324 | 0.84 | 1.00 | - | - | - | 5e+04 | 40 | - | - | - | yes |
| gauss_d16 | blackjax_nss | GPU H100 | 500 | 20/20 | -0.016 ± 0.064 | 0.278 | 0.285 | 0.297 | 0.96 | 1.00 | - | - | - | 3.82e+06 | 14 | 1.0 | 13.1 | 21 | yes |
| gauss_d16 | dynesty | CPU 4c | 500 | 20/20 | +0.245 ± 0.058 | 0.351 | 0.258 | 0.340 | 0.76 | 1.00 | - | - | - | 8.93e+05 | 133 | - | - | - | no: Z |
| gauss_d16 | dynesty:rwalk100 | CPU 4c | 500 | 20/20 | +0.093 ± 0.059 | 0.273 | 0.263 | 0.340 | 0.77 | 1.00 | - | - | - | 2.46e+06 | 341 | - | - | - | yes |
| gauss_d16 | jaxns | GPU H100 | 480 | 20/20 | -0.075 ± 0.091 | 0.405 | 0.408 | 0.301 | 1.36 | 1.00 | - | - | - | 1.06e+07 | 64 | - | - | 114 | yes |
| gauss_d16 | jaxns:nlive | GPU H100 | 500 | 20/20 | +0.066 ± 0.072 | 0.319 | 0.320 | 0.294 | 1.09 | 1.00 | - | - | - | 1.13e+07 | 67 | - | - | 111 | yes |
| gauss_d16 | nautilus | CPU 4c | 2000 | 20/20 | -0.073 ± 0.002 | 0.073 | 0.007 | - | - | - | - | - | - | 1.64e+05 | 1160 | - | - | - | yes |
| gauss_d16 | nautilus:discard | CPU 4c | 2000 | 20/20 | +0.003 ± 0.002 | 0.007 | 0.007 | - | - | - | - | - | - | 1.8e+05 | 1154 | - | - | - | yes |
| gauss_d16 | tinyns_v02 | GPU H100 | 500 | 20/20 | -0.021 ± 0.053 | 0.231 | 0.236 | 0.295 | 0.80 | 1.00 | - | - | - | 2.2e+06 | 45 | 2.1 | 42.5 | 47 | yes |
| gauss_d16 | tinyns_v1 | GPU H100 | 500 | 20/20 | -0.050 ± 0.059 | 0.260 | 0.262 | 0.310 | 0.85 | 1.00 | - | - | - | 2.4e+06 | 4 | 2.3 | 2.1 | 11 | yes |
| gauss_d16 | tinyns_v1 | GPU H100 | 1000 | 20/20 | -0.058 ± 0.046 | 0.207 | 0.203 | 0.219 | 0.93 | 1.00 | - | - | - | 4.81e+06 | 5 | 2.4 | 2.2 | 11 | yes |
| gauss_d16 | ultranest:slice | CPU 4c | >=500 | 20/20 | +0.168 ± 0.092 | 0.433 | 0.410 | 0.422 | 0.97 | 1.00 | - | - | - | 3.37e+06 | 460 | - | - | - | yes |
| gauss_d32 | blackjax_nss | GPU H100 | 500 | 20/20 | +0.445 ± 0.079 | 0.562 | 0.352 | 0.422 | 0.83 | 0.95 | - | - | - | 1.47e+07 | 27 | 1.0 | 26.2 | 34 | no: Z |
| gauss_d32 | dynesty | CPU 4c | 500 | 10/10 | +1.191 ± 0.134 | 1.257 | 0.423 | 0.470 | 0.90 | 0.70 | - | - | - | 8.78e+06 | 1084 | - | - | - | no: Z |
| gauss_d32 | dynesty | CPU 4c | 1600 | 5/5 | +0.361 ± 0.114 | 0.427 | 0.255 | 0.266 | 0.96 | 1.00 | - | - | - | 2.86e+07 | 3521 | - | - | - | no: Z |
| gauss_d32 | dynesty:rwalk100 | CPU 4c | 500 | 10/10 | +0.421 ± 0.135 | 0.584 | 0.426 | 0.466 | 0.92 | 1.00 | - | - | - | 4.83e+06 | 705 | - | - | - | no: Z |
| gauss_d32 | dynesty:rwalk100 [walks=192] | CPU 4c | 500 | 5/5 | -0.054 ± 0.126 | 0.257 | 0.281 | 0.465 | 0.60 | 1.00 | - | - | - | 9.3e+06 | 1343 | - | - | - | yes |
| gauss_d32 | jaxns | GPU H100 | 960 | 20/20 | +0.051 ± 0.062 | 0.276 | 0.278 | 0.301 | 0.93 | 1.00 | - | - | - | 8.51e+07 | 256 | - | - | 316 | yes |
| gauss_d32 | jaxns:nlive | GPU H100 | 500 | 20/20 | -0.033 ± 0.125 | 0.546 | 0.559 | 0.416 | 1.34 | 1.00 | - | - | - | 4.46e+07 | 147 | - | - | 191 | yes |
| gauss_d32 | nautilus | CPU 4c | 2000 | 10/10 | -0.153 ± 0.002 | 0.153 | 0.005 | - | - | - | - | - | - | 3.8e+05 | 4181 | - | - | - | no: Z |
| gauss_d32 | nautilus:discard | CPU 4c | 2000 | 10/10 | +0.000 ± 0.002 | 0.006 | 0.007 | - | - | - | - | - | - | 4.05e+05 | 4302 | - | - | - | yes |
| gauss_d32 | tinyns_v02 | GPU H100 | 500 | 20/20 | +0.243 ± 0.093 | 0.472 | 0.415 | 0.417 | 0.99 | 1.00 | - | - | - | 8.36e+06 | 142 | 2.1 | 139.4 | 143 | yes |
| gauss_d32 | tinyns_v1 | GPU H100 | 500 | 20/20 | +0.094 ± 0.110 | 0.487 | 0.490 | 0.436 | 1.13 | 1.00 | - | - | - | 9.13e+06 | 6 | 2.3 | 4.1 | 14 | yes |
| gauss_d32 | tinyns_v1 | GPU H100 | 1000 | 20/20 | +0.009 ± 0.062 | 0.272 | 0.279 | 0.308 | 0.90 | 1.00 | - | - | - | 1.83e+07 | 7 | 2.4 | 4.8 | 15 | yes |
| gauss_d32 | ultranest:slice | CPU 4c | >=500 | 10/10 | +0.143 ± 0.123 | 0.396 | 0.389 | 0.612 | 0.64 | 1.00 | - | - | - | 1.32e+07 | 1836 | - | - | - | yes |
| gauss_d64 | blackjax_nss | GPU H100 | 500 | 20/20 | +3.271 ± 0.159 | 3.344 | 0.710 | 0.594 | 1.20 | 0.00 | - | - | - | 5.55e+07 | 78 | 1.0 | 77.4 | 85 | no: Z |
| gauss_d64 | dynesty | CPU 4c | 500 | 10/10 | +7.430 ± 0.119 | 7.439 | 0.377 | 0.645 | 0.59 | 0.00 | - | - | - | 3.21e+07 | 4082 | - | - | - | no: Z |
| gauss_d64 | dynesty:rwalk100 | CPU 4c | 500 | 10/10 | +12.144 ± 0.160 | 12.153 | 0.505 | 0.632 | 0.80 | 0.00 | - | - | - | 8.92e+06 | 1489 | - | - | - | no: Z |
| gauss_d64 | dynesty:rwalk100 [walks=384] | CPU 4c | 500 | 5/5 | +1.760 ± 0.368 | 1.908 | 0.823 | 0.649 | 1.27 | 0.60 | - | - | - | 3.59e+07 | 5249 | - | - | - | no: Z |
| gauss_d64 | jaxns | GPU H100 | 1920 | 20/20 | -0.019 ± 0.067 | 0.292 | 0.299 | 0.301 | 0.99 | 1.00 | - | - | - | 6.69e+08 | 1140 | - | - | 1209 | yes |
| gauss_d64 | jaxns:nlive | GPU H100 | 500 | 20/20 | +0.058 ± 0.102 | 0.448 | 0.455 | 0.588 | 0.77 | 1.00 | - | - | - | 1.75e+08 | 360 | - | - | 401 | yes |
| gauss_d64 | nautilus | CPU 4c | own | 0/10 (10T 0E) | - | - | - | - | - | - | - | - | - | - | >14405 | - | - | - | no: T |
| gauss_d64 | nautilus:discard | CPU 4c | own | 0/10 (10T 0E) | - | - | - | - | - | - | - | - | - | - | >14405 | - | - | - | no: T |
| gauss_d64 | tinyns_v02 | GPU H100 | 500 | 20/20 | +0.913 ± 0.120 | 1.051 | 0.535 | 0.590 | 0.91 | 1.00 | - | - | - | 3.2e+07 | 515 | 2.3 | 513.3 | 518 | no: Z |
| gauss_d64 | tinyns_v1 | GPU H100 | 500 | 20/20 | +0.209 ± 0.126 | 0.587 | 0.562 | 0.613 | 0.92 | 1.00 | - | - | - | 6.21e+07 | 22 | 2.4 | 19.1 | 30 | yes |
| gauss_d64 | tinyns_v1 | GPU H100 | 1000 | 20/20 | -0.043 ± 0.119 | 0.520 | 0.532 | 0.436 | 1.22 | 1.00 | - | - | - | 1.25e+08 | 23 | 2.4 | 20.1 | 31 | yes |
| gauss_d64 | ultranest:slice | CPU 4c | >=500 | 10/10 | +1.571 ± 0.142 | 1.627 | 0.449 | 0.807 | 0.56 | 1.00 | - | - | - | 7.54e+07 | 10951 | - | - | - | no: Z |
| rosen_d2 | blackjax_nss | GPU H100 | 500 | 20/20 | -0.009 ± 0.027 | 0.118 | 0.120 | 0.102 | 1.18 | 0.95 | - | - | - | 1.04e+05 | 10 | 1.1 | 9.2 | 17 | yes |
| rosen_d2 | dynesty | CPU 4c | 500 | 20/20 | -0.014 ± 0.022 | 0.095 | 0.096 | 0.124 | 0.77 | 1.00 | - | - | - | 2.35e+04 | 10 | - | - | - | yes |
| rosen_d2 | dynesty:rwalk100 | CPU 4c | 500 | 20/20 | -0.001 ± 0.024 | 0.105 | 0.107 | 0.124 | 0.86 | 1.00 | - | - | - | 2.46e+05 | 35 | - | - | - | yes |
| rosen_d2 | jaxns | GPU H100 | 60 | 20/20 | +0.048 ± 0.069 | 0.305 | 0.309 | 0.283 | 1.09 | 1.00 | - | - | - | 5.34e+04 | 7 | - | - | 18 | yes |
| rosen_d2 | jaxns:nlive | GPU H100 | 500 | 20/20 | +0.025 ± 0.021 | 0.096 | 0.095 | 0.100 | 0.95 | 1.00 | - | - | - | 3.55e+05 | 21 | - | - | 43 | yes |
| rosen_d2 | nautilus | CPU 4c | 2000 | 20/20 | +0.001 ± 0.002 | 0.009 | 0.009 | - | - | - | - | - | - | 3.3e+04 | 101 | - | - | - | yes |
| rosen_d2 | nautilus:discard | CPU 4c | 2000 | 20/20 | +0.005 ± 0.002 | 0.010 | 0.009 | - | - | - | - | - | - | 4.45e+04 | 110 | - | - | - | yes |
| rosen_d2 | tinyns_v02 | GPU H100 | 500 | 20/20 | -0.036 ± 0.027 | 0.124 | 0.122 | 0.099 | 1.23 | 0.95 | - | - | - | 9.12e+04 | 9 | 2.2 | 6.5 | 10 | yes |
| rosen_d2 | tinyns_v1 | GPU H100 | 500 | 20/20 | +0.016 ± 0.030 | 0.130 | 0.133 | 0.101 | 1.32 | 0.95 | - | - | - | 9.74e+04 | 4 | 2.3 | 1.6 | 9 | yes |
| rosen_d2 | tinyns_v1 | GPU H100 | 1000 | 20/20 | -0.007 ± 0.018 | 0.079 | 0.081 | 0.070 | 1.16 | 1.00 | - | - | - | 1.96e+05 | 4 | 2.3 | 1.6 | 10 | yes |
| rosen_d2 | ultranest | CPU 4c | >=500 | 20/20 | -0.005 ± 0.020 | 0.086 | 0.088 | 0.153 | 0.58 | 1.00 | - | - | - | 1.15e+04 | 15 | - | - | - | yes |
| rosen_d10 | blackjax_nss | GPU H100 | 500 | 20/20 | -0.058 ± 0.149 | 0.652 | 0.666 | 0.274 | 2.43 | 0.75 | - | - | - | 2.04e+06 | 12 | 1.0 | 11.0 | 18 | yes |
| rosen_d10 | dynesty | CPU 4c | 500 | 20/20 | -0.026 ± 0.192 | 0.838 | 0.859 | 0.315 | 2.73 | 0.85 | - | - | - | 6.32e+05 | 96 | - | - | - | yes |
| rosen_d10 | dynesty:rwalk100 | CPU 4c | 500 | 20/20 | -0.169 ± 0.133 | 0.605 | 0.596 | 0.317 | 1.88 | 0.95 | - | - | - | 2.1e+06 | 296 | - | - | - | yes |
| rosen_d10 | jaxns | GPU H100 | 300 | 20/20 | -0.325 ± 0.189 | 0.885 | 0.844 | 0.346 | 2.44 | 0.80 | - | - | - | 4.39e+06 | 39 | - | - | 68 | yes |
| rosen_d10 | jaxns:nlive | GPU H100 | 500 | 20/20 | -0.175 ± 0.197 | 0.874 | 0.879 | 0.269 | 3.27 | 0.50 | - | - | - | 7.12e+06 | 56 | - | - | 105 | yes |
| rosen_d10 | nautilus | CPU 4c | 2000 | 20/20 | -0.036 ± 0.004 | 0.039 | 0.017 | - | - | - | - | - | - | 1.62e+05 | 2104 | - | - | - | yes |
| rosen_d10 | nautilus:discard | CPU 4c | 2000 | 20/20 | -0.070 ± 0.004 | 0.072 | 0.019 | - | - | - | - | - | - | 1.78e+05 | 2096 | - | - | - | yes |
| rosen_d10 | tinyns_v02 | GPU H100 | 500 | 20/20 | -0.382 ± 0.100 | 0.579 | 0.447 | 0.271 | 1.65 | 0.85 | - | - | - | 1.24e+06 | 34 | 2.2 | 31.3 | 34 | no: Z |
| rosen_d10 | tinyns_v1 | GPU H100 | 500 | 20/20 | -0.215 ± 0.058 | 0.331 | 0.258 | 0.286 | 0.90 | 1.00 | - | - | - | 1.29e+06 | 5 | 2.3 | 2.6 | 12 | no: Z |
| rosen_d10 | tinyns_v1 | GPU H100 | 1000 | 20/20 | -0.056 ± 0.048 | 0.216 | 0.214 | 0.202 | 1.05 | 1.00 | - | - | - | 2.56e+06 | 5 | 2.3 | 2.7 | 12 | yes |
| rosen_d10 | ultranest | CPU 4c | >=500 | 0/20 (20T 0E) | - | - | - | - | - | - | - | - | - | - | >14405 | - | - | - | no: T |
| rosen_d10 | ultranest:slice | CPU 4c | >=500 | 10/10 | -0.208 ± 0.137 | 0.462 | 0.435 | 0.365 | 1.19 | 1.00 | - | - | - | 2.23e+06 | 311 | - | - | - | yes |
| funnel_d10 | blackjax_nss | GPU H100 | 500 | 20/20 | +0.014 ± 0.050 | 0.219 | 0.225 | 0.194 | 1.16 | 1.00 | - | - | - | 1.99e+06 | 13 | 1.1 | 11.5 | 18 | yes |
| funnel_d10 | dynesty | CPU 4c | 500 | 20/20 | +0.035 ± 0.060 | 0.264 | 0.268 | 0.256 | 1.05 | 1.00 | - | - | - | 8.52e+05 | 129 | - | - | - | yes |
| funnel_d10 | dynesty:rwalk100 | CPU 4c | 500 | 20/20 | -0.008 ± 0.044 | 0.191 | 0.195 | 0.253 | 0.77 | 1.00 | - | - | - | 2.86e+06 | 374 | - | - | - | yes |
| funnel_d10 | jaxns | GPU H100 | 300 | 20/20 | -0.030 ± 0.061 | 0.268 | 0.274 | 0.243 | 1.13 | 1.00 | - | - | - | 4.38e+06 | 40 | - | - | 54 | yes |
| funnel_d10 | jaxns:nlive | GPU H100 | 500 | 20/20 | -0.078 ± 0.033 | 0.163 | 0.147 | 0.188 | 0.78 | 1.00 | - | - | - | 7.32e+06 | 59 | - | - | 81 | yes |
| funnel_d10 | nautilus | CPU 4c | 2000 | 20/20 | -0.039 ± 0.001 | 0.039 | 0.003 | - | - | - | - | - | - | 1.66e+05 | 844 | - | - | - | yes |
| funnel_d10 | nautilus:discard | CPU 4c | 2000 | 20/20 | -0.005 ± 0.001 | 0.007 | 0.005 | - | - | - | - | - | - | 1.78e+05 | 865 | - | - | - | yes |
| funnel_d10 | tinyns_v02 | GPU H100 | 500 | 20/20 | +0.024 ± 0.047 | 0.205 | 0.209 | 0.212 | 0.98 | 1.00 | - | - | - | 1.66e+06 | 37 | 2.1 | 34.3 | 37 | yes |
| funnel_d10 | tinyns_v1 | GPU H100 | 500 | 20/20 | +0.052 ± 0.045 | 0.205 | 0.203 | 0.254 | 0.80 | 1.00 | - | - | - | 1.68e+06 | 5 | 2.4 | 2.6 | 11 | yes |
| funnel_d10 | tinyns_v1 | GPU H100 | 1000 | 20/20 | +0.015 ± 0.028 | 0.124 | 0.126 | 0.183 | 0.69 | 1.00 | - | - | - | 3.45e+06 | 5 | 2.4 | 2.8 | 11 | yes |
| funnel_d10 | ultranest | CPU 4c | >=500 | 20/20 | -0.093 ± 0.038 | 0.188 | 0.168 | 0.286 | 0.59 | 1.00 | - | - | - | 2.42e+05 | 71 | - | - | - | yes |
| loggamma_d2 | blackjax_nss | GPU H100 | 500 | 20/20 | -0.034 ± 0.015 | 0.075 | 0.068 | 0.074 | 0.92 | 1.00 | +0.03 ± 0.036 | 0.16 [0.11, 0.20] | 0.00 | 8.29e+04 | 10 | 1.3 | 9.0 | 17 | yes |
| loggamma_d2 | dynesty | CPU 4c | 500 | 20/20 | -0.027 ± 0.020 | 0.093 | 0.092 | 0.091 | 1.00 | 1.00 | +0.00 ± 0.012 | 0.05 [0.04, 0.07] | 0.00 | 2.03e+04 | 6 | - | - | - | yes |
| loggamma_d2 | dynesty:rwalk100 | CPU 4c | 500 | 20/20 | -0.026 ± 0.020 | 0.093 | 0.091 | 0.091 | 1.00 | 1.00 | -0.01 ± 0.019 | 0.09 [0.05, 0.11] | 0.00 | 1.31e+05 | 19 | - | - | - | yes |
| loggamma_d2 | jaxns | GPU H100 | 60 | 20/20 | -0.050 ± 0.040 | 0.181 | 0.179 | 0.204 | 0.88 | 1.00 | -0.01 ± 0.064 | 0.29 [0.20, 0.37] | 0.00 | 2.62e+04 | 7 | - | - | 18 | yes |
| loggamma_d2 | jaxns:nlive | GPU H100 | 500 | 20/20 | +0.028 ± 0.015 | 0.072 | 0.068 | 0.071 | 0.96 | 1.00 | -0.01 ± 0.027 | 0.12 [0.08, 0.15] | 0.00 | 1.57e+05 | 16 | - | - | 43 | yes |
| loggamma_d2 | nautilus | CPU 4c | 2000 | 20/20 | -0.004 ± 0.002 | 0.009 | 0.008 | - | - | - | +0.00 ± 0.004 | 0.02 [0.01, 0.02] | 0.00 | 2.48e+04 | 78 | - | - | - | yes |
| loggamma_d2 | nautilus:discard | CPU 4c | 2000 | 20/20 | +0.001 ± 0.002 | 0.007 | 0.007 | - | - | - | -0.00 ± 0.005 | 0.02 [0.01, 0.03] | 0.00 | 3.59e+04 | 91 | - | - | - | yes |
| loggamma_d2 | tinyns_v02 | GPU H100 | 500 | 20/20 | -0.003 ± 0.013 | 0.056 | 0.058 | 0.070 | 0.83 | 1.00 | -0.00 ± 0.016 | 0.07 [0.05, 0.09] | 0.00 | 6.22e+04 | 9 | 2.2 | 6.9 | 13 | yes |
| loggamma_d2 | tinyns_v1 | GPU H100 | 500 | 20/20 | -0.002 ± 0.018 | 0.077 | 0.079 | 0.070 | 1.13 | 1.00 | -0.01 ± 0.014 | 0.06 [0.05, 0.07] | 0.00 | 7.05e+04 | 4 | 2.4 | 1.6 | 10 | yes |
| loggamma_d2 | tinyns_v1 | GPU H100 | 1000 | 20/20 | +0.002 ± 0.012 | 0.053 | 0.054 | 0.049 | 1.12 | 1.00 | +0.00 ± 0.011 | 0.05 [0.03, 0.06] | 0.00 | 1.41e+05 | 4 | 2.4 | 1.6 | 10 | yes |
| loggamma_d2 | ultranest | CPU 4c | >=500 | 20/20 | +0.017 ± 0.014 | 0.065 | 0.064 | 0.109 | 0.59 | 1.00 | +0.01 ± 0.013 | 0.06 [0.04, 0.07] | 0.00 | 7.72e+03 | 24 | - | - | - | yes |
| loggamma_d10 | blackjax_nss | GPU H100 | 500 | 20/20 | +0.140 ± 0.031 | 0.195 | 0.139 | 0.191 | 0.73 | 1.00 | +0.04 ± 0.051 | 0.23 [0.15, 0.28] | 0.00 | 1.16e+06 | 12 | 1.3 | 10.6 | 19 | no: Z |
| loggamma_d10 | dynesty | CPU 4c | 500 | 20/20 | +0.004 ± 0.055 | 0.239 | 0.245 | 0.224 | 1.09 | 1.00 | -0.04 ± 0.097 | 0.43 [0.32, 0.52] | 0.00 | 3.37e+05 | 54 | - | - | - | yes |
| loggamma_d10 | dynesty:rwalk100 | CPU 4c | 500 | 20/20 | -0.014 ± 0.033 | 0.145 | 0.148 | 0.225 | 0.66 | 1.00 | -0.10 ± 0.077 | 0.34 [0.22, 0.42] | 0.00 | 1.08e+06 | 152 | - | - | - | yes |
| loggamma_d10 | jaxns | GPU H100 | 300 | 20/20 | -0.102 ± 0.036 | 0.186 | 0.159 | 0.243 | 0.65 | 1.00 | -0.18 ± 0.117 | 0.52 [0.34, 0.67] | 0.00 | 1.61e+06 | 24 | - | - | 42 | yes |
| loggamma_d10 | jaxns:nlive | GPU H100 | 500 | 20/20 | +0.013 ± 0.051 | 0.221 | 0.226 | 0.188 | 1.20 | 0.95 | -0.06 ± 0.095 | 0.43 [0.26, 0.56] | 0.00 | 2.52e+06 | 32 | - | - | 62 | yes |
| loggamma_d10 | nautilus | CPU 4c | 2000 | 20/20 | -0.021 ± 0.001 | 0.022 | 0.006 | - | - | - | +0.00 ± 0.004 | 0.02 [0.01, 0.03] | 0.00 | 8.58e+04 | 732 | - | - | - | yes |
| loggamma_d10 | nautilus:discard | CPU 4c | 2000 | 20/20 | +0.002 ± 0.002 | 0.007 | 0.007 | - | - | - | +0.00 ± 0.005 | 0.02 [0.01, 0.03] | 0.00 | 9.91e+04 | 764 | - | - | - | yes |
| loggamma_d10 | tinyns_v02 | GPU H100 | 500 | 20/20 | +0.018 ± 0.031 | 0.134 | 0.137 | 0.188 | 0.73 | 1.00 | +0.08 ± 0.040 | 0.18 [0.10, 0.23] | 0.00 | 6.43e+05 | 35 | 2.3 | 32.2 | 36 | yes |
| loggamma_d10 | tinyns_v1 | GPU H100 | 500 | 20/20 | +0.072 ± 0.038 | 0.180 | 0.169 | 0.200 | 0.85 | 1.00 | -0.02 ± 0.020 | 0.09 [0.06, 0.11] | 0.00 | 7.06e+05 | 4 | 2.4 | 2.1 | 11 | yes |
| loggamma_d10 | tinyns_v1 | GPU H100 | 1000 | 20/20 | +0.035 ± 0.026 | 0.120 | 0.118 | 0.143 | 0.83 | 1.00 | -0.00 ± 0.010 | 0.05 [0.03, 0.06] | 0.00 | 1.42e+06 | 5 | 2.5 | 2.2 | 11 | yes |
| loggamma_d10 | ultranest | CPU 4c | >=500 | 20/20 | +0.023 ± 0.041 | 0.181 | 0.184 | 0.316 | 0.58 | 1.00 | -0.01 ± 0.010 | 0.04 [0.02, 0.06] | 0.00 | 1.8e+06 | 227 | - | - | - | yes |
| loggamma_d30 | blackjax_nss | GPU H100 | 500 | 20/20 | +0.939 ± 0.082 | 1.005 | 0.367 | 0.334 | 1.10 | 0.55 | +0.06 ± 0.079 | 0.35 [0.22, 0.46] | 0.00 | 8.96e+06 | 24 | 1.2 | 22.3 | 31 | no: Z |
| loggamma_d30 | dynesty | CPU 4c | 500 | 20/20 | +1.665 ± 0.083 | 1.705 | 0.373 | 0.372 | 1.00 | 0.15 | -0.05 ± 0.131 | 0.58 [0.41, 0.73] | 0.00 | 5.64e+06 | 701 | - | - | - | no: Z |
| loggamma_d30 | dynesty:rwalk100 | CPU 4c | 500 | 20/20 | +0.970 ± 0.078 | 1.028 | 0.348 | 0.374 | 0.93 | 0.70 | -0.21 ± 0.116 | 0.52 [0.36, 0.62] | 0.00 | 3.24e+06 | 448 | - | - | - | no: Z |
| loggamma_d30 | jaxns | GPU H100 | 900 | 20/20 | +0.034 ± 0.064 | 0.281 | 0.286 | 0.248 | 1.16 | 1.00 | -0.23 ± 0.198 | 0.88 [0.59, 1.12] | 0.05 | 3.67e+07 | 153 | - | - | 196 | yes |
| loggamma_d30 | jaxns:nlive | GPU H100 | 500 | 20/20 | +0.012 ± 0.064 | 0.279 | 0.286 | 0.332 | 0.86 | 1.00 | -0.32 ± 0.178 | 0.80 [0.51, 1.02] | 0.00 | 2.09e+07 | 92 | - | - | 129 | yes |
| loggamma_d30 | nautilus | CPU 4c | 2000 | 20/20 | -0.031 ± 0.008 | 0.048 | 0.037 | - | - | - | -0.01 ± 0.003 | 0.01 [0.01, 0.02] | 0.00 | 3.97e+05 | 13104 | - | - | - | yes |
| loggamma_d30 | nautilus:discard | CPU 4c | 2000 | 20/20 | -0.004 ± 0.002 | 0.010 | 0.010 | - | - | - | -0.00 ± 0.006 | 0.03 [0.02, 0.03] | 0.00 | 4.25e+05 | 13368 | - | - | - | yes |
| loggamma_d30 | tinyns_v02 | GPU H100 | 500 | 20/20 | +0.347 ± 0.110 | 0.593 | 0.493 | 0.333 | 1.48 | 0.95 | -0.12 ± 0.073 | 0.33 [0.23, 0.40] | 0.00 | 5.33e+06 | 185 | 2.4 | 182.9 | 188 | no: Z |
| loggamma_d30 | tinyns_v1 | GPU H100 | 500 | 20/20 | +0.191 ± 0.078 | 0.388 | 0.347 | 0.351 | 0.99 | 1.00 | -0.08 ± 0.079 | 0.35 [0.27, 0.41] | 0.00 | 5.85e+06 | 6 | 2.4 | 4.1 | 14 | yes |
| loggamma_d30 | tinyns_v1 | GPU H100 | 1000 | 20/20 | -0.019 ± 0.056 | 0.244 | 0.249 | 0.249 | 1.00 | 1.00 | -0.06 ± 0.047 | 0.21 [0.12, 0.29] | 0.00 | 1.18e+07 | 7 | 2.4 | 4.2 | 14 | yes |
| loggamma_d30 | ultranest:slice | CPU 4c | >=500 | 20/20 | +0.402 ± 0.084 | 0.543 | 0.373 | 0.504 | 0.74 | 1.00 | +0.01 ± 0.045 | 0.20 [0.13, 0.25] | 0.00 | 8.14e+06 | 1142 | - | - | - | no: Z |
| eggbox_d2 | blackjax_nss | GPU H100 | 500 | 20/20 | +0.011 ± 0.029 | 0.126 | 0.129 | 0.113 | 1.14 | 1.00 | - | - | - | 2.18e+05 | 10 | 1.1 | 9.1 | 13 | yes |
| eggbox_d2 | dynesty | CPU 4c | 500 | 20/20 | -0.012 ± 0.018 | 0.079 | 0.080 | 0.139 | 0.58 | 1.00 | - | - | - | 7.5e+04 | 56 | - | - | - | yes |
| eggbox_d2 | dynesty:rwalk100 | CPU 4c | 500 | 20/20 | -0.020 ± 0.019 | 0.085 | 0.085 | 0.139 | 0.61 | 1.00 | - | - | - | 3.15e+05 | 43 | - | - | - | yes |
| eggbox_d2 | jaxns | GPU H100 | 60 | 20/20 | -0.087 ± 0.072 | 0.324 | 0.320 | 0.320 | 1.00 | 1.00 | - | - | - | 5.67e+04 | 7 | - | - | 20 | yes |
| eggbox_d2 | jaxns:nlive | GPU H100 | 500 | 20/20 | -0.048 ± 0.023 | 0.113 | 0.105 | 0.112 | 0.93 | 1.00 | - | - | - | 4.76e+05 | 23 | - | - | 49 | yes |
| eggbox_d2 | nautilus | CPU 4c | 2000 | 20/20 | -0.010 ± 0.003 | 0.016 | 0.012 | - | - | - | - | - | - | 6.82e+04 | 5538 | - | - | - | yes |
| eggbox_d2 | nautilus:discard | CPU 4c | 2000 | 20/20 | -0.002 ± 0.002 | 0.010 | 0.010 | - | - | - | - | - | - | 8.77e+04 | 6099 | - | - | - | yes |
| eggbox_d2 | tinyns_v02 | GPU H100 | 500 | 20/20 | -0.028 ± 0.032 | 0.140 | 0.141 | 0.111 | 1.27 | 0.95 | - | - | - | 9.7e+04 | 9 | 2.1 | 6.7 | 10 | yes |
| eggbox_d2 | tinyns_v1 | GPU H100 | 500 | 20/20 | +0.011 ± 0.023 | 0.099 | 0.101 | 0.112 | 0.90 | 1.00 | - | - | - | 1.14e+05 | 4 | 2.3 | 1.6 | 10 | yes |
| eggbox_d2 | tinyns_v1 | GPU H100 | 1000 | 20/20 | -0.007 ± 0.022 | 0.096 | 0.098 | 0.079 | 1.24 | 1.00 | - | - | - | 2.27e+05 | 4 | 2.3 | 1.6 | 10 | yes |
| eggbox_d2 | ultranest | CPU 4c | >=500 | 15/15 | -0.039 ± 0.020 | 0.084 | 0.077 | 0.170 | 0.45 | 1.00 | - | - | - | 2.18e+04 | 22 | - | - | - | yes |
| sepW_d4 | blackjax_nss | GPU H100 | 500 | 40/40 | -0.037 ± 0.025 | 0.158 | 0.155 | 0.170 | 0.92 | 1.00 | +0.04 ± 0.065 | 0.41 [0.32, 0.48] | 0.00 | 3.53e+05 | 11 | 1.2 | 9.4 | 18 | yes |
| sepW_d4 | dynesty | CPU 4c | 500 | 39/40 (1T 0E) | +0.033 ± 0.022 | 0.139 | 0.137 | 0.204 | 0.67 | 1.00 | -0.01 ± 0.013 | 0.08 [0.07, 0.09] | 0.00 | 5e+06 | 1508 | - | - | - | yes |
| sepW_d4 | dynesty:rwalk100 | CPU 4c | 500 | 40/40 | +0.021 ± 0.022 | 0.139 | 0.139 | 0.204 | 0.68 | 1.00 | -0.05 ± 0.067 | 0.42 [0.31, 0.53] | 0.00 | 7.43e+05 | 103 | - | - | - | yes |
| sepW_d4 | jaxns | GPU H100 | 120 | 40/40 | +0.033 ± 0.065 | 0.405 | 0.409 | 0.335 | 1.22 | 1.00 | -0.07 ± 0.072 | 0.45 [0.36, 0.53] | 0.00 | 3.59e+05 | 10 | - | - | 30 | yes |
| sepW_d4 | jaxns:nlive | GPU H100 | 500 | 40/40 | +0.050 ± 0.039 | 0.250 | 0.248 | 0.166 | 1.50 | 0.97 | -0.09 ± 0.042 | 0.27 [0.21, 0.31] | 0.00 | 1.43e+06 | 30 | - | - | 76 | yes |
| sepW_d4 | nautilus | CPU 4c | 2000 | 40/40 | -0.011 ± 0.001 | 0.013 | 0.007 | - | - | - | -0.02 ± 0.006 | 0.04 [0.03, 0.05] | 0.00 | 6.06e+04 | 190 | - | - | - | yes |
| sepW_d4 | nautilus:discard | CPU 4c | 2000 | 40/40 | -0.000 ± 0.001 | 0.007 | 0.008 | - | - | - | -0.01 ± 0.006 | 0.04 [0.03, 0.04] | 0.00 | 7.29e+04 | 199 | - | - | - | yes |
| sepW_d4 | tinyns_v02 | GPU H100 | 500 | 40/40 | +0.004 ± 0.028 | 0.174 | 0.176 | 0.165 | 1.07 | 1.00 | -0.03 ± 0.017 | 0.11 [0.09, 0.12] | 0.00 | 1.76e+05 | 13 | 2.2 | 10.7 | 16 | yes |
| sepW_d4 | tinyns_v1 | GPU H100 | 500 | 40/40 | -0.018 ± 0.028 | 0.174 | 0.176 | 0.173 | 1.02 | 1.00 | -0.01 ± 0.012 | 0.08 [0.06, 0.09] | 0.00 | 2.16e+05 | 4 | 2.3 | 1.2 | 10 | yes |
| sepW_d4 | tinyns_v1 | GPU H100 | 1000 | 40/40 | +0.012 ± 0.017 | 0.109 | 0.110 | 0.123 | 0.90 | 1.00 | -0.01 ± 0.012 | 0.08 [0.06, 0.09] | 0.00 | 4.31e+05 | 4 | 2.3 | 1.3 | 11 | yes |
| sepW_d4 | tinyns_v1 | GPU H100 | 2000 | 40/40 | +0.013 ± 0.013 | 0.084 | 0.084 | 0.087 | 0.96 | 0.97 | -0.00 ± 0.009 | 0.06 [0.05, 0.07] | 0.00 | 8.62e+05 | 4 | 2.4 | 1.3 | 11 | yes |
| sepW_d4 | tinyns_v1 | GPU H100 | 4000 | 40/40 | +0.003 ± 0.009 | 0.054 | 0.055 | 0.062 | 0.89 | 1.00 | -0.00 ± 0.006 | 0.04 [0.03, 0.05] | 0.00 | 1.72e+06 | 4 | 2.4 | 1.3 | 11 | yes |
| sepW_d4 | ultranest | CPU 4c | >=500 | 40/40 | +0.016 ± 0.026 | 0.161 | 0.162 | 0.258 | 0.63 | 1.00 | +0.01 ± 0.014 | 0.09 [0.07, 0.10] | 0.00 | 4.32e+04 | 34 | - | - | - | yes |
| sepW_d10 | blackjax_nss | GPU H100 | 500 | 40/40 | +0.078 ± 0.034 | 0.228 | 0.217 | 0.270 | 0.80 | 1.00 | -0.04 ± 0.108 | 0.65 [0.51, 0.75] | 0.10 | 1.95e+06 | 13 | 1.1 | 12.3 | 17 | yes |
| sepW_d10 | dynesty | CPU 4c | 500 | 40/40 | +0.091 ± 0.042 | 0.279 | 0.267 | 0.313 | 0.85 | 1.00 | -0.23 ± 0.138 | 0.87 [0.65, 1.03] | 0.00 | 5.94e+05 | 91 | - | - | - | yes |
| sepW_d10 | dynesty:rwalk100 | CPU 4c | 500 | 40/40 | +0.039 ± 0.047 | 0.297 | 0.298 | 0.313 | 0.95 | 1.00 | -0.14 ± 0.104 | 0.64 [0.50, 0.76] | 0.05 | 1.94e+06 | 265 | - | - | - | yes |
| sepW_d10 | jaxns | GPU H100 | 300 | 40/40 | +0.040 ± 0.068 | 0.425 | 0.428 | 0.341 | 1.25 | 1.00 | -0.41 ± 0.210 | 1.17 [0.96, 1.32] | 0.23 | 4.5e+06 | 36 | - | - | 68 | no: M |
| sepW_d10 | jaxns:nlive | GPU H100 | 500 | 40/40 | +0.028 ± 0.089 | 0.555 | 0.561 | 0.265 | 2.12 | 0.85 | -0.31 ± 0.133 | 0.82 [0.61, 0.99] | 0.05 | 7.34e+06 | 55 | - | - | 77 | yes |
| sepW_d10 | nautilus | CPU 4c | 2000 | 40/40 | -0.033 ± 0.001 | 0.034 | 0.007 | - | - | - | -0.02 ± 0.005 | 0.03 [0.03, 0.04] | 0.00 | 1.38e+05 | 961 | - | - | - | yes |
| sepW_d10 | nautilus:discard | CPU 4c | 2000 | 40/40 | -0.003 ± 0.001 | 0.007 | 0.007 | - | - | - | -0.00 ± 0.007 | 0.04 [0.04, 0.05] | 0.00 | 1.54e+05 | 1001 | - | - | - | yes |
| sepW_d10 | tinyns_v02 | GPU H100 | 500 | 40/40 | -0.011 ± 0.031 | 0.194 | 0.196 | 0.265 | 0.74 | 1.00 | -0.02 ± 0.021 | 0.13 [0.10, 0.15] | 0.00 | 9.78e+05 | 35 | 2.2 | 32.7 | 38 | yes |
| sepW_d10 | tinyns_v1 | GPU H100 | 500 | 40/40 | -0.026 ± 0.038 | 0.242 | 0.243 | 0.278 | 0.87 | 1.00 | -0.02 ± 0.022 | 0.14 [0.09, 0.18] | 0.00 | 1.2e+06 | 4 | 2.3 | 1.9 | 10 | yes |
| sepW_d10 | tinyns_v1 | GPU H100 | 1000 | 40/40 | -0.016 ± 0.031 | 0.192 | 0.194 | 0.197 | 0.98 | 1.00 | -0.01 ± 0.010 | 0.06 [0.05, 0.08] | 0.00 | 2.41e+06 | 4 | 2.4 | 2.0 | 10 | yes |
| sepW_d10 | tinyns_v1 | GPU H100 | 2000 | 40/40 | -0.040 ± 0.019 | 0.124 | 0.118 | 0.139 | 0.85 | 1.00 | -0.00 ± 0.004 | 0.03 [0.02, 0.03] | 0.00 | 4.81e+06 | 5 | 2.4 | 2.1 | 11 | yes |
| sepW_d10 | tinyns_v1 | GPU H100 | 4000 | 40/40 | -0.005 ± 0.018 | 0.109 | 0.111 | 0.099 | 1.12 | 1.00 | -0.00 ± 0.003 | 0.02 [0.02, 0.02] | 0.00 | 9.63e+06 | 5 | 2.3 | 2.4 | 12 | yes |
| sepW_d10 | ultranest | CPU 4c | >=500 | 35/40 (5T 0E) | -0.015 ± 0.032 | 0.188 | 0.191 | 0.400 | 0.48 | 1.00 | -0.01 ± 0.011 | 0.07 [0.05, 0.08] | 0.00 | 1.37e+07 | 1837 | - | - | - | no: T |
| sepW_d10 | ultranest:slice | CPU 4c | >=500 | 10/10 | -0.008 ± 0.073 | 0.219 | 0.231 | 0.384 | 0.60 | 1.00 | -0.06 ± 0.128 | 0.41 [0.24, 0.51] | 0.00 | 1.62e+06 | 235 | - | - | - | yes |
| sepW_d18 | blackjax_nss | GPU H100 | 500 | 40/40 | +0.419 ± 0.055 | 0.543 | 0.351 | 0.362 | 0.97 | 1.00 | +0.16 ± 0.135 | 0.81 [0.65, 0.93] | 0.10 | 5.95e+06 | 18 | 1.3 | 17.0 | 25 | no: Z |
| sepW_d18 | dynesty | CPU 4c | 500 | 40/40 | +0.941 ± 0.054 | 0.999 | 0.338 | 0.405 | 0.84 | 0.80 | +0.33 ± 0.095 | 0.59 [0.46, 0.72] | 0.03 | 1.33e+06 | 204 | - | - | - | no: Z,M |
| sepW_d18 | dynesty:rwalk100 | CPU 4c | 500 | 40/40 | +0.334 ± 0.057 | 0.488 | 0.360 | 0.408 | 0.88 | 0.97 | +0.11 ± 0.125 | 0.78 [0.55, 0.96] | 0.03 | 3.49e+06 | 485 | - | - | - | no: Z |
| sepW_d18 | jaxns | GPU H100 | 540 | 40/40 | -0.152 ± 0.158 | 0.997 | 0.998 | 0.343 | 2.91 | 0.65 | -0.03 ± 0.209 | 1.14 [0.91, 1.32] | 0.25 | 2.43e+07 | 104 | - | - | 155 | no: M |
| sepW_d18 | jaxns:nlive | GPU H100 | 500 | 40/40 | -0.497 ± 0.186 | 1.261 | 1.174 | 0.357 | 3.29 | 0.68 | +0.03 ± 0.183 | 0.97 [0.74, 1.15] | 0.30 | 2.29e+07 | 98 | - | - | 142 | no: M |
| sepW_d18 | nautilus | CPU 4c | 2000 | 40/40 | -0.064 ± 0.001 | 0.064 | 0.008 | - | - | - | -0.22 ± 0.013 | 0.08 [0.06, 0.10] | 0.00 | 2.65e+05 | 3481 | - | - | - | no: M |
| sepW_d18 | nautilus:discard | CPU 4c | 2000 | 40/40 | -0.002 ± 0.001 | 0.008 | 0.008 | - | - | - | -0.01 ± 0.012 | 0.08 [0.06, 0.09] | 0.00 | 2.87e+05 | 3600 | - | - | - | yes |
| sepW_d18 | tinyns_v02 | GPU H100 | 500 | 40/40 | +0.109 ± 0.056 | 0.363 | 0.351 | 0.357 | 0.98 | 1.00 | -0.07 ± 0.043 | 0.27 [0.21, 0.32] | 0.00 | 3.09e+06 | 82 | 2.2 | 79.6 | 86 | yes |
| sepW_d18 | tinyns_v1 | GPU H100 | 500 | 40/40 | +0.051 ± 0.048 | 0.304 | 0.303 | 0.372 | 0.81 | 1.00 | -0.14 ± 0.062 | 0.40 [0.22, 0.56] | 0.00 | 3.76e+06 | 6 | 2.3 | 3.5 | 14 | yes |
| sepW_d18 | tinyns_v1 | GPU H100 | 1000 | 40/40 | -0.003 ± 0.039 | 0.242 | 0.245 | 0.264 | 0.93 | 1.00 | -0.01 ± 0.010 | 0.06 [0.04, 0.08] | 0.00 | 7.53e+06 | 6 | 2.4 | 3.6 | 14 | yes |
| sepW_d18 | tinyns_v1 | GPU H100 | 2000 | 40/40 | +0.027 ± 0.031 | 0.193 | 0.194 | 0.187 | 1.04 | 1.00 | -0.00 ± 0.006 | 0.04 [0.03, 0.04] | 0.00 | 1.51e+07 | 6 | 2.4 | 3.9 | 14 | yes |
| sepW_d18 | tinyns_v1 | GPU H100 | 4000 | 40/40 | +0.013 ± 0.018 | 0.116 | 0.117 | 0.133 | 0.88 | 1.00 | -0.00 ± 0.003 | 0.02 [0.02, 0.02] | 0.00 | 3.02e+07 | 7 | 2.3 | 4.4 | 15 | yes |
| sepW_d18 | ultranest:slice | CPU 4c | >=500 | 40/40 | +0.276 ± 0.067 | 0.500 | 0.422 | 0.554 | 0.76 | 1.00 | -0.21 ± 0.130 | 0.80 [0.56, 1.00] | 0.05 | 4.97e+06 | 738 | - | - | - | no: Z |
| sepW_d32 | blackjax_nss | GPU H100 | 500 | 40/40 | +2.158 ± 0.087 | 2.225 | 0.548 | 0.483 | 1.13 | 0.12 | +1.99 ± 0.091 | 0.57 [0.26, 0.87] | 0.00 | 1.79e+07 | 34 | 1.1 | 32.7 | 41 | no: Z,M |
| sepW_d32 | dynesty | CPU 4c | 500 | 10/10 | +4.015 ± 0.122 | 4.032 | 0.384 | 0.528 | 0.73 | 0.00 | +1.69 ± 0.281 | 0.89 [0.32, 1.14] | 0.00 | 1.12e+07 | 1352 | - | - | - | no: Z,M |
| sepW_d32 | dynesty:rwalk100 | CPU 4c | 500 | 10/10 | +2.689 ± 0.141 | 2.723 | 0.447 | 0.532 | 0.84 | 0.00 | +1.51 ± 0.236 | 0.75 [0.52, 0.84] | 0.00 | 6.04e+06 | 863 | - | - | - | no: Z,M |
| sepW_d32 | jaxns | GPU H100 | 960 | 40/40 | -1.022 ± 0.236 | 1.795 | 1.495 | 0.344 | 4.34 | 0.55 | +0.37 ± 0.331 | 1.44 [1.07, 1.73] | 0.53 | 1.35e+08 | 344 | - | - | 398 | no: Z,M |
| sepW_d32 | nautilus | CPU 4c | 2000 | 10/10 | -0.215 ± 0.002 | 0.215 | 0.006 | - | - | - | - | - [-, -] | 1.00 | 5.37e+05 | 9644 | - | - | - | no: Z,M |
| sepW_d32 | nautilus:discard | CPU 4c | 2000 | 10/10 | -0.062 ± 0.001 | 0.062 | 0.004 | - | - | - | - | - [-, -] | 1.00 | 5.65e+05 | 9571 | - | - | - | no: M |
| sepW_d32 | tinyns_v02 | GPU H100 | 500 | 40/40 | +0.775 ± 0.077 | 0.913 | 0.489 | 0.476 | 1.03 | 0.90 | +1.23 ± 0.097 | 0.60 [0.45, 0.74] | 0.05 | 9.5e+06 | 200 | 2.3 | 197.4 | 200 | no: Z,M |
| sepW_d32 | tinyns_v1 | GPU H100 | 500 | 40/40 | +0.110 ± 0.066 | 0.427 | 0.418 | 0.497 | 0.84 | 1.00 | -0.19 ± 0.144 | 0.79 [0.60, 0.95] | 0.25 | 1.16e+07 | 10 | 2.4 | 7.4 | 16 | no: M |
| sepW_d32 | tinyns_v1 | GPU H100 | 1000 | 40/40 | -0.062 ± 0.055 | 0.351 | 0.350 | 0.351 | 1.00 | 1.00 | -0.01 ± 0.030 | 0.19 [0.10, 0.28] | 0.03 | 2.32e+07 | 11 | 2.4 | 8.3 | 18 | yes |
| sepW_d32 | tinyns_v1 | GPU H100 | 2000 | 40/40 | +0.068 ± 0.031 | 0.204 | 0.194 | 0.249 | 0.78 | 1.00 | +0.00 ± 0.006 | 0.04 [0.03, 0.05] | 0.00 | 4.64e+07 | 11 | 2.4 | 8.2 | 19 | yes |
| sepW_d32 | tinyns_v1 | GPU H100 | 4000 | 40/40 | -0.016 ± 0.035 | 0.217 | 0.219 | 0.176 | 1.24 | 1.00 | +0.00 ± 0.003 | 0.02 [0.01, 0.02] | 0.00 | 9.3e+07 | 11 | 2.3 | 9.1 | 19 | yes |
| sepW_d32 | ultranest:slice | CPU 4c | >=500 | 10/10 | +1.044 ± 0.190 | 1.189 | 0.600 | 0.788 | 0.76 | 0.90 | +0.73 ± 0.272 | 0.77 [0.41, 0.95] | 0.20 | 1.49e+07 | 2253 | - | - | - | no: Z,M |
| connW_d4 | blackjax_nss | GPU H100 | 500 | 40/40 | -0.041 ± 0.029 | 0.185 | 0.183 | 0.170 | 1.08 | 1.00 | +0.02 ± 0.035 | 0.22 [0.17, 0.25] | 0.00 | 3.54e+05 | 11 | 1.2 | 9.3 | 13 | yes |
| connW_d4 | dynesty | CPU 4c | 500 | 40/40 | +0.025 ± 0.028 | 0.178 | 0.178 | 0.204 | 0.87 | 1.00 | -0.05 ± 0.014 | 0.09 [0.07, 0.10] | 0.00 | 6.14e+04 | 32 | - | - | - | yes |
| connW_d4 | dynesty:rwalk100 | CPU 4c | 500 | 40/40 | +0.042 ± 0.026 | 0.166 | 0.163 | 0.204 | 0.80 | 1.00 | -0.11 ± 0.042 | 0.26 [0.19, 0.32] | 0.00 | 7.42e+05 | 103 | - | - | - | yes |
| connW_d4 | nautilus | CPU 4c | 2000 | 40/40 | -0.010 ± 0.001 | 0.013 | 0.008 | - | - | - | -0.01 ± 0.005 | 0.03 [0.02, 0.04] | 0.00 | 6.08e+04 | 204 | - | - | - | yes |
| connW_d4 | nautilus:discard | CPU 4c | 2000 | 40/40 | -0.000 ± 0.001 | 0.008 | 0.008 | - | - | - | -0.01 ± 0.006 | 0.04 [0.03, 0.04] | 0.00 | 7.32e+04 | 212 | - | - | - | yes |
| connW_d4 | tinyns_v1 | GPU H100 | 500 | 40/40 | +0.004 ± 0.022 | 0.137 | 0.139 | 0.173 | 0.80 | 1.00 | -0.04 ± 0.014 | 0.09 [0.07, 0.11] | 0.00 | 2.16e+05 | 4 | 2.3 | 1.2 | 10 | yes |
| connW_d4 | tinyns_v1 | GPU H100 | 1000 | 40/40 | -0.007 ± 0.016 | 0.098 | 0.099 | 0.122 | 0.81 | 1.00 | -0.00 ± 0.010 | 0.06 [0.05, 0.07] | 0.00 | 4.31e+05 | 4 | 2.4 | 1.3 | 10 | yes |
| connW_d4 | tinyns_v1 | GPU H100 | 2000 | 40/40 | -0.007 ± 0.013 | 0.083 | 0.084 | 0.087 | 0.96 | 1.00 | -0.02 ± 0.007 | 0.05 [0.04, 0.05] | 0.00 | 8.62e+05 | 4 | 2.4 | 1.2 | 10 | yes |
| connW_d4 | tinyns_v1 | GPU H100 | 4000 | 40/40 | +0.008 ± 0.009 | 0.058 | 0.058 | 0.062 | 0.94 | 1.00 | -0.01 ± 0.006 | 0.04 [0.03, 0.05] | 0.00 | 1.72e+06 | 4 | 2.4 | 1.3 | 10 | yes |
| connW_d4 | ultranest | CPU 4c | >=500 | 40/40 | -0.013 ± 0.024 | 0.152 | 0.153 | 0.248 | 0.62 | 1.00 | +0.02 ± 0.014 | 0.09 [0.06, 0.11] | 0.00 | 3.2e+04 | 33 | - | - | - | yes |
| connW_d10 | blackjax_nss | GPU H100 | 500 | 40/40 | +0.121 ± 0.037 | 0.258 | 0.231 | 0.270 | 0.85 | 1.00 | -0.08 ± 0.072 | 0.45 [0.33, 0.55] | 0.00 | 1.96e+06 | 13 | 1.1 | 12.1 | 16 | no: Z |
| connW_d10 | dynesty | CPU 4c | 500 | 40/40 | +0.117 ± 0.050 | 0.332 | 0.315 | 0.313 | 1.01 | 1.00 | -0.30 ± 0.075 | 0.48 [0.35, 0.58] | 0.00 | 5.94e+05 | 90 | - | - | - | no: M |
| connW_d10 | dynesty:rwalk100 | CPU 4c | 500 | 40/40 | +0.072 ± 0.036 | 0.239 | 0.231 | 0.313 | 0.74 | 1.00 | +0.01 ± 0.065 | 0.41 [0.34, 0.48] | 0.00 | 1.94e+06 | 267 | - | - | - | yes |
| connW_d10 | nautilus | CPU 4c | 2000 | 40/40 | -0.007 ± 0.001 | 0.011 | 0.008 | - | - | - | -0.06 ± 0.006 | 0.04 [0.03, 0.05] | 0.00 | 1.39e+05 | 1012 | - | - | - | yes |
| connW_d10 | nautilus:discard | CPU 4c | 2000 | 40/40 | -0.002 ± 0.001 | 0.008 | 0.008 | - | - | - | -0.01 ± 0.007 | 0.05 [0.03, 0.05] | 0.00 | 1.55e+05 | 1032 | - | - | - | yes |
| connW_d10 | tinyns_v1 | GPU H100 | 500 | 40/40 | -0.068 ± 0.047 | 0.300 | 0.296 | 0.278 | 1.06 | 1.00 | -0.03 ± 0.020 | 0.13 [0.08, 0.17] | 0.00 | 1.2e+06 | 4 | 2.3 | 1.9 | 10 | yes |
| connW_d10 | tinyns_v1 | GPU H100 | 1000 | 40/40 | -0.018 ± 0.025 | 0.160 | 0.161 | 0.197 | 0.82 | 1.00 | -0.01 ± 0.009 | 0.06 [0.05, 0.07] | 0.00 | 2.41e+06 | 4 | 2.4 | 1.9 | 10 | yes |
| connW_d10 | tinyns_v1 | GPU H100 | 2000 | 40/40 | -0.022 ± 0.026 | 0.165 | 0.165 | 0.139 | 1.18 | 0.97 | -0.00 ± 0.006 | 0.04 [0.03, 0.04] | 0.00 | 4.81e+06 | 4 | 2.4 | 2.0 | 10 | yes |
| connW_d10 | tinyns_v1 | GPU H100 | 4000 | 40/40 | +0.012 ± 0.017 | 0.105 | 0.105 | 0.099 | 1.07 | 1.00 | -0.00 ± 0.003 | 0.02 [0.02, 0.03] | 0.00 | 9.63e+06 | 5 | 2.3 | 2.3 | 11 | yes |
| connW_d10 | ultranest | CPU 4c | >=500 | 39/40 (1T 0E) | -0.068 ± 0.037 | 0.240 | 0.233 | 0.432 | 0.54 | 1.00 | -0.00 ± 0.011 | 0.07 [0.05, 0.08] | 0.00 | 7.48e+06 | 708 | - | - | - | yes |
| connW_d10 | ultranest:slice | CPU 4c | >=500 | 10/10 | +0.207 ± 0.089 | 0.337 | 0.280 | 0.391 | 0.72 | 1.00 | -0.46 ± 0.234 | 0.74 [0.21, 1.08] | 0.00 | 1.57e+06 | 234 | - | - | - | yes |
| connW_d18 | blackjax_nss | GPU H100 | 500 | 40/40 | +0.358 ± 0.048 | 0.469 | 0.307 | 0.363 | 0.85 | 1.00 | -0.17 ± 0.122 | 0.75 [0.54, 0.94] | 0.05 | 5.98e+06 | 18 | 1.3 | 16.9 | 21 | no: Z |
| connW_d18 | dynesty | CPU 4c | 500 | 40/40 | +0.838 ± 0.059 | 0.916 | 0.374 | 0.407 | 0.92 | 0.85 | +0.10 ± 0.093 | 0.59 [0.41, 0.74] | 0.00 | 1.33e+06 | 201 | - | - | - | no: Z |
| connW_d18 | dynesty:rwalk100 | CPU 4c | 500 | 40/40 | +0.250 ± 0.055 | 0.423 | 0.346 | 0.408 | 0.85 | 1.00 | -0.16 ± 0.115 | 0.72 [0.54, 0.87] | 0.03 | 3.49e+06 | 490 | - | - | - | no: Z |
| connW_d18 | nautilus | CPU 4c | 2000 | 40/40 | +0.002 ± 0.003 | 0.017 | 0.018 | - | - | - | -1.06 ± 0.035 | 0.22 [0.17, 0.26] | 0.00 | 2.67e+05 | 3825 | - | - | - | no: M |
| connW_d18 | nautilus:discard | CPU 4c | 2000 | 40/40 | -0.007 ± 0.002 | 0.012 | 0.010 | - | - | - | -0.13 ± 0.022 | 0.14 [0.10, 0.18] | 0.00 | 2.92e+05 | 3882 | - | - | - | no: M |
| connW_d18 | tinyns_v1 | GPU H100 | 500 | 40/40 | -0.040 ± 0.050 | 0.318 | 0.319 | 0.373 | 0.86 | 1.00 | -0.12 ± 0.075 | 0.47 [0.26, 0.66] | 0.03 | 3.76e+06 | 6 | 2.3 | 3.6 | 12 | yes |
| connW_d18 | tinyns_v1 | GPU H100 | 1000 | 40/40 | +0.022 ± 0.043 | 0.271 | 0.273 | 0.264 | 1.03 | 1.00 | -0.01 ± 0.011 | 0.07 [0.05, 0.08] | 0.00 | 7.53e+06 | 6 | 2.4 | 3.6 | 12 | yes |
| connW_d18 | tinyns_v1 | GPU H100 | 2000 | 40/40 | -0.031 ± 0.027 | 0.173 | 0.172 | 0.187 | 0.92 | 1.00 | -0.01 ± 0.006 | 0.03 [0.03, 0.04] | 0.00 | 1.51e+07 | 6 | 2.4 | 3.6 | 12 | yes |
| connW_d18 | tinyns_v1 | GPU H100 | 4000 | 40/40 | +0.017 ± 0.021 | 0.130 | 0.131 | 0.132 | 0.99 | 1.00 | +0.00 ± 0.004 | 0.02 [0.02, 0.03] | 0.00 | 3.02e+07 | 6 | 2.3 | 4.0 | 13 | yes |
| connW_d18 | ultranest:slice | CPU 4c | >=500 | 40/40 | +0.264 ± 0.042 | 0.373 | 0.266 | 0.548 | 0.48 | 1.00 | -0.27 ± 0.084 | 0.52 [0.35, 0.66] | 0.05 | 4.9e+06 | 725 | - | - | - | no: Z,M |
| connW_d32 | blackjax_nss | GPU H100 | 500 | 20/20 | +1.940 ± 0.094 | 1.982 | 0.420 | 0.478 | 0.88 | 0.10 | +1.66 ± 0.090 | 0.39 [0.24, 0.48] | 0.05 | 1.8e+07 | 34 | 1.0 | 32.6 | 36 | no: Z,M |
| connW_d32 | dynesty | CPU 4c | 500 | 10/10 | +3.532 ± 0.147 | 3.559 | 0.464 | 0.525 | 0.88 | 0.00 | +1.67 ± 0.327 | 1.03 [0.22, 1.57] | 0.00 | 1.12e+07 | 1365 | - | - | - | no: Z,M |
| connW_d32 | dynesty:rwalk100 | CPU 4c | 500 | 10/10 | +2.475 ± 0.158 | 2.521 | 0.501 | 0.528 | 0.95 | 0.00 | +1.49 ± 0.274 | 0.87 [0.40, 1.07] | 0.00 | 6.05e+06 | 862 | - | - | - | no: Z,M |
| connW_d32 | nautilus | CPU 4c | 2000 | 10/10 | -0.222 ± 0.002 | 0.222 | 0.008 | - | - | - | - | - [-, -] | 1.00 | 5.42e+05 | 9645 | - | - | - | no: Z,M |
| connW_d32 | nautilus:discard | CPU 4c | 2000 | 10/10 | -0.066 ± 0.004 | 0.067 | 0.013 | - | - | - | - | - [-, -] | 1.00 | 5.7e+05 | 9784 | - | - | - | no: M |
| connW_d32 | tinyns_v1 | GPU H100 | 500 | 20/20 | +0.232 ± 0.102 | 0.501 | 0.455 | 0.495 | 0.92 | 1.00 | -0.11 ± 0.318 | 1.19 [0.70, 1.46] | 0.30 | 1.16e+07 | 10 | 2.4 | 7.9 | 16 | no: M |
| connW_d32 | tinyns_v1 | GPU H100 | 1000 | 20/20 | +0.110 ± 0.079 | 0.363 | 0.355 | 0.350 | 1.01 | 1.00 | -0.11 ± 0.149 | 0.65 [0.26, 0.90] | 0.05 | 2.31e+07 | 11 | 2.4 | 8.7 | 18 | yes |
| connW_d32 | tinyns_v1 | GPU H100 | 2000 | 20/20 | +0.018 ± 0.050 | 0.217 | 0.222 | 0.248 | 0.90 | 1.00 | -0.01 ± 0.051 | 0.23 [0.11, 0.30] | 0.00 | 4.64e+07 | 11 | 2.4 | 8.2 | 17 | yes |
| connW_d32 | tinyns_v1 | GPU H100 | 4000 | 20/20 | +0.016 ± 0.035 | 0.153 | 0.156 | 0.176 | 0.89 | 1.00 | +0.01 ± 0.030 | 0.13 [0.09, 0.16] | 0.00 | 9.32e+07 | 11 | 2.3 | 8.3 | 17 | yes |
| connW_d32 | ultranest:slice | CPU 4c | >=500 | 10/10 | +0.789 ± 0.144 | 0.900 | 0.455 | 0.752 | 0.61 | 1.00 | +0.29 ± 0.235 | 0.66 [0.30, 0.79] | 0.20 | 1.47e+07 | 2150 | - | - | - | no: Z,M |
| sepM_d10 | blackjax_nss | GPU H100 | 500 | 20/20 | +0.061 ± 0.064 | 0.286 | 0.287 | 0.270 | 1.06 | 1.00 | +0.07 ± 0.254 | 0.88 [0.34, 1.25] | 0.40 | 1.97e+06 | 13 | 1.1 | 12.3 | 17 | no: M |
| sepM_d10 | dynesty | CPU 4c | 500 | 20/20 | +0.126 ± 0.057 | 0.281 | 0.257 | 0.311 | 0.83 | 1.00 | +0.23 ± 0.293 | 1.02 [0.65, 1.19] | 0.40 | 5.97e+05 | 94 | - | - | - | no: M |
| sepM_d10 | dynesty:rwalk100 | CPU 4c | 500 | 20/20 | +0.006 ± 0.054 | 0.235 | 0.241 | 0.311 | 0.78 | 1.00 | -0.15 ± 0.270 | 0.97 [0.58, 1.19] | 0.35 | 1.95e+06 | 269 | - | - | - | no: M |
| sepM_d10 | nautilus | CPU 4c | 2000 | 20/20 | -0.107 ± 0.002 | 0.108 | 0.008 | - | - | - | -2.10 ± - | - [-, -] | 0.95 | 1.34e+05 | 785 | - | - | - | no: Z,M |
| sepM_d10 | nautilus:discard | CPU 4c | 2000 | 20/20 | -0.061 ± 0.002 | 0.061 | 0.007 | - | - | - | - | - [-, -] | 1.00 | 1.49e+05 | 811 | - | - | - | no: M |
| sepM_d10 | nautilus:discard [n_live=8000] | CPU 4c | 8000 | 5/5 | -0.004 ± 0.004 | 0.010 | 0.010 | - | - | - | -0.01 ± 0.030 | 0.07 [0.02, 0.08] | 0.00 | 5.55e+05 | 4177 | - | - | - | yes |
| sepM_d10 | tinyns_v1 | GPU H100 | 500 | 20/20 | +0.002 ± 0.058 | 0.251 | 0.258 | 0.278 | 0.93 | 1.00 | -0.06 ± 0.262 | 0.74 [0.12, 1.09] | 0.60 | 1.2e+06 | 4 | 2.3 | 1.9 | 10 | no: M |
| sepM_d10 | tinyns_v1 | GPU H100 | 1000 | 20/20 | -0.020 ± 0.049 | 0.214 | 0.218 | 0.196 | 1.11 | 1.00 | +0.01 ± 0.010 | 0.04 [0.03, 0.05] | 0.05 | 2.42e+06 | 4 | 2.4 | 2.0 | 10 | yes |
| sepM_d10 | tinyns_v1 | GPU H100 | 2000 | 20/20 | +0.016 ± 0.030 | 0.130 | 0.133 | 0.140 | 0.95 | 1.00 | +0.01 ± 0.007 | 0.03 [0.02, 0.04] | 0.00 | 4.86e+06 | 5 | 2.4 | 2.1 | 10 | yes |
| sepM_d10 | tinyns_v1 | GPU H100 | 4000 | 20/20 | +0.018 ± 0.020 | 0.090 | 0.091 | 0.099 | 0.92 | 1.00 | -0.00 ± 0.005 | 0.02 [0.02, 0.03] | 0.00 | 9.75e+06 | 5 | 2.3 | 2.4 | 11 | yes |
| sepM_d10 | ultranest | CPU 4c | >=500 | 8/20 (12T 0E) | +0.214 ± 0.088 | 0.316 | 0.249 | 0.364 | 0.68 | 1.00 | +0.02 ± 0.014 | 0.04 [0.02, 0.05] | 0.00 | 2.28e+07 | >14405 | - | - | - | no: T |
| sepM_d10 | ultranest:slice | CPU 4c | >=500 | 10/10 | -0.036 ± 0.088 | 0.268 | 0.279 | 0.407 | 0.69 | 1.00 | -0.05 ± 0.259 | 0.69 [0.27, 0.83] | 0.30 | 1.61e+06 | 243 | - | - | - | no: M |
| sepM_d18 | blackjax_nss | GPU H100 | 500 | 40/40 | +0.226 ± 0.061 | 0.442 | 0.385 | 0.364 | 1.06 | 0.97 | +0.64 ± 0.331 | 1.37 [1.02, 1.59] | 0.57 | 6.04e+06 | 18 | 1.3 | 16.9 | 21 | no: Z,M |
| sepM_d18 | dynesty | CPU 4c | 500 | 40/40 | +0.689 ± 0.076 | 0.837 | 0.480 | 0.407 | 1.18 | 0.88 | +0.41 ± 0.269 | 1.17 [0.79, 1.47] | 0.53 | 1.33e+06 | 200 | - | - | - | no: Z,M |
| sepM_d18 | dynesty:rwalk100 | CPU 4c | 500 | 40/40 | +0.198 ± 0.070 | 0.480 | 0.443 | 0.410 | 1.08 | 0.97 | +0.41 ± 0.260 | 1.35 [1.07, 1.54] | 0.33 | 3.5e+06 | 489 | - | - | - | no: M |
| sepM_d18 | nautilus | CPU 4c | 2000 | 40/40 | -0.142 ± 0.001 | 0.143 | 0.005 | - | - | - | - | - [-, -] | 1.00 | 2.38e+05 | 1993 | - | - | - | no: Z,M |
| sepM_d18 | nautilus:discard | CPU 4c | 2000 | 40/40 | -0.061 ± 0.001 | 0.062 | 0.008 | - | - | - | - | - [-, -] | 1.00 | 2.58e+05 | 2059 | - | - | - | no: M |
| sepM_d18 | tinyns_v1 | GPU H100 | 500 | 40/40 | +0.028 ± 0.075 | 0.468 | 0.473 | 0.373 | 1.27 | 1.00 | +0.02 ± 0.300 | 1.08 [0.56, 1.36] | 0.68 | 3.76e+06 | 6 | 2.3 | 3.3 | 12 | no: M |
| sepM_d18 | tinyns_v1 | GPU H100 | 1000 | 40/40 | -0.023 ± 0.048 | 0.301 | 0.304 | 0.264 | 1.15 | 1.00 | -0.16 ± 0.070 | 0.40 [0.11, 0.58] | 0.20 | 7.54e+06 | 6 | 2.4 | 4.0 | 12 | no: M |
| sepM_d18 | tinyns_v1 | GPU H100 | 2000 | 40/40 | -0.014 ± 0.033 | 0.204 | 0.206 | 0.187 | 1.10 | 1.00 | -0.01 ± 0.004 | 0.03 [0.02, 0.03] | 0.00 | 1.51e+07 | 6 | 2.4 | 4.1 | 13 | yes |
| sepM_d18 | tinyns_v1 | GPU H100 | 4000 | 40/40 | -0.009 ± 0.020 | 0.124 | 0.125 | 0.132 | 0.94 | 1.00 | -0.01 ± 0.003 | 0.02 [0.01, 0.02] | 0.00 | 3.02e+07 | 7 | 2.3 | 4.4 | 13 | yes |
| sepM_d18 | ultranest:slice | CPU 4c | >=500 | 40/40 | +0.023 ± 0.050 | 0.311 | 0.314 | 0.586 | 0.54 | 1.00 | +0.10 ± 0.225 | 0.87 [0.56, 1.06] | 0.62 | 4.78e+06 | 724 | - | - | - | no: M |
| sepM_d32 | blackjax_nss | GPU H100 | 500 | 40/40 | +1.449 ± 0.128 | 1.656 | 0.812 | 0.485 | 1.67 | 0.55 | +2.68 ± 0.145 | 0.58 [0.26, 0.78] | 0.60 | 1.84e+07 | 34 | 1.0 | 32.6 | 37 | no: Z,M |
| sepM_d32 | dynesty | CPU 4c | 500 | 10/10 | +3.398 ± 0.357 | 3.562 | 1.128 | 0.529 | 2.13 | 0.00 | +2.95 ± 0.216 | 0.53 [0.15, 0.71] | 0.40 | 1.13e+07 | 1372 | - | - | - | no: Z,M |
| sepM_d32 | dynesty:rwalk100 | CPU 4c | 500 | 10/10 | +1.699 ± 0.273 | 1.886 | 0.863 | 0.529 | 1.63 | 0.50 | +1.57 ± 1.095 | 2.19 [-, -] | 0.60 | 6.12e+06 | 827 | - | - | - | no: Z,M |
| sepM_d32 | nautilus | CPU 4c | 2000 | 10/10 | -0.216 ± 0.002 | 0.216 | 0.006 | - | - | - | - | - [-, -] | 1.00 | 4.81e+05 | 5671 | - | - | - | no: Z,M |
| sepM_d32 | nautilus:discard | CPU 4c | 2000 | 10/10 | -0.061 ± 0.003 | 0.062 | 0.009 | - | - | - | - | - [-, -] | 1.00 | 5.09e+05 | 5579 | - | - | - | no: M |
| sepM_d32 | tinyns_v1 | GPU H100 | 500 | 40/40 | +0.148 ± 0.094 | 0.605 | 0.594 | 0.495 | 1.20 | 1.00 | +0.51 ± 0.427 | 1.48 [0.51, 1.82] | 0.70 | 1.15e+07 | 9 | 2.4 | 6.5 | 16 | no: M |
| sepM_d32 | tinyns_v1 | GPU H100 | 1000 | 40/40 | -0.032 ± 0.046 | 0.286 | 0.288 | 0.351 | 0.82 | 1.00 | +0.26 ± 0.200 | 0.85 [0.50, 1.11] | 0.55 | 2.32e+07 | 10 | 2.4 | 8.1 | 18 | no: M |
| sepM_d32 | tinyns_v1 | GPU H100 | 2000 | 40/40 | -0.002 ± 0.037 | 0.233 | 0.236 | 0.249 | 0.95 | 1.00 | -0.01 ± 0.063 | 0.34 [0.03, 0.54] | 0.28 | 4.64e+07 | 12 | 2.3 | 9.2 | 17 | no: M |
| sepM_d32 | tinyns_v1 | GPU H100 | 4000 | 40/40 | -0.003 ± 0.032 | 0.202 | 0.205 | 0.176 | 1.16 | 0.97 | -0.01 ± 0.013 | 0.08 [0.02, 0.14] | 0.03 | 9.3e+07 | 13 | 2.3 | 10.3 | 19 | yes |
| sepM_d32 | ultranest:slice | CPU 4c | >=500 | 10/10 | +0.672 ± 0.116 | 0.757 | 0.368 | 0.710 | 0.52 | 1.00 | +1.95 ± 0.535 | 0.93 [-, -] | 0.70 | 1.42e+07 | 2195 | - | - | - | no: Z,M |
| mix3_d10 | blackjax_nss | GPU H100 | 500 | 40/40 | +0.168 ± 0.043 | 0.315 | 0.269 | 0.270 | 1.00 | 1.00 | -0.21 ± 0.140 | 0.87 [0.70, 1.01] | 0.12 | 1.95e+06 | 13 | 1.1 | 12.3 | 16 | no: Z,M |
| mix3_d10 | dynesty | CPU 4c | 500 | 40/40 | +0.132 ± 0.040 | 0.281 | 0.251 | 0.311 | 0.81 | 1.00 | -0.12 ± 0.151 | 0.93 [0.73, 1.10] | 0.10 | 5.9e+05 | 91 | - | - | - | no: Z |
| mix3_d10 | dynesty:rwalk100 | CPU 4c | 500 | 40/40 | +0.036 ± 0.044 | 0.278 | 0.279 | 0.311 | 0.90 | 1.00 | -0.35 ± 0.097 | 0.61 [0.50, 0.71] | 0.03 | 1.93e+06 | 266 | - | - | - | no: M |
| mix3_d10 | nautilus | CPU 4c | 2000 | 40/40 | -0.057 ± 0.001 | 0.058 | 0.007 | - | - | - | -0.01 ± 0.003 | 0.02 [0.01, 0.02] | 0.00 | 1.48e+05 | 1507 | - | - | - | yes |
| mix3_d10 | nautilus:discard | CPU 4c | 2000 | 40/40 | -0.000 ± 0.001 | 0.008 | 0.008 | - | - | - | -0.00 ± 0.005 | 0.03 [0.02, 0.04] | 0.00 | 1.65e+05 | 1543 | - | - | - | yes |
| mix3_d10 | tinyns_v1 | GPU H100 | 500 | 40/40 | +0.062 ± 0.035 | 0.226 | 0.220 | 0.277 | 0.79 | 1.00 | -0.02 ± 0.009 | 0.06 [0.05, 0.07] | 0.00 | 1.19e+06 | 4 | 2.4 | 2.0 | 10 | yes |
| mix3_d10 | tinyns_v1 | GPU H100 | 1000 | 40/40 | -0.011 ± 0.026 | 0.160 | 0.162 | 0.196 | 0.83 | 1.00 | -0.02 ± 0.008 | 0.05 [0.04, 0.06] | 0.00 | 2.4e+06 | 4 | 2.4 | 2.1 | 11 | yes |
| mix3_d10 | tinyns_v1 | GPU H100 | 2000 | 40/40 | -0.013 ± 0.019 | 0.122 | 0.122 | 0.139 | 0.88 | 1.00 | -0.00 ± 0.004 | 0.03 [0.02, 0.03] | 0.00 | 4.79e+06 | 5 | 2.4 | 2.3 | 11 | yes |
| mix3_d10 | tinyns_v1 | GPU H100 | 4000 | 40/40 | -0.001 ± 0.013 | 0.084 | 0.085 | 0.099 | 0.86 | 1.00 | +0.00 ± 0.003 | 0.02 [0.01, 0.02] | 0.00 | 9.6e+06 | 5 | 2.4 | 2.6 | 11 | yes |
| mix3_d10 | ultranest | CPU 4c | >=500 | 0/40 (40T 0E) | - | - | - | - | - | - | - | - | - | - | >14405 | - | - | - | no: T |
| mix3_d10 | ultranest:slice | CPU 4c | >=500 | 10/10 | +0.085 ± 0.089 | 0.279 | 0.280 | 0.356 | 0.79 | 1.00 | +0.15 ± 0.181 | 0.54 [0.27, 0.65] | 0.10 | 1.82e+06 | 263 | - | - | - | yes |

