#!/usr/bin/env bash
# Competitor baseline sweep on js2h100: environment build and queue lines.
#
# Nothing in this script runs by itself. Each step is a subcommand, so it can
# be reviewed and then run one at a time:
#
#   bench/h100_plan.sh build-envs   # clones + env/bench + env/bench_jaxns (~15 min)
#   bench/h100_plan.sh smoke        # 1 seed of each sampler on gauss_d2 (~5 min)
#   bench/h100_plan.sh emit-jobs    # writes queue/bench_{gpu,cpu}_jobs.txt
#   bench/h100_plan.sh cpu-pool [SLOTS] [CORES_PER_SLOT]   # runs the CPU lines
#   bench/h100_plan.sh summarize    # table.md from runs/bench_baseline/*.jsonl
#
# The existing env/core and src/tinyns (3c768e5, used by the reference runs)
# are left alone. The new clones go to src/tinyns-bench (this branch; v1 after
# the PR merges) and src/tinyns-0x (release/0.x = 2f67111).
#
# How the work is split:
# - GPU samplers (tinyns 0.x, BlackJAX NSS, JAXNS): one queue line per
#   (sampler, target, all seeds). The lines go to the existing GPU queue, with
#   cat queue/bench_gpu_jobs.txt >> queue/jobs.txt. Each line runs one process
#   at a time on the GPU, and the two qworkers keep the GPU busy.
# - CPU samplers (dynesty, UltraNest, Nautilus): one line per
#   (sampler, target, block of 5 seeds), in queue/bench_cpu_jobs.txt. They are
#   run by `cpu-pool`: SLOTS workers, each pinned with taskset to its own
#   CORES_PER_SLOT cores. Cores 0-3 stay free for the GPU jobs' host threads.
#
# Seeds:
# - 20 per target for logZ calibration (gauss, rosen, funnel, loggamma, eggbox);
# - 40 per target for the mixtures (sepW, connW, sepM, mix3);
# - CPU samplers get 10 seeds at d >= 32.
#
# Runtime estimate (nlive 500, x64): about 2 days of GPU queue and 2 days
# of CPU pool, running side by side. Treat it as good to a factor of 2-3.
# - Model: calls per run = nlive * (H + 2) * (calls per iteration), where H is
#   the information (5 for gauss_d2 up to 175 for gauss_d64; 14-114 for the
#   mixtures). The calls per iteration and the per-call costs come from the
#   Hilda CPU validation (gauss_d2, gauss_d8, sepW_d4).
# - GPU costs are assumed, not measured: 15 us per sequential walk step for
#   tinyns 0.x (one chain), and ~1 us per call, amortised over the vmapped
#   lanes, for BlackJAX and JAXNS.
#
#   GPU queue (one H100, sequential), runs x per-run time:
#     tinyns 0.x   ~17 h   (gauss_d64 ~9 min/seed; sepW_d32 ~3 min/seed)
#     BlackJAX NSS ~ 3 h
#     JAXNS        ~16 h   (30d roots x 5d slices: calls grow as d^2 H)
#     JAXNS nlive  ~11 h
#     total        ~47 h
#   CPU pool, in slot-hours of 4 cores:
#     dynesty default ~35, dynesty rwalk100 ~25, UltraNest ~4 (d <= 10),
#     UltraNest slice ~46 (d > 10), Nautilus ~113, Nautilus discard ~113
#     total ~335 slot-h: ~42 h on 8 slots, ~48 h on 7.
# - Nautilus at d >= 32 (35-110 min per seed, mostly network training) and
#   dynesty's rslice at d > 20 (estimated, not validated) are the uncertain
#   tail. dynesty default can also stall on mixtures through its bootstrap
#   expansion: on sepW_d4, two of three seeds took 4.1M and 7.4M calls (11 and
#   18 min), against 74k calls (20 s) for the third. The
#   4 h per-seed CPU timeout caps such runs, and EMIT_CPU_HIGH_D=0 drops the
#   CPU samplers at d >= 32.
#
# Timeouts: 1 h per seed on the GPU, 4 h per seed on the CPU. A timeout is
# recorded as status "timeout", so a stuck sampler never blocks the queue.

set -euo pipefail

P=${P:-/media/volume/datasets/tinyns-darksirens}
PYTHON=${PYTHON:-python3}                # needs >= 3.10
REPO=https://github.com/ignaciomagana/tinyns.git
BENCH_BRANCH=${BENCH_BRANCH:-agent/v1-bench}   # switch to v1 once merged
SRC=$P/src/tinyns-bench
SRC0X=$P/src/tinyns-0x
OUT=runs/bench_baseline                  # relative to $P, like other queue lines
NLIVE=${NLIVE:-500}
EMIT_CPU_HIGH_D=${EMIT_CPU_HIGH_D:-1}    # 0: skip CPU samplers at d >= 32

CALIB="gauss_d2 gauss_d8 gauss_d16 gauss_d32 gauss_d64 rosen_d2 rosen_d10
funnel_d10 loggamma_d2 loggamma_d10 loggamma_d30 eggbox_d2"
MIXT="sepW_d4 sepW_d10 sepW_d18 sepW_d32 connW_d4 connW_d10 connW_d18 connW_d32
sepM_d10 sepM_d18 sepM_d32 mix3_d10"

ndim() { echo "${1##*_d}"; }

build_envs() {
  cd "$P/src"
  [ -d "$SRC" ] || git clone "$REPO" "$SRC"
  git -C "$SRC" fetch origin
  git -C "$SRC" checkout "$BENCH_BRANCH"
  git -C "$SRC" pull --ff-only
  [ -d "$SRC0X" ] || git -C "$SRC" worktree add "$SRC0X" origin/release/0.x
  git -C "$SRC0X" log -1 --format=%h        # expect 2f67111

  "$PYTHON" -m venv "$P/env/bench"
  "$P/env/bench/bin/pip" install --upgrade pip
  "$P/env/bench/bin/pip" install -r "$SRC/bench/requirements-bench.txt"
  "$P/env/bench/bin/pip" install -e "$SRC"           # tinyns v1 (once PR 1 lands)

  "$PYTHON" -m venv "$P/env/bench_jaxns"
  "$P/env/bench_jaxns/bin/pip" install --upgrade pip
  "$P/env/bench_jaxns/bin/pip" install -r "$SRC/bench/requirements-bench_jaxns.txt"

  for e in bench bench_jaxns; do
    "$P/env/$e/bin/python" -c "import jax; print('$e', jax.__version__, jax.devices())"
  done
  mkdir -p "$P/$OUT" "$P/cache/jax_bench"
}

# Command prefix for a sampler (relative to $P, as queue lines are).
runner() {
  case $1 in
    jaxns*) echo "env/bench_jaxns/bin/python src/tinyns-bench/bench/run.py" ;;
    tinyns_v02*) echo "env TINYNS_V02_SRC=$SRC0X/src env/bench/bin/python src/tinyns-bench/bench/run.py" ;;
    *) echo "env/bench/bin/python src/tinyns-bench/bench/run.py" ;;
  esac
}

smoke() {
  cd "$P"
  for s in tinyns_v02 blackjax_nss dynesty ultranest nautilus jaxns; do
    $(runner $s) --sampler $s --target gauss_d2 --seeds 0 --nlive 100 \
      --opt n_live=200 --out $OUT/smoke.jsonl --timeout 900 || true
  done
  env/bench/bin/python src/tinyns-bench/bench/summarize.py $OUT/smoke.jsonl
}

gpu_line() {  # sampler target seeds
  echo "$(runner $1) --sampler $1 --target $2 --seeds $3 --nlive $NLIVE" \
       "--out $OUT/gpu.jsonl --timeout 3600 --exclusive --jax-cache cache/jax_bench" \
       "--tag baseline-h100"
}

cpu_lines() {  # sampler target nseeds
  local s=$1 t=$2 n=$3 a
  for ((a = 0; a < n; a += 5)); do
    echo "$(runner $s) --sampler $s --target $t --seeds $a-$((a + 4 < n - 1 ? a + 4 : n - 1))" \
         "--nlive $NLIVE --out $OUT/cpu.jsonl --timeout 14400 --exclusive --tag baseline-h100-cpu"
  done
}

emit_jobs() {
  local g=$P/queue/bench_gpu_jobs.txt c=$P/queue/bench_cpu_jobs.txt t d n
  : > "$g"; : > "$c"
  for t in $CALIB $MIXT; do
    d=$(ndim $t)
    case " $MIXT " in *" $t "*) n=40 ;; *) n=20 ;; esac
    for s in tinyns_v02 blackjax_nss jaxns jaxns:nlive; do
      gpu_line $s $t "0-$((n - 1))" >> "$g"
    done
    local nc=$n
    if ((d >= 32)); then
      nc=10
      ((EMIT_CPU_HIGH_D)) || continue
    fi
    for s in dynesty dynesty:rwalk100 nautilus nautilus:discard; do
      cpu_lines $s $t $nc >> "$c"
    done
    if ((d <= 10)); then cpu_lines ultranest $t $nc >> "$c"; else cpu_lines ultranest:slice $t $nc >> "$c"; fi
  done
  wc -l "$g" "$c"
  echo "review, then: cat $g >> $P/queue/jobs.txt ; $0 cpu-pool"
}

# SLOTS workers; slot i is pinned to cores 4+i*CPS .. 4+(i+1)*CPS-1 and claims
# the next unclaimed line of bench_cpu_jobs.txt under flock.
cpu_pool() {
  local slots=${1:-8} cps=${2:-4} q=$P/queue/bench_cpu_jobs.txt
  local ctr=$P/queue/bench_cpu_jobs.next log=$P/queue/bench_cpu.log
  [ -f "$ctr" ] || echo 1 > "$ctr"
  cd "$P"
  for ((i = 0; i < slots; i++)); do
    local lo=$((4 + i * cps)) hi=$((4 + (i + 1) * cps - 1))
    (
      while true; do
        line=$(flock "$ctr" bash -c "k=\$(cat '$ctr'); l=\$(sed -n \"\${k}p\" '$q');
               [ -n \"\$l\" ] && echo \$((k + 1)) > '$ctr'; echo \"\$l\"")
        [ -z "$line" ] && break
        echo "$(date -u +%FT%TZ) slot$i start $line" >> "$log"
        OMP_NUM_THREADS=$cps OPENBLAS_NUM_THREADS=$cps taskset -c $lo-$hi bash -c "$line" \
          >> "$log" 2>&1 || true
        echo "$(date -u +%FT%TZ) slot$i done" >> "$log"
      done
    ) &
  done
  echo "cpu pool: $slots slots x $cps cores; log $log"
  wait
}

case ${1:-} in
  build-envs) build_envs ;;
  smoke) smoke ;;
  emit-jobs) emit_jobs ;;
  cpu-pool) shift; cpu_pool "$@" ;;
  summarize)
    cd "$P"
    env/bench/bin/python src/tinyns-bench/bench/summarize.py $OUT/gpu.jsonl $OUT/cpu.jsonl \
      --out $OUT/table.md && echo "$P/$OUT/table.md" ;;
  *) sed -n '2,12p' "$0"; exit 1 ;;
esac
