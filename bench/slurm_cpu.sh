#!/bin/bash
# The CPU sampler lines as a Slurm array: task i runs line i of JOBS from ROOT.
#
#   P=$ROOT BENCH_PY=<cpu env>/bin/python BENCH_RUN=<checkout>/bench/run.py \
#     OUT=$ROOT JOBDIR=$ROOT bench/h100_plan.sh emit-jobs
#   n=$(wc -l < $ROOT/bench_cpu_jobs.txt)
#   sbatch --array=1-$n%60 -A <account> -p RM -c 4 --mem=16G -t 22:00:00 \
#     -o $ROOT/logs/%A_%a.out bench/slurm_cpu.sh $ROOT/bench_cpu_jobs.txt $ROOT
#
# Each line is (sampler, target, block of 5 seeds) with a 4 h per-seed
# timeout, so a task needs at most ~20 h. The task's cores come from
# --cpus-per-task; the BLAS/OpenMP thread counts are set to match, and run.py
# records the core count (cpu_threads), the CPU model and the Slurm job in hw.
set -euo pipefail
JOBS=$1
ROOT=${2:-$PWD}
line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBS")
[ -n "$line" ] || { echo "no line ${SLURM_ARRAY_TASK_ID} in $JOBS"; exit 0; }
cd "$ROOT"
unset PYTHONPATH JAX_PLATFORMS
n=${SLURM_CPUS_PER_TASK:-4}
export OMP_NUM_THREADS=$n OPENBLAS_NUM_THREADS=$n MKL_NUM_THREADS=$n
echo "$(date -u +%FT%TZ) $(hostname) task ${SLURM_ARRAY_TASK_ID}: $line"
rc=0
bash -c "$line" || rc=$?
echo "$(date -u +%FT%TZ) done rc=$rc"
exit $rc
