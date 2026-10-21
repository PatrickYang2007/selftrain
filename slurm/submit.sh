#!/bin/bash
# Submit one or more arms as separate jobs:  slurm/submit.sh oracle self_judge majority
# Extra sbatch flags via SBATCH_EXTRA, e.g. SBATCH_EXTRA="--constraint=a100" slurm/submit.sh oracle
# Loop overrides via LOOP_SET, e.g. LOOP_SET="run_name=oracle_star continue_from_previous=false"
set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/env.sh
for ARM in "$@"; do
    NAME=$ARM
    for kv in ${LOOP_SET:-}; do [[ $kv == run_name=* ]] && NAME=${kv#run_name=}; done
    mkdir -p "runs/$NAME"
    # shellcheck disable=SC2086
    # --export=NONE: a clean job env (env.sh sets PATH), even when submitted from inside a job.
    sbatch --export=NONE ${SBATCH_EXTRA:-} --job-name="st-$NAME" --output="runs/$NAME/slurm-%j.out" \
        slurm/run_arm.sbatch "$ARM" ${LOOP_SET:+--set $LOOP_SET}
done
