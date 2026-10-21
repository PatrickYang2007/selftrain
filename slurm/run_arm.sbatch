#!/bin/bash
# One arm of the self-training loop. Submit through slurm/submit.sh, which sets the log path
# to runs/<arm>/slurm-%j.out. loop.py resumes, so a requeued or resubmitted job picks up where it
# stopped. The 5.5h limit keeps gpu_devel (6h cap) eligible. scavenge_gpu is left out: on the
# scavenge nodes tried, jobs could not initialize CUDA at all.
#SBATCH --job-name=selftrain
#SBATCH --partition=gpu,gpu_devel
#SBATCH --constraint=a100|a5000
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=5:30:00
#SBATCH --requeue
set -euo pipefail
ARM=${1:?usage: run_arm.sbatch <config-name> [--set key=value ...]}
shift
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
echo "job $SLURM_JOB_ID on $(hostname), gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -m src.loop --config "configs/${ARM}.yaml" "$@"
