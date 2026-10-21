# Cluster environment for selftrain jobs. Set SELFTRAIN_ENV (the conda env) and
# SELFTRAIN_SCRATCH (large-file storage) in the environment or in slurm/env.local.sh, which is
# not tracked by git:
#   SELFTRAIN_ENV=/path/to/envs/selftrain
#   SELFTRAIN_SCRATCH=/path/to/scratch/selftrain
if [ -f slurm/env.local.sh ]; then
    source slurm/env.local.sh
fi
: "${SELFTRAIN_ENV:?set SELFTRAIN_ENV (see slurm/env.sh)}"
: "${SELFTRAIN_SCRATCH:?set SELFTRAIN_SCRATCH (see slurm/env.sh)}"
export HF_HOME="${HF_HOME:-$SELFTRAIN_SCRATCH/hf_cache}"
export PATH="$SELFTRAIN_ENV/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
# Run vLLM's engine in the calling process: one process per stage, nothing left holding GPU memory.
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTHONUNBUFFERED=1
# runs/ must live on scratch: one checkpoint per round per arm adds up fast.
if [ ! -e runs ]; then
    mkdir -p "$SELFTRAIN_SCRATCH/runs"
    ln -s "$SELFTRAIN_SCRATCH/runs" runs
fi
