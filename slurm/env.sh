# Cluster environment for selftrain jobs. Every path can be overridden before sourcing.
: "${SELFTRAIN_ENV:=/gpfs/gibbs/project/garg/py97/envs/selftrain}"
: "${SELFTRAIN_SCRATCH:=/vast/palmer/scratch/garg/py97/selftrain}"
export HF_HOME="${HF_HOME:-$SELFTRAIN_SCRATCH/hf_cache}"
export PATH="$SELFTRAIN_ENV/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# runs/ must live on scratch: one checkpoint per round per arm adds up fast.
if [ ! -e runs ]; then
    mkdir -p "$SELFTRAIN_SCRATCH/runs"
    ln -s "$SELFTRAIN_SCRATCH/runs" runs
fi
