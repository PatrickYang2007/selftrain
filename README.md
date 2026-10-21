# selftrain

A multi-round self-training loop for a small LLM (Qwen2.5-0.5B-Instruct on GSM8K). The question
is where self-improvement breaks when the model filters its own training data. The loop is
fixed and only the filter changes between arms:

| arm | keeps a sampled solution when... | uses ground truth to filter? |
|---|---|---|
| `oracle` | its final answer matches the ground truth | yes |
| `self_judge` | the current model says "Yes" when asked whether it is correct | no |
| `majority` | its answer matches the majority answer across the model's 8 samples | no |

Ground truth is used for evaluation in every arm. The design is in [PLAN.md](PLAN.md).

## Layout

```
configs/      base.yaml (all hyperparameters), one file per arm, debug.yaml (tiny)
src/          data, answer extraction, sampling, filters, judge, training, eval, loop, plots
tests/        answer extraction, filters/metrics, loss masking against the real tokenizer
slurm/        env.sh, run_arm.sbatch, submit.sh
runs/         per-run outputs (symlink to scratch on the cluster; not in git)
results/      written by src.plots: summary.csv, figures/, logs/ (not in git)
```

Each round writes `runs/<run>/round_<r>/` with `samples.jsonl`, `sample_stats.json`,
`kept.jsonl`, `filter_stats.json`, `judged.jsonl` (self-judge only), `model/`,
`train_log.json`, `eval.json`, `eval_greedy.jsonl`, `eval_sampled.jsonl`, `timing.json`.

## Running

Cluster (Slurm; one GPU per arm). Install `requirements.txt` into a conda env, then point the
scripts at it and at a large-file directory in `slurm/env.local.sh` (untracked):

```bash
SELFTRAIN_ENV=/path/to/envs/selftrain           # conda env with requirements.txt installed
SELFTRAIN_SCRATCH=/path/to/scratch/selftrain    # HF cache and runs/ (checkpoints) go here
```

```bash
slurm/submit.sh oracle self_judge majority      # three jobs, logs in runs/<arm>/slurm-<id>.out
python -m src.plots                             # results/: summary.csv + figures (partial runs ok)
```

`loop.py` resumes from the last round with an `eval.json`, and from the last finished stage
inside a round, so a requeued or timed-out job can be resubmitted as is. The first launch
freezes the resolved config in `runs/<run>/config.yaml` and copies `src/` to `runs/<run>/code/`,
and every stage runs from that copy, so editing the repo cannot change a run halfway through.
A relaunch with different settings is refused.

Variants use overrides and a new run name:

```bash
LOOP_SET="run_name=oracle_star continue_from_previous=false" slurm/submit.sh oracle
```

Laptop (no vLLM; transformers `generate` on MPS/CPU):

```bash
pip install torch transformers accelerate datasets pandas matplotlib pyyaml pytest
python -m src.loop --config configs/debug.yaml                     # oracle, tiny
python -m src.loop --config configs/debug.yaml --set arm=self_judge
```

Tests: `python -m pytest`. The masking tests download the Qwen tokenizer on first use.

## Stages

Every stage is its own process (`python -m src.<stage> --config runs/<run>/config.yaml --round r`),
so vLLM frees GPU memory before training starts.

1. `sample`: model r-1 draws 8 samples (temperature 0.8, 512 tokens max) for each of the 2000
   pool problems.
2. `filters`: the arm's accept decision on every sample, then the same selection for all arms:
   drop truncated samples, dedupe identical responses, keep at most 2 per problem.
3. `train`: one epoch of full fine-tuning on the kept set with loss on assistant tokens only
   (lr 1e-5, effective batch 32, cosine schedule, bf16 mixed precision, fp32 checkpoints).
4. `evaluate`: greedy and 8-sample metrics on 500 GSM8K test problems, and greedy accuracy on
   all 1000 SVAMP problems.
