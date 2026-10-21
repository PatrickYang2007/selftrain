# Self-Training Loop: Where Does LLM Self-Improvement Break?

## Goal

Build a multi-round self-improvement loop for a small LLM and measure how it behaves over rounds. The model solves problems, we keep the solutions that pass a filter, fine-tune on them, and repeat.

The research question: how does the loop behave when the filter is ground truth vs when the model judges itself? Where does it plateau, lose diversity, or start fooling itself?

Three arms, same loop, different filter:

1. **oracle**: keep solutions whose final answer matches the ground truth
2. **self_judge**: keep solutions the current model says are correct (no ground truth used)
3. **majority**: keep solutions whose answer matches the majority vote across the model's own samples (no ground truth used)

Ground truth is always used for evaluation in every arm, just never for filtering in arms 2 and 3.

## Setup

- Model: `Qwen/Qwen2.5-0.5B-Instruct` for development and the main runs. Switch to the 1.5B version only if everything works and there is time left.
- Train data: GSM8K train split. Use a fixed random subset of 2000 problems as the self-training pool.
- Eval data:
  - In-distribution: GSM8K test, fixed random subset of 500 problems
  - Out-of-distribution: SVAMP (full test set, or 500 problems)
- Libraries: `vllm` for sampling, `transformers` + `trl` (SFTTrainer) for fine-tuning, `datasets`, `pandas`, `matplotlib`
- Hardware: one GPU (A100/H100 class). Run sampling and training as separate processes so vLLM releases GPU memory between stages.
- Seeds fixed everywhere. Every run writes to its own directory.

## Where things run

- **Local (MacBook)**: development and debugging only. vLLM does not run on macOS, so `sample.py` needs a `backend` option: `vllm` on the cluster, plain `transformers` `generate` on MPS locally. Use a `configs/debug.yaml` with tiny settings (pool_size 50, samples_per_problem 2, rounds 1, eval_subset 50) to check the whole pipeline end to end on the laptop.
- **Yale Slurm cluster**: all real runs.
  - One conda env with vllm, transformers, trl, datasets. Pin versions in `requirements.txt`.
  - Put the HF cache and `runs/` on scratch storage, not home (models and checkpoints will blow past home quotas).
  - `slurm/run_arm.sbatch` takes the arm name and runs `loop.py` for it. Request 1 GPU (A100/H100 if the partition has them), ~8 CPUs, ~64 GB RAM, and a time limit with margin.
  - Submit the three arms as three separate jobs so they run in parallel if GPUs are free.
  - `loop.py` must resume: on start, find the last round whose `eval.json` exists and continue from there. Jobs can hit time limits or get preempted.
  - Inside the job, still run sample / train / eval as separate subprocesses so vLLM frees GPU memory between stages.
  - Write logs to `runs/<arm>/slurm-%j.out`.

## Repo layout

```
selftrain/
  PLAN.md
  configs/
    base.yaml            # shared hyperparameters
    oracle.yaml
    self_judge.yaml
    majority.yaml
  src/
    data.py              # load GSM8K / SVAMP, fixed subsets, prompt formatting
    answer.py            # final answer extraction + normalization (CRITICAL, unit tested)
    sample.py            # vLLM sampling: K samples per problem
    filters.py           # oracle / self_judge / majority filters
    judge.py             # self-judge prompt + parsing
    train.py             # SFT on filtered samples
    evaluate.py          # pass@1, pass@k, diversity, OOD
    loop.py              # runs N rounds for one arm
    plots.py             # all figures from run logs
  tests/
    test_answer.py
    test_filters.py
  runs/
    <arm>/round_<r>/     # samples.jsonl, kept.jsonl, model/, eval.json
  results/
    figures/
    summary.csv
  REPORT.md
```

## Prompt format

Use the model's chat template via `tokenizer.apply_chat_template`. Same template for sampling, training, and eval.

System: "Solve the math problem. Show your reasoning step by step, then give the final answer on the last line as: Answer: <number>"

Answer extraction: take the last line matching `Answer:`, strip commas, `$`, and trailing periods, parse as a number. If no match, fall back to the last number in the response. Log the extraction-failure rate every round, since a rising failure rate is itself a signal.

GSM8K ground truth is the number after `####` in the solution field.

## Loop (per arm)

Hyperparameters in `configs/base.yaml`:

```yaml
rounds: 5
pool_size: 2000
samples_per_problem: 8
sample_temperature: 0.8
max_new_tokens: 512
keep_per_problem: 2        # cap to avoid easy problems dominating
lr: 1.0e-5
epochs: 1
batch_size: 32             # effective, use grad accumulation if needed
continue_from_previous: true
eval_subset: 500
eval_k: 8
```

Each round `r`:

1. **Sample**: current model generates 8 samples for each of the 2000 pool problems. Save to `samples.jsonl` with problem id, response, extracted answer, and ground-truth correctness (correctness is logged in every arm for analysis).
2. **Filter**: apply the arm's filter. Dedupe identical responses. Keep at most 2 per problem. Save `kept.jsonl`.
3. **Train**: SFT the current model on the kept samples (loss on assistant tokens only). Save to `round_<r>/model/`. If `continue_from_previous` is false, train from the original base model on the union of all kept data so far (STaR style). Default is true since that is the more RSI-like setting.
4. **Evaluate**: run `evaluate.py` on the new model. Save `eval.json`.

Round 0 is the untouched base model, evaluated before any training.

## Filters

- **oracle**: extracted answer equals ground truth.
- **majority**: for each problem, take the most common extracted answer among the 8 samples. Keep samples matching it. Skip the problem if the top answer has fewer than 3 votes.
- **self_judge**: show the current model the problem and one sample, ask "Is this solution correct? Reply with only Yes or No." Keep samples where it says Yes. Use temperature 0. Parse Yes/No robustly and log the unparseable rate.

For self_judge and majority, also log the filter's precision and recall against ground truth every round. This is the key data for the "does the model fool itself" result.

## Metrics (every round, every arm)

In-distribution (GSM8K test subset):
- pass@1: greedy decoding accuracy
- pass@8: 8 samples at temp 0.8, unbiased pass@k estimator
- maj@8: accuracy of the majority-vote answer
- mean response length in tokens
- answer extraction failure rate

Diversity (on the 8 test samples per problem):
- distinct final answers per problem
- distinct-2 / distinct-3 n-gram ratio across samples

Out-of-distribution (SVAMP):
- pass@1 greedy

Filter quality (self_judge and majority only):
- filter precision and recall vs ground truth
- "self-reported accuracy" (fraction of pool samples the filter accepts) vs true pool accuracy

Training data:
- number of kept samples
- fraction of pool problems with at least one kept sample

## Figures for the report

1. pass@1 vs round, one line per arm
2. pass@1 and pass@8 vs round for the oracle arm (looking for pass@1 up while pass@8 flat or down)
3. diversity vs round, one line per arm
4. self-reported accuracy vs true accuracy vs round for self_judge (the self-deception plot)
5. filter precision vs round for self_judge and majority
6. OOD pass@1 vs round, one line per arm
7. response length vs round, one line per arm

## Build order and checkpoints

Do not move on until each checkpoint passes.

**Phase 1: plumbing (target ~2 hours)**
- `data.py`, `answer.py` with unit tests covering commas, decimals, negatives, `$`, missing answer lines, and GSM8K `####` parsing
- `evaluate.py` on the base model
- Checkpoint: base model pass@1 on GSM8K subset is in a believable range (roughly 30 to 50 percent for the 0.5B instruct model). Manually read 20 responses and their extracted answers and confirm extraction is right.

**Phase 2: one oracle round end to end (target ~1.5 hours)**
- `sample.py`, `filters.py` (oracle only), `train.py`, `loop.py` for 1 round
- Checkpoint: kept set is non-trivial (hundreds to low thousands of samples), training loss decreases, round 1 eval runs. Read 10 kept samples by hand to confirm they are actually correct and formatted properly.

**Phase 3: the other filters (target ~1.5 hours)**
- `majority` and `self_judge` filters, `judge.py`
- Checkpoint: on round 0 samples, print precision/recall of each filter vs ground truth. Read 20 judge outputs by hand.

**Phase 4: full runs (target ~4 to 5 hours of mostly waiting)**
- 5 rounds per arm. Run arms back to back. While they run, write `plots.py` against partial logs.

**Phase 5: results (target ~2 hours)**
- All figures, `results/summary.csv`, and `REPORT.md` with: question, setup, results with figures, what broke and when, limitations, next steps.

## Things that will silently ruin the day

- Answer extraction bugs. A parser that scores everything wrong or right makes every curve meaningless. Test it first and keep checking the failure rate.
- Chat template mismatch between sampling, training, and eval.
- Training loss computed on the prompt tokens too. Mask to assistant tokens only.
- vLLM holding GPU memory when training starts. Separate processes per stage.
- Eval leaking into the training pool. Pool comes from GSM8K train only.
- Comparing arms that used different seeds or eval subsets.

## If time is short

Cut in this order: OOD eval, the 1.5B model, the majority arm, rounds down from 5 to 3. Keep oracle vs self_judge no matter what, since that comparison is the main result.

## If time is left over

- Run with `continue_from_previous: false` and compare
- Repeat on the 1.5B model and check if the self-judge drifts less
- Have the model judge with a stronger external model as a fourth arm, to see how much judge quality matters
