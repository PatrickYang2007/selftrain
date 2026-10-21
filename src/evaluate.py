"""Evaluation: GSM8K test subset (in-distribution) and SVAMP (out-of-distribution).

Metrics (all on fixed subsets and fixed seeds, identical across arms and rounds):
- pass@1 greedy, plus its binomial standard error
- pass@1 sampled (mean accuracy of the k temperature samples), pass@k (unbiased estimator),
  maj@k (accuracy of the most common extracted answer among the k samples)
- mean response length in tokens, answer-extraction failure rate (no number at all), which
  extraction route was used (Answer: line / \\boxed / last-number fallback), truncation rate
- diversity on the k samples per problem: distinct final answers, distinct-2 / distinct-3
- SVAMP pass@1 greedy

Stage CLI:     python -m src.evaluate --config runs/<arm>/config.yaml --round r
Standalone:    python -m src.evaluate --config configs/base.yaml --model <path> --out <dir>
"""
from __future__ import annotations

import argparse
import math
import time
from collections import Counter
from pathlib import Path

from src.answer import canonical, is_correct
from src.common import (
    load_config,
    model_path,
    round_dir,
    seed_everything,
    write_json,
    write_jsonl,
)
from src.sample import load_tokenizer, make_generator, score_completion


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k (Chen et al. 2021): 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def majority_answer(preds: list[float | None]) -> float | None:
    """Most common extracted answer; ties go to the answer seen first. None if no answers."""
    keyed = [(canonical(p), p) for p in preds if p is not None]
    if not keyed:
        return None
    counts = Counter(k for k, _ in keyed)
    top = max(counts.values())
    for key, value in keyed:
        if counts[key] == top:
            return value
    return None


def distinct_n(texts: list[str], n: int) -> float:
    """Unique word n-grams / total word n-grams, pooled over the given texts."""
    grams = []
    for t in texts:
        words = t.split()
        grams.extend(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    return len(set(grams)) / len(grams) if grams else 0.0


def _rate(rows, pred) -> float:
    return sum(pred(r) for r in rows) / len(rows) if rows else float("nan")


def greedy_metrics(rows: list[dict]) -> dict:
    acc = _rate(rows, lambda r: r["correct"])
    return {
        "n": len(rows),
        "pass@1": acc,
        "pass@1_se": math.sqrt(acc * (1 - acc) / len(rows)),
        "mean_tokens": _rate(rows, lambda r: r["n_tokens"]),
        "extract_fail": _rate(rows, lambda r: r["pred"] is None),
        "format_answer_line": _rate(rows, lambda r: r["extract_method"] == "answer_line"),
        "format_boxed": _rate(rows, lambda r: r["extract_method"] == "boxed"),
        "format_last_number": _rate(rows, lambda r: r["extract_method"] == "last_number"),
        "truncated": _rate(rows, lambda r: r["finish_reason"] == "length"),
    }


def sampled_metrics(groups: list[list[dict]], gts: list[float], k: int) -> dict:
    flat = [r for g in groups for r in g]
    out = {
        "pass@1_sampled": _rate(flat, lambda r: r["correct"]),
        "mean_tokens_sampled": _rate(flat, lambda r: r["n_tokens"]),
        "extract_fail_sampled": _rate(flat, lambda r: r["pred"] is None),
        "truncated_sampled": _rate(flat, lambda r: r["finish_reason"] == "length"),
    }
    # pass@1 from samples is "pass@1_sampled" above; plain "pass@1" is the greedy metric.
    for kk in sorted({4, k} & set(range(2, k + 1))):
        out[f"pass@{kk}"] = sum(pass_at_k(len(g), sum(r["correct"] for r in g), kk)
                                for g in groups) / len(groups)
    out[f"maj@{k}"] = sum(is_correct(majority_answer([r["pred"] for r in g]), gt)
                          for g, gt in zip(groups, gts)) / len(groups)
    out["distinct_answers"] = sum(len({canonical(r["pred"]) for r in g if r["pred"] is not None})
                                  for g in groups) / len(groups)
    for n in (2, 3):
        out[f"distinct_{n}"] = sum(distinct_n([r["response"] for r in g], n)
                                   for g in groups) / len(groups)
    return out


def evaluate_model(model: str, cfg: dict, out_dir: Path) -> dict:
    from src.data import get_eval_set, get_ood_set, solve_prompt_ids

    seed = cfg["eval_seed"]
    seed_everything(seed)
    t0 = time.time()
    test = get_eval_set(cfg)
    ood = get_ood_set(cfg) if cfg.get("ood_eval", True) else []
    tok = load_tokenizer(model)
    gen = make_generator(model, cfg)

    problems = test + ood
    prompts = [solve_prompt_ids(tok, p.question) for p in problems]
    greedy = gen.generate(prompts, 1, 0.0, cfg["max_new_tokens"], seed)
    greedy_rows = [{"problem_id": p.id, "gt": p.answer, **score_completion(c[0], p.answer)}
                   for p, c in zip(problems, greedy)]

    k = cfg["eval_k"]
    sampled = gen.generate(prompts[:len(test)], k, cfg["sample_temperature"],
                           cfg["max_new_tokens"], seed)
    groups = [[{"problem_id": p.id, "sample_idx": j, "gt": p.answer,
                **score_completion(c, p.answer)} for j, c in enumerate(comps)]
              for p, comps in zip(test, sampled)]

    result = {
        "model": model,
        "gsm8k": greedy_metrics(greedy_rows[:len(test)])
        | sampled_metrics(groups, [p.answer for p in test], k),
        "seconds": None,
    }
    if ood:
        result["svamp"] = greedy_metrics(greedy_rows[len(test):])
    result["seconds"] = time.time() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "eval_greedy.jsonl", greedy_rows)
    write_jsonl(out_dir / "eval_sampled.jsonl", [r for g in groups for r in g])
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--round", type=int)
    ap.add_argument("--model", help="standalone mode: model to evaluate")
    ap.add_argument("--out", help="standalone mode: output directory")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    if args.round is not None:
        model, out_dir = model_path(cfg, args.round), round_dir(cfg, args.round)
    else:
        if not (args.model and args.out):
            ap.error("give --round, or both --model and --out")
        model, out_dir = args.model, Path(args.out)
    result = evaluate_model(model, cfg, out_dir)
    if args.round is not None:
        result["round"] = args.round
    # eval.json is written last: its existence marks the round as complete (see loop.py).
    write_json(out_dir / "eval.json", result)
    g, k = result["gsm8k"], cfg["eval_k"]
    svamp = result.get("svamp", {}).get("pass@1", float("nan"))
    print(f"[eval] {model}: gsm8k pass@1={g['pass@1']:.3f} pass@{k}={g[f'pass@{k}']:.3f} "
          f"maj@{k}={g[f'maj@{k}']:.3f} svamp pass@1={svamp:.3f}")


if __name__ == "__main__":
    main()
