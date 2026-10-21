"""The three filters and the shared training-set selection.

A filter maps every pool sample to accepted / rejected:
- oracle:     extracted answer equals ground truth.
- majority:   extracted answer equals the problem's majority answer among its samples. Problems
              whose top answer has fewer than `majority_min_votes` votes, or whose top answer is
              tied with another, get nothing accepted. Truncated samples have no final answer:
              they neither vote nor get accepted.
- self_judge: the current model says "Yes" when asked whether the sample is correct.
- external_judge: the same Yes/No question answered by a fixed, stronger model (`judge_model`),
              to separate "the judge is weak" from "the model is judging itself".

Filter quality (precision / recall vs ground truth, self-reported vs true accuracy) is measured
on that raw accept decision over the whole pool. Then the same selection runs for every arm:
drop truncated samples, dedupe identical responses per problem, keep at most
`keep_per_problem` per problem (random, seeded).

Stage CLI:  python -m src.filters --config runs/<arm>/config.yaml --round r
Reads round_r/samples.jsonl, writes round_r/kept.jsonl and round_r/filter_stats.json.
"""
from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict

from src.answer import canonical
from src.common import (
    is_done,
    load_config,
    mark_done,
    model_path,
    read_jsonl,
    round_dir,
    write_json,
    write_jsonl,
)

ARMS = ("oracle", "majority", "self_judge", "external_judge")


def group_by_problem(samples: list[dict]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        groups[s["problem_id"]].append(i)
    return groups


def oracle_accept(samples: list[dict]) -> list[bool]:
    return [bool(s["correct"]) for s in samples]


def majority_vote(preds: list[float | None], min_votes: int) -> str | None:
    """Canonical majority answer, or None if no answer has a clear majority of >= min_votes."""
    counts = Counter(canonical(p) for p in preds if p is not None).most_common(2)
    if not counts or counts[0][1] < min_votes:
        return None
    if len(counts) > 1 and counts[1][1] == counts[0][1]:
        return None  # tie at the top: no majority
    return counts[0][0]


def final_answer(sample: dict) -> float | None:
    """The sample's answer for voting: None when it was cut off at max_new_tokens."""
    return None if sample["finish_reason"] == "length" else sample["pred"]


def majority_accept(samples: list[dict], min_votes: int) -> list[bool]:
    accept = [False] * len(samples)
    for idx in group_by_problem(samples).values():
        winner = majority_vote([final_answer(samples[i]) for i in idx], min_votes)
        if winner is None:
            continue
        for i in idx:
            answer = final_answer(samples[i])
            accept[i] = answer is not None and canonical(answer) == winner
    return accept


def self_judge_accept(verdicts: list[bool | None]) -> list[bool]:
    return [v is True for v in verdicts]


def select_training(samples: list[dict], accept: list[bool], keep_per_problem: int, seed: int,
                    drop_truncated: bool = True, dedupe: bool = True) -> list[dict]:
    """Accepted samples -> training rows, at most `keep_per_problem` per problem."""
    rng = random.Random(seed)
    kept = []
    for pid, idx in group_by_problem(samples).items():
        chosen, seen = [], set()
        for i in idx:
            s = samples[i]
            if not accept[i]:
                continue
            if drop_truncated and s["finish_reason"] == "length":
                continue
            key = s["response"].strip()
            if dedupe and key in seen:
                continue
            seen.add(key)
            chosen.append(s)
        if len(chosen) > keep_per_problem:
            chosen = rng.sample(chosen, keep_per_problem)
        kept.extend(chosen)
    return kept


def auroc(scores: list[float], labels: list[bool]) -> float | None:
    """Probability a random positive outscores a random negative (ties count half)."""
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    # Mann-Whitney U with average ranks for ties.
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        rank_sum += avg_rank * sum(lbl for _, lbl in pairs[i:j])
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def filter_quality(samples: list[dict], accept: list[bool]) -> dict:
    n = len(samples)
    correct = [bool(s["correct"]) for s in samples]
    tp = sum(a and c for a, c in zip(accept, correct))
    n_acc, n_cor = sum(accept), sum(correct)
    return {
        "n_samples": n,
        "n_accepted": n_acc,
        "n_correct": n_cor,
        "precision": tp / n_acc if n_acc else None,
        "recall": tp / n_cor if n_cor else None,
        "self_reported_acc": n_acc / n,
        "true_pool_acc": n_cor / n,
    }


def training_stats(kept: list[dict], n_problems: int) -> dict:
    problems = {s["problem_id"] for s in kept}
    return {
        "n_kept": len(kept),
        "n_problems_with_kept": len(problems),
        "frac_problems_with_kept": len(problems) / n_problems,
        "kept_precision": sum(s["correct"] for s in kept) / len(kept) if kept else None,
        "kept_mean_tokens": sum(s["n_tokens"] for s in kept) / len(kept) if kept else None,
    }


def apply_filter(cfg: dict, r: int) -> None:
    rd = round_dir(cfg, r)
    out_path = rd / "kept.jsonl"
    if is_done(out_path):
        print(f"[filter] {out_path} already done, skipping")
        return
    arm = cfg["arm"]
    samples = read_jsonl(rd / "samples.jsonl")
    extra: dict = {}
    if arm == "oracle":
        accept = oracle_accept(samples)
    elif arm == "majority":
        accept = majority_accept(samples, cfg["majority_min_votes"])
        votes = [majority_vote([final_answer(samples[i]) for i in idx], cfg["majority_min_votes"])
                 for idx in group_by_problem(samples).values()]
        extra["frac_problems_with_majority"] = sum(v is not None for v in votes) / len(votes)
    elif arm in ("self_judge", "external_judge"):
        judge = model_path(cfg, r - 1) if arm == "self_judge" else cfg["judge_model"]
        accept, extra = _judge_filter(cfg, r, samples, judge)
        extra["judge_model"] = judge
    else:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")

    kept = select_training(samples, accept, cfg["keep_per_problem"], cfg["seed"] + r,
                           cfg["drop_truncated"], cfg["dedupe"])
    n_problems = len(group_by_problem(samples))
    stats = {"arm": arm, "round": r} | filter_quality(samples, accept) \
        | training_stats(kept, n_problems) | extra
    write_jsonl(out_path, [{"problem_id": s["problem_id"], "question": s["question"],
                            "response": s["response"], "correct": s["correct"],
                            "n_tokens": s["n_tokens"]} for s in kept])
    write_json(rd / "filter_stats.json", stats)
    mark_done(out_path)
    print(f"[filter] round {r} {arm}: {stats}")


def _judge_filter(cfg: dict, r: int, samples: list[dict], judge: str) -> tuple[list[bool], dict]:
    """Judge every unique (problem, response) once with `judge`."""
    from src.judge import run_judge

    rd = round_dir(cfg, r)
    judged_path = rd / "judged.jsonl"
    if is_done(judged_path):
        judged = read_jsonl(judged_path)
    else:
        unique: dict[tuple[str, str], int] = {}
        items = []
        for s in samples:
            key = (s["problem_id"], s["response"])
            if key not in unique:
                unique[key] = len(items)
                items.append((s["question"], s["response"]))
        results = run_judge(judge, cfg, items)
        # Temperature-0 verdicts are a function of the input, so duplicates share one verdict.
        judged = [{"problem_id": s["problem_id"], "sample_idx": s["sample_idx"],
                   "correct": s["correct"], **results[unique[(s["problem_id"], s["response"])]]}
                  for s in samples]
        write_jsonl(judged_path, judged)
        mark_done(judged_path)
    verdicts = [j["verdict"] for j in judged]
    scored = [(j["p_yes"], j["correct"]) for j in judged if j["p_yes"] is not None]
    extra = {
        "judge_yes_rate": sum(v is True for v in verdicts) / len(verdicts),
        "judge_unparseable_rate": sum(v is None for v in verdicts) / len(verdicts),
        "judge_p_yes_coverage": len(scored) / len(judged),
        "judge_auroc": auroc([s for s, _ in scored], [c for _, c in scored]) if scored else None,
    }
    return self_judge_accept(verdicts), extra


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--round", type=int, required=True)
    args = ap.parse_args()
    apply_filter(load_config(args.config), args.round)


if __name__ == "__main__":
    main()
