import pytest

from src.evaluate import distinct_n, majority_answer, pass_at_k
from src.filters import (
    auroc,
    filter_quality,
    majority_accept,
    majority_vote,
    oracle_accept,
    select_training,
    self_judge_accept,
    training_stats,
)
from src.judge import p_yes, parse_verdict


def s(pid, pred, gt=5.0, response=None, finish="stop", idx=0):
    return {
        "problem_id": pid,
        "sample_idx": idx,
        "question": f"q{pid}",
        "pred": pred,
        "gt": gt,
        "correct": pred is not None and abs(pred - gt) < 1e-9,
        "response": response if response is not None else f"{pid}-{idx}-{pred}",
        "finish_reason": finish,
        "n_tokens": 10,
    }


def problem(pid, preds, gt=5.0):
    return [s(pid, p, gt, idx=i) for i, p in enumerate(preds)]


# ------------------------------------------------------------------ oracle

def test_oracle_accepts_exactly_correct():
    samples = problem("a", [5, 4, None, 5.0])
    assert oracle_accept(samples) == [True, False, False, True]


# ------------------------------------------------------------------ majority

def test_majority_vote_basic():
    assert majority_vote([5, 5, 5, 4, None, 3, 2, 1], 3) == "5"


def test_majority_vote_needs_min_votes():
    assert majority_vote([5, 5, 4, 3, 2, 1, None, None], 3) is None


def test_majority_vote_tie_is_skipped():
    assert majority_vote([5, 5, 5, 4, 4, 4, 1, 2], 3) is None


def test_majority_vote_ignores_none():
    assert majority_vote([None, None, None, None, 7, 7, 7, 1], 3) == "7"


def test_majority_vote_normalizes_floats():
    assert majority_vote([5, 5.0, 5.0000000001, 4], 3) == "5"


def test_majority_ignores_truncated_samples():
    samples = problem("a", [5, 5, 5, 4])
    samples[0]["finish_reason"] = "length"  # only two complete votes for 5 remain
    assert majority_accept(samples, 3) == [False] * 4
    samples = problem("b", [5, 5, 5, 5])
    samples[3]["finish_reason"] = "length"
    assert majority_accept(samples, 3) == [True, True, True, False]


def test_majority_accept_can_be_wrong():
    # Majority answer 4 is wrong: the filter keeps the wrong samples and drops the right one.
    samples = problem("a", [4, 4, 4, 5]) + problem("b", [1, 2, 3, 5])
    assert majority_accept(samples, 3) == [True, True, True, False, False, False, False, False]


# ------------------------------------------------------------------ self-judge

@pytest.mark.parametrize(
    "text, verdict",
    [
        ("Yes", True),
        ("No", False),
        ("yes.", True),
        ("No.", False),
        ("**Yes**", True),
        ("  YES, the solution is correct.", True),
        ("No, the final answer should be 12.", False),
        ('"No"', False),
        ("The solution is correct.", True),
        ("The solution is incorrect.", False),
        ("Not correct", False),
        ("The answer is wrong", False),
        ("I think yes", True),
        ("Yes or no? Hard to say.", True),  # leading Yes wins
        ("Maybe yes, maybe no", None),
        ("", None),
        ("Let me check the steps", None),
        ("Nope", None),
    ],
)
def test_parse_verdict(text, verdict):
    assert parse_verdict(text) is verdict


def test_self_judge_accept_only_true():
    assert self_judge_accept([True, False, None]) == [True, False, False]


def test_p_yes():
    import math

    assert p_yes({"Yes": math.log(0.6), "No": math.log(0.2), "The": math.log(0.2)}) \
        == pytest.approx(0.75)
    assert p_yes({" yes": math.log(0.3), "YES": math.log(0.3), "no": math.log(0.6)}) \
        == pytest.approx(0.5)
    assert p_yes({"The": 0.0}) is None
    assert p_yes(None) is None


# ------------------------------------------------------------------ selection

def test_select_caps_per_problem_and_dedupes():
    samples = [s("a", 5, response="same", idx=0), s("a", 5, response="same", idx=1),
               s("a", 5, response="other", idx=2), s("a", 5, response="third", idx=3),
               s("b", 5, idx=0)]
    kept = select_training(samples, [True] * 5, keep_per_problem=2, seed=0)
    by_pid = {}
    for k in kept:
        by_pid.setdefault(k["problem_id"], []).append(k["response"])
    assert len(by_pid["a"]) == 2
    assert len(set(by_pid["a"])) == 2
    assert len(by_pid["b"]) == 1


def test_select_dedupe_toggle():
    samples = [s("a", 5, response="same", idx=0), s("a", 5, response="same", idx=1)]
    assert len(select_training(samples, [True, True], 5, 0, dedupe=True)) == 1
    assert len(select_training(samples, [True, True], 5, 0, dedupe=False)) == 2


def test_select_drops_truncated_and_rejected():
    samples = [s("a", 5, finish="length", idx=0), s("a", 5, idx=1), s("a", 4, idx=2)]
    kept = select_training(samples, [True, False, True], 5, 0)
    assert [k["sample_idx"] for k in kept] == [2]
    kept = select_training(samples, [True, False, True], 5, 0, drop_truncated=False)
    assert sorted(k["sample_idx"] for k in kept) == [0, 2]


def test_select_is_seeded():
    samples = [s("a", 5, idx=i) for i in range(8)]
    a = select_training(samples, [True] * 8, 2, seed=1)
    b = select_training(samples, [True] * 8, 2, seed=1)
    assert a == b


# ------------------------------------------------------------------ stats

def test_filter_quality():
    samples = problem("a", [5, 5, 4, 4])  # 2 correct
    q = filter_quality(samples, [True, False, True, False])
    assert q["precision"] == 0.5
    assert q["recall"] == 0.5
    assert q["self_reported_acc"] == 0.5
    assert q["true_pool_acc"] == 0.5


def test_filter_quality_nothing_accepted():
    q = filter_quality(problem("a", [5, 4]), [False, False])
    assert q["precision"] is None
    assert q["recall"] == 0.0


def test_training_stats():
    kept = [s("a", 5), s("a", 5, idx=1), s("b", 4)]
    t = training_stats(kept, n_problems=4)
    assert t["n_kept"] == 3
    assert t["frac_problems_with_kept"] == 0.5
    assert t["kept_precision"] == pytest.approx(2 / 3)


def test_auroc():
    assert auroc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
    assert auroc([0.9, 0.8, 0.2, 0.1], [False, False, True, True]) == 0.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [False, True, False, True]) == 0.5
    assert auroc([0.1, 0.4, 0.35, 0.8], [False, False, True, True]) == 0.75
    assert auroc([0.1, 0.2], [True, True]) is None


# ------------------------------------------------------------------ eval metrics

def test_pass_at_k():
    assert pass_at_k(8, 0, 8) == 0.0
    assert pass_at_k(8, 1, 8) == 1.0
    assert pass_at_k(8, 2, 1) == pytest.approx(0.25)
    assert pass_at_k(8, 1, 4) == pytest.approx(0.5)


def test_majority_answer_ties_go_to_first_seen():
    assert majority_answer([3, 4, 4, 3]) == 3
    assert majority_answer([None, None]) is None
    assert majority_answer([2, 5, 5]) == 5


def test_distinct_n():
    assert distinct_n(["a b c", "a b c"], 2) == 0.5
    assert distinct_n(["a b c", "d e f"], 2) == 1.0
    assert distinct_n(["a"], 2) == 0.0
    assert distinct_n(["a b c d", "a b x y"], 2, max_words=2) == 0.5


def test_paired_bootstrap():
    from src.plots import paired_bootstrap

    a = {str(i): i % 2 == 0 for i in range(200)}
    d, lo, hi = paired_bootstrap(a, a)
    assert d == 0 and lo == 0 and hi == 0
    b = {k: False for k in a}
    d, lo, hi = paired_bootstrap(a, b)
    assert d == 0.5 and lo < 0.5 < hi
