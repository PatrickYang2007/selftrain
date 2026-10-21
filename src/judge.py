"""Self-judge: the current model grades its own solutions, with no access to ground truth.

The judge sees the problem and one sampled solution and is asked
"Is this solution correct? Reply with only Yes or No." at temperature 0.

Besides the parsed verdict we record P(Yes) from the first generated token's logprobs
(Yes-like vs No-like tokens, renormalized). The verdict is what the filter uses; P(Yes) only
feeds the AUROC diagnostic, which separates "the judge has no signal" from "the judge has
signal but says Yes too often".
"""
from __future__ import annotations

import math
import re

JUDGE_SYSTEM = "You are a helpful assistant."
JUDGE_QUESTION = "Is this solution correct? Reply with only Yes or No."
JUDGE_TEMPLATE = "Problem:\n{question}\n\nProposed solution:\n{response}\n\n" + JUDGE_QUESTION

_LEADING_JUNK = re.compile(r"^[\s*_#>`\"'(\[]+")
_FIRST_WORD = re.compile(r"(yes|no)\b")
_YES_NO_WORD = re.compile(r"\b(yes|no)\b")
_NEGATIVE = re.compile(r"\b(incorrect|not correct|wrong|false)\b")
_POSITIVE = re.compile(r"\b(correct|right|true)\b")


def judge_messages(question: str, response: str) -> list[dict]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_TEMPLATE.format(question=question, response=response)},
    ]


def parse_verdict(text: str) -> bool | None:
    """Yes -> True, No -> False, anything we cannot read -> None.

    Order: a leading Yes/No (after markdown or quotes) wins. Otherwise, if exactly one of the
    words yes / no appears, use it. Otherwise fall back to correct / incorrect wording.
    """
    t = _LEADING_JUNK.sub("", text.strip().lower())
    m = _FIRST_WORD.match(t)
    if m:
        return m.group(1) == "yes"
    words = set(_YES_NO_WORD.findall(t))
    if len(words) == 1:
        return words.pop() == "yes"
    if words:
        return None
    if _NEGATIVE.search(t):
        return False
    if _POSITIVE.search(t):
        return True
    return None


def p_yes(first_logprobs: dict[str, float] | None) -> float | None:
    """P(Yes) / (P(Yes) + P(No)) over the top first-token candidates; None if neither shows up."""
    if not first_logprobs:
        return None
    yes = no = 0.0
    for token, lp in first_logprobs.items():
        word = token.strip().lower()
        if word == "yes":
            yes += math.exp(lp)
        elif word == "no":
            no += math.exp(lp)
    if yes + no == 0:
        return None
    return yes / (yes + no)


def run_judge(model: str, cfg: dict, items: list[tuple[str, str]]) -> list[dict]:
    """Judge (question, response) pairs with `model`. Returns one dict per pair."""
    from src.data import encode, prompt_text
    from src.sample import load_tokenizer, make_generator

    tok = load_tokenizer(model)
    prompts = [encode(tok, prompt_text(tok, judge_messages(q, r))) for q, r in items]
    gen = make_generator(model, cfg)
    outputs = gen.generate(prompts, 1, 0.0, cfg["judge_max_new_tokens"], cfg["seed"],
                           top_logprobs=cfg.get("judge_top_logprobs", 0))
    results = []
    for (comp,) in outputs:
        results.append({
            "judge_raw": comp.text,
            "verdict": parse_verdict(comp.text),
            "p_yes": p_yes(comp.first_logprobs),
        })
    return results
