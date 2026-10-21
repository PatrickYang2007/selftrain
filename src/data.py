"""Datasets, fixed subsets, and the one prompt format used everywhere.

Sampling, training, judging and evaluation all build prompts through `build_messages` and
`prompt_text`, so the chat template cannot drift between stages.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from src.answer import parse_gsm8k_answer

SYSTEM_PROMPT = (
    "Solve the math problem. Show your reasoning step by step, then give the final answer "
    "on the last line as: Answer: <number>"
)


@dataclass(frozen=True)
class Problem:
    id: str
    question: str
    answer: float


def load_gsm8k(split: str) -> list[Problem]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    return [
        Problem(f"gsm8k_{split}_{i}", ex["question"], parse_gsm8k_answer(ex["answer"]))
        for i, ex in enumerate(ds)
    ]


def load_svamp() -> list[Problem]:
    """All 1000 SVAMP problems (the HF copy splits them 700/300; none are used for training)."""
    from datasets import concatenate_datasets, load_dataset

    ds = load_dataset("ChilleD/SVAMP")
    ds = concatenate_datasets([ds["train"], ds["test"]])
    return [Problem(f"svamp_{ex['ID']}", svamp_question(ex["Body"], ex["Question"]),
                    float(ex["Answer"])) for ex in ds]


def svamp_question(body: str, question: str) -> str:
    # The dataset's own `question_concat` often runs the two together with no period.
    body = body.strip()
    if not body.endswith((".", "?", "!")):
        body += "."
    return f"{body} {question.strip()}"


def fixed_subset(problems: list[Problem], size: int | None, seed: int) -> list[Problem]:
    """Deterministic subset, kept in original order. size=None or >= len returns everything."""
    if size is None or size >= len(problems):
        return list(problems)
    idx = sorted(random.Random(seed).sample(range(len(problems)), size))
    return [problems[i] for i in idx]


def get_pool(cfg: dict) -> list[Problem]:
    return fixed_subset(load_gsm8k("train"), cfg["pool_size"], cfg["data_seed"])


def get_eval_set(cfg: dict) -> list[Problem]:
    return fixed_subset(load_gsm8k("test"), cfg["eval_subset"], cfg["data_seed"])


def get_ood_set(cfg: dict) -> list[Problem]:
    return fixed_subset(load_svamp(), cfg.get("ood_subset"), cfg["data_seed"])


# ---------------------------------------------------------------- prompts

def build_messages(question: str, response: str | None = None) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    if response is not None:
        messages.append({"role": "assistant", "content": response})
    return messages


def prompt_text(tokenizer, messages: list[dict]) -> str:
    """Chat-templated prompt string ending at the start of the assistant turn."""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def encode(tokenizer, text: str) -> list[int]:
    # The chat template already contains every special token we need.
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def solve_prompt_ids(tokenizer, question: str) -> list[int]:
    return encode(tokenizer, prompt_text(tokenizer, build_messages(question)))
