"""Loss masking and chat-template consistency, checked against the real Qwen tokenizer."""
import pytest

from src.data import build_messages, prompt_text, solve_prompt_ids
from src.train import IGNORE, Collator, tokenize_example

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(MODEL)
    except OSError:
        pytest.skip(f"{MODEL} tokenizer not available offline")


def test_prompt_tokens_are_masked(tok):
    q, resp = "What is 2 + 3?", "2 + 3 = 5.\nAnswer: 5"
    ex = tokenize_example(tok, q, resp)
    prompt = solve_prompt_ids(tok, q)
    n = len(prompt)
    # The training prompt is token-for-token the sampling prompt.
    assert ex["input_ids"][:n] == prompt
    assert all(label == IGNORE for label in ex["labels"][:n])
    # Everything after the prompt is trained on: the response plus the end-of-turn token.
    completion = ex["labels"][n:]
    assert IGNORE not in completion
    assert completion == ex["input_ids"][n:]
    text = tok.decode(completion)
    assert text.startswith(resp)
    assert "<|im_end|>" in text


def test_system_prompt_is_ours(tok):
    text = prompt_text(tok, build_messages("q"))
    assert "Answer: <number>" in text
    assert "Alibaba" not in text  # Qwen's default system prompt must not sneak in
    assert text.endswith("<|im_start|>assistant\n")


def test_collator_pads_labels_with_ignore(tok):
    a = tokenize_example(tok, "q1", "Answer: 1")
    b = tokenize_example(tok, "a much longer question here", "longer reasoning.\nAnswer: 2")
    batch = Collator(tok.pad_token_id)([a, b])
    width = batch["input_ids"].shape[1]
    short = len(a["input_ids"])
    assert width == max(len(a["input_ids"]), len(b["input_ids"]))
    assert (batch["labels"][0, short:] == IGNORE).all()
    assert batch["attention_mask"][0, :short].all()
    assert not batch["attention_mask"][0, short:].any()
