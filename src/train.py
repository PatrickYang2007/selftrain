"""SFT on the kept samples, loss on assistant tokens only.

Masking is done here rather than inside a trainer so it can be unit tested
(tests/test_train.py): the prompt is the exact token sequence used at sampling time
(src.data.solve_prompt_ids) and gets label -100; the completion is the response followed by
the chat template's end-of-turn tokens, so the model also learns to stop.

Which model and data a round trains on:
- continue_from_previous: true  -> start from round r-1's model, train on round r's kept set.
- continue_from_previous: false -> start from the base model, train on kept sets 1..r (STaR).

Stage CLI:  python -m src.train --config runs/<arm>/config.yaml --round r
Writes round_r/model/ and round_r/train_log.json.
"""
from __future__ import annotations

import argparse
import shutil
import time

from src.common import (
    is_done,
    load_config,
    mark_done,
    model_path,
    read_jsonl,
    round_dir,
    seed_everything,
    write_json,
)
from src.data import build_messages, encode, prompt_text

IGNORE = -100


def tokenize_example(tokenizer, question: str, response: str) -> dict:
    messages = build_messages(question, response)
    prompt = prompt_text(tokenizer, messages[:-1])
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    if not full.startswith(prompt):
        raise ValueError("chat template: prompt is not a prefix of the full conversation")
    prompt_ids = encode(tokenizer, prompt)
    completion_ids = encode(tokenizer, full[len(prompt):])
    return {
        "input_ids": prompt_ids + completion_ids,
        "labels": [IGNORE] * len(prompt_ids) + completion_ids,
    }


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        import torch

        width = max(len(b["input_ids"]) for b in batch)

        def pad(seq, value):
            return seq + [value] * (width - len(seq))

        return {
            "input_ids": torch.tensor([pad(b["input_ids"], self.pad_id) for b in batch]),
            "labels": torch.tensor([pad(b["labels"], IGNORE) for b in batch]),
            "attention_mask": torch.tensor([pad([1] * len(b["input_ids"]), 0) for b in batch]),
        }


def training_data(cfg: dict, r: int) -> list[dict]:
    rounds = [r] if cfg["continue_from_previous"] else range(1, r + 1)
    rows = []
    for k in rounds:
        rows.extend(read_jsonl(round_dir(cfg, k) / "kept.jsonl"))
    return rows


def train_round(cfg: dict, r: int) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    rd = round_dir(cfg, r)
    out = rd / "model"
    if is_done(out):
        print(f"[train] {out} already done, skipping")
        return
    seed = cfg["seed"] + r
    seed_everything(seed)
    init = model_path(cfg, r - 1) if cfg["continue_from_previous"] else cfg["model_name"]
    rows = training_data(cfg, r)
    # The tokenizer always comes from the base model: one chat template for every round.
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    t0 = time.time()

    if out.exists():
        shutil.rmtree(out)
    model = AutoModelForCausalLM.from_pretrained(init, torch_dtype=torch.float32)
    log: dict = {"init": init, "n_examples": len(rows), "trained": bool(rows)}
    if rows:
        dataset = [tokenize_example(tok, x["question"], x["response"]) for x in rows]
        log["n_train_tokens"] = sum(sum(t != IGNORE for t in d["labels"]) for d in dataset)
        accum = max(1, cfg["batch_size"] // cfg["micro_batch_size"])
        use_cuda = torch.cuda.is_available()
        args = TrainingArguments(
            output_dir=str(rd / "trainer_tmp"),
            per_device_train_batch_size=cfg["micro_batch_size"],
            gradient_accumulation_steps=accum,
            learning_rate=cfg["lr"],
            num_train_epochs=cfg["epochs"],
            lr_scheduler_type=cfg["lr_scheduler"],
            warmup_ratio=cfg["warmup_ratio"],
            weight_decay=cfg["weight_decay"],
            max_grad_norm=cfg["max_grad_norm"],
            bf16=use_cuda,  # mixed precision; master weights stay fp32
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            seed=seed,
            data_seed=seed,
            remove_unused_columns=False,
            dataloader_num_workers=0,
        )
        trainer = Trainer(model=model, args=args, train_dataset=dataset,
                          data_collator=Collator(tok.pad_token_id))
        trainer.train()
        log["log_history"] = trainer.state.log_history
        losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
        if losses:
            k = max(1, len(losses) // 10)
            log["loss_first10pct"] = sum(losses[:k]) / k
            log["loss_last10pct"] = sum(losses[-k:]) / k
        shutil.rmtree(rd / "trainer_tmp", ignore_errors=True)
    else:
        print(f"[train] round {r}: no kept samples, carrying {init} forward unchanged")

    # Saved in fp32: rounding to bf16 between rounds would erase per-weight updates smaller than
    # half a bf16 step, which at lr 1e-5 is most of them. Inference still runs in bf16.
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    log["seconds"] = time.time() - t0
    write_json(rd / "train_log.json", log)
    mark_done(out)
    print(f"[train] round {r}: {len(rows)} examples, "
          f"loss {log.get('loss_first10pct')} -> {log.get('loss_last10pct')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--round", type=int, required=True)
    args = ap.parse_args()
    train_round(load_config(args.config), args.round)


if __name__ == "__main__":
    main()
