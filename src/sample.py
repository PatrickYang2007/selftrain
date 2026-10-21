"""Generation backends and the pool-sampling stage.

Two backends behind one interface:
- `vllm` for the cluster.
- `hf` (plain transformers `generate`) for laptops, where vLLM does not run.

Both take pre-tokenized prompts, so the exact token ids built by src/data.py reach the model,
and both override the model's generation_config (Qwen ships top_k=20, top_p=0.8 and
repetition_penalty=1.05 there) so that "temperature 0.8" means plain temperature sampling.

Stage CLI:  python -m src.sample --config runs/<arm>/config.yaml --round r
Samples the pool with the model from round r-1 and writes round_r/samples.jsonl.
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

from src.answer import extract_answer, is_correct
from src.common import (
    is_done,
    load_config,
    mark_done,
    model_path,
    round_dir,
    seed_everything,
    write_json,
    write_jsonl,
)


@dataclass
class Completion:
    text: str
    n_tokens: int
    finish_reason: str  # "stop" or "length"
    first_logprobs: dict[str, float] | None = None  # decoded token -> logprob at position 0


def _merge_logprob(d: dict[str, float], key: str, lp: float) -> None:
    # Several token ids can decode to the same string; combine them in probability space.
    if key in d:
        hi, lo = max(d[key], lp), min(d[key], lp)
        d[key] = hi + math.log1p(math.exp(lo - hi))
    else:
        d[key] = lp


class VLLMGenerator:
    def __init__(self, model: str, cfg: dict):
        from vllm import LLM

        self.llm = LLM(
            model=model,
            tokenizer=model,
            dtype="bfloat16",
            seed=cfg["seed"],
            gpu_memory_utilization=cfg["gpu_memory_utilization"],
            max_model_len=cfg["max_model_len"],
            enable_prefix_caching=True,
        )

    def generate(self, prompts: list[list[int]], n: int, temperature: float,
                 max_new_tokens: int, seed: int, top_logprobs: int = 0) -> list[list[Completion]]:
        from vllm import SamplingParams

        # One seed per prompt keeps each problem's samples independent of batching order.
        params = [
            SamplingParams(
                n=n,
                temperature=temperature,
                top_p=1.0,
                repetition_penalty=1.0,
                max_tokens=max_new_tokens,
                seed=None if temperature == 0 else seed + i,
                logprobs=top_logprobs or None,
            )
            for i in range(len(prompts))
        ]
        outputs = self.llm.generate(
            [{"prompt_token_ids": ids} for ids in prompts], params, use_tqdm=True
        )
        results = []
        for out in outputs:
            comps = []
            for o in out.outputs:
                first = None
                if top_logprobs and o.logprobs:
                    first = {}
                    for lp in o.logprobs[0].values():
                        _merge_logprob(first, lp.decoded_token or "", lp.logprob)
                comps.append(Completion(o.text, len(o.token_ids), o.finish_reason or "stop", first))
            results.append(comps)
        return results


class HFGenerator:
    def __init__(self, model: str, cfg: dict):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.cuda.is_available():
            self.device, dtype = "cuda", torch.bfloat16
        elif torch.backends.mps.is_available():
            self.device, dtype = "mps", torch.float16
        else:
            self.device, dtype = "cpu", torch.float32
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype).to(self.device)
        self.model.eval()
        self.batch_size = cfg.get("hf_batch_size", 16)
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else min(self.eos_ids)

    def generate(self, prompts: list[list[int]], n: int, temperature: float,
                 max_new_tokens: int, seed: int, top_logprobs: int = 0) -> list[list[Completion]]:
        import torch

        torch.manual_seed(seed)
        results: list[list[Completion]] = []
        for start in range(0, len(prompts), self.batch_size):
            batch = prompts[start:start + self.batch_size]
            width = max(len(p) for p in batch)
            ids = torch.tensor([[self.pad_id] * (width - len(p)) + p for p in batch])
            mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in batch])
            kwargs = dict(do_sample=False)
            if temperature > 0:
                kwargs = dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
            with torch.no_grad():
                out = self.model.generate(
                    input_ids=ids.to(self.device),
                    attention_mask=mask.to(self.device),
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=n,
                    repetition_penalty=1.0,
                    pad_token_id=self.pad_id,
                    eos_token_id=sorted(self.eos_ids),
                    output_scores=bool(top_logprobs),
                    return_dict_in_generate=True,
                    **kwargs,
                )
            new = out.sequences[:, width:].tolist()
            first_lp = None
            if top_logprobs:
                first_lp = torch.log_softmax(out.scores[0].float(), dim=-1).topk(top_logprobs)
            for b in range(len(batch)):
                comps = []
                for j in range(n):
                    row = b * n + j
                    toks = new[row]
                    stop = next((k for k, t in enumerate(toks) if t in self.eos_ids), None)
                    gen = toks if stop is None else toks[:stop]
                    first = None
                    if first_lp is not None:
                        first = {}
                        for lp, tid in zip(first_lp.values[row].tolist(), first_lp.indices[row].tolist()):
                            _merge_logprob(first, self.tok.decode([tid]), lp)
                    comps.append(Completion(
                        self.tok.decode(gen, skip_special_tokens=True),
                        len(gen) + (0 if stop is None else 1),
                        "length" if stop is None else "stop",
                        first,
                    ))
                results.append(comps)
        return results


def make_generator(model: str, cfg: dict):
    backend = cfg.get("backend", "vllm")
    if backend == "vllm":
        return VLLMGenerator(model, cfg)
    if backend == "hf":
        return HFGenerator(model, cfg)
    raise ValueError(f"unknown backend {backend!r}")


def load_tokenizer(model: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model)


def score_completion(comp: Completion, gt: float) -> dict:
    ex = extract_answer(comp.text)
    return {
        "response": comp.text,
        "n_tokens": comp.n_tokens,
        "finish_reason": comp.finish_reason,
        "pred": ex.value,
        "extract_method": ex.method,
        "correct": is_correct(ex.value, gt),
    }


def sample_stats(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "n_samples": n,
        "accuracy": sum(r["correct"] for r in rows) / n,
        "extract_fail_rate": sum(r["pred"] is None for r in rows) / n,
        "format_answer_line": sum(r["extract_method"] == "answer_line" for r in rows) / n,
        "format_boxed": sum(r["extract_method"] == "boxed" for r in rows) / n,
        "format_last_number": sum(r["extract_method"] == "last_number" for r in rows) / n,
        "truncated_rate": sum(r["finish_reason"] == "length" for r in rows) / n,
        "mean_tokens": sum(r["n_tokens"] for r in rows) / n,
    }


def sample_pool(cfg: dict, r: int) -> None:
    from src.data import get_pool, solve_prompt_ids

    out_path = round_dir(cfg, r) / "samples.jsonl"
    if is_done(out_path):
        print(f"[sample] {out_path} already done, skipping")
        return
    seed = cfg["seed"] + 1000 * r
    seed_everything(seed)
    model = model_path(cfg, r - 1)
    pool = get_pool(cfg)
    tok = load_tokenizer(model)
    prompts = [solve_prompt_ids(tok, p.question) for p in pool]
    t0 = time.time()
    gen = make_generator(model, cfg)
    outputs = gen.generate(prompts, cfg["samples_per_problem"], cfg["sample_temperature"],
                           cfg["max_new_tokens"], seed)
    rows = []
    for p, comps in zip(pool, outputs):
        for j, comp in enumerate(comps):
            rows.append({"problem_id": p.id, "sample_idx": j, "question": p.question,
                         "gt": p.answer, **score_completion(comp, p.answer)})
    write_jsonl(out_path, rows)
    stats = sample_stats(rows) | {"model": model, "seconds": time.time() - t0}
    write_json(round_dir(cfg, r) / "sample_stats.json", stats)
    mark_done(out_path)
    print(f"[sample] round {r}: {stats}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--round", type=int, required=True)
    args = ap.parse_args()
    sample_pool(load_config(args.config), args.round)


if __name__ == "__main__":
    main()
