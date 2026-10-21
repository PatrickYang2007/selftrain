"""Run the self-training loop for one arm, resuming wherever it stopped.

    python -m src.loop --config configs/oracle.yaml [--set key=value ...]

Round 0 evaluates the untouched base model. Round r >= 1:
    sample (model r-1) -> filter -> train (-> model r) -> evaluate (model r)
Each stage runs as its own subprocess so vLLM's GPU memory is released before training.

Resume: a round is complete when its eval.json exists. Within an unfinished round, stages whose
outputs carry a done marker are skipped, so a preempted job picks up at the stage it was in.

The resolved config is frozen in runs/<run_name>/config.yaml on the first launch, and so is the
code: src/ is copied to runs/<run_name>/code/src and every stage runs from that copy. Editing the
repo therefore cannot change a run halfway through. A relaunch with a different config is refused
unless --allow-config-change is given; --refresh-code re-copies src/ on purpose.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

from src.common import (
    REPO_ROOT,
    atomic_write_text,
    load_config,
    read_json,
    round_dir,
    run_dir,
    runs_root,
    save_config,
    write_json,
)

STAGES = ("sample", "filters", "train", "evaluate")


def completed_rounds(cfg: dict) -> int:
    """Index of the last round with an eval.json, or -1 if none (rounds must be contiguous)."""
    last = -1
    while (round_dir(cfg, last + 1) / "eval.json").exists():
        last += 1
    return last


def run_stage(stage: str, config_path: str, r: int, code_dir: Path) -> float:
    t0 = time.time()
    cmd = [sys.executable, "-m", f"src.{stage}", "--config", config_path, "--round", str(r)]
    print(f"\n=== round {r}: {stage} ===", flush=True)
    subprocess.run(cmd, cwd=code_dir, check=True)
    return time.time() - t0


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def snapshot_code(cfg: dict, refresh: bool) -> Path:
    """Copy src/ into the run directory once; stages import from the copy."""
    code_dir = run_dir(cfg) / "code"
    if refresh and code_dir.exists():
        shutil.rmtree(code_dir)
    if not (code_dir / "src").exists():
        shutil.copytree(REPO_ROOT / "src", code_dir / "src",
                        ignore=shutil.ignore_patterns("__pycache__"))
        dirty = _git("status", "--porcelain", "src")
        atomic_write_text(code_dir / "VERSION",
                          f"commit {_git('rev-parse', 'HEAD')}\n"
                          f"uncommitted changes in src/: {'yes' if dirty else 'no'}\n{dirty}\n")
    return code_dir


def freeze_config(cfg: dict, allow_change: bool) -> str:
    path = run_dir(cfg) / "config.yaml"
    if path.exists():
        frozen = load_config(path)
        if frozen != cfg:
            diff = {k: (frozen.get(k), cfg.get(k)) for k in set(frozen) | set(cfg)
                    if frozen.get(k) != cfg.get(k)}
            if not allow_change:
                raise SystemExit(f"config differs from frozen {path}: {diff}\n"
                                 "use --allow-config-change to overwrite it")
            print(f"[loop] overwriting frozen config, changes: {diff}")
            save_config(cfg, path)
    else:
        save_config(cfg, path)
    return str(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--allow-config-change", action="store_true")
    ap.add_argument("--refresh-code", action="store_true",
                    help="re-copy src/ into the run (the code of later stages changes)")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    if not cfg.get("arm"):
        raise SystemExit("config has no `arm`")
    # Absolute, so stages running from the code snapshot write to the same place.
    cfg["runs_dir"] = str(runs_root(cfg))
    config_path = freeze_config(cfg, args.allow_config_change)
    code_dir = snapshot_code(cfg, args.refresh_code)

    done = completed_rounds(cfg)
    print(f"[loop] run {cfg['run_name']} (arm {cfg['arm']}): rounds 0..{done} complete, "
          f"target {cfg['rounds']}", flush=True)
    if done < 0:
        write_json(round_dir(cfg, 0) / "timing.json",
                   {"evaluate": run_stage("evaluate", config_path, 0, code_dir)})
    for r in range(max(done, 0) + 1, cfg["rounds"] + 1):
        timing = {stage: run_stage(stage, config_path, r, code_dir) for stage in STAGES}
        write_json(round_dir(cfg, r) / "timing.json", timing)
        g = read_json(round_dir(cfg, r) / "eval.json")["gsm8k"]
        print(f"[loop] round {r} done: pass@1={g['pass@1']:.3f} in {sum(timing.values()):.0f}s",
              flush=True)
    print("[loop] all rounds complete")


if __name__ == "__main__":
    main()
