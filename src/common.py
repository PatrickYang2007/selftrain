"""Shared helpers: config loading, run-directory layout, jsonl io, stage markers."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- config

def load_config(path: str | os.PathLike, overrides: Iterable[str] = ()) -> dict:
    """Load a yaml config. A top-level `inherit: other.yaml` key merges the parent first.

    `overrides` are `key=value` strings; values are parsed as yaml (so `rounds=3` is an int).
    """
    cfg = _load_raw(Path(path))
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"override must be key=value, got {item!r}")
        cfg[key] = yaml.safe_load(value)
    # run_name may reference other keys, e.g. "debug_{arm}"; it defaults to the arm name.
    if cfg.get("arm"):
        cfg["run_name"] = (cfg.get("run_name") or "{arm}").format(**cfg)
    return cfg


def _load_raw(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        merged = _load_raw(path.parent / parent)
        merged.update(cfg)
        cfg = merged
    return cfg


def save_config(cfg: dict, path: str | os.PathLike) -> None:
    atomic_write_text(path, yaml.safe_dump(cfg, sort_keys=True))


# ---------------------------------------------------------------- run layout

def runs_root(cfg: dict) -> Path:
    root = Path(cfg.get("runs_dir", "runs"))
    return root if root.is_absolute() else REPO_ROOT / root


def run_dir(cfg: dict) -> Path:
    return runs_root(cfg) / cfg["run_name"]


def round_dir(cfg: dict, r: int) -> Path:
    return run_dir(cfg) / f"round_{r}"


def model_path(cfg: dict, r: int) -> str:
    """Weights of the model after round r. Round 0 is the untouched base model."""
    if r == 0:
        return cfg["model_name"]
    return str(round_dir(cfg, r) / "model")


def done_marker(path: str | os.PathLike) -> Path:
    path = Path(path)
    return path.parent / f".{path.name}.done"


def mark_done(path: str | os.PathLike) -> None:
    done_marker(path).touch()


def is_done(path: str | os.PathLike) -> bool:
    return Path(path).exists() and done_marker(path).exists()


# ---------------------------------------------------------------- io

def atomic_write_text(path: str | os.PathLike, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_json(path: str | os.PathLike, obj) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")


def read_json(path: str | os.PathLike):
    with open(path) as f:
        return json.load(f)


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict]) -> None:
    atomic_write_text(path, "".join(json.dumps(r) + "\n" for r in rows))


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
