"""Figures, summary table and archived logs from runs/.

    python -m src.plots [--runs oracle self_judge majority] [--extra-runs oracle_star ...]

Writes:
- results/summary.csv          one row per (run, round): eval, sampling, filter and training stats
- results/paired.csv           paired bootstrap CIs on greedy accuracy differences (same problems):
                               each run vs its own round 0, and each run vs oracle at each round
- results/figures/*.png        the seven report figures (plus a few diagnostics)
- results/logs/<run>/round_r/  small json logs copied out of runs/ (scratch is purged)

Works on partial runs: missing rounds are just absent from the curves.

Round convention: eval metrics at round r describe model r (round 0 = base). Sampling and filter
stats at round r describe the pool samples drawn from model r-1 and the filter applied to them.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.common import REPO_ROOT  # noqa: E402

RESULTS = REPO_ROOT / "results"
FIGURES = RESULTS / "figures"
ARMS = ("oracle", "self_judge", "majority", "external_judge")
LOG_FILES = ("eval.json", "sample_stats.json", "filter_stats.json", "train_log.json",
             "timing.json")

# Reference categorical palette, first three slots (validated all-pairs for CVD), with a
# distinct marker per series as secondary encoding. Text stays in ink colors.
SERIES = [
    {"color": "#2a78d6", "marker": "o"},
    {"color": "#eb6834", "marker": "s"},
    {"color": "#1baf7a", "marker": "^"},
    {"color": "#eda100", "marker": "D"},
]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
LABELS = {"oracle": "oracle", "self_judge": "self-judge", "majority": "majority",
          "external_judge": "7B judge", "oracle_star": "oracle (STaR)",
          "self_judge_star": "self-judge (STaR)"}


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK,
        "font.size": 10, "lines.linewidth": 2, "lines.markersize": 7,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })


# ---------------------------------------------------------------- collection

def _flatten(prefix: str, d: dict, out: dict) -> None:
    for k, v in d.items():
        if k == "log_history":
            continue
        if isinstance(v, dict):
            _flatten(f"{prefix}{k}.", v, out)
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[f"{prefix}{k}"] = v


def collect(runs_root: Path, runs: list[str]) -> pd.DataFrame:
    rows = []
    for run in runs:
        for rd in sorted((runs_root / run).glob("round_*"),
                         key=lambda p: int(p.name.split("_")[1])):
            r = int(rd.name.split("_")[1])
            row: dict = {"run": run, "round": r}
            for name, prefix in (("eval.json", "eval."), ("sample_stats.json", "pool."),
                                 ("filter_stats.json", "filter."), ("train_log.json", "train."),
                                 ("timing.json", "seconds.")):
                path = rd / name
                if path.exists():
                    _flatten(prefix, json.loads(path.read_text()), row)
            if len(row) > 2:
                rows.append(row)
    return pd.DataFrame(rows)


def archive_logs(runs_root: Path, runs: list[str]) -> None:
    for run in runs:
        src_run = runs_root / run
        if not src_run.exists():
            continue
        dst_run = RESULTS / "logs" / run
        if (src_run / "config.yaml").exists():
            dst_run.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_run / "config.yaml", dst_run / "config.yaml")
        for rd in src_run.glob("round_*"):
            for name in LOG_FILES:
                if (rd / name).exists():
                    (dst_run / rd.name).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(rd / name, dst_run / rd.name / name)


def _greedy_correct(runs_root: Path, run: str, r: int) -> dict[str, dict[str, bool]] | None:
    path = runs_root / run / f"round_{r}" / "eval_greedy.jsonl"
    if not path.exists():
        return None
    out: dict[str, dict[str, bool]] = {"gsm8k": {}, "svamp": {}}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        out[row["problem_id"].split("_")[0]][row["problem_id"]] = bool(row["correct"])
    return out


def paired_bootstrap(a: dict[str, bool], b: dict[str, bool], n_boot: int = 2000,
                     seed: int = 0) -> tuple[float, float, float]:
    """Mean of (a - b) over shared problems, with a 95% percentile bootstrap CI."""
    import numpy as np

    keys = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def paired_table(runs_root: Path, runs: list[str], max_round: int = 10) -> pd.DataFrame:
    rows = []
    for run in runs:
        base = _greedy_correct(runs_root, run, 0)
        for r in range(1, max_round + 1):
            cur = _greedy_correct(runs_root, run, r)
            if cur is None or base is None:
                continue
            ref = _greedy_correct(runs_root, "oracle", r)
            for ds in ("gsm8k", "svamp"):
                if not cur[ds]:
                    continue
                d, lo, hi = paired_bootstrap(cur[ds], base[ds])
                rows.append({"run": run, "round": r, "dataset": ds, "vs": "own round 0",
                             "diff": d, "ci_low": lo, "ci_high": hi})
                if ref is not None and run != "oracle" and ref[ds]:
                    d, lo, hi = paired_bootstrap(cur[ds], ref[ds])
                    rows.append({"run": run, "round": r, "dataset": ds, "vs": "oracle",
                                 "diff": d, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- drawing

def _series(df: pd.DataFrame, run: str, col: str) -> pd.DataFrame:
    if col not in df:
        return pd.DataFrame(columns=["round", col])
    sub = df[(df["run"] == run) & df[col].notna()][["round", col]]
    return sub.sort_values("round")


def lines(ax, df, runs, col, labels=None, se_col=None, end_labels=True, fmt="{:.2f}"):
    """One line per run. SE band if se_col given. End labels only when they won't collide."""
    ends, xs = [], set()
    for i, run in enumerate(runs):
        sub = _series(df, run, col)
        if sub.empty:
            continue
        spec = SERIES[i % len(SERIES)]
        label = (labels or {}).get(run, LABELS.get(run, run))
        ax.plot(sub["round"], sub[col], color=spec["color"], marker=spec["marker"],
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
        if se_col and se_col in df:
            se = _series(df, run, se_col).set_index("round")[se_col]
            y = sub.set_index("round")[col]
            ax.fill_between(y.index, y - se.reindex(y.index), y + se.reindex(y.index),
                            color=spec["color"], alpha=0.10, linewidth=0, zorder=1)
        ends.append((sub["round"].iloc[-1], sub[col].iloc[-1], label))
        xs.update(sub["round"])
    ax.set_xlabel("round")
    if xs:
        ax.set_xticks(sorted(xs))
    if end_labels and ends:
        lo, hi = ax.get_ylim()
        ys = sorted(y for _, y, _ in ends)
        gaps = [b - a for a, b in zip(ys, ys[1:])]
        if not gaps or min(gaps) > 0.06 * (hi - lo):
            for x, y, label in ends:
                ax.annotate(f"{label} {fmt.format(y)}", (x, y), xytext=(8, 0),
                            textcoords="offset points", va="center", fontsize=9, color=INK)
            ax.set_xlim(right=max(x for x, _, _ in ends) + 0.9)
    return ax


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", dpi=160)
    plt.close(fig)


def fig_pass1(df, runs):
    fig, ax = plt.subplots(figsize=(6.4, 4))
    lines(ax, df, runs, "eval.gsm8k.pass@1", se_col="eval.gsm8k.pass@1_se")
    ax.set_title("GSM8K pass@1 (greedy) by round")
    ax.set_ylabel("accuracy, 500 test problems (band = ±1 SE)")
    ax.legend(loc="lower left")
    save(fig, "1_pass1_by_round")


def fig_oracle_pass1_vs_passk(df, k):
    if "oracle" not in set(df["run"]):
        return
    one = df[df["run"] == "oracle"].copy()
    cols = {"eval.gsm8k.pass@1": "pass@1 greedy", "eval.gsm8k.pass@1_sampled": "pass@1 sampled",
            f"eval.gsm8k.maj@{k}": f"maj@{k}", f"eval.gsm8k.pass@{k}": f"pass@{k}"}
    long = pd.concat([one[["round", c]].rename(columns={c: "v"}).assign(run=name)
                      for c, name in cols.items() if c in one])
    fig, ax = plt.subplots(figsize=(6.4, 4))
    lines(ax, long, list(cols.values()), "v", labels={v: v for v in cols.values()})
    ax.set_title(f"Oracle arm: pass@1 vs pass@{k}")
    ax.set_ylabel("accuracy, GSM8K test subset")
    ax.legend(loc="lower left")
    save(fig, "2_oracle_pass1_vs_passk")


def fig_diversity(df, runs):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, col, title in zip(
        axes,
        ("eval.gsm8k.distinct_answers", "eval.gsm8k.distinct_3",
         "eval.gsm8k.distinct_3_first100w"),
        ("Distinct final answers per problem", "Distinct-3 across samples",
         "Distinct-3, first 100 words only"),
    ):
        lines(ax, df, runs, col, end_labels=False)
        ax.set_title(title, fontsize=11)
    axes[0].legend(loc="best")
    fig.suptitle("Diversity of the 8 temperature-0.8 test samples per problem", x=0.01,
                 ha="left", fontweight="bold", color=INK)
    save(fig, "3_diversity_by_round")


def fig_self_deception(df, runs):
    arms = [a for a in ("self_judge", "majority", "external_judge")
            if a in runs and a in set(df["run"])]
    if not arms:
        return
    fig, axes = plt.subplots(1, len(arms), figsize=(6.2 * len(arms), 4), squeeze=False)
    for ax, arm in zip(axes[0], arms):
        one = df[df["run"] == arm]
        long = pd.concat([
            one[["round", "filter.self_reported_acc"]].rename(
                columns={"filter.self_reported_acc": "v"}).assign(run="filter acceptance rate"),
            one[["round", "filter.true_pool_acc"]].rename(
                columns={"filter.true_pool_acc": "v"}).assign(run="true pool accuracy"),
        ])
        names = ["filter acceptance rate", "true pool accuracy"]
        lines(ax, long, names, "v", labels={n: n for n in names})
        ax.set_title(f"{LABELS[arm]}: self-reported vs true accuracy")
        ax.set_ylabel("fraction of pool samples")
        ax.set_xlabel("round (samples from model r−1)")
        ax.legend(loc="lower left")
    save(fig, "4_self_reported_vs_true")


def fig_filter_quality(df, runs):
    arms = [a for a in ("self_judge", "majority", "external_judge") if a in runs]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, col, title in zip(axes, ("filter.precision", "filter.recall"),
                              ("Filter precision vs ground truth",
                               "Filter recall vs ground truth")):
        lines(ax, df, arms, col)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("round (samples from model r−1)")
    axes[0].legend(loc="lower left")
    save(fig, "5_filter_precision")


def fig_judge(df, runs):
    arms = [a for a in ("self_judge", "external_judge") if a in runs and a in set(df["run"])]
    if not arms:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, col, title in zip(axes, ("filter.judge_yes_rate", "filter.judge_auroc"),
                              ("Judge says Yes", "Judge AUROC: P(Yes) vs correctness")):
        lines(ax, df, arms, col)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("round (samples from model r−1)")
    axes[1].axhline(0.5, color=INK_2, linewidth=1, zorder=0)
    axes[1].annotate("chance", (axes[1].get_xlim()[0], 0.5), xytext=(4, 4),
                     textcoords="offset points", fontsize=8, color=INK_2)
    axes[0].legend(loc="lower left")
    save(fig, "11_judge_diagnostics")


def fig_simple(df, runs, col, title, ylabel, name, fmt="{:.2f}"):
    fig, ax = plt.subplots(figsize=(6.4, 4))
    lines(ax, df, runs, col, fmt=fmt)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    save(fig, name)


def make_all(df: pd.DataFrame, runs: list[str], k: int) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    style()
    fig_pass1(df, runs)
    fig_oracle_pass1_vs_passk(df, k)
    fig_diversity(df, runs)
    fig_self_deception(df, runs)
    fig_filter_quality(df, runs)
    fig_judge(df, runs)
    if "eval.svamp.pass@1" in df:
        fig_simple(df, runs, "eval.svamp.pass@1", "SVAMP pass@1 (greedy, out of distribution)",
                   "accuracy, 1000 SVAMP problems", "6_ood_pass1_by_round")
    fig_simple(df, runs, "eval.gsm8k.mean_tokens", "Mean response length (greedy, GSM8K test)",
               "tokens", "7_response_length_by_round", fmt="{:.0f}")
    # Diagnostics beyond the seven planned figures.
    fig_simple(df, runs, "eval.gsm8k.format_last_number",
               "Responses with neither an Answer: line nor \\boxed{}",
               "fraction (greedy, GSM8K test)", "8_unformatted_answers", fmt="{:.3f}")
    fig_simple(df, runs, "filter.n_kept", "Training examples kept per round", "examples",
               "9_kept_examples", fmt="{:.0f}")
    fig_simple(df, runs, "filter.kept_precision",
               "Fraction of training examples that are actually correct", "fraction",
               "10_kept_precision")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(REPO_ROOT / "runs"))
    ap.add_argument("--runs", nargs="*", default=list(ARMS))
    ap.add_argument("--extra-runs", nargs="*", default=[],
                    help="also collected into summary.csv and logs, not drawn in main figures")
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()
    root = Path(args.runs_dir)
    all_runs = args.runs + args.extra_runs
    df = collect(root, all_runs)
    if df.empty:
        raise SystemExit(f"no results under {root}")
    RESULTS.mkdir(exist_ok=True)
    cols = ["run", "round"] + sorted(c for c in df.columns if c not in ("run", "round"))
    df[cols].to_csv(RESULTS / "summary.csv", index=False)
    archive_logs(root, all_runs)
    paired = paired_table(root, all_runs)
    if not paired.empty:
        paired.round(4).to_csv(RESULTS / "paired.csv", index=False)
    main_df = df[df["run"].isin(args.runs)]
    make_all(main_df, [r for r in args.runs if r in set(df["run"])], args.k)
    print(f"wrote {RESULTS / 'summary.csv'} ({len(df)} rows) and figures to {FIGURES}")


if __name__ == "__main__":
    main()
