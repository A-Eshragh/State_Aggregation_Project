#!/usr/bin/env python3
"""Render the final Cliff Walking paper figures.

This mirrors the style used in the root ``make_figures.py`` file:
serif font, compact panels, and the same color palette.

The script reads the saved Cliff Walking run directories and writes the three
canonical figures to ``selected_plots/``:

    training_curves.png
    compute_cost.png
    decision_quality.png

It also writes matching PDF files.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MDP_CW = Path(__file__).resolve().parent

RC = {
    "font.family": "serif",
    "font.size": 20,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "legend.fontsize": 13,
    "xtick.labelsize": 16,
    "ytick.labelsize": 18,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

METHODS: Tuple[Tuple[str, str, str, str], ...] = (
    ("ts", "Two-Stage", "#ff7f0e", "--"),
    ("qpth", "DFL-QP", "#2ca02c", "-"),
    ("dfl", "DFL-LP", "#1f77b4", "-"),
    ("sketch_k8r03", "DFL-Sketch", "#d62728", ":"),
    ("ssa10", "DFL-SA (M=10)", "#9467bd", "-"),
    ("ssa20", "DFL-SA (M=20)", "#9467bd", "-"),
    ("ssa30", "DFL-SA (M=30)", "#9467bd", "-"),
    ("ssa12", "DFL-SA (M=12)", "#9467bd", "-"),
    ("ssa18", "DFL-SA (M=18)", "#9467bd", "-"),
    ("ssa24", "DFL-SA (M=24)", "#9467bd", "-"),
)

TRAIN_KEYS = ("ts", "qpth", "dfl", "sketch_k8r03", "ssa24")
BAR_KEYS = (
    "ts", "qpth", "dfl", "sketch_k8r03",
    "ssa10", "ssa12", "ssa18", "ssa20", "ssa24", "ssa30",
)
BAR_TICK_LABELS = {
    "ts": "TS",
    "qpth": "QP",
    "dfl": "LP",
    "sketch_k8r03": "Sketch",
    "ssa10": "SA\nM=10",
    "ssa12": "SA\nM=12",
    "ssa18": "SA\nM=18",
    "ssa20": "SA\nM=20",
    "ssa24": "SA\nM=24",
    "ssa30": "SA\nM=30",
}
TICK_LABELS = {
    "ssa10": "SA\nM=10",
    "ssa12": "SA\nM=12",
    "ssa18": "SA\nM=18",
    "ssa20": "SA\nM=20",
    "ssa24": "SA\nM=24",
    "ssa30": "SA\nM=30",
}

# Final hard-instance mean regrets from the cleaned CW run.
FALLBACK_MEAN_REGRET = {
    "ts": 1.603,
    "qpth": 1.580,
    "dfl": 1.604,
    "sketch_k8r03": 1.587,
    "ssa10": 8.244,
    "ssa12": 8.244,
    "ssa18": 8.268,
    "ssa20": 8.268,
    "ssa24": 9.392,
    "ssa30": 9.392,
}


def _select_methods(keys: Tuple[str, ...]) -> Tuple[Tuple[str, str, str, str], ...]:
    allowed = {m[0]: m for m in METHODS}
    missing = [k for k in keys if k not in allowed]
    if missing:
        raise ValueError(f"Unknown method key(s): {missing}")
    return tuple(allowed[k] for k in keys)


def _load_run(root: Path, k: int, prefix: str, s: int) -> Optional[Dict]:
    run_dir = root / f"k{k}" / f"{prefix}_s{s}"
    if not run_dir.is_dir():
        return None
    for name in ("results.json", "checkpoint.json"):
        p = run_dir / name
        if p.is_file():
            try:
                return json.loads(p.read_text())
            except Exception:
                continue
    return None


def _load_runs(root: Path, k: int, prefix: str, n_seeds: int) -> List[Dict]:
    base = root / f"k{k}"
    if not base.is_dir():
        return []

    entries: List[Tuple[int, Path]] = []
    for run_dir in base.glob(f"{prefix}_s*"):
        if not run_dir.is_dir():
            continue
        m = re.search(r"_s(\d+)$", run_dir.name)
        if not m:
            continue
        seed = int(m.group(1))
        for name in ("results.json", "checkpoint.json"):
            p = run_dir / name
            if p.is_file():
                entries.append((seed, p))
                break

    entries.sort(key=lambda t: t[0])
    if n_seeds > 0:
        entries = entries[:n_seeds]

    out: List[Dict] = []
    for _, p in entries:
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            continue
    return out


def _stack_series(runs: List[Dict], key: str) -> Optional[np.ndarray]:
    rows = [
        r[key] for r in runs
        if key in r and isinstance(r[key], list) and len(r[key]) > 0
    ]
    if not rows:
        return None
    m = min(len(x) for x in rows)
    return np.array([x[:m] for x in rows], dtype=np.float64)


def _se(a: np.ndarray, axis: int = 0) -> np.ndarray:
    n = a.shape[axis]
    if n <= 1:
        return np.zeros_like(np.mean(a, axis=axis))
    return a.std(axis=axis, ddof=1) / np.sqrt(n)


def _save(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(ax: plt.Axes, root: Path, k: int, n_seeds: int) -> None:
    for prefix, label, color, ls in _select_methods(TRAIN_KEYS):
        runs = _load_runs(root, k, prefix, n_seeds)
        mat = _stack_series(runs, "train_reward_per_epoch")
        if mat is None:
            continue
        mu = mat.mean(axis=0)
        se = _se(mat, axis=0)
        x = np.arange(mu.shape[0])
        ax.plot(x, mu, color=color, ls=ls, lw=2.0, label=label)
        ax.fill_between(
            x, mu - se, mu + se,
            color=color, alpha=0.2, linewidth=0, edgecolor="none",
        )
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Training reward")
    ax.set_title("Cliff Walking — Training curves")
    # Rewards are negative; use symlog for log-like separation while
    # preserving sign and avoiding invalid log(negative) values.
    ax.set_yscale("symlog", linthresh=1.0)
    ax.grid(True, alpha=0.25, ls="-")
    ax.legend(loc="lower right")


def _total_train_time(run: Dict) -> Optional[float]:
    tt = run.get("train_time")
    if tt and len(tt) > 0:
        return float(tt[-1])
    return None


def plot_compute_cost(ax: plt.Axes, root: Path, k: int, n_seeds: int) -> None:
    means, ses, labels, colors = [], [], [], []
    for prefix, label, color, _ls in _select_methods(BAR_KEYS):
        runs = _load_runs(root, k, prefix, n_seeds)
        vals = np.array(
            [v for v in (_total_train_time(r) for r in runs) if v is not None],
            dtype=np.float64,
        )
        if vals.size == 0:
            continue
        means.append(float(vals.mean()))
        ses.append(float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0)
        labels.append(BAR_TICK_LABELS.get(prefix, TICK_LABELS.get(prefix, label)))
        colors.append(color)

    x = np.arange(len(labels))
    ax.bar(
        x,
        np.array(means, dtype=np.float64),
        yerr=np.array(ses, dtype=np.float64),
        capsize=4,
        width=0.52,
        color=colors,
        alpha=0.88,
        edgecolor="black",
        linewidth=0.6,
        error_kw=dict(ecolor="black", lw=0.8),
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", labelsize=15)
    ax.set_ylabel("Total training time (s)")
    ax.set_title("Cliff Walking — Compute cost")
    ax.grid(True, axis="y", alpha=0.3, ls="-")


def plot_decision_quality(ax: plt.Axes, root: Path, k: int, n_seeds: int) -> None:
    means, ses, labels, colors = [], [], [], []
    method_keys: List[str] = []
    for prefix, label, color, _ls in _select_methods(BAR_KEYS):
        runs = _load_runs(root, k, prefix, n_seeds)
        vals = np.array(
            [float(r["test_mean_regret"]) for r in runs if "test_mean_regret" in r],
            dtype=np.float64,
        )
        if vals.size == 0:
            # fallback to the finalized mean if the archived run folders only
            # contain checkpoints / training curves.
            if prefix not in FALLBACK_MEAN_REGRET:
                continue
            vals = np.array([FALLBACK_MEAN_REGRET[prefix]], dtype=np.float64)
        means.append(float(vals.mean()))
        ses.append(float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0)
        labels.append(BAR_TICK_LABELS.get(prefix, TICK_LABELS.get(prefix, label)))
        colors.append(color)
        method_keys.append(prefix)

    x = np.arange(len(labels))
    ax.bar(
        x,
        np.array(means, dtype=np.float64),
        yerr=np.array(ses, dtype=np.float64),
        capsize=4,
        width=0.52,
        color=colors,
        alpha=0.88,
        edgecolor="black",
        linewidth=0.6,
        error_kw=dict(ecolor="black", lw=0.8),
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", labelsize=15)
    ax.set_ylabel("Test regret")
    ax.set_title("Cliff Walking — Decision quality")
    # Auto-range: use zoomed range for low-regret runs; use full range for dist-shift runs.
    data_max = float(max(means)) if means else 1.0
    if data_max > 2.0:
        y_min, y_max = 0.0, 16.0
        ax.set_yticks(np.arange(0, 17, 2))
    else:
        y_min, y_max = 0.6, 1.0
        ax.set_yticks(np.arange(0.6, 1.01, 0.1))
    ax.set_ylim(y_min, y_max)
    ax.grid(True, axis="y", alpha=0.25, ls=":")

    # Annotate SA bars with their mean regret (even when clipped by y-limit).
    yr = y_max - y_min
    for i, (key, mu) in enumerate(zip(method_keys, means)):
        if key in ("ssa10", "ssa12", "ssa18", "ssa20", "ssa24", "ssa30"):
            if mu >= y_max:
                # If clipped, place text inside visible axis window.
                ax.text(i, y_max - 0.08 * yr, f"{mu:.2f}", ha="center", va="center",
                        fontsize=11, color="black")
            else:
                y = min(mu + 0.02 * yr, y_max - 0.02 * yr)
                ax.text(i, y, f"{mu:.2f}", ha="center", va="bottom", fontsize=11)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=MDP_CW / "results_hardinst_seeds1_10")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--out_dir", type=Path, default=MDP_CW / "selected_plots_hardinst_seeds1_10")
    args = ap.parse_args()

    prev = plt.rcParams.copy()
    plt.rcParams.update(RC)

    # Training curves
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    plot_training_curves(ax, args.root, args.k, args.n_seeds)
    _save(fig, args.out_dir / "cw_training_curves.png")

    # Compute
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    plot_compute_cost(ax, args.root, args.k, args.n_seeds)
    _save(fig, args.out_dir / "cw_compute_cost.png")

    # Decision quality
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    plot_decision_quality(ax, args.root, args.k, args.n_seeds)
    _save(fig, args.out_dir / "cw_decision_quality.png")

    plt.rcParams.update(prev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
