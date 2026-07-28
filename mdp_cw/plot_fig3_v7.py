"""Figure-3 style decision-quality bar chart for the v7 gate-trap setting.

Reads the raw per-seed ``results.json`` files and plots mean test regret with
standard-error bars, in the same visual style as ``make_figures.py`` /
``plot_report_three_panel_best_nk.py`` so it sits beside the existing figures.

Each method's source directory and seed window is declared explicitly in
``SERIES`` below, because the methods were tuned with different training
budgets; the mismatch is annotated rather than hidden.

Usage::

    python3 mdp_cw/plot_fig3_v7.py --out mdp_cw/selected_plots/fig3_v7.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Match make_figures.py exactly.
RC = {
    "font.family": "serif", "font.size": 20, "axes.labelsize": 18,
    "axes.titlesize": 20, "legend.fontsize": 13, "xtick.labelsize": 16,
    "ytick.labelsize": 18, "axes.linewidth": 1.2,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}

# (label, results dir, run prefix, seed list, bar colour)
# Colours reuse the per-method identity from plot_report_three_panel_best_nk.py.
SERIES = [
    ("Two-Stage",      "mdp_cw/results_tsconv", "ts600",   range(10),  "#ff7f0e"),
    ("DFL-LP",         "mdp_cw/results_epochs", "lp62",    range(10),  "#1f77b4"),
    ("DFL-Sketch",     "mdp_cw/results_final7", "sk",      range(10),  "#d62728"),
    ("DFL-SA\n(M=12)", "mdp_cw/results_final7", "ssa",     range(10),  "#9467bd"),
    ("DFL-QP",         "mdp_cw/results_final7", "qpe0002", range(2, 12), "#2ca02c"),
]


def load(dirname: str, prefix: str, seeds) -> Optional[np.ndarray]:
    vals = []
    for s in seeds:
        f = ROOT / dirname / f"{prefix}_s{s}" / "results.json"
        if not f.is_file():
            return None
        vals.append(float(json.loads(f.read_text())["test_mean_regret"]))
    return np.asarray(vals, dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=ROOT / "mdp_cw/selected_plots/fig3_v7.png")
    args = ap.parse_args()

    labels, means, errs, colors = [], [], [], []
    for label, d, p, seeds, color in SERIES:
        v = load(d, p, seeds)
        if v is None:
            print(f"  skip {label}: incomplete ({d}/{p})")
            continue
        labels.append(label)
        means.append(v.mean())
        errs.append(v.std(ddof=1) / np.sqrt(v.size))
        colors.append(color)
        print(f"  {label:16s} n={v.size:2d}  mean={v.mean():.3f}  sem={errs[-1]:.3f}")

    if not labels:
        print("no data")
        return 1

    prev = plt.rcParams.copy()
    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=errs, capsize=4, width=0.6,
           color=colors, alpha=0.88, edgecolor="black", linewidth=0.6,
           error_kw=dict(ecolor="black", lw=1.0))

    # Two-stage reference line makes "better than baseline" readable at a glance.
    ax.axhline(means[0], color="#ff7f0e", ls="--", lw=1.4, alpha=0.8, zorder=0)

    # Direct value labels (few enough bars that labelling all of them is fine).
    for xi, m, e in zip(x, means, errs):
        ax.text(xi, m + e + 0.06, f"{m:.2f}", ha="center", va="bottom", fontsize=15)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test regret")
    ax.set_title("Cliff Walking — Decision quality")
    ax.set_ylim(0, max(m + e for m, e in zip(means, errs)) * 1.20)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update(prev)
    print(f"wrote {args.out} (+ .pdf)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
