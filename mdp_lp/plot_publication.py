#!/usr/bin/env python3
"""
Publication-quality plots for elasticity DFL vs SSA experiments.

Generates 4 figures (PDF + PNG):
  1. reward_curves      – train reward per epoch (learning curves)
  2. regret_boxplot     – per-sample test regret distribution per method
  3. compute_profile    – avg wall-clock compute breakdown per epoch
  4. nagg_ablation      – test regret & compute vs n_agg (M) for DFL-LP and DFL-qpth

Usage:
  python -m mdp_lp.plot_publication \
      --results_root mdp_lp/100samples_results \
      --out_dir mdp_lp/publication_plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# NeurIPS-style global rcParams
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "#cccccc",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linewidth":     0.5,
    "lines.linewidth":    1.6,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
})

# ---------------------------------------------------------------------------
# Variant catalogue
# ---------------------------------------------------------------------------
_VARIANTS: List[Dict[str, Any]] = [
    dict(key="twostage",       algo="twostage",  n_agg=-1,  qp=False, label="Two-Stage",          color="#E41A1C", ls="-",  marker="o",  zorder=5),
    dict(key="dfl",            algo="dfl",       n_agg=-1,  qp=False, label="DFL (LP)",            color="#377EB8", ls="-",  marker="s",  zorder=6),
    dict(key="dfl_ssa5",       algo="dfl",       n_agg=5,   qp=False, label="DFL+SSA (M=5)",       color="#4DAF4A", ls="--", marker="^",  zorder=4),
    dict(key="dfl_ssa10",      algo="dfl",       n_agg=10,  qp=False, label="DFL+SSA (M=10)",      color="#00896A", ls="--", marker="D",  zorder=4),
    dict(key="dfl_ssa20",      algo="dfl",       n_agg=20,  qp=False, label="DFL+SSA (M=20)",      color="#006400", ls="--", marker="v",  zorder=4),
    dict(key="qpth",           algo="dfl_qpth",  n_agg=-1,  qp=True,  label="DFL (qpth)",          color="#984EA3", ls="-",  marker="s",  zorder=3),
    dict(key="qpth_ssa5",      algo="dfl_qpth",  n_agg=5,   qp=True,  label="DFL+qpth+SSA (M=5)", color="#FF7F00", ls=":",  marker="^",  zorder=2),
    dict(key="qpth_ssa10",     algo="dfl_qpth",  n_agg=10,  qp=True,  label="DFL+qpth+SSA (M=10)",color="#A65628", ls=":",  marker="D",  zorder=2),
    dict(key="qpth_ssa20",     algo="dfl_qpth",  n_agg=20,  qp=True,  label="DFL+qpth+SSA (M=20)",color="#C17817", ls=":",  marker="v",  zorder=2),
]
_KEY_MAP = {v["key"]: v for v in _VARIANTS}

# Condition catalogue
_CONDITIONS = [
    dict(exp="b", cost="normal_margin", beta2=None,   title="Exp B  (β₂=0)"),
    dict(exp="c", cost="normal_margin", beta2=0.09,   title="Exp C  (β₂=+0.09, substitutes)"),
    dict(exp="c", cost="normal_margin", beta2=-0.09,  title="Exp C  (β₂=−0.09, complements)"),
]

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _results_path(root: Path, exp: str, cost: str, beta2: Optional[float],
                  algo: str, n_agg: int, qp: bool, qp_reg: float, epochs: int) -> Path:
    base = root / f"exp_{exp}" / cost / algo
    if exp == "c" and beta2 is not None:
        base = base / f"beta2_{beta2}"
    if n_agg != -1:
        base = base / f"n_agg_{n_agg}"
    if qp:
        base = base / f"qp_reg_{qp_reg}"
    return base / f"epochs_{epochs}" / "results.json"


def load_condition(root: Path, cond: Dict, qp_reg: float, epochs: int
                   ) -> Dict[str, Optional[Dict[str, Any]]]:
    """Load all variant results for one experimental condition."""
    out: Dict[str, Optional[Dict[str, Any]]] = {}
    for v in _VARIANTS:
        p = _results_path(root, cond["exp"], cond["cost"], cond["beta2"],
                          v["algo"], v["n_agg"], v["qp"], qp_reg, epochs)
        out[v["key"]] = _load(p)
    return out


def _safe(d: Optional[Dict], key: str) -> Optional[np.ndarray]:
    if d is None:
        return None
    val = d.get(key)
    if not isinstance(val, list) or len(val) == 0:
        return None
    arr = np.asarray(val, dtype=np.float64)
    return arr if np.any(np.isfinite(arr)) else None


def _diverged(regret_arr: Optional[np.ndarray], threshold: float = 1500.0) -> bool:
    if regret_arr is None or regret_arr.size == 0:
        return True
    finite = regret_arr[np.isfinite(regret_arr)]
    return finite.size == 0 or float(np.median(finite)) > threshold


# ---------------------------------------------------------------------------
# Figure 1: Reward Learning Curves
# ---------------------------------------------------------------------------

def plot_reward_curves(conditions_data: List[Tuple[Dict, Dict[str, Optional[Dict]]]],
                       out_path: Path) -> None:
    ncols = len(conditions_data)
    fig, axes = plt.subplots(1, ncols, figsize=(3.5 * ncols, 3.0), sharey=False)
    if ncols == 1:
        axes = [axes]

    for ax, (cond, data) in zip(axes, conditions_data):
        oracle = None
        for vkey, d in data.items():
            if d is not None and d.get("test_mean_oracle_reward") is not None:
                oracle = float(d["test_mean_oracle_reward"])
                break

        plotted = False
        for v in _VARIANTS:
            d = data.get(v["key"])
            rw = _safe(d, "train_reward_per_epoch")
            if rw is None or len(rw) < 10:
                continue
            regret_arr = _safe(d, "test_regret")
            if _diverged(regret_arr):
                continue
            xs = np.arange(len(rw))
            ax.plot(xs, rw, label=v["label"], color=v["color"],
                    linestyle=v["ls"], linewidth=1.5, zorder=v["zorder"])
            plotted = True

        if oracle is not None:
            ax.axhline(oracle, color="#555555", linewidth=1.0, linestyle="--",
                       label=f"Oracle ({oracle:.0f})", zorder=1)

        ax.set_title(cond["title"], pad=4)
        ax.set_xlabel("Epoch")
        if ax is axes[0]:
            ax.set_ylabel("Train objective value")
        ax.legend(loc="lower right", ncol=1, fontsize=7)
        if not plotted:
            ax.text(0.5, 0.5, "No converged runs", transform=ax.transAxes,
                    ha="center", va="center", color="#888888", fontsize=9)

    fig.suptitle("Training Reward per Epoch", fontsize=11, y=1.01)
    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Figure 2: Regret Boxplots
# ---------------------------------------------------------------------------

def plot_regret_boxplots(conditions_data: List[Tuple[Dict, Dict[str, Optional[Dict]]]],
                         out_path: Path, diverge_thresh: float = 1500.0) -> None:
    ncols = len(conditions_data)
    fig, axes = plt.subplots(1, ncols, figsize=(4.0 * ncols, 3.5), sharey=False)
    if ncols == 1:
        axes = [axes]

    for ax, (cond, data) in zip(axes, conditions_data):
        plot_data, tick_labels, colors = [], [], []
        for v in _VARIANTS:
            d = data.get(v["key"])
            regret = _safe(d, "test_regret")
            if regret is None or regret.size == 0:
                continue
            if _diverged(regret, diverge_thresh):
                continue
            plot_data.append(regret)
            tick_labels.append(v["label"])
            colors.append(v["color"])

        if not plot_data:
            ax.text(0.5, 0.5, "No converged runs", transform=ax.transAxes,
                    ha="center", va="center", color="#888888", fontsize=9)
            ax.set_title(cond["title"], pad=4)
            continue

        bp = ax.boxplot(plot_data, patch_artist=True, showfliers=True,
                        flierprops=dict(marker=".", markersize=3, alpha=0.5),
                        medianprops=dict(color="#111111", linewidth=1.8),
                        widths=0.55)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.30)
            patch.set_edgecolor(c)
            patch.set_linewidth(1.5)
        for whisker in bp["whiskers"]:
            whisker.set_linewidth(1.0)
            whisker.set_color("#555555")
        for cap in bp["caps"]:
            cap.set_linewidth(1.0)
            cap.set_color("#555555")

        # Overlay mean dots
        for i, (arr, c) in enumerate(zip(plot_data, colors), start=1):
            finite = arr[np.isfinite(arr)]
            if finite.size:
                ax.scatter([i], [float(np.mean(finite))], color=c, zorder=10,
                           s=18, marker="D", edgecolors="white", linewidths=0.5)

        ax.set_xticks(range(1, len(tick_labels) + 1))
        ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=7)
        ax.set_title(cond["title"], pad=4)
        if ax is axes[0]:
            ax.set_ylabel("Test regret  (oracle − predicted)")
        ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)

    # Legend: diamond = mean
    mean_patch = Line2D([0], [0], marker="D", color="gray", markersize=5,
                        linestyle="none", label="mean")
    axes[-1].legend(handles=[mean_patch], loc="upper right", fontsize=7)

    fig.suptitle("Test Regret Distribution per Method", fontsize=11, y=1.01)
    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Figure 3: Compute Profile
# ---------------------------------------------------------------------------

def plot_compute_profile(exp_b_data: Dict[str, Optional[Dict]],
                         out_path: Path, warmup_skip: int = 1) -> None:
    """
    Stacked bar chart: avg solver time + sa time + backward time per epoch.
    Skips warmup (first warmup_skip epochs) to show steady-state cost.
    """
    labels, solver_means, sa_means, bwd_means = [], [], [], []

    for v in _VARIANTS:
        d = exp_b_data.get(v["key"])
        if d is None:
            continue
        solver = _safe(d, "solver_time_sec_per_epoch")
        sa     = _safe(d, "sa_time_sec_per_epoch")
        bwd    = _safe(d, "backward_time_sec_per_epoch")

        if solver is None or len(solver) <= warmup_skip:
            continue

        s = float(np.mean(solver[warmup_skip:]))
        a = float(np.mean(sa[warmup_skip:])) if sa is not None and len(sa) > warmup_skip else 0.0
        b = float(np.mean(bwd[warmup_skip:])) if bwd is not None and len(bwd) > warmup_skip else 0.0

        labels.append(v["label"])
        solver_means.append(s)
        sa_means.append(a)
        bwd_means.append(b)

    if not labels:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "No timing data available", transform=ax.transAxes,
                ha="center", va="center", color="#888888", fontsize=10)
        _save(fig, out_path)
        return

    x = np.arange(len(labels))
    bar_w = 0.55
    fig, ax = plt.subplots(figsize=(max(5.5, 1.1 * len(labels)), 3.5))

    bars_solver = ax.bar(x, solver_means, bar_w, label="LP/QP forward solve",
                         color="#377EB8", alpha=0.85, zorder=3)
    bars_sa = ax.bar(x, sa_means, bar_w, bottom=solver_means,
                     label="State aggregation (SA)", color="#4DAF4A", alpha=0.85, zorder=3)
    bwd_bottom = np.array(solver_means) + np.array(sa_means)
    bars_bwd = ax.bar(x, bwd_means, bar_w, bottom=bwd_bottom,
                      label="Backward pass", color="#FF7F00", alpha=0.85, zorder=3)

    # Annotate total
    for xi, (s, a, b) in enumerate(zip(solver_means, sa_means, bwd_means)):
        total = s + a + b
        ax.text(xi, total + 0.5, f"{total:.1f}s", ha="center", va="bottom",
                fontsize=7, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("Avg wall-clock time per epoch (s)")
    ax.set_title("Compute Profile — Exp B (steady-state, epochs 2–100)", pad=4)
    ax.legend(loc="upper right", fontsize=8)
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    if any(b == 0 for b in bwd_means):
        ax.text(0.01, 0.98, "* Backward time available only in new runs (field added to train_elasticity.py)",
                transform=ax.transAxes, fontsize=6.5, va="top", color="#888888", style="italic")

    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Figure 4: n_agg Ablation (regret + compute vs M)
# ---------------------------------------------------------------------------

def plot_nagg_ablation(conditions_data: List[Tuple[Dict, Dict[str, Optional[Dict]]]],
                       out_path: Path, diverge_thresh: float = 1500.0) -> None:
    """
    Two-row figure:
      Row 1: median test regret vs n_agg for DFL-LP and DFL-qpth
      Row 2: avg solver time vs n_agg for DFL-LP and DFL-qpth
    One column per experimental condition.
    """
    ncols = len(conditions_data)
    fig, axes = plt.subplots(2, ncols, figsize=(3.8 * ncols, 5.0), sharex=False)
    if ncols == 1:
        axes = axes.reshape(2, 1)

    nagg_vals = [-1, 5, 10, 20]
    nagg_labels = ["no-SSA", "M=5", "M=10", "M=20"]

    def _variant_key(algo_qp: bool, n: int) -> str:
        if not algo_qp:
            return "dfl" if n == -1 else f"dfl_ssa{n}"
        else:
            return "qpth" if n == -1 else f"qpth_ssa{n}"

    for col, (cond, data) in enumerate(conditions_data):
        ax_reg, ax_cmp = axes[0, col], axes[1, col]

        for algo_qp, color, label_base in [
            (False, "#377EB8", "DFL (LP)"),
            (True,  "#984EA3", "DFL (qpth)"),
        ]:
            regrets, solve_times = [], []
            for n in nagg_vals:
                vkey = _variant_key(algo_qp, n)
                d = data.get(vkey)
                reg = _safe(d, "test_regret")
                if reg is None or _diverged(reg, diverge_thresh):
                    regrets.append(np.nan)
                else:
                    regrets.append(float(np.median(reg[np.isfinite(reg)])))

                sol = _safe(d, "solver_time_sec_per_epoch")
                sa  = _safe(d, "sa_time_sec_per_epoch")
                if sol is not None and len(sol) > 1:
                    t = float(np.mean(sol[1:])) + (float(np.mean(sa[1:])) if sa is not None and len(sa) > 1 else 0.0)
                else:
                    t = np.nan
                solve_times.append(t)

            xs = np.arange(len(nagg_vals))
            # Regret plot
            valid = [(xi, r) for xi, r in zip(xs, regrets) if np.isfinite(r)]
            if valid:
                xv, rv = zip(*valid)
                ax_reg.plot(list(xv), list(rv), marker="o", color=color,
                            label=label_base, linewidth=1.5, markersize=5, zorder=4)

            # Compute plot
            valid_c = [(xi, t) for xi, t in zip(xs, solve_times) if np.isfinite(t)]
            if valid_c:
                xv, tv = zip(*valid_c)
                ax_cmp.plot(list(xv), list(tv), marker="s", color=color,
                            label=label_base, linewidth=1.5, markersize=5, linestyle="--", zorder=4)

        ax_reg.set_xticks(xs)
        ax_reg.set_xticklabels(nagg_labels, fontsize=8)
        ax_reg.set_ylabel("Median test regret")
        ax_reg.set_title(cond["title"], pad=4)
        ax_reg.legend(fontsize=7, loc="best")

        ax_cmp.set_xticks(xs)
        ax_cmp.set_xticklabels(nagg_labels, fontsize=8)
        ax_cmp.set_ylabel("Avg solver+SA time / epoch (s)")
        ax_cmp.set_xlabel("n_agg (M)")
        ax_cmp.legend(fontsize=7, loc="best")

    axes[0, 0].set_title(f"{_CONDITIONS[0]['title']}\n(quality)", pad=4)
    fig.suptitle("Effect of State Aggregation: Regret & Compute vs  M", fontsize=11, y=1.01)
    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Figure 5: Summary table (regret + compute) as a heatmap-style figure
# ---------------------------------------------------------------------------

def plot_summary_table(conditions_data: List[Tuple[Dict, Dict[str, Optional[Dict]]]],
                       out_path: Path, diverge_thresh: float = 1500.0) -> None:
    """
    Color-coded table: rows = methods, cols = conditions.
    Cell value = median test regret (gray if diverged).
    """
    row_labels = [v["label"] for v in _VARIANTS]
    col_labels = [cond["title"] for cond, _ in conditions_data]

    table = np.full((len(_VARIANTS), len(conditions_data)), np.nan)
    for ci, (cond, data) in enumerate(conditions_data):
        oracle_val = None
        for d in data.values():
            if d and d.get("test_mean_oracle_reward") is not None:
                oracle_val = float(d["test_mean_oracle_reward"])
                break
        for ri, v in enumerate(_VARIANTS):
            d = data.get(v["key"])
            reg = _safe(d, "test_regret")
            if reg is None or _diverged(reg, diverge_thresh):
                continue
            finite = reg[np.isfinite(reg)]
            table[ri, ci] = float(np.median(finite))

    fig, ax = plt.subplots(figsize=(2.5 + 2.0 * len(conditions_data), 0.45 * len(_VARIANTS) + 1.2))
    ax.axis("off")

    # Build cell texts and colors
    cell_text, cell_colors = [], []
    for ri in range(len(_VARIANTS)):
        row_text, row_color = [], []
        for ci in range(len(conditions_data)):
            val = table[ri, ci]
            if np.isnan(val):
                row_text.append("—")
                row_color.append("#f0f0f0")
            else:
                row_text.append(f"{val:.1f}")
                # Color scale: green (low regret) → red (high regret)
                col_min = np.nanmin(table[:, ci])
                col_max = np.nanmax(table[:, ci])
                normed = (val - col_min) / max(col_max - col_min, 1e-9)
                r = min(1.0, 0.2 + 0.8 * normed)
                g = min(1.0, 0.8 - 0.6 * normed)
                row_color.append((r, g, 0.2, 0.35))
        cell_text.append(row_text)
        cell_colors.append(row_color)

    tbl = ax.table(cellText=cell_text, cellColours=cell_colors,
                   rowLabels=row_labels, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.5)

    ax.set_title("Median Test Regret  (lower is better; — = diverged)", pad=8, fontsize=10)
    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    # Also save PDF for vector quality
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}  +  {pdf_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-quality plots for DFL/SSA experiments.")
    parser.add_argument("--results_root", default="mdp_lp/100samples_results",
                        help="Root directory of experiment results (default: mdp_lp/100samples_results)")
    parser.add_argument("--out_dir", default="mdp_lp/publication_plots",
                        help="Output directory for plots (default: mdp_lp/publication_plots)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs used (default: 100)")
    parser.add_argument("--qp_reg", type=float, default=10.0, help="QP regularization used (default: 10.0)")
    parser.add_argument("--diverge_thresh", type=float, default=1500.0,
                        help="Median regret threshold above which a run is considered diverged (default: 1500)")
    args = parser.parse_args()

    root = Path(args.results_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {root}")
    conditions_data: List[Tuple[Dict, Dict[str, Optional[Dict]]]] = []
    for cond in _CONDITIONS:
        data = load_condition(root, cond, args.qp_reg, args.epochs)
        n_loaded = sum(1 for d in data.values() if d is not None)
        print(f"  {cond['title']}: {n_loaded}/{len(_VARIANTS)} variants loaded")
        conditions_data.append((cond, data))

    print("\nFigure 1: Reward learning curves...")
    plot_reward_curves(conditions_data, out / "fig1_reward_curves.png")

    print("Figure 2: Test regret boxplots...")
    plot_regret_boxplots(conditions_data, out / "fig2_regret_boxplot.png",
                         diverge_thresh=args.diverge_thresh)

    print("Figure 3: Compute profile (exp_b)...")
    _, exp_b_data = conditions_data[0]
    plot_compute_profile(exp_b_data, out / "fig3_compute_profile.png")

    print("Figure 4: n_agg ablation...")
    plot_nagg_ablation(conditions_data, out / "fig4_nagg_ablation.png",
                       diverge_thresh=args.diverge_thresh)

    print("Figure 5: Summary table...")
    plot_summary_table(conditions_data, out / "fig5_summary_table.png",
                       diverge_thresh=args.diverge_thresh)

    print(f"\nAll plots saved to: {out}")


if __name__ == "__main__":
    main()
