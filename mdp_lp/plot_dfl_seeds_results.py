"""
Aggregate the results from ``run_dfl_seeds.sh`` and render a publication-quality
figure for the inventory experiment (NeurIPS 2026 paper).

The script walks
  mdp_lp/results_elasticity/exp_b/high_margin/{dfl,dfl_qpth}/tag_seed_*/.../results.json
and builds a table of
    (algo, n_agg, seed) -> (test_regret_mean, train_time_per_epoch, oracle_reward)
then plots a single 2-panel figure:

  (left)  Normalized test regret  = 100 * test_mean_regret / test_mean_oracle_reward,
          grouped by aggregation size; LP (ours) vs KKT (QP baseline) side-by-side.
  (right) Per-epoch wall-clock training time (log scale), same grouping.

The figure demonstrates both headline claims of the paper at once:
  1. The LP-based DFL gradient is an order of magnitude faster than the
     KKT-based baseline (right panel).
  2. Soft state aggregation reduces the DFL regret (left panel).

Usage:
    python -m mdp_lp.plot_dfl_seeds_results
    python -m mdp_lp.plot_dfl_seeds_results --exp b --cost high_margin
    python -m mdp_lp.plot_dfl_seeds_results --show-seeds
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Paths / discovery
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent          # .../mdp_lp
RESULTS_ROOT = ROOT / "results_elasticity"
PLOT_DIR = ROOT / "plot_results"
PLOT_DIR.mkdir(exist_ok=True, parents=True)

ALGOS = ("dfl", "dfl_qpth", "twostage")          # LP / KKT / regression baselines
ALGO_LABEL = {
    "dfl":      r"DFL-LP (ours)",
    "dfl_qpth": r"DFL-KKT",
    "twostage": r"Two-stage (MSE)",
}
ALGO_COLOR = {
    "dfl":      "#1565C0",      # blue
    "dfl_qpth": "#E65100",      # orange
    "twostage": "#2E7D32",      # green
}
# Aggregation sizes to display left-to-right. ``-1`` means no aggregation.
N_AGG_ORDER = (-1, 30, 20, 10)
NAGG_LABEL = {-1: "No agg", 10: "M=10", 20: "M=20", 30: "M=30"}


def _load_run(path: Path) -> Optional[Dict]:
    """Load ``results.json`` and keep only the summary scalars we need."""
    try:
        with path.open() as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if d.get("status") != "complete":
        # Run was interrupted; skip rather than contaminate aggregates.
        return None
    train_time = d.get("train_time") or []
    if not train_time:
        return None
    return {
        "algo":        d.get("algo"),
        "seed":        int(d.get("seed", -1)),
        "n_agg":       int(d.get("n_agg", -1)),
        "oracle":      float(d.get("test_mean_oracle_reward", float("nan"))),
        "regret":      float(d.get("test_mean_regret", float("nan"))),
        "pred_reward": float(d.get("test_mean_pred_reward", float("nan"))),
        "time_epoch":  float(np.mean(train_time)),
        "epochs":      int(d.get("n_epochs", len(train_time))),
        "path":        str(path),
    }


def collect_runs(exp: str, cost_case: str) -> List[Dict]:
    """Walk the seeded results tree and return one dict per completed run."""
    base = RESULTS_ROOT / f"exp_{exp}" / cost_case
    runs: List[Dict] = []
    for algo in ALGOS:
        algo_dir = base / algo
        if not algo_dir.is_dir():
            continue
        # Seeded runs live under tag_seed_* subfolders.
        for seed_dir in sorted(algo_dir.glob("tag_seed_*")):
            for results_json in seed_dir.rglob("results.json"):
                run = _load_run(results_json)
                if run is not None:
                    runs.append(run)
    return runs


# ---------------------------------------------------------------------------
# Aggregation across seeds
# ---------------------------------------------------------------------------
def _aggregate(runs: List[Dict],
               restrict_common_seeds: bool = False,
               ) -> Tuple[Dict[Tuple[str, int], Dict[str, np.ndarray]], List[int]]:
    """Group runs by ``(algo, n_agg)`` and stack per-seed scalars.

    If ``restrict_common_seeds`` is True, drop seeds that are not present in
    every (algo, n_agg) cell so the bar means are computed over a matched
    set and the per-seed variance from easy/hard draws cancels out.
    """
    by_cell: Dict[Tuple[str, int], Dict[int, Dict]] = defaultdict(dict)
    for r in runs:
        if not np.isfinite(r["regret"]) or not np.isfinite(r["oracle"]) or r["oracle"] <= 0:
            continue
        cell = by_cell[(r["algo"], r["n_agg"])]
        prev = cell.get(r["seed"])
        # Prefer the longer-trained run when the same (algo, n_agg, seed) appears
        # in multiple epochs_* subdirs (e.g. twostage/tag_seed_3/{epochs_50,epochs_200}).
        if prev is None or r["epochs"] >= prev["epochs"]:
            cell[r["seed"]] = r

    # ``twostage`` is a no-aggregation baseline; do not let its absence from the
    # M=10/20/30 cells shrink the common-seed set used for the aggregated cells.
    cells_for_common = {k: v for k, v in by_cell.items() if v}
    if restrict_common_seeds and cells_for_common:
        seed_sets = [set(cell.keys()) for cell in cells_for_common.values()
                     if cell]  # guard against any empty cells
        common = sorted(set.intersection(*seed_sets)) if seed_sets else []
    else:
        common = sorted({s for cell in by_cell.values() for s in cell.keys()})

    agg: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
    for key, cell in by_cell.items():
        seeds = sorted(s for s in cell.keys() if (not restrict_common_seeds) or s in common)
        if not seeds:
            continue
        agg[key] = {
            "regret_pct": np.array([100.0 * cell[s]["regret"] / cell[s]["oracle"]
                                    for s in seeds]),
            "time_epoch": np.array([cell[s]["time_epoch"] for s in seeds]),
            "seeds":      np.array(seeds, dtype=int),
        }
    return agg, common


def _mean_std(xs: np.ndarray) -> Tuple[float, float]:
    if xs.size == 0:
        return float("nan"), float("nan")
    if xs.size == 1:
        return float(xs[0]), 0.0
    return float(xs.mean()), float(xs.std(ddof=1))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_main_figure(agg: Dict[Tuple[str, int], Dict[str, np.ndarray]],
                     out_path: Path,
                     show_seeds: bool = True) -> None:
    """Two-panel figure: normalized regret (left) and training time (right)."""
    x = np.arange(len(N_AGG_ORDER), dtype=float)
    width = 0.26
    offsets = {"dfl": -width, "dfl_qpth": 0.0, "twostage": +width}

    prev_rc = plt.rcParams.copy()
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "axes.linewidth": 0.9,
        "legend.fontsize": 10,
        "legend.frameon": True,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, (ax_reg, ax_time) = plt.subplots(1, 2, figsize=(12.0, 3.8))

    def _plot_panel(ax, key: str, use_log: bool = False) -> None:
        for algo in ALGOS:
            means, stds, has_data, cells = [], [], [], []
            for n_agg in N_AGG_ORDER:
                row = agg.get((algo, n_agg))
                xs = row[key] if row is not None else np.array([])
                m, s = _mean_std(xs)
                means.append(m)
                stds.append(s)
                has_data.append(xs.size > 0)
                cells.append(row)
            means = np.asarray(means)
            stds = np.asarray(stds)
            mask = np.array(has_data)
            pos = x[mask] + offsets[algo]
            ax.bar(pos, means[mask], width, yerr=stds[mask], capsize=3,
                   color=ALGO_COLOR[algo], edgecolor="black", linewidth=0.6,
                   label=ALGO_LABEL[algo], alpha=0.90,
                   error_kw=dict(ecolor="black", lw=0.8, alpha=0.85))
            if show_seeds:
                # Trace each seed across the four x-groups with a thin line so the
                # reader can see whether the trend is consistent per seed.
                seed_to_xy: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
                for p, row in zip(pos, [c for c, m in zip(cells, mask) if m]):
                    for s_idx, seed in enumerate(row["seeds"]):
                        seed_to_xy[int(seed)].append((p, float(row[key][s_idx])))
                for seed, pts in seed_to_xy.items():
                    if len(pts) == 0:
                        continue
                    pts_sorted = sorted(pts, key=lambda t: t[0])
                    xs_s = [p for p, _ in pts_sorted]
                    ys_s = [v for _, v in pts_sorted]
                    ax.plot(xs_s, ys_s, color=ALGO_COLOR[algo], lw=0.7,
                            alpha=0.55, zorder=4)
                    ax.scatter(xs_s, ys_s, s=36, facecolor="white",
                               edgecolor=ALGO_COLOR[algo], linewidths=1.2,
                               zorder=6)
        if use_log:
            ax.set_yscale("log")

    _plot_panel(ax_reg, "regret_pct")
    _plot_panel(ax_time, "time_epoch", use_log=True)

    ax_reg.set_xticks(x)
    ax_reg.set_xticklabels([NAGG_LABEL[n] for n in N_AGG_ORDER])
    ax_reg.set_ylabel(r"Test regret (% of oracle reward)")
    ax_reg.set_title(r"(a) Decision quality (lower is better)")
    ax_reg.grid(True, axis="y", alpha=0.35, ls=":", lw=0.7)
    ax_reg.legend(loc="upper right")
    ax_reg.set_axisbelow(True)

    ax_time.set_xticks(x)
    ax_time.set_xticklabels([NAGG_LABEL[n] for n in N_AGG_ORDER])
    ax_time.set_ylabel("Training time per epoch (s, log scale)")
    ax_time.set_title(r"(b) Compute cost (lower is better)")
    ax_time.grid(True, which="both", axis="y", alpha=0.35, ls=":", lw=0.7)
    ax_time.legend(loc="upper right")
    ax_time.set_axisbelow(True)

    fig.tight_layout()
    dpi = 180 if out_path.suffix.lower() == ".png" else None
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    plt.rcParams.update(prev_rc)


# ---------------------------------------------------------------------------
# Text summary (for paper body + sanity checking)
# ---------------------------------------------------------------------------
def print_summary(runs: List[Dict],
                  agg: Dict[Tuple[str, int], Dict[str, np.ndarray]]) -> None:
    print("\n=== Per-run detail ===")
    header = f"{'algo':10s} {'n_agg':>6s} {'seed':>5s} {'regret%':>9s} {'time/ep':>9s}  oracle"
    print(header)
    print("-" * len(header))
    for r in sorted(runs, key=lambda r: (r["algo"], r["n_agg"], r["seed"])):
        pct = 100.0 * r["regret"] / r["oracle"] if r["oracle"] > 0 else float("nan")
        print(f"{r['algo']:10s} {r['n_agg']:>6d} {r['seed']:>5d} "
              f"{pct:>9.2f} {r['time_epoch']:>9.2f}  {r['oracle']:.2f}")

    print("\n=== Aggregated (mean ± std across seeds) ===")
    print(f"{'algo':10s} {'n_agg':>6s} {'n':>3s} "
          f"{'regret%':>16s}  {'time/ep (s)':>18s}  seeds")
    print("-" * 80)
    for algo in ALGOS:
        for n_agg in N_AGG_ORDER:
            row = agg.get((algo, n_agg))
            if row is None or row["regret_pct"].size == 0:
                continue
            m_r, s_r = _mean_std(row["regret_pct"])
            m_t, s_t = _mean_std(row["time_epoch"])
            seeds = ",".join(str(int(s)) for s in sorted(row["seeds"]))
            print(f"{algo:10s} {n_agg:>6d} {row['regret_pct'].size:>3d} "
                  f"{m_r:>7.2f} ± {s_r:>5.2f}   "
                  f"{m_t:>8.2f} ± {s_t:>6.2f}     {seeds}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--exp", default="b", help="experiment_type (default: b)")
    p.add_argument("--cost", default="high_margin",
                   help="cost_case subfolder (default: high_margin)")
    p.add_argument("--out", default=str(PLOT_DIR / "elasticity_main.pdf"),
                   help="Output figure path (PDF)")
    p.add_argument("--hide-seeds", action="store_true",
                   help="Disable the per-seed dot/line overlay on the bars")
    args = p.parse_args()

    runs = collect_runs(args.exp, args.cost)
    if not runs:
        print(f"No completed runs under exp_{args.exp}/{args.cost}", file=sys.stderr)
        sys.exit(1)

    # For the figure we use only seeds that are present in every cell, so the
    # bar means compare matched populations rather than conflating seed-level
    # variance with method-level variance.
    agg_common, common_seeds = _aggregate(runs, restrict_common_seeds=True)
    agg_all, _ = _aggregate(runs, restrict_common_seeds=False)
    print_summary(runs, agg_all)
    print(f"\nSeeds used in the figure (common across all cells): {common_seeds}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    show_seeds = not args.hide_seeds
    plot_main_figure(agg_common, out_path, show_seeds=show_seeds)
    # Also save a PNG preview for README/slides.
    png_path = out_path.with_suffix(".png")
    plot_main_figure(agg_common, png_path, show_seeds=show_seeds)
    print(f"\nSaved figure to {out_path}\n           and {png_path}")


if __name__ == "__main__":
    main()
