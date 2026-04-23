"""
Aggregate the results from ``mdp_continuous/run_dfl_seeds.sh`` and render the
CartPole learning-curve figure (NeurIPS 2026 paper, Figure 1).

The script walks
  mdp_continuous/results_cartpole/{neural_dfl,two_stage}/tag_seed_*/.../results.json
and builds per-algo arrays of shape (n_seeds, n_iterations), then plots a
single panel with mean ± 1 std shading across seeds.

This mirrors ``mdp_lp/plot_dfl_seeds_results.py``: the bash driver produces
one ``results.json`` per (algo, seed); this script is the renderer. No
training happens here, so it is safe to rerun freely.

Usage:
    python -m mdp_continuous.plot_dfl_seeds_results
    python -m mdp_continuous.plot_dfl_seeds_results --show-seeds
    python -m mdp_continuous.plot_dfl_seeds_results --out my_fig.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Paths / discovery
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent          # .../mdp_continuous
RESULTS_ROOT = ROOT / "results_cartpole"
PLOT_DIR = ROOT
PLOT_DIR.mkdir(exist_ok=True, parents=True)

ALGOS = ("neural_dfl", "two_stage")
ALGO_LABEL = {
    "neural_dfl": r"Neural DFL (ours)",
    "two_stage":  r"Two-stage (predict-then-optimize)",
}
ALGO_COLOR = {
    "neural_dfl": "#1f77b4",     # blue
    "two_stage":  "#ff7f0e",     # orange
}
ALGO_LINESTYLE = {
    "neural_dfl": "-",
    "two_stage":  "--",
}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def _load_run(path: Path) -> Optional[Dict]:
    """Load ``results.json`` and keep only the fields the plotter needs."""
    try:
        with path.open() as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if d.get("status") != "complete":
        return None
    rewards = d.get("avg_reward_per_iter") or []
    if not rewards:
        return None
    return {
        "algo":    d.get("algo"),
        "seed":    int(d.get("seed", -1)),
        "n_iter":  int(d.get("n_iterations", len(rewards))),
        "rewards": np.asarray(rewards, dtype=float),
        "lengths": np.asarray(d.get("avg_length_per_iter") or [], dtype=float),
        "time":    float(d.get("total_train_time", float("nan"))),
        "path":    str(path),
    }


def collect_runs() -> List[Dict]:
    """Walk the seeded results tree and return one dict per completed run."""
    runs: List[Dict] = []
    for algo in ALGOS:
        algo_dir = RESULTS_ROOT / algo
        if not algo_dir.is_dir():
            continue
        for seed_dir in sorted(algo_dir.glob("tag_seed_*")):
            for results_json in seed_dir.rglob("results.json"):
                run = _load_run(results_json)
                if run is not None:
                    runs.append(run)
    return runs


# ---------------------------------------------------------------------------
# Aggregation across seeds
# ---------------------------------------------------------------------------
def _aggregate(runs: List[Dict]) -> Dict[str, Dict[str, np.ndarray]]:
    """Group by algo and stack per-seed learning curves to a common length.

    If seeds disagree on ``n_iterations`` we truncate to the minimum common
    length so the mean/std are computed over a matched set of iterations.
    """
    by_algo: Dict[str, Dict[int, Dict]] = defaultdict(dict)
    for r in runs:
        algo = r["algo"]
        if algo not in ALGO_LABEL:
            continue
        prev = by_algo[algo].get(r["seed"])
        # Prefer the longer curve if the same (algo, seed) appears twice.
        if prev is None or r["rewards"].size >= prev["rewards"].size:
            by_algo[algo][r["seed"]] = r

    agg: Dict[str, Dict[str, np.ndarray]] = {}
    for algo, cell in by_algo.items():
        if not cell:
            continue
        seeds = sorted(cell.keys())
        min_len = min(cell[s]["rewards"].size for s in seeds)
        curves = np.stack([cell[s]["rewards"][:min_len] for s in seeds], axis=0)
        times = np.array([cell[s]["time"] for s in seeds], dtype=float)
        agg[algo] = {
            "seeds":  np.asarray(seeds, dtype=int),
            "curves": curves,               # (n_seeds, min_len)
            "mean":   curves.mean(axis=0),
            "std":    curves.std(axis=0, ddof=1) if curves.shape[0] > 1
                      else np.zeros(curves.shape[1]),
            "time":   times,
        }
    return agg


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_main_figure(agg: Dict[str, Dict[str, np.ndarray]],
                     out_path: Path,
                     show_seeds: bool = False) -> None:
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

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 4.0))

    for algo in ALGOS:
        row = agg.get(algo)
        if row is None:
            continue
        xs = np.arange(row["mean"].size)
        ax.plot(xs, row["mean"],
                color=ALGO_COLOR[algo],
                linestyle=ALGO_LINESTYLE[algo],
                lw=2.0, label=ALGO_LABEL[algo])
        ax.fill_between(xs, row["mean"] - row["std"], row["mean"] + row["std"],
                        color=ALGO_COLOR[algo], alpha=0.20, linewidth=0)
        if show_seeds:
            for curve in row["curves"]:
                ax.plot(xs, curve[:xs.size], color=ALGO_COLOR[algo],
                        lw=0.7, alpha=0.35, zorder=2)

    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Avg. discounted episode reward")
    ax.set_title("CartPole: Neural DFL vs. two-stage")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.set_axisbelow(True)

    fig.tight_layout()
    dpi = 180 if out_path.suffix.lower() == ".png" else None
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    plt.rcParams.update(prev_rc)


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------
def print_summary(runs: List[Dict],
                  agg: Dict[str, Dict[str, np.ndarray]]) -> None:
    print("\n=== Per-run detail ===")
    header = f"{'algo':12s} {'seed':>5s} {'n_iter':>7s} {'final':>9s} {'best':>9s} {'time(s)':>9s}"
    print(header)
    print("-" * len(header))
    for r in sorted(runs, key=lambda r: (r["algo"], r["seed"])):
        final = float(r["rewards"][-1]) if r["rewards"].size else float("nan")
        best = float(r["rewards"].max()) if r["rewards"].size else float("nan")
        print(f"{r['algo']:12s} {r['seed']:>5d} {r['n_iter']:>7d} "
              f"{final:>9.2f} {best:>9.2f} {r['time']:>9.1f}")

    print("\n=== Aggregated (mean ± std across seeds) ===")
    print(f"{'algo':12s} {'n':>3s} {'final_reward':>18s} {'best_reward':>17s} {'time/run (s)':>15s}  seeds")
    print("-" * 88)
    for algo in ALGOS:
        row = agg.get(algo)
        if row is None:
            continue
        finals = row["curves"][:, -1]
        bests = row["curves"].max(axis=1)
        n = row["curves"].shape[0]
        seeds_str = ",".join(str(int(s)) for s in row["seeds"])
        t_mean = float(np.nanmean(row["time"])) if row["time"].size else float("nan")
        t_std = float(np.nanstd(row["time"], ddof=1)) if n > 1 else 0.0
        print(f"{algo:12s} {n:>3d} "
              f"{finals.mean():>8.2f} ± {finals.std(ddof=1) if n > 1 else 0.0:>6.2f}   "
              f"{bests.mean():>7.2f} ± {bests.std(ddof=1) if n > 1 else 0.0:>6.2f}   "
              f"{t_mean:>6.1f} ± {t_std:>5.1f}    {seeds_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=str(PLOT_DIR / "figure1_cartpole.pdf"),
                   help="Output figure path (PDF)")
    p.add_argument("--show-seeds", action="store_true",
                   help="Overlay thin per-seed traces on top of the mean curve")
    args = p.parse_args()

    runs = collect_runs()
    if not runs:
        print(f"No completed runs under {RESULTS_ROOT}", file=sys.stderr)
        sys.exit(1)

    agg = _aggregate(runs)
    print_summary(runs, agg)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_main_figure(agg, out_path, show_seeds=args.show_seeds)
    png_path = out_path.with_suffix(".png")
    plot_main_figure(agg, png_path, show_seeds=args.show_seeds)
    print(f"\nSaved figure to {out_path}\n           and {png_path}")


if __name__ == "__main__":
    main()
