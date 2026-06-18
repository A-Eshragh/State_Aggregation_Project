"""
Three-panel summary figure for the inventory-MDP DFL experiments
(mirrors mdp_continuous/make_figure_3panel.py for the LP/SAA setting).

Panel 1 — Training curves (per-epoch training regret %, mean +/- SE across seeds)
    Two-stage, DFL-LP, DFL-QPTH, DFL-Sketch (soft state aggregation).

Panel 2 — Decision quality (test regret %, bar chart, lower = better)
    Same four methods, mean +/- SE across seeds.

Panel 3 — Training cost (solver time per epoch, bar chart, log scale)
    DFL methods report the LP/QP solver+backward time; Two-stage has no
    decision-aware solver, so its full per-epoch wall-clock is shown for
    reference.

Reads:
    mdp_lp/results/exp_{exp}/{cost_case}/{algo}/tag_seed_*/epochs_*/results.json

Usage (from project root):
    python -m mdp_lp.make_figure_3panel
    python -m mdp_lp.make_figure_3panel --exp b --cost normal_margin
    python -m mdp_lp.make_figure_3panel --sketch_dir dfl_sketch_ngg_10 \\
        --sketch_K 8 --sketch_r 0.2
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


HERE = Path(__file__).resolve().parent
RESULTS_ROOT = HERE / "results"
PLOT_DIR = HERE / "plot_results"


METHODS = ("twostage", "dfl", "dfl_qpth", "dfl_sketch")
LABELS = {
    "twostage":   "Two-stage",
    "dfl":        "DFL",
    "dfl_qpth":   "DFL-QP",
    "dfl_sketch": "DFL-SA",
}
# Order of methods for the compute-cost bar chart.
COST_BAR_ORDER = ("dfl_qpth", "dfl", "dfl_sketch", "twostage")
COLORS = {
    "twostage":   "#ff7f0e",   # orange
    "dfl":        "#1f77b4",   # blue
    "dfl_qpth":   "#2ca02c",   # green
    "dfl_sketch": "#9467bd",   # purple
}
LINESTYLES = {
    "twostage":   "--",
    "dfl":        "-",
    "dfl_qpth":   "-",
    "dfl_sketch": "-",
}


# ── Run discovery ───────────────────────────────────────────────────────

def _load_run(path: Path) -> Optional[Dict]:
    try:
        with path.open() as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if d.get("status") != "complete":
        return None
    return d


def _matches_sketch(d: Dict, K: Optional[int], r: Optional[float]) -> bool:
    if K is not None and int(d.get("n_sketches", -1)) != K:
        return False
    if r is not None:
        if abs(float(d.get("sketch_ratio", -1.0)) - r) > 1e-6:
            return False
    return True


def collect_runs(exp: str, cost: str,
                 sketch_K: Optional[int], sketch_r: Optional[float],
                 seeds: Optional[List[int]],
                 sketch_canonical_nagg: int,
                 results_root: Optional[Path] = None,
                 ) -> Tuple[Dict[str, List[Dict]], Dict[int, List[Dict]]]:
    """Walk the seeded results tree and return:
        grouped:        {method: [run, ...]} for the four methods. DFL-Sketch
                        is restricted to ``sketch_canonical_nagg`` so panels
                        1 & 3 use a single representative configuration.
        sketch_by_nagg: {n_agg: [run, ...]} for every DFL-Sketch n_agg
                        directory found on disk (panel 2 sweep).
    """
    root = results_root if results_root is not None else RESULTS_ROOT
    base = root / f"exp_{exp}" / cost
    requested = set(int(s) for s in seeds) if seeds else None

    grouped: Dict[str, List[Dict]] = {m: [] for m in METHODS}

    # Non-sketch methods: read straight from the algo subdirectory.
    for method in ("twostage", "dfl", "dfl_qpth"):
        algo_root = base / method
        if not algo_root.is_dir():
            continue
        best: Dict[int, Dict] = {}
        for results_json in algo_root.rglob("results.json"):
            d = _load_run(results_json)
            if d is None:
                continue
            seed = int(d.get("seed", -1))
            if requested is not None and seed not in requested:
                continue
            if int(d.get("requested_n_samples", -1)) not in (-1, 10):
                continue
            d["_path"] = str(results_json)
            prev = best.get(seed)
            if prev is None or int(d.get("n_epochs", 0)) >= int(prev.get("n_epochs", 0)):
                best[seed] = d
        grouped[method] = [best[s] for s in sorted(best.keys())]

    # DFL-Sketch: collect across every dfl_sketch_ngg_<N> directory.
    sketch_by_nagg: Dict[int, Dict[int, Dict]] = defaultdict(dict)
    for sketch_root in sorted(base.glob("dfl_sketch_ngg_*")):
        if not sketch_root.is_dir():
            continue
        for results_json in sketch_root.rglob("results.json"):
            d = _load_run(results_json)
            if d is None:
                continue
            if d.get("algo") != "dfl_sketch":
                continue
            if not _matches_sketch(d, sketch_K, sketch_r):
                continue
            seed = int(d.get("seed", -1))
            if requested is not None and seed not in requested:
                continue
            if int(d.get("requested_n_samples", -1)) not in (-1, 10):
                continue
            n_agg = int(d.get("n_agg", -1))
            d["_path"] = str(results_json)
            cell = sketch_by_nagg[n_agg]
            prev = cell.get(seed)
            if prev is None or int(d.get("n_epochs", 0)) >= int(prev.get("n_epochs", 0)):
                cell[seed] = d
    sketch_grouped: Dict[int, List[Dict]] = {
        n_agg: [cell[s] for s in sorted(cell.keys())]
        for n_agg, cell in sorted(sketch_by_nagg.items())
    }

    grouped["dfl_sketch"] = sketch_grouped.get(sketch_canonical_nagg, [])
    return grouped, sketch_grouped


# ── Curve helpers ───────────────────────────────────────────────────────

def _stack_truncated(arrays: List[List[float]]) -> Optional[np.ndarray]:
    if not arrays:
        return None
    L = min(len(a) for a in arrays)
    if L == 0:
        return None
    return np.array([a[:L] for a in arrays], dtype=float)


def _per_epoch_train_reward(runs: List[Dict]) -> Optional[np.ndarray]:
    """Stack per-epoch training reward across seeds (raw, no normalization)."""
    series = []
    for d in runs:
        rew = d.get("train_reward_per_epoch")
        if not rew:
            continue
        series.append(list(rew))
    return _stack_truncated(series)


def _draw_curve(ax, arr: np.ndarray, color: str, ls: str, label: str) -> None:
    mu = arr.mean(axis=0)
    se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 \
        else np.zeros_like(mu)
    xs = np.arange(arr.shape[1])
    ax.plot(xs, mu, color=color, lw=2.0, ls=ls, label=label)
    ax.fill_between(xs, mu - se, mu + se, color=color, alpha=0.20)


def _test_regret_pct(runs: List[Dict]) -> np.ndarray:
    vals = []
    for d in runs:
        oracle = d.get("test_mean_oracle_reward")
        regret = d.get("test_mean_regret")
        if oracle is None or oracle <= 0 or regret is None:
            continue
        vals.append(100.0 * float(regret) / float(oracle))
    return np.array(vals, dtype=float)


def _epoch_time(runs: List[Dict], method: str) -> np.ndarray:
    """Per-seed average training time per epoch.

    DFL methods: ``solver_time_sec_per_epoch`` (LP/QP forward + backward).
    Two-stage:   ``train_time`` (full wall-clock; no decision-aware solver).
    """
    key = "train_time" if method == "twostage" else "solver_time_sec_per_epoch"
    vals = []
    for d in runs:
        ts = d.get(key) or []
        if not ts:
            # Fallback to train_time if solver-only is missing/zeroed.
            ts = d.get("train_time") or []
        if not ts:
            continue
        ts = np.asarray(ts, dtype=float)
        # Drop the zero-only edge case (e.g. solver=0 for runs that didn't
        # log it) so we don't show a misleading 0-bar.
        if np.allclose(ts, 0.0):
            ts = np.asarray(d.get("train_time") or [], dtype=float)
            if ts.size == 0:
                continue
        vals.append(float(ts.mean()))
    return np.array(vals, dtype=float)


def _mean_se(xs: np.ndarray) -> Tuple[float, float]:
    if xs.size == 0:
        return float("nan"), float("nan")
    if xs.size == 1:
        return float(xs[0]), 0.0
    return float(xs.mean()), float(xs.std(ddof=1) / np.sqrt(xs.size))


# ── Plot ────────────────────────────────────────────────────────────────

_RC_PARAMS = {
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _draw_panel_training_curves(ax, grouped: Dict[str, List[Dict]]) -> None:
    drew_any_curve = False
    for method in METHODS:
        arr = _per_epoch_train_reward(grouped.get(method, []))
        if arr is None:
            continue
        _draw_curve(ax, arr, COLORS[method], LINESTYLES[method], LABELS[method])
        drew_any_curve = True
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Training reward")
    ax.set_title("Training curves")
    ax.grid(True, alpha=0.3)
    if drew_any_curve:
        ax.legend(loc="lower right", framealpha=0.9)
    else:
        ax.text(0.5, 0.5, "No training curves cached",
                ha="center", va="center", transform=ax.transAxes)


def _draw_panel_sketch_sweep(ax, sketch_by_nagg: Dict[int, List[Dict]]) -> None:
    sketch_xs, sketch_mu, sketch_se = [], [], []
    for n_agg in sorted(sketch_by_nagg.keys()):
        if n_agg < 0:
            continue
        xs = _test_regret_pct(sketch_by_nagg.get(n_agg, []))
        if xs.size == 0:
            continue
        m, s = _mean_se(xs)
        sketch_xs.append(int(n_agg))
        sketch_mu.append(m)
        sketch_se.append(s)
    if sketch_xs:
        sx = np.asarray(sketch_xs)
        sm = np.asarray(sketch_mu)
        ss = np.asarray(sketch_se)
        ax.plot(sx, sm, color=COLORS["dfl_sketch"],
                ls=LINESTYLES["dfl_sketch"], lw=2.0, marker="o",
                markersize=8, label=LABELS["dfl_sketch"])
        ax.fill_between(sx, sm - ss, sm + ss,
                        color=COLORS["dfl_sketch"], alpha=0.20)
        ax.set_xticks(sx)
        ax.set_xticklabels([f"M={x}" for x in sx])
    else:
        ax.text(0.5, 0.5, "No DFL-SA sweep cached",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Aggregation size  M")
    ax.set_ylabel("Test regret (% of oracle reward)")
    ax.set_title("Decision quality vs aggregation size")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_axisbelow(True)


def _draw_panel_compute_cost(ax, grouped: Dict[str, List[Dict]]) -> None:
    bar_methods, bar_means, bar_ses, bar_labels = [], [], [], []
    for method in COST_BAR_ORDER:
        xs = _epoch_time(grouped.get(method, []), method)
        if xs.size == 0:
            continue
        m, s = _mean_se(xs)
        bar_methods.append(method)
        bar_means.append(m)
        bar_ses.append(s)
        bar_labels.append(LABELS[method])
    if bar_methods:
        x = np.arange(len(bar_methods))
        ax.bar(x, bar_means, yerr=bar_ses, capsize=4,
               color=[COLORS[m] for m in bar_methods], alpha=0.90,
               edgecolor="black", linewidth=0.6,
               error_kw=dict(ecolor="black", lw=0.8))
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels)
        ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "No timing data cached",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_ylabel("Training time per epoch (s, log scale)")
    ax.set_title("Compute cost")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    ax.set_axisbelow(True)


def _save_panel_pdf(draw_fn, save_path: Path) -> None:
    prev_rc = plt.rcParams.copy()
    plt.rcParams.update(_RC_PARAMS)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    draw_fn(ax)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update(prev_rc)
    print(f"Saved figure to {save_path}")


def plot_three_panel(grouped: Dict[str, List[Dict]],
                     sketch_by_nagg: Dict[int, List[Dict]],
                     save_path: Path,
                     title_suffix: str = "",
                     sketch_canonical_nagg: int = 10) -> None:
    """Save three separate single-panel PDFs derived from `save_path`'s stem."""
    base = save_path.with_suffix("")
    suffix = save_path.suffix or ".pdf"
    p1 = base.with_name(base.name + "_training_curves").with_suffix(suffix)
    p2 = base.with_name(base.name + "_decision_quality").with_suffix(suffix)
    p3 = base.with_name(base.name + "_compute_cost").with_suffix(suffix)
    _save_panel_pdf(lambda ax: _draw_panel_training_curves(ax, grouped), p1)
    _save_panel_pdf(lambda ax: _draw_panel_sketch_sweep(ax, sketch_by_nagg), p2)
    _save_panel_pdf(lambda ax: _draw_panel_compute_cost(ax, grouped), p3)


# ── Text summary (mirrors plot_dfl_seeds_results.print_summary) ─────────

def print_summary(grouped: Dict[str, List[Dict]],
                  sketch_by_nagg: Dict[int, List[Dict]]) -> None:
    print("\n=== Per-run detail ===")
    header = f"{'method':14s} {'seed':>5s} {'n_agg':>6s} " \
             f"{'regret%':>9s} {'time/ep':>9s}  oracle"
    print(header)
    print("-" * len(header))
    for method in METHODS:
        runs = (sum((sketch_by_nagg[k] for k in sorted(sketch_by_nagg)), [])
                if method == "dfl_sketch" else grouped.get(method, []))
        for d in sorted(runs, key=lambda r: (int(r.get("n_agg", -1)),
                                             int(r.get("seed", -1)))):
            oracle = float(d.get("test_mean_oracle_reward", 0.0)) or float("nan")
            pct = (100.0 * float(d.get("test_mean_regret", float("nan"))) / oracle
                   if oracle and oracle > 0 else float("nan"))
            t_key = "train_time" if method == "twostage" else "solver_time_sec_per_epoch"
            ts = d.get(t_key) or d.get("train_time") or []
            t_ep = float(np.mean(ts)) if ts else float("nan")
            print(f"{method:14s} {int(d.get('seed', -1)):>5d} "
                  f"{int(d.get('n_agg', -1)):>6d} {pct:>9.2f} "
                  f"{t_ep:>9.2f}  {oracle:.2f}")

    print("\n=== Aggregated (mean ± SE across seeds) ===")
    print(f"{'method':14s} {'n_agg':>6s} {'n':>3s} "
          f"{'regret%':>16s}  {'time/ep (s)':>18s}")
    print("-" * 70)
    for method in ("twostage", "dfl", "dfl_qpth"):
        runs = grouped.get(method, [])
        regret = _test_regret_pct(runs)
        times = _epoch_time(runs, method)
        if regret.size == 0 and times.size == 0:
            continue
        m_r, s_r = _mean_se(regret)
        m_t, s_t = _mean_se(times)
        print(f"{method:14s} {'-':>6s} {regret.size:>3d} "
              f"{m_r:>7.2f} ± {s_r:>5.2f}   {m_t:>8.3f} ± {s_t:>6.3f}")
    for n_agg in sorted(sketch_by_nagg.keys()):
        runs = sketch_by_nagg[n_agg]
        regret = _test_regret_pct(runs)
        times = _epoch_time(runs, "dfl_sketch")
        if regret.size == 0 and times.size == 0:
            continue
        m_r, s_r = _mean_se(regret)
        m_t, s_t = _mean_se(times)
        print(f"{'dfl_sketch':14s} {int(n_agg):>6d} {regret.size:>3d} "
              f"{m_r:>7.2f} ± {s_r:>5.2f}   {m_t:>8.3f} ± {s_t:>6.3f}")


# ── CLI ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--exp", default="b", help="experiment_type (default: b)")
    p.add_argument("--cost", default="normal_margin",
                   help="cost_case subfolder (default: normal_margin)")
    p.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="If set, restrict to these seeds; otherwise use every "
                        "completed seed found on disk.")
    p.add_argument("--sketch_K", type=int, default=8,
                   help="Filter DFL-Sketch runs to this n_sketches.")
    p.add_argument("--sketch_r", type=float, default=0.2,
                   help="Filter DFL-Sketch runs to this sketch_ratio.")
    p.add_argument("--sketch_canonical_nagg", type=int, default=10,
                   help="Which DFL-Sketch n_agg to use in panels 1 and 3 "
                        "(panel 2 plots all available n_agg values).")
    p.add_argument("--save_path", default=str(PLOT_DIR / "figure_3panel.pdf"))
    p.add_argument("--results_root", default=None,
                   help="Override the default results root "
                        "(default: mdp_lp/results).")
    return p.parse_args()


def main():
    args = parse_args()

    results_root = Path(args.results_root) if args.results_root else None
    grouped, sketch_by_nagg = collect_runs(
        exp=args.exp, cost=args.cost,
        sketch_K=args.sketch_K, sketch_r=args.sketch_r,
        seeds=args.seeds,
        sketch_canonical_nagg=args.sketch_canonical_nagg,
        results_root=results_root,
    )
    if not any(grouped.values()) and not sketch_by_nagg:
        print(f"No completed runs under exp_{args.exp}/{args.cost}",
              file=sys.stderr)
        sys.exit(1)

    print_summary(grouped, sketch_by_nagg)

    title_suffix = (f"sketch K={args.sketch_K}, r={args.sketch_r}"
                    if sketch_by_nagg else "")
    plot_three_panel(grouped, sketch_by_nagg, Path(args.save_path),
                     title_suffix=title_suffix,
                     sketch_canonical_nagg=args.sketch_canonical_nagg)


if __name__ == "__main__":
    main()
