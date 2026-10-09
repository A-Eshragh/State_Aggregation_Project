"""Render the 3-panel figures for both inventory MDP and CartPole.

Single rendering pipeline; data-loading helpers are imported from each
dataset's existing module so this file only owns the plotting logic.

Usage:
    python make_figures.py            # render both datasets (default)
    python make_figures.py inventory  # only mdp_lp
    python make_figures.py cartpole   # only mdp_continuous
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt

from mdp_lp import make_figure_3panel as inv_mod
from mdp_continuous import make_figure_3panel as cp_mod
from mdp_continuous import make_figure1_cartpole as m1
from mdp_continuous import sweep_n_features as snf


ROOT = Path(__file__).resolve().parent

RC = {
    "font.family": "serif", "font.size": 20, "axes.labelsize": 18,
    "axes.titlesize": 20, "legend.fontsize": 13, "xtick.labelsize": 16,
    "ytick.labelsize": 18, "pdf.fonttype": 42, "ps.fonttype": 42,
}
LABELS = {
    "twostage": "Two-stage", "dfl": "DFL-LP", "dfl_qpth": "DFL-QP",
    "dfl_sketch": "DFL-Sketch", "neural_dfl": "DFL-LP", "dfl_dr": "DFL-SA",
    "cp_sketch": "DFL-Sketch", "two_stage_mbrl": "Two-stage",
    "dfl_sa_M10": "DFL-SA (M = 10)",
}
COLORS = {
    "twostage": "#ff7f0e", "dfl": "#1f77b4", "dfl_qpth": "#2ca02c",
    "dfl_sketch": "#d62728", "neural_dfl": "#1f77b4", "dfl_dr": "#9467bd",
    "cp_sketch": "#d62728", "two_stage_mbrl": "#ff7f0e",
    "dfl_sa_M10": "#9467bd",
}
LS = {
    "twostage": "--", "dfl": "-", "dfl_qpth": "-", "dfl_sketch": ":",
    "neural_dfl": "-", "dfl_dr": "-", "cp_sketch": ":",
    "two_stage_mbrl": "--",
    "dfl_sa_M10": "-",
}
INV_COST_ORDER = ("twostage", "dfl", "dfl_qpth", "dfl_sketch")
CP_COST_ORDER  = ("two_stage_mbrl", "dfl_qpth", "neural_dfl", "dfl_dr",
                  "cp_sketch")


# ── Shared rendering primitives ──────────────────────────────────────────

def _save(draw, path: Path) -> None:
    prev = plt.rcParams.copy()
    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    draw(ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update(prev)
    print(f"Saved {path}")


def _curve(ax, arr: np.ndarray, key: str, label: str) -> None:
    mu = arr.mean(axis=0)
    se = (arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
          if arr.shape[0] > 1 else np.zeros_like(mu))
    xs = np.arange(arr.shape[1])
    ax.plot(xs, mu, color=COLORS[key], ls=LS[key], lw=2.0, label=label)
    ax.fill_between(xs, mu - se, mu + se, color=COLORS[key], alpha=0.2)


def _href(ax, mu: float, se: float, x_lo: float, x_hi: float,
          key: str, label: str) -> None:
    """Horizontal reference band — used for non-sweep methods in panel 2."""
    ax.axhline(mu, color=COLORS[key], ls="--", lw=1.6,
               alpha=0.85, label=label)
    ax.fill_between([x_lo, x_hi], [mu - se, mu - se],
                    [mu + se, mu + se], color=COLORS[key], alpha=0.12)


def _bars(ax, keys: List[str], means: List[float], ses: List[float],
          *, log: bool = False,
          labels: Optional[List[str]] = None) -> None:
    if not keys:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return
    x = np.arange(len(keys))
    ax.bar(x, means, yerr=ses, capsize=4, width=0.65,
           color=[COLORS[k] for k in keys], alpha=0.88,
           edgecolor="black", linewidth=0.6,
           error_kw=dict(ecolor="black", lw=0.8))
    ax.set_xticks(x)
    ax.set_xticklabels(labels if labels is not None
                       else [LABELS[k] for k in keys])
    if log:
        ax.set_yscale("log")


def _mse(xs: np.ndarray):
    if xs.size == 0: return float("nan"), float("nan")
    if xs.size == 1: return float(xs[0]), 0.0
    return float(xs.mean()), float(xs.std(ddof=1) / np.sqrt(xs.size))


# ── Inventory MDP (mdp_lp) ───────────────────────────────────────────────

def render_inventory(out: Path, results_root: Path,
                     seeds: Optional[List[int]] = None) -> None:
    grouped, sketch_r01 = inv_mod.collect_runs(
        exp="b", cost="normal_margin", sketch_K=8, sketch_r=0.1,
        seeds=seeds, sketch_canonical_nagg=-1, results_root=results_root,
    )
    _, sketch_r02 = inv_mod.collect_runs(
        exp="b", cost="normal_margin", sketch_K=8, sketch_r=0.2,
        seeds=seeds, sketch_canonical_nagg=-1, results_root=results_root,
    )

    def curves(ax):
        for m in ("twostage", "dfl_qpth", "dfl", "dfl_sketch"):
            arr = inv_mod._per_epoch_train_reward(grouped.get(m, []))
            if arr is not None:
                _curve(ax, arr, m, LABELS[m])
        sa10 = inv_mod._per_epoch_train_reward(sketch_r02.get(10, []))
        if sa10 is not None:
            _curve(ax, sa10, "dfl_sa_M10", LABELS["dfl_sa_M10"])
        ax.set(xlabel="Training epoch", ylabel="Training reward",
               title="Inventory MDP — Training curves")
        ax.set_ylim(bottom=100)
        ax.grid(True, alpha=0.3); ax.legend(loc="lower right")

    def quality(ax):
        # Non-sketch methods + new DFL-Sketch (K=8, r=0.1) +
        # DFL-Sketch sweep over M=10/20/30 (K=8, r=0.2).
        keys, labels, means, ses = [], [], [], []
        def _add(key, label, vals):
            if vals.size == 0: return
            mu, se = _mse(vals)
            keys.append(key); labels.append(label)
            means.append(mu); ses.append(se)
        _add("twostage", "TS",
             inv_mod._test_regret_pct(grouped.get("twostage", [])))
        _add("dfl_qpth", "QP",
             inv_mod._test_regret_pct(grouped.get("dfl_qpth", [])))
        _add("dfl", "LP",
             inv_mod._test_regret_pct(grouped.get("dfl", [])))
        if sketch_r01.get(-1):
            _add("dfl_sketch", "Sketch",
                 inv_mod._test_regret_pct(sketch_r01[-1]))
        for n_agg in (10, 20, 30):
            if sketch_r02.get(n_agg):
                _add("dfl_sa_M10", f"SA\nM={n_agg}",
                     inv_mod._test_regret_pct(sketch_r02[n_agg]))
        _bars(ax, keys, means, ses, labels=labels)
        ax.set(ylabel="Test regret",
               title="Inventory MDP — Decision quality")
        ax.set_ylim(10, 18)
        ax.tick_params(axis="x", labelsize=RC["xtick.labelsize"] - 1)
        ax.grid(True, axis="y", alpha=0.3, ls=":")
        ax.set_axisbelow(True)

    def cost(ax):
        # Full per-epoch wall-clock so two-stage's MSE-only training is
        # compared on the same accounting as DFL methods' MSE + solver
        # passes. DFL-Sketch contributes one bar per (K=8 r=0.1) and
        # M=10/20/30 (K=8 r=0.2).
        keys, labels, means, ses = [], [], [], []
        def _add(key, label, runs):
            totals = [float(np.sum(d.get("train_time") or []))
                      for d in runs if d.get("train_time")]
            v = np.array(totals)
            if v.size:
                mu, s = _mse(v)
                keys.append(key); labels.append(label)
                means.append(mu); ses.append(s)
        _add("twostage", "TS", grouped.get("twostage", []))
        _add("dfl_qpth", "QP", grouped.get("dfl_qpth", []))
        _add("dfl",      "LP", grouped.get("dfl", []))
        if sketch_r01.get(-1):
            _add("dfl_sketch", "Sketch", sketch_r01[-1])
        for n_agg in (10, 20, 30):
            if sketch_r02.get(n_agg):
                _add("dfl_sa_M10", f"SA\nM={n_agg}",
                     sketch_r02[n_agg])
        _bars(ax, keys, means, ses, labels=labels, log=False)
        ax.set(ylabel="Total training time (s)",
               title="Inventory MDP — Compute cost")
        ax.tick_params(axis="x", labelsize=RC["xtick.labelsize"] - 1)
        ax.grid(True, axis="y", alpha=0.3)

    _save(curves, out / "inventory_training_curves.pdf")
    _save(quality, out / "inventory_decision_quality.pdf")
    _save(cost,    out / "inventory_compute_cost.pdf")


# ── CartPole (mdp_continuous) ────────────────────────────────────

def _cp_args():
    """Match the args that produced mdp_continuous/figure_3panel.pdf."""
    saved = sys.argv
    try:
        sys.argv = [
            "x", "--plot_only",
            "--n_iterations", "60",
            "--mbrl_n_iterations", "60",
            "--seeds", *[str(s) for s in range(1, 21)],
        ]
        return cp_mod.parse_args()
    finally:
        sys.argv = saved


def render_cartpole(out: Path) -> None:
    args = _cp_args()
    requested = set(int(s) for s in args.seeds)
    m1_ns = cp_mod._m1_args(args)
    snf_ns = cp_mod._snf_args(args)

    # Which DFL loss variant to render. Defaults reproduce the paper's Eq.(5)
    # penalty curves exactly; set CARTPOLE_LOSS_MODE=auglag (plus the damping
    # env vars, if used) to render the aug-Lagrangian runs instead. The cache
    # keys carry a matching tag, so both sets coexist on disk.
    import os
    _lm = os.environ.get("CARTPOLE_LOSS_MODE", "penalty")
    _dw = float(os.environ.get("CARTPOLE_DUAL_WEIGHT", "1.0"))
    _dfo = os.environ.get("CARTPOLE_DUAL_FEASIBLE_ONLY", "0") not in ("0", "", "false")
    for _ns in (m1_ns, snf_ns):
        _ns.loss_mode, _ns.dual_weight, _ns.dual_feasible_only = _lm, _dw, _dfo
    if _lm != "penalty":
        print(f"[cartpole] rendering DFL arms with loss_mode={_lm} "
              f"dual_weight={_dw} dual_feasible_only={_dfo}")

    dr_nf = cp_mod.pick_best_dr(args, snf_ns)

    # Per-dim (K, ratio) for the SA bars — each dim uses its best-tuned config
    # over the grid K in {4,8} x r in {0.3,0.5,0.7,0.8}. Must stay in sync with
    # the DFL-SA rows of the results table; mdp_continuous/run_dfl_seeds.sh
    # generates exactly these three caches and verifies them.
    SA_DIM_CONFIGS = ((30, 4, 0.8), (60, 8, 0.5), (90, 4, 0.8))
    # Training-curves panel: pin to d=90 with its best (K, r).
    SA_CURVE_D, SA_CURVE_K, SA_CURVE_R = 90, 4, 0.8

    def _sa_payload(d: int, K: int, r: float):
        ns = copy.copy(snf_ns)
        ns.sa_K, ns.sa_ratio = K, r
        return snf._load_or_none(snf._sweep_cache_key("curve_neural_dfl", ns, d))

    neural_p = m1._load_or_none(m1._cache_key("curve_neural", m1_ns))
    qpth_p   = m1._load_or_none(m1._cache_key("curve_qpth", m1_ns))
    mbrl_p   = m1._load_or_none(
        m1._cache_key("curve_two_stage_mbrl", cp_mod._mbrl_args(args)))
    dr_p     = _sa_payload(SA_CURVE_D, SA_CURVE_K, SA_CURVE_R)
    dr_nf    = SA_CURVE_D
    sketch_p = m1._load_or_none(
        m1._cache_key("curve_sketch", m1_ns) + "_K8_r0.2_mgaussian")

    arr = lambda p: cp_mod._curve_from_payload(p, requested)
    neural_a, qpth_a, mbrl_a, dr_a, sketch_a = (
        arr(neural_p), arr(qpth_p), arr(mbrl_p), arr(dr_p), arr(sketch_p))

    def curves(ax):
        dr_label = (f"DFL-SA (d = {dr_nf})"
                    if dr_nf is not None else LABELS["dfl_dr"])
        series = [(k, a, lbl) for k, a, lbl in (
                      ("two_stage_mbrl", mbrl_a, LABELS["two_stage_mbrl"]),
                      ("dfl_qpth", qpth_a, LABELS["dfl_qpth"]),
                      ("neural_dfl", neural_a, LABELS["neural_dfl"]),
                      ("cp_sketch", sketch_a, LABELS["cp_sketch"]),
                      ("dfl_dr", dr_a, dr_label))
                  if a is not None]
        L = min(a.shape[1] for _, a, _ in series) if series else 0
        for k, a, lbl in series:
            d = a[:, :L]
            if args.show_running_max:
                d = np.maximum.accumulate(d, axis=1)
            _curve(ax, d, k, lbl)
        ax.set(xlabel="Training iteration",
               ylabel=("Avg. episode reward"
                       if args.show_running_max else "Avg. episode reward"),
               title="CartPole — Training curves")
        ax.grid(True, alpha=0.3); ax.set_ylim(100, 205)
        ax.legend(loc="lower right")

    def quality(ax):
        metric = "best" if args.show_running_max else "final"

        def _final(arr):
            v = (np.maximum.accumulate(arr, axis=1)[:, -1]
                 if metric == "best" else arr[:, -1])
            return 200.0 - v.astype(float)  # reward → regret

        # All four methods as bars; DFL-SA expands across d=30/60/90.
        keys, labels, means, ses = [], [], [], []
        def _add(key, label, vals):
            if vals is None or vals.size == 0: return
            mu, se = _mse(vals)
            keys.append(key); labels.append(label)
            means.append(mu); ses.append(se)
        _add("two_stage_mbrl", "TS",
             _final(mbrl_a) if mbrl_a is not None else np.array([]))
        _add("dfl_qpth", "QP",
             _final(qpth_a) if qpth_a is not None else np.array([]))
        _add("neural_dfl", "LP",
             _final(neural_a) if neural_a is not None else np.array([]))
        _add("cp_sketch", "Sketch",
             _final(sketch_a) if sketch_a is not None else np.array([]))
        for d, K, r in SA_DIM_CONFIGS:
            payload = _sa_payload(d, K, r)
            runs = cp_mod._filtered_runs(payload, requested) if payload else None
            if not runs:
                continue
            vals = (np.array([max(rr) for rr in runs], dtype=float)
                    if metric == "best"
                    else np.array([rr[-1] for rr in runs], dtype=float))
            _add("dfl_dr", f"SA\nd={d}", 200.0 - vals)
        _bars(ax, keys, means, ses, labels=labels)
        ax.set(ylabel=f"Test regret",
               title="CartPole — Decision quality")
        ax.set_ylim(0, 65)
        ax.tick_params(axis="x", labelsize=RC["xtick.labelsize"] - 2)
        ax.grid(True, axis="y", alpha=0.3, ls=":")
        ax.set_axisbelow(True)

    def cost(ax):
        # Full wall-clock per seed for every method (apples-to-apples).
        # SA expands across d=30/60/90 (matching the decision-quality panel).
        fixed = (("two_stage_mbrl", mbrl_p, "TS"),
                 ("dfl_qpth", qpth_p, "QP"),
                 ("neural_dfl", neural_p, "LP"),
                 ("cp_sketch", sketch_p, "Sketch"))
        keys, labels, means, ses = [], [], [], []
        def _add(key, payload, label):
            ts = cp_mod._times_from_payload(payload, requested,
                                            key="times_per_seed")
            if ts is None or ts.size == 0:
                return
            keys.append(key); labels.append(label)
            means.append(float(ts.mean()))
            ses.append(float(ts.std() / np.sqrt(max(ts.size, 1))))
        for k, p, lbl in fixed:
            _add(k, p, lbl)
        for d, K, r in SA_DIM_CONFIGS:
            _add("dfl_dr", _sa_payload(d, K, r), f"SA\nd={d}")
        _bars(ax, keys, means, ses, labels=labels, log=False)
        ax.set(ylabel="Total training time (s)",
               title="CartPole — Compute cost")
        ax.grid(True, axis="y", alpha=0.3)

    _save(curves,  out / "cartpole_training_curves.pdf")
    _save(quality, out / "cartpole_decision_quality.pdf")
    _save(cost,    out / "cartpole_compute_cost.pdf")


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("which", choices=["inventory", "cartpole", "all"],
                   nargs="?", default="all")
    p.add_argument("--out_dir", default=str(ROOT / "figures"))
    p.add_argument("--inv_results_root",
                   default=str(ROOT / "mdp_lp/results/normal_margin_pretrain_30"))
    p.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="Inventory MDP only: restrict to these seeds.")
    args = p.parse_args()
    out = Path(args.out_dir)
    if args.which in ("inventory", "all"):
        render_inventory(out, Path(args.inv_results_root), args.seeds)
    if args.which in ("cartpole", "all"):
        render_cartpole(out)


if __name__ == "__main__":
    main()
