"""
Three-panel summary figure for offline-CartPole DFL.

Panel 1 — Training curves (per-iter avg reward, mean +/- SE across seeds)
    DFL (lp), DFL-QP (qpth), DFL-DR (lp, best small n_features), Two-stage.
    All four use eval_policy=lp so they share an iter-0 anchor.

Panel 2 — Decision quality vs feature dimension
    Best-so-far reward across n_features for DFL (lp) and Two-stage.

Panel 3 — Training cost (mean wall-clock seconds per run)
    Bar chart: DFL, DFL-QP, DFL-DR, Two-stage.

Reuses the experiment runners from make_figure1_cartpole and sweep_n_features
so the figure shares its caches with those scripts.

Usage (from project root):
    python -m mdp_continuous.make_figure_3panel --seeds 1 2 3 4 5 6 7 8
    python -m mdp_continuous.make_figure_3panel --plot_only
    python -m mdp_continuous.make_figure_3panel --quick
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt

from . import make_figure1_cartpole as m1
from . import sweep_n_features as snf


HERE = Path(__file__).resolve().parent

LABELS = {
    "neural_dfl":      "DFL",
    "dfl_qpth":        "DFL-QP",
    "dfl_dr":          "DFL-DR",
    "two_stage_mbrl":  "Two-stage MBRL",
}
COLORS = {
    "neural_dfl":      "#1f77b4",   # blue
    "dfl_qpth":        "#2ca02c",   # green
    "dfl_dr":          "#9467bd",   # purple
    "two_stage_mbrl":  "#ff7f0e",   # orange
}
LINESTYLES = {
    "neural_dfl":      "-",
    "dfl_qpth":        "-",
    "dfl_dr":          "-",
    "two_stage_mbrl":  "--",
}


# ── Argument bridges ────────────────────────────────────────────────────

_M1_FIELDS = (
    "seeds n_workers behavior eps td_weight n_trajectories "
    "pretrain_epochs_ts pretrain_epochs_dfl n_iterations n_eval_episodes "
    "n_ce_steps n_dfl_steps dfl_lr n_features bins_per_dim gamma lr qp_eta "
    "ridge rho tau eval_policy use_world_model world_epochs world_lr "
    "include_qpth force"
).split()


def _m1_args(args):
    """Build a Namespace usable by make_figure1_cartpole.run_all_curves."""
    ns = SimpleNamespace()
    for f in _M1_FIELDS:
        setattr(ns, f, getattr(args, f))
    ns.include_qpth = True  # we always need DFL-QP for the 3-panel figure
    return ns


def _mbrl_args(args):
    """m1_ns with optional MBRL-only overrides (e.g., longer n_iterations)
    so the MBRL cache key is distinct from the DFL/QP iters=20 caches."""
    ns = _m1_args(args)
    if args.mbrl_n_iterations is not None:
        ns.n_iterations = args.mbrl_n_iterations
    return ns


_SNF_FIELDS = _M1_FIELDS + ["n_features_list"]


def _snf_args(args):
    ns = SimpleNamespace()
    for f in _SNF_FIELDS:
        if f == "n_features_list":
            setattr(ns, f, args.n_features_list)
        elif f == "include_qpth":
            # Sweep DFL + Two-stage only by default (qpth-sweep is opt-in
            # via --include_qpth_sweep).
            setattr(ns, f, args.include_qpth_sweep)
        else:
            setattr(ns, f, getattr(args, f))
    return ns


# ── Pick best dimension-reduced n_features ───────────────────────────────

def _filtered_runs(payload, requested):
    if not payload:
        return None
    runs = [r for s, r in zip(payload["seeds"], payload["rewards_per_seed"])
            if int(s) in requested]
    return runs or None


def pick_best_dr(args, sweep_args):
    """Best n_features <= dr_max_features for DFL (highest mean best-so-far).

    Falls back to None if nothing is cached in that range.
    """
    requested = set(int(s) for s in args.seeds)
    candidates = []
    for nf in sweep_args.n_features_list:
        if nf > args.dr_max_features:
            continue
        payload = snf._load_or_none(
            snf._sweep_cache_key("curve_neural_dfl", sweep_args, nf))
        runs = _filtered_runs(payload, requested)
        if not runs:
            continue
        bests = np.array([max(r) for r in runs], dtype=float)
        candidates.append((float(bests.mean()), nf))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


# ── Plot helpers ────────────────────────────────────────────────────────

def _stack(runs):
    L = min(len(r) for r in runs)
    return np.array([r[:L] for r in runs], dtype=float)


def _draw_curve(ax, arr, color, ls, label):
    mu = arr.mean(axis=0)
    se = arr.std(axis=0) / np.sqrt(max(arr.shape[0], 1))
    xs = np.arange(arr.shape[1])
    ax.plot(xs, mu, color=color, lw=2.0, ls=ls, label=label)
    ax.fill_between(xs, mu - se, mu + se, color=color, alpha=0.20)


def _curve_from_payload(payload, requested):
    runs = _filtered_runs(payload, requested)
    if not runs:
        return None
    return _stack(runs)


def _times_from_payload(payload, requested, key="solver_times_per_seed"):
    """Pull per-seed timings from a cache payload.

    Default key is the *solver-only* time (forward solver + backward in
    the DFL phase) — what differs across methods. Pass key="times_per_seed"
    to get full wall-clock instead.
    """
    if not payload or not payload.get(key):
        return None
    times = [t for s, t in zip(payload["seeds"], payload[key])
             if int(s) in requested and t is not None]
    return np.array(times, dtype=float) if times else None


def _decision_quality_curve(args, sweep_args, algo, metric):
    requested = set(int(s) for s in args.seeds)
    xs, mu, se = [], [], []
    for nf in sweep_args.n_features_list:
        payload = snf._load_or_none(
            snf._sweep_cache_key(f"curve_{algo}", sweep_args, nf))
        runs = _filtered_runs(payload, requested)
        if not runs:
            continue
        if metric == "best":
            vals = np.array([max(r) for r in runs], dtype=float)
        else:
            vals = np.array([r[-1] for r in runs], dtype=float)
        xs.append(nf)
        mu.append(vals.mean())
        se.append(vals.std() / np.sqrt(max(len(vals), 1)))
    return np.array(xs), np.array(mu), np.array(se)


# ── Plot ────────────────────────────────────────────────────────────────

def plot_three_panel(args, m1_ns, snf_ns, dr_nf, save_path):
    requested = set(int(s) for s in args.seeds)

    # Pull cached curves at default n_features.
    neural_payload = m1._load_or_none(m1._cache_key("curve_neural", m1_ns))
    qpth_payload = m1._load_or_none(m1._cache_key("curve_qpth", m1_ns))
    mbrl_payload = m1._load_or_none(
        m1._cache_key("curve_two_stage_mbrl", _mbrl_args(args)))

    neural_arr = _curve_from_payload(neural_payload, requested)
    qpth_arr = _curve_from_payload(qpth_payload, requested)
    mbrl_arr = _curve_from_payload(mbrl_payload, requested)

    # DFL-DR: training curve at chosen reduced n_features.
    dr_arr = None
    dr_payload = None
    if dr_nf is not None:
        dr_payload = snf._load_or_none(
            snf._sweep_cache_key("curve_neural_dfl", snf_ns, dr_nf))
        dr_arr = _curve_from_payload(dr_payload, requested)

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.4))

    # ── Panel 1: training curves ─────────────────────────────────────
    # DFL/DFL-QP/Two-stage share the same architecture (n_features=128)
    # and so naturally align at iter 0. DFL-DR uses a smaller architecture
    # (n_features=dr_nf), which gives it an intrinsically lower iter-0
    # LP-derived policy — this gap is preserved in the plot.
    ax = axes[0]
    panel1 = [
        ("neural_dfl",     LABELS["neural_dfl"],     neural_arr),
        ("dfl_qpth",       LABELS["dfl_qpth"],       qpth_arr),
        ("dfl_dr",
         f"{LABELS['dfl_dr']} (n_feat={dr_nf})" if dr_nf is not None
         else LABELS["dfl_dr"],
         dr_arr),
        ("two_stage_mbrl", LABELS["two_stage_mbrl"], mbrl_arr),
    ]
    for key, label, arr in panel1:
        if arr is None:
            continue
        a = np.maximum.accumulate(arr, axis=1) if args.show_running_max else arr
        _draw_curve(ax, a, COLORS[key], LINESTYLES[key], label)
    ax.set_xlabel("Training iteration")
    ylab = ("Avg. episode reward (best-so-far)"
            if args.show_running_max else "Avg. episode reward")
    ax.set_ylabel(ylab)
    ax.set_title("Training curves")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 210)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    # ── Panel 2: decision quality vs n_features ──────────────────────
    # MBRL doesn't use the LP feature parameterisation, so it has no
    # n_features axis — plot it as a horizontal reference instead.
    ax = axes[1]
    metric = "best" if args.show_running_max else "final"
    panel2_algos = ["neural_dfl"]
    if args.include_qpth_sweep:
        panel2_algos.append("dfl_qpth")
    for algo in panel2_algos:
        xs, mu, se = _decision_quality_curve(args, snf_ns, algo, metric)
        if len(xs) == 0:
            continue
        ax.plot(xs, mu, color=COLORS[algo], lw=2.0,
                ls=LINESTYLES[algo], marker="o", label=LABELS[algo])
        ax.fill_between(xs, mu - se, mu + se, color=COLORS[algo], alpha=0.20)
    if mbrl_arr is not None:
        if metric == "best":
            mbrl_vals = np.maximum.accumulate(mbrl_arr, axis=1)[:, -1]
        else:
            mbrl_vals = mbrl_arr[:, -1]
        mu_m = float(mbrl_vals.mean())
        se_m = float(mbrl_vals.std() / np.sqrt(max(mbrl_vals.size, 1)))
        ax.axhline(mu_m, color=COLORS["two_stage_mbrl"],
                   ls=LINESTYLES["two_stage_mbrl"], lw=2.0,
                   label=f"{LABELS['two_stage_mbrl']} (no n_feat dep.)")
        ax.fill_between(snf_ns.n_features_list,
                        [mu_m - se_m] * len(snf_ns.n_features_list),
                        [mu_m + se_m] * len(snf_ns.n_features_list),
                        color=COLORS["two_stage_mbrl"], alpha=0.15)
    if dr_nf is not None:
        ax.axvline(dr_nf, color=COLORS["dfl_dr"], ls=":", lw=1.2,
                   alpha=0.8, label=f"DR pick (n_feat={dr_nf})")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("n_features")
    ylab = "best-so-far reward" if metric == "best" else "final-iter reward"
    ax.set_ylabel(f"Avg. episode reward ({ylab})")
    ax.set_title("Decision quality vs feature dimension")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(0, 210)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    # ── Panel 3: training time ──────────────────────────────────────
    # Use solver-only time = forward solver + backward inside the DFL
    # phase. This is the part that differs across DFL/DFL-QP/DFL-DR/TS;
    # pretrain, evaluation, and the iter-end LP solve are identical
    # across methods and so excluded.
    ax = axes[2]
    bar_keys, bar_means, bar_ses = [], [], []
    time_sources = [
        ("neural_dfl",     neural_payload),
        ("dfl_qpth",       qpth_payload),
        ("dfl_dr",         dr_payload),
        ("two_stage_mbrl", mbrl_payload),
    ]
    for key, payload in time_sources:
        # MBRL has no LP solver; report its full wall-clock instead. For
        # DFL/DFL-QP/DFL-DR we keep the solver-only number (forward solve
        # + backward through the LP/QP), which is the part that differs
        # across those methods.
        time_key = ("times_per_seed" if key == "two_stage_mbrl"
                    else "solver_times_per_seed")
        ts = _times_from_payload(payload, requested, key=time_key)
        if ts is None or ts.size == 0:
            ts = _times_from_payload(payload, requested,
                                     key="times_per_seed")
            if ts is None or ts.size == 0:
                continue
        bar_keys.append(key)
        bar_means.append(float(ts.mean()))
        bar_ses.append(float(ts.std() / np.sqrt(max(ts.size, 1))))
    source_label = ("forward solver + backward (DFL phase only); "
                    "MBRL: full wall-clock")
    if bar_keys:
        xs = np.arange(len(bar_keys))
        labels = [
            f"{LABELS[k]} (n_feat={dr_nf})" if k == "dfl_dr" and dr_nf is not None
            else LABELS[k]
            for k in bar_keys
        ]
        ax.bar(xs, bar_means, yerr=bar_ses, capsize=4,
               color=[COLORS[k] for k in bar_keys], alpha=0.85,
               edgecolor="black", linewidth=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=15, ha="right")
    else:
        ax.text(0.5, 0.5,
                "No timing data in cache.\nRe-run with --force.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11)
    ax.set_ylabel("Wall-clock training time (s)")
    title = "Training cost"
    if source_label:
        title += f"\n({source_label})"
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Offline CartPole — DFL vs DFL-QP vs DFL-DR vs Two-stage MBRL",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved figure to {save_path} and {pdf_path}")


# ── CLI ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Three-panel figure for CartPole DFL")
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[1, 2, 3, 4, 5, 6, 7, 8])
    p.add_argument("--n_workers", type=int, default=8)
    p.add_argument("--n_features_list", type=int, nargs="+",
                   default=[8, 16, 32, 64, 128, 256, 1024])
    p.add_argument("--dr_max_features", type=int, default=64,
                   help="DFL-DR is the best DFL run with n_features <= this.")

    # offline data + behavior policy
    p.add_argument("--behavior", type=str, default="biased",
                   choices=["expert", "noisy", "random",
                            "angle_only", "biased"])
    p.add_argument("--eps", type=float, default=0.5)
    p.add_argument("--td_weight", type=float, default=0.013)
    p.add_argument("--n_trajectories", type=int, default=100)

    # Symmetric pretrain so DFL/DFL-QP/TS see identical pretrained features
    # and therefore identical iter-0 LP-derived policies. With same seed
    # and same architecture, iter-0 of all three is byte-identical.
    p.add_argument("--pretrain_epochs_ts", type=int, default=30)
    p.add_argument("--pretrain_epochs_dfl", type=int, default=30)
    p.add_argument("--n_iterations", type=int, default=20)
    p.add_argument("--mbrl_n_iterations", type=int, default=None,
                   help="Override n_iterations for MBRL only. Lets MBRL run "
                        "longer than DFL/QP/TS without invalidating their "
                        "iters=20 caches. None → use --n_iterations.")
    p.add_argument("--n_eval_episodes", type=int, default=100)
    p.add_argument("--n_ce_steps", type=int, default=0)
    p.add_argument("--n_dfl_steps", type=int, default=5)
    p.add_argument("--dfl_lr", type=float, default=None,
                   help="None = use cfg.lr=1e-3.")

    p.add_argument("--n_features", type=int, default=1024,
                   help="Feature dim for DFL/DFL-QP/Two-stage curves in "
                        "panels 1 and 3 — 'large/no-aggregation' anchor. "
                        "LP size (m,d) = (n_features/A, n_features) = "
                        "(512, 1024) at default. Plain DFL is noisy at this "
                        "dim (basis-restricted backward is sensitive); the "
                        "best-so-far view still puts it above Two-stage.")
    p.add_argument("--bins_per_dim", type=int, default=8)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--qp_eta", type=float, default=0.01)
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--rho", type=float, default=10.0)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--eval_policy", type=str, default="lp",
                   choices=["ce", "lp"])

    # World-model fields preserved so _job() works unchanged.
    p.add_argument("--use_world_model", action="store_true")
    p.add_argument("--world_epochs", type=int, default=400)
    p.add_argument("--world_lr", type=float, default=1e-3)

    # Flags
    p.add_argument("--include_qpth", action="store_true", default=True,
                   help="Always on for 3-panel — DFL-QP needs caching.")
    p.add_argument("--include_qpth_sweep", action="store_true",
                   help="Also sweep DFL-QP across n_features for Panel 2.")
    p.add_argument("--show_running_max", dest="show_running_max",
                   action="store_true", default=True,
                   help="Plot best-so-far per seed (matches train_cartpole's "
                        "end-of-run best-state restoration). On by default.")
    p.add_argument("--per_iter", dest="show_running_max", action="store_false",
                   help="Plot raw per-iter rewards instead of best-so-far.")

    p.add_argument("--save_path", type=str,
                   default=str(HERE / "figure_3panel.png"))
    p.add_argument("--plot_only", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--quick", action="store_true",
                   help="Smoke test: 2 seeds, small list, fewer iters.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.quick:
        args.seeds = [1, 2]
        args.n_features_list = [16, 64, 256]
        args.n_features = 256  # "large" anchor for the smoke test
        args.dr_max_features = 32
        args.n_iterations = 6
        args.pretrain_epochs_ts = 30
        args.pretrain_epochs_dfl = 15
        args.n_workers = min(args.n_workers, 4)

    m1_ns = _m1_args(args)
    snf_ns = _snf_args(args)

    if not args.plot_only:
        # Panel-1 / Panel-3 caches — DFL, DFL-QP, (and the legacy Two-stage
        # entry, which is unused in the new figure but kept cached for
        # parity with other scripts) at default n_features.
        m1.run_all_curves(m1_ns)
        # Two-stage MBRL — may use a longer n_iterations via mbrl_ns.
        m1.run_mbrl_curve(_mbrl_args(args))
        # Panel-2 / DFL-DR caches — sweep over n_features.
        snf.run_sweep(snf_ns)

    dr_nf = pick_best_dr(args, snf_ns)
    if dr_nf is None:
        print("[warn] no DFL sweep results within dr_max_features="
              f"{args.dr_max_features}; DFL-DR curve will be omitted.")
    else:
        print(f"[info] DFL-DR pick: n_features={dr_nf}")

    plot_three_panel(args, m1_ns, snf_ns, dr_nf, save_path=args.save_path)


if __name__ == "__main__":
    main()
