#!/usr/bin/env python3
"""
Paper Figure 2 reproduction: run experiments for inventory + CartPole,
then generate the 2x3 panel figure (training curves, decision quality,
compute cost for each domain).

Usage:
    python run_paper_repro.py --seeds 1 2 3
    python run_paper_repro.py --plot_only --seeds 1 2 3
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


# ── helpers ────────────────────────────────────────────────────────────────

def run_cmd(cmd: list[str], label: str) -> tuple[str, bool, float]:
    t0 = time.time()
    print(f"[START] {label}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    elapsed = time.time() - t0
    ok = r.returncode == 0
    print(f"[{'OK' if ok else 'FAIL'}] {label}  ({elapsed:.0f}s)", flush=True)
    if not ok:
        print(f"  stdout: {r.stdout[-400:]}")
        print(f"  stderr: {r.stderr[-400:]}")
    return label, ok, elapsed


# ── Inventory experiment jobs ──────────────────────────────────────────────

INV_RESULTS_ROOT = "mdp_lp/results/normal_margin_pretrain_30"
INV_BASE_ARGS = [
    "python", "-m", "mdp_lp.run_elasticity_experiments",
    "--experiment_type", "b",
    "--cost_case", "normal_margin",
    "--bc", "10",
    "--n_samples", "10",
    "--n_epochs", "50",
    "--pretrain_epochs", "30",
    "--b_max", "5", "--s_max", "30", "--q_max", "30",
    "--results_root", INV_RESULTS_ROOT,
]
DIM_TAG = "s30q30"


def inventory_jobs(seeds: list[int]) -> list[tuple[list[str], str]]:
    jobs = []
    for seed in seeds:
        tag = f"seed_{seed}_{DIM_TAG}"
        # Two-Stage
        jobs.append((INV_BASE_ARGS + [
            "--algo", "twostage", "--seed", str(seed),
            "--tag", tag, "--aug_lag_rho", "10000",
        ], f"inv/twostage/s{seed}"))
        # DFL-LP
        jobs.append((INV_BASE_ARGS + [
            "--algo", "dfl", "--seed", str(seed),
            "--tag", tag, "--aug_lag_rho", "10000",
        ], f"inv/dfl/s{seed}"))
        # DFL-QP
        jobs.append((INV_BASE_ARGS + [
            "--algo", "dfl_qpth", "--seed", str(seed),
            "--tag", tag, "--qp_reg", "0.001", "--aug_lag_rho", "10000",
        ], f"inv/dfl_qpth/s{seed}"))
        # DFL-Sketch (K=8, r=0.1, n_agg=-1) — canonical Sketch
        jobs.append((INV_BASE_ARGS + [
            "--algo", "dfl_sketch", "--n_agg", "-1",
            "--n_sketches", "8", "--sketch_ratio", "0.1",
            "--sketch_mode", "gaussian",
            "--aug_lag_rho", "10.0", "--loss_weight", "1",
            "--seed", str(seed),
            "--tag", tag + "_K8_r0.1_gaussian",
        ], f"inv/sketch_r01/s{seed}"))
        # DFL-SA M=10 (K=8, r=0.2, n_agg=10)
        for n_agg in (10, 20, 30):
            jobs.append((INV_BASE_ARGS + [
                "--algo", "dfl_sketch", "--n_agg", str(n_agg),
                "--n_sketches", "8", "--sketch_ratio", "0.2",
                "--sketch_mode", "gaussian",
                "--aug_lag_rho", "0.1", "--loss_weight", "1",
                "--seed", str(seed),
                "--tag", tag + "_K8_r0.2_gaussian",
            ], f"inv/sa_m{n_agg}/s{seed}"))
    return jobs


# ── CartPole experiment command ────────────────────────────────────────────

def cartpole_cmd(seeds: list[int]) -> list[str]:
    # n_features=128 and pretrain_ts=30, pretrain_dfl=30 match the
    # make_figure_3panel defaults used by make_figures.py for the cache key.
    return [
        "python", "-m", "mdp_continuous.make_figure1_cartpole",
        "--seeds", *[str(s) for s in seeds],
        "--n_iterations", "60",
        "--n_features", "128",
        "--pretrain_epochs_ts", "30",
        "--pretrain_epochs_dfl", "30",
        "--n_workers", str(min(len(seeds) * 3, 8)),
        "--include_qpth",
        "--save_path", "figures/repro/cartpole_learning_curves.pdf",
    ]


# ── Cliff experiment jobs ──────────────────────────────────────────────────

def cliff_jobs(seeds: list[int]) -> list[tuple[list[str], str]]:
    jobs = []
    base = [
        "python", "-m", "mdp_cw.run_cliff_experiments",
        "--n_samples", "120",
        "--n_epochs", "50",
        "--pretrain_epochs", "60",
        "--regime_set", "v4",
        "--cliff_penalty", "-10",
        "--noise_std", "2.5",
        "--center_scale", "0.88",
    ]
    for seed in seeds:
        for algo, extra in [
            ("twostage", []),
            ("dfl",      []),
            ("dfl_qpth", ["--qp_reg", "0.01"]),
            ("dfl_sketch", ["--n_agg", "12"]),
        ]:
            save_dir = f"mdp_cw/results_repro/n120/k7/{algo}_s{seed}"
            jobs.append((base + [
                "--algo", algo, "--seed", str(seed),
                "--save_dir", save_dir,
            ] + extra, f"cliff/{algo}/s{seed}"))
    return jobs


# ── Figure 2 plotting ──────────────────────────────────────────────────────

RC = {
    "font.family": "serif", "font.size": 20, "axes.labelsize": 18,
    "axes.titlesize": 20, "legend.fontsize": 13, "xtick.labelsize": 16,
    "ytick.labelsize": 18, "pdf.fonttype": 42, "ps.fonttype": 42,
}

COLORS = {
    "twostage":       "#ff7f0e",
    "dfl":            "#1f77b4",
    "dfl_qpth":       "#2ca02c",
    "dfl_sketch_r01": "#d62728",
    "dfl_sa":         "#9467bd",
    "neural_dfl":     "#1f77b4",
    "two_stage_mbrl": "#ff7f0e",
    "cp_qpth":        "#2ca02c",
}
LS = {
    "twostage": "--", "dfl": "-", "dfl_qpth": "-",
    "dfl_sketch_r01": ":", "dfl_sa": "-",
    "neural_dfl": "-", "two_stage_mbrl": "--", "cp_qpth": "-",
}


def _mse(xs):
    if xs.size == 0: return float("nan"), 0.0
    if xs.size == 1: return float(xs[0]), 0.0
    return float(xs.mean()), float(xs.std(ddof=1) / np.sqrt(xs.size))


def _curve(ax, arr, color, ls, label):
    mu = arr.mean(axis=0)
    se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros_like(mu)
    xs = np.arange(arr.shape[1])
    ax.plot(xs, mu, color=color, ls=ls, lw=2, label=label)
    ax.fill_between(xs, mu - se, mu + se, color=color, alpha=0.2)


def _bars(ax, keys, labels, means, ses, colors):
    x = np.arange(len(keys))
    ax.bar(x, means, yerr=ses, capsize=4, width=0.65,
           color=colors, alpha=0.88, edgecolor="black", linewidth=0.6,
           error_kw=dict(ecolor="black", lw=0.8))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)


# ── Inventory data loading ─────────────────────────────────────────────────

def _load_inv_runs(seeds: list[int]):
    root = ROOT / INV_RESULTS_ROOT / "exp_b" / "normal_margin"
    requested = set(seeds)
    grouped = defaultdict(list)

    def _best_per_seed(algo_dir):
        best = {}
        for p in algo_dir.rglob("results.json"):
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            if d.get("status") != "complete":
                continue
            seed = int(d.get("seed", -1))
            if seed not in requested:
                continue
            prev = best.get(seed)
            if prev is None or d.get("n_epochs", 0) >= prev.get("n_epochs", 0):
                best[seed] = d
        return [best[s] for s in sorted(best)]

    for method in ("twostage", "dfl", "dfl_qpth"):
        d = root / method
        if d.is_dir():
            grouped[method] = _best_per_seed(d)

    # Sketch: K=8 r=0.1 n_agg=-1
    for sketch_dir in root.glob("dfl_sketch_ngg_*"):
        n_agg_str = sketch_dir.name.split("_")[-1]
        try:
            n_agg = int(n_agg_str)
        except ValueError:
            continue
        for p in sketch_dir.rglob("results.json"):
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            if d.get("status") != "complete":
                continue
            seed = int(d.get("seed", -1))
            if seed not in requested:
                continue
            K = int(d.get("n_sketches", -1))
            r_val = float(d.get("sketch_ratio", -1))
            key = f"sketch_K{K}_r{r_val:.1f}_nagg{n_agg}"
            prev = grouped.get(key + f"_s{seed}")
            if not prev or d.get("n_epochs", 0) >= prev[0].get("n_epochs", 0):
                # Store by (key, seed)
                grouped[key].append(d)

    return grouped


def _inv_train_reward(runs):
    """Return (n_seeds, n_epochs+1) array including initial_reward at epoch 0."""
    arrs = []
    for d in runs:
        r = d.get("train_reward_per_epoch", [])
        if r:
            arrs.append(np.array(r, dtype=float))
    if not arrs:
        return None
    L = min(len(a) for a in arrs)
    return np.array([a[:L] for a in arrs])


def _inv_test_regret_pct(runs, oracle_key="train_optimal_reward"):
    vals = []
    for d in runs:
        oracle = d.get("train_optimal_reward") or d.get("test_mean_oracle_reward")
        pred = d.get("test_mean_pred_reward")
        if oracle and pred:
            regret = oracle - pred
            # express as % of oracle magnitude
            vals.append(100.0 * regret / abs(oracle))
    return np.array(vals)


# ── CartPole data loading ──────────────────────────────────────────────────

CACHE_DIR = ROOT / "mdp_continuous" / "results" / "dfl_vs_qpth"


def _cp_cache_key(prefix, n_iterations=60, n_features=128,
                  pretrain_ts=30, pretrain_dfl=30, qp_eta=0.01):
    qpth_tag = f"_qpeta{qp_eta}" if "qpth" in prefix else ""
    return (
        f"{prefix}"
        f"_beh{'biased'}_td{0.013}_eps{0.5}"
        f"_ntraj{100}"
        f"_iters{n_iterations}"
        f"_nfeat{n_features}"
        f"_preTS{pretrain_ts}_preDFL{pretrain_dfl}"
        f"_nce{0}_ndfl{5}"
        f"_dfllrauto"
        f"_eval{'lp'}"
        f"{qpth_tag}"
    )


def _load_cp(prefix, seeds, n_iterations=60, n_features=128,
             pretrain_ts=30, pretrain_dfl=30):
    key = _cp_cache_key(prefix, n_iterations, n_features, pretrain_ts, pretrain_dfl)
    p = CACHE_DIR / f"{key}.json"
    if not p.exists():
        print(f"  [warn] cache missing: {p.name}")
        return None
    payload = json.loads(p.read_text())
    requested = set(seeds)
    runs = [(s, rw) for s, rw in zip(payload["seeds"], payload["rewards_per_seed"])
            if int(s) in requested]
    if not runs:
        return None
    return np.array([rw for _, rw in runs], dtype=float)


def _load_cp_times(prefix, seeds, n_iterations=60, n_features=128,
                   pretrain_ts=30, pretrain_dfl=30):
    key = _cp_cache_key(prefix, n_iterations, n_features, pretrain_ts, pretrain_dfl)
    p = CACHE_DIR / f"{key}.json"
    if not p.exists():
        return None
    payload = json.loads(p.read_text())
    requested = set(seeds)
    times = [t for s, t in zip(payload["seeds"],
                                payload.get("solver_times_per_seed", [None]*len(payload["seeds"])))
             if int(s) in requested and t is not None]
    if not times:
        # fall back to total time
        times = [t for s, t in zip(payload["seeds"],
                                     payload.get("times_per_seed", [None]*len(payload["seeds"])))
                  if int(s) in requested and t is not None]
    return np.array(times) if times else None


# ── Figure 2 ───────────────────────────────────────────────────────────────

def plot_figure2(seeds: list[int], out_dir: Path):
    prev = copy.deepcopy(plt.rcParams)
    plt.rcParams.update(RC)

    fig, axes = plt.subplots(2, 3, figsize=(21, 10))
    plt.subplots_adjust(wspace=0.35, hspace=0.45)

    inv_runs = _load_inv_runs(seeds)

    # ── Row 0: Inventory ────────────────────────────────────────────────

    ax = axes[0, 0]
    # Training curves
    curve_specs = [
        ("twostage",       "Two-stage", "twostage",       "--"),
        ("dfl_qpth",       "DFL-QP",    "dfl_qpth",       "-"),
        ("dfl",            "DFL-LP",    "dfl",            "-"),
        ("sketch_K8_r0.1_nagg-1", "DFL-Sketch", "dfl_sketch_r01", ":"),
        ("sketch_K8_r0.2_nagg10", "DFL-SA (M=10)", "dfl_sa", "-"),
    ]
    any_inv = False
    for key, label, ck, ls in curve_specs:
        runs = inv_runs.get(key, [])
        arr = _inv_train_reward(runs)
        if arr is not None:
            _curve(ax, arr, COLORS[ck], ls, label)
            any_inv = True
    ax.set(xlabel="Training epoch", ylabel="Training reward",
           title="Inventory MDP — Training curves")
    ax.set_ylim(bottom=50)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    if not any_inv:
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center", transform=ax.transAxes, fontsize=16)

    ax = axes[0, 1]
    # Decision quality bars
    bar_specs = [
        ("twostage",             "TS",     "twostage"),
        ("dfl_qpth",             "QP",     "dfl_qpth"),
        ("dfl",                  "LP",     "dfl"),
        ("sketch_K8_r0.1_nagg-1","Sketch", "dfl_sketch_r01"),
        ("sketch_K8_r0.2_nagg10","SA\nM=10","dfl_sa"),
        ("sketch_K8_r0.2_nagg20","SA\nM=20","dfl_sa"),
        ("sketch_K8_r0.2_nagg30","SA\nM=30","dfl_sa"),
    ]
    bkeys, blabels, bmeans, bses, bcolors = [], [], [], [], []
    for key, label, ck in bar_specs:
        vals = _inv_test_regret_pct(inv_runs.get(key, []))
        if vals.size > 0:
            mu, se = _mse(vals)
            bkeys.append(key); blabels.append(label)
            bmeans.append(mu); bses.append(se); bcolors.append(COLORS[ck])
    _bars(ax, bkeys, blabels, bmeans, bses, bcolors)
    ax.set(ylabel="Test regret (%)", title="Inventory MDP — Decision quality")
    ax.set_ylim(0, 25)
    ax.grid(True, axis="y", alpha=0.3, ls=":")
    ax.set_axisbelow(True)
    if not bkeys:
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center", transform=ax.transAxes, fontsize=16)

    ax = axes[0, 2]
    # Compute cost bars (total training time)
    tkeys, tlabels, tmeans, tses, tcolors = [], [], [], [], []
    for key, label, ck in bar_specs:
        runs = inv_runs.get(key, [])
        totals = [float(np.sum(d.get("train_time") or [])) for d in runs if d.get("train_time")]
        if totals:
            mu, se = _mse(np.array(totals))
            tkeys.append(key); tlabels.append(label)
            tmeans.append(mu); tses.append(se); tcolors.append(COLORS[ck])
    _bars(ax, tkeys, tlabels, tmeans, tses, tcolors)
    ax.set(ylabel="Total training time (s)", title="Inventory MDP — Compute cost")
    ax.grid(True, axis="y", alpha=0.3)
    if not tkeys:
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center", transform=ax.transAxes, fontsize=16)

    # ── Row 1: CartPole ─────────────────────────────────────────────────

    cp_neural = _load_cp("curve_neural", seeds)
    cp_qpth   = _load_cp("curve_qpth", seeds)
    cp_mbrl   = _load_cp("curve_two_stage_mbrl", seeds)

    ax = axes[1, 0]
    cp_specs = [
        (cp_mbrl,   "Two-stage",  "two_stage_mbrl", "--"),
        (cp_qpth,   "DFL-QP",     "cp_qpth",        "-"),
        (cp_neural, "DFL-LP",     "neural_dfl",     "-"),
    ]
    any_cp = False
    for arr, label, ck, ls in cp_specs:
        if arr is not None:
            # show running max (best so far), matching paper
            arr_rm = np.maximum.accumulate(arr, axis=1)
            _curve(ax, arr_rm, COLORS[ck], ls, label)
            any_cp = True
    ax.set(xlabel="Training iteration", ylabel="Avg. episode reward",
           title="CartPole — Training curves")
    ax.set_ylim(100, 205)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    if not any_cp:
        ax.text(0.5, 0.5, "No CartPole data yet\n(run CartPole script first)",
                ha="center", va="center", transform=ax.transAxes, fontsize=14)

    ax = axes[1, 1]
    # CartPole decision quality (best-of-run regret = 200 - best_reward)
    cp_bar_specs = [
        (cp_mbrl,   "TS",     "two_stage_mbrl"),
        (cp_qpth,   "QP",     "cp_qpth"),
        (cp_neural, "LP",     "neural_dfl"),
    ]
    bkeys, blabels, bmeans, bses, bcolors = [], [], [], [], []
    for arr, label, ck in cp_bar_specs:
        if arr is not None:
            best = np.maximum.accumulate(arr, axis=1)[:, -1]
            regret = 200.0 - best
            mu, se = _mse(regret)
            bkeys.append(ck); blabels.append(label)
            bmeans.append(mu); bses.append(se); bcolors.append(COLORS[ck])
    _bars(ax, bkeys, blabels, bmeans, bses, bcolors)
    ax.set(ylabel="Test regret", title="CartPole — Decision quality")
    ax.set_ylim(0, 65)
    ax.grid(True, axis="y", alpha=0.3, ls=":")
    ax.set_axisbelow(True)
    if not bkeys:
        ax.text(0.5, 0.5, "No CartPole data yet",
                ha="center", va="center", transform=ax.transAxes, fontsize=14)

    ax = axes[1, 2]
    # CartPole compute cost (solver time per seed)
    tkeys, tlabels, tmeans, tses, tcolors = [], [], [], [], []
    for prefix, label, ck in [
        ("curve_two_stage_mbrl", "TS",  "two_stage_mbrl"),
        ("curve_qpth",           "QP",  "cp_qpth"),
        ("curve_neural",         "LP",  "neural_dfl"),
    ]:
        ts = _load_cp_times(prefix, seeds)
        if ts is not None and ts.size > 0:
            mu, se = _mse(ts)
            tkeys.append(ck); tlabels.append(label)
            tmeans.append(mu); tses.append(se); tcolors.append(COLORS[ck])
    _bars(ax, tkeys, tlabels, tmeans, tses, tcolors)
    ax.set(ylabel="Total training time (s)", title="CartPole — Compute cost")
    ax.grid(True, axis="y", alpha=0.3)
    if not tkeys:
        ax.text(0.5, 0.5, "No CartPole data yet",
                ha="center", va="center", transform=ax.transAxes, fontsize=14)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "figure2_repro.pdf"
    png_path = out_dir / "figure2_repro.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    plt.rcParams.update(prev)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--skip_inventory", action="store_true")
    p.add_argument("--skip_cartpole", action="store_true")
    p.add_argument("--skip_cliff", action="store_true")
    p.add_argument("--plot_only", action="store_true")
    args = p.parse_args()

    Path("figures/repro").mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    # ── Inventory ──────────────────────────────────────────────────────────
    if not args.skip_inventory and not args.plot_only:
        inv_jobs = inventory_jobs(args.seeds)
        print(f"\n=== Inventory: {len(inv_jobs)} runs, {args.workers} workers ===")
        ok = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_cmd, cmd, lbl): lbl for cmd, lbl in inv_jobs}
            for fut in as_completed(futs):
                _, success, _ = fut.result()
                if success: ok += 1
        print(f"Inventory: {ok}/{len(inv_jobs)} OK")

    # ── CartPole ───────────────────────────────────────────────────────────
    if not args.skip_cartpole and not args.plot_only:
        print(f"\n=== CartPole: {len(args.seeds)} seeds, 60 iters ===")
        _, ok, t = run_cmd(cartpole_cmd(args.seeds), "cartpole")
        print(f"CartPole: {'OK' if ok else 'FAIL'} ({t:.0f}s)")

    # ── Cliff (optional) ───────────────────────────────────────────────────
    if not args.skip_cliff and not args.plot_only:
        c_jobs = cliff_jobs(args.seeds)
        print(f"\n=== Cliff: {len(c_jobs)} runs, {args.workers} workers ===")
        ok = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_cmd, cmd, lbl): lbl for cmd, lbl in c_jobs}
            for fut in as_completed(futs):
                _, success, _ = fut.result()
                if success: ok += 1
        print(f"Cliff: {ok}/{len(c_jobs)} OK")

    # ── Figure 2 ───────────────────────────────────────────────────────────
    print("\n=== Generating Figure 2 ===")
    plot_figure2(args.seeds, ROOT / "figures" / "repro")

    print(f"\nTotal elapsed: {time.time() - t_total:.0f}s")


if __name__ == "__main__":
    main()
