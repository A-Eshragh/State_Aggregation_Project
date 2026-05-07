from __future__ import annotations

import argparse
import copy
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .make_figure1_cartpole import _job, _run_one_seed, COLORS


HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "results" / "dfl_vs_qpth"


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{name}.json"


def _load_or_none(name: str):
    p = _cache_path(name)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def _save(name: str, data):
    with open(_cache_path(name), "w") as f:
        json.dump(data, f)


def _sweep_cache_key(prefix: str, args, n_features: int) -> str:
    dfl_lr_tag = "auto" if args.dfl_lr is None else f"{args.dfl_lr}"
    qpth_tag = f"_qpeta{args.qp_eta}" if "qpth" in prefix else ""
    # Only tag SA sketch params when they differ from the historical defaults
    # (K=8, ratio=0.2) so pre-existing caches under those defaults stay
    # reachable from this key.
    sa_K = int(getattr(args, "sa_K", 8))
    sa_r = float(getattr(args, "sa_ratio", 0.2))
    sa_tag = ""
    if "neural_dfl" in prefix and (sa_K != 8 or abs(sa_r - 0.2) > 1e-9):
        sa_tag = f"_saK{sa_K}_sar{sa_r}"
    return (
        f"{prefix}_nfeat{n_features}"
        f"_beh{args.behavior}_td{args.td_weight}_eps{args.eps}"
        f"_ntraj{args.n_trajectories}"
        f"_iters{args.n_iterations}"
        f"_preTS{args.pretrain_epochs_ts}_preDFL{args.pretrain_epochs_dfl}"
        f"_nce{args.n_ce_steps}_ndfl{args.n_dfl_steps}"
        f"_dfllr{dfl_lr_tag}"
        f"_eval{args.eval_policy}"
        f"{qpth_tag}{sa_tag}"
    )


ALGOS_BASE = ("neural_dfl",)


def _algos_to_run(args):
    return ALGOS_BASE + (("dfl_qpth",) if args.include_qpth else ())


def _build_jobs(args):
    """One (algo, n_features, seed) per job, skipping cache hits."""
    jobs = []
    pending_targets = {}  # (algo, n_features) -> list of (seed, idx in jobs)
    for nf in args.n_features_list:
        for algo in _algos_to_run(args):
            cache_name = _sweep_cache_key(f"curve_{algo}", args, nf)
            if not args.force and _load_or_none(cache_name):
                continue
            args_for_job = copy.copy(args)
            args_for_job.n_features = nf
            for s in args.seeds:
                j = _job(args_for_job, algo, s)
                j["__nfeat"] = nf
                jobs.append(j)
                pending_targets.setdefault((algo, nf), []).append(s)
    return jobs, pending_targets


def run_sweep(args):
    jobs, pending = _build_jobs(args)
    if not jobs:
        print("[sweep] all (algo, n_features, seed) results cached — skip run")
        return

    print(f"[sweep] {len(jobs)} jobs across {args.n_workers} workers, "
          f"n_features={args.n_features_list}, "
          f"algos={list(_algos_to_run(args))}, seeds={args.seeds}")

    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futures = {ex.submit(_run_one_seed, {k: v for k, v in j.items()
                                             if not k.startswith("__")}): j
                   for j in jobs}
        for fut in as_completed(futures):
            j = futures[fut]
            r = fut.result()
            r["__nfeat"] = j["__nfeat"]
            print(f"[done] algo={r['algo']:<10s} nfeat={r['__nfeat']:>4d} "
                  f"seed={r['seed']} best={max(r['rewards']):.2f}")
            results.append(r)
    print(f"[sweep] {len(results)} jobs finished in {time.time() - t0:.1f}s")

    # Group by (algo, n_features) and write a cache file each.
    groups = {}
    for r in results:
        key = (r["algo"], r["__nfeat"])
        groups.setdefault(key, []).append(r)
    for (algo, nf), runs in groups.items():
        runs = sorted(runs, key=lambda r: r["seed"])
        payload = {
            "n_features": nf,
            "seeds": [r["seed"] for r in runs],
            "rewards_per_seed": [r["rewards"] for r in runs],
            "lengths_per_seed": [r["lengths"] for r in runs],
            "times_per_seed": [r.get("total_time") for r in runs],
            "solver_times_per_seed":
                [r.get("solver_total_time", 0.0) for r in runs],
        }
        _save(_sweep_cache_key(f"curve_{algo}", args, nf), payload)


# ── Plotting ────────────────────────────────────────────────────────────

LABELS = {
    "neural_dfl": "DFL",
    "dfl_qpth":   "DFL-qpth",
}
LINESTYLES = {
    "neural_dfl": "-",
    "dfl_qpth":   "-.",
}


def _summary_for_algo(args, algo, metric):
    """Return (xs, mu, se) across n_features for an algo, picking metric per seed."""
    xs, mus, ses = [], [], []
    requested = set(int(s) for s in args.seeds)
    for nf in args.n_features_list:
        payload = _load_or_none(_sweep_cache_key(f"curve_{algo}", args, nf))
        if not payload:
            continue
        runs = [r for s, r in zip(payload["seeds"], payload["rewards_per_seed"])
                if int(s) in requested]
        if not runs:
            continue
        if metric == "best":
            vals = np.array([max(r) for r in runs], dtype=float)
        elif metric == "final":
            vals = np.array([r[-1] for r in runs], dtype=float)
        else:
            raise ValueError(metric)
        xs.append(nf)
        mus.append(vals.mean())
        ses.append(vals.std() / np.sqrt(max(len(vals), 1)))
    return np.array(xs), np.array(mus), np.array(ses)


def plot_sweep(args, save_path: str):
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.4))
    metric = "best" if args.show_running_max else "final"
    for algo in _algos_to_run(args):
        xs, mu, se = _summary_for_algo(args, algo, metric)
        if len(xs) == 0:
            continue
        ax.plot(xs, mu, color=COLORS[algo], lw=2.0,
                ls=LINESTYLES[algo], marker="o", label=LABELS[algo])
        ax.fill_between(xs, mu - se, mu + se, color=COLORS[algo], alpha=0.20)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("n_features (feature/aggregation dimension)")
    ax.set_ylabel(f"Avg. episode reward ({'best-so-far' if metric=='best' else 'final-iter'})")
    ax.set_title("CartPole: feature dimension as soft-aggregation analog")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(0, 210)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved figure to {save_path} and {pdf_path}")


# ── CLI ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="n_features sweep for CartPole DFL")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--n_workers", type=int, default=8)
    p.add_argument("--n_features_list", type=int, nargs="+",
                   default=[8, 16, 32, 64, 128, 256, 1024])

    # Mirror the offline-data + behavior knobs from make_figure1_cartpole
    p.add_argument("--behavior", type=str, default="biased",
                   choices=["expert", "noisy", "random",
                            "angle_only", "biased"])
    p.add_argument("--eps", type=float, default=0.5)
    p.add_argument("--td_weight", type=float, default=0.013)
    p.add_argument("--n_trajectories", type=int, default=100)

    p.add_argument("--pretrain_epochs_ts", type=int, default=30)
    p.add_argument("--pretrain_epochs_dfl", type=int, default=30)
    p.add_argument("--n_iterations", type=int, default=60)
    p.add_argument("--n_eval_episodes", type=int, default=100)
    p.add_argument("--n_ce_steps", type=int, default=0)
    p.add_argument("--n_dfl_steps", type=int, default=5)
    p.add_argument("--dfl_lr", type=float, default=None)

    p.add_argument("--bins_per_dim", type=int, default=8)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--qp_eta", type=float, default=0.01)
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--rho", type=float, default=10.0)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--sa_K", type=int, default=8,
                   help="number of sketches for DFL-SA (n_sketches in differentiable_sketched_lp)")
    p.add_argument("--sa_ratio", type=float, default=0.2,
                   help="row-sketch ratio for DFL-SA (k = round(ratio * m))")
    p.add_argument("--eval_policy", type=str, default="lp",
                   choices=["ce", "lp"])

    # World-model fields preserved so _job() works unchanged.
    p.add_argument("--use_world_model", action="store_true")
    p.add_argument("--world_epochs", type=int, default=400)
    p.add_argument("--world_lr", type=float, default=1e-3)

    p.add_argument("--include_qpth", action="store_true",
                   help="Also sweep the DFL-qpth variant.")
    p.add_argument("--show_running_max", action="store_true",
                   help="Plot best-so-far per seed instead of final-iter reward.")
    p.add_argument("--save_path", type=str,
                   default=str(HERE / "figure_nfeat_sweep.png"))
    p.add_argument("--plot_only", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--quick", action="store_true",
                   help="Smoke test: 2 seeds, small n_features list, fewer iters.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.quick:
        args.seeds = [1, 2]
        args.n_features_list = [16, 64, 256]
        args.n_iterations = 6
        args.pretrain_epochs_ts = 30
        args.pretrain_epochs_dfl = 15
        args.n_workers = min(args.n_workers, 4)

    if not args.plot_only:
        run_sweep(args)

    plot_sweep(args, save_path=args.save_path)


if __name__ == "__main__":
    main()
