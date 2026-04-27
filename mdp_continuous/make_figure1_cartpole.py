"""
Generate Figure 1 for the offline-CartPole DFL experiment.

Single-panel learning curve: avg episode reward vs. iteration
(mean +/- std over seeds) for two methods:
    - Neural DFL (CE pretrain + LP-DFL gradient)
    - Two-stage (CE pretrain only, predict-then-optimize)

Setting: offline buffer collected with a *systematically suboptimal*
biased heuristic (theta_dot weight = 0.013 << expert's 0.5). BC therefore
caps near the behavior-policy reward (~120) regardless of pretrain budget;
DFL's LP-objective gradient bypasses this ceiling.

Seeds run in parallel via a process pool — set --n_workers to control.

Usage (run from project root):
    python -m mdp_continuous.make_figure1_cartpole --seeds 1 2 3 4 5 6 7 8
    python -m mdp_continuous.make_figure1_cartpole --plot_only   # use cache
    python -m mdp_continuous.make_figure1_cartpole --quick       # smoke test
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from .configs import CartPoleConfig
from .train_cartpole import run_neural_dfl, run_two_stage, set_seed
from .cartpole_env import (
    heuristic_policy,
    random_policy,
    epsilon_greedy_policy,
)


# ── Cache layout ────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "figure1_cache"


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


# ── Behavior policies (must be picklable for multiprocessing) ──────────

def _angle_only_policy(states):
    return (states[:, 2] > 0).long()


class _BiasedPolicy:
    """Biased PD controller — picklable so it crosses process boundaries."""
    def __init__(self, theta_dot_weight: float):
        self.td_w = float(theta_dot_weight)

    def __call__(self, states):
        return ((states[:, 2] + self.td_w * states[:, 3]) > 0).long()


def make_behavior_policy(kind: str, eps: float, td_weight: float):
    if kind == "expert":
        return heuristic_policy
    if kind == "noisy":
        return epsilon_greedy_policy(heuristic_policy, epsilon=eps)
    if kind == "random":
        return random_policy
    if kind == "angle_only":
        return _angle_only_policy
    if kind == "biased":
        return _BiasedPolicy(td_weight)
    raise ValueError(kind)


# ── Single-seed runners (top-level => picklable) ──────────────────────

def _run_one_seed(args_dict: dict) -> dict:
    """Run one (algo, seed) configuration. Returns a small JSON-able dict.

    Each worker re-imports torch and re-seeds, so runs are deterministic
    and isolated. Workers are CPU-only.
    """
    algo = args_dict["algo"]
    seed = args_dict["seed"]

    torch.set_num_threads(args_dict.get("torch_threads", 1))

    set_seed(seed)
    cfg = CartPoleConfig(
        n_iterations=args_dict["n_iterations"],
        gamma=args_dict["gamma"],
        lr=args_dict["lr"],
        qp_eta=args_dict["qp_eta"],
        reward_type="cos_theta",
        n_features=args_dict["n_features"],
        n_trajectories=args_dict["n_trajectories"],
        n_epochs=args_dict["n_iterations"],
        ridge=args_dict["ridge"],
        rho=args_dict["rho"],
        n_eval_episodes=args_dict["n_eval_episodes"],
    )
    behavior = make_behavior_policy(
        args_dict["behavior"], args_dict["eps"], args_dict["td_weight"])

    if algo == "neural_dfl":
        runner = run_neural_dfl
        pretrain = args_dict["pretrain_epochs_dfl"]
    elif algo == "two_stage":
        runner = run_two_stage
        pretrain = args_dict["pretrain_epochs_ts"]
    else:
        raise ValueError(algo)

    out = runner(
        cfg,
        verbose=False,
        bins_per_dim=args_dict["bins_per_dim"],
        tau=args_dict["tau"],
        pretrain_epochs=pretrain,
        eval_policy_type=args_dict["eval_policy"],
        behavior_policy=behavior,
        n_ce_steps=args_dict["n_ce_steps"],
        n_dfl_steps=args_dict["n_dfl_steps"],
        dfl_lr=args_dict["dfl_lr"],
    )
    rewards = [float(h["avg_reward"]) for h in out["history"]]
    lengths = [float(h["avg_length"]) for h in out["history"]]
    return {
        "algo": algo,
        "seed": int(seed),
        "rewards": rewards,
        "lengths": lengths,
    }


def _job(args, algo: str, seed: int) -> dict:
    return {
        "algo": algo,
        "seed": int(seed),
        "n_iterations": args.n_iterations,
        "gamma": args.gamma,
        "lr": args.lr,
        "qp_eta": args.qp_eta,
        "n_features": args.n_features,
        "n_trajectories": args.n_trajectories,
        "ridge": args.ridge,
        "rho": args.rho,
        "n_eval_episodes": args.n_eval_episodes,
        "bins_per_dim": args.bins_per_dim,
        "tau": args.tau,
        "behavior": args.behavior,
        "eps": args.eps,
        "td_weight": args.td_weight,
        "pretrain_epochs_ts": args.pretrain_epochs_ts,
        "pretrain_epochs_dfl": args.pretrain_epochs_dfl,
        "eval_policy": args.eval_policy,
        "n_ce_steps": args.n_ce_steps,
        "n_dfl_steps": args.n_dfl_steps,
        "dfl_lr": args.dfl_lr,
        "torch_threads": 1,
    }


def _cache_key(prefix: str, args) -> str:
    dfl_lr_tag = "auto" if args.dfl_lr is None else f"{args.dfl_lr}"
    return (
        f"{prefix}"
        f"_beh{args.behavior}_td{args.td_weight}_eps{args.eps}"
        f"_ntraj{args.n_trajectories}"
        f"_iters{args.n_iterations}"
        f"_preTS{args.pretrain_epochs_ts}_preDFL{args.pretrain_epochs_dfl}"
        f"_nce{args.n_ce_steps}_ndfl{args.n_dfl_steps}"
        f"_dfllr{dfl_lr_tag}"
        f"_eval{args.eval_policy}"
    )


def run_all_curves(args):
    """Run both methods x all seeds in parallel; return (neural, twostage)."""
    name_neural = _cache_key("curve_neural", args)
    name_twostage = _cache_key("curve_twostage", args)
    neural_cached = _load_or_none(name_neural) if not args.force else None
    ts_cached = _load_or_none(name_twostage) if not args.force else None

    if neural_cached and ts_cached:
        return neural_cached, ts_cached

    jobs = []
    if not neural_cached:
        for s in args.seeds:
            jobs.append(_job(args, "neural_dfl", s))
    if not ts_cached:
        for s in args.seeds:
            jobs.append(_job(args, "two_stage", s))

    print(f"[run] {len(jobs)} jobs across {args.n_workers} workers")
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futures = {ex.submit(_run_one_seed, j): j for j in jobs}
        for fut in as_completed(futures):
            j = futures[fut]
            r = fut.result()
            print(f"[done] algo={r['algo']:<10s} seed={r['seed']} "
                  f"best={max(r['rewards']):.2f} "
                  f"final={r['rewards'][-1]:.2f}")
            results.append(r)
    print(f"[run] {len(results)} jobs finished in {time.time() - t0:.1f}s")

    neural_runs = sorted([r for r in results if r["algo"] == "neural_dfl"],
                         key=lambda r: r["seed"])
    ts_runs = sorted([r for r in results if r["algo"] == "two_stage"],
                     key=lambda r: r["seed"])

    if not neural_cached and neural_runs:
        neural_cached = {
            "seeds": [r["seed"] for r in neural_runs],
            "rewards_per_seed": [r["rewards"] for r in neural_runs],
            "lengths_per_seed": [r["lengths"] for r in neural_runs],
        }
        _save(name_neural, neural_cached)
    if not ts_cached and ts_runs:
        ts_cached = {
            "seeds": [r["seed"] for r in ts_runs],
            "rewards_per_seed": [r["rewards"] for r in ts_runs],
            "lengths_per_seed": [r["lengths"] for r in ts_runs],
        }
        _save(name_twostage, ts_cached)

    return neural_cached, ts_cached


# ── Plotting ────────────────────────────────────────────────────────────

COLORS = {
    "neural_dfl": "#1f77b4",   # blue
    "two_stage":  "#ff7f0e",   # orange
}


def _running_max(arr2d: np.ndarray) -> np.ndarray:
    """Per-seed cummax — what the algorithm would *return* (best-so-far)."""
    return np.maximum.accumulate(arr2d, axis=1)


def _stack_curves(runs):
    if not runs:
        return None, None
    L = min(len(r) for r in runs)
    arr = np.array([r[:L] for r in runs], dtype=float)
    return arr, L


def plot_figure1(args, save_path: str):
    neural = _load_or_none(_cache_key("curve_neural", args))
    twostage = _load_or_none(_cache_key("curve_twostage", args))

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.2))

    for label, key, payload, ls in [
        ("Neural DFL (ours)", "neural_dfl", neural, "-"),
        ("Two-stage (predict-then-optimize)", "two_stage", twostage, "--"),
    ]:
        if not payload:
            continue
        arr, L = _stack_curves(payload["rewards_per_seed"])
        if args.show_running_max:
            arr = _running_max(arr)
        mu = arr.mean(axis=0)
        sd = arr.std(axis=0)
        xs = np.arange(L)
        ax.plot(xs, mu, color=COLORS[key], lw=2.0, ls=ls, label=label)
        ax.fill_between(xs, mu - sd, mu + sd, color=COLORS[key], alpha=0.20)

    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Avg. episode reward "
                  + ("(best-so-far)" if args.show_running_max else "(per iter)"))
    ax.set_title(
        "CartPole offline: Neural DFL vs. two-stage\n"
        f"(biased behavior, td_weight={args.td_weight}; "
        f"n_seeds={len(args.seeds)})"
    )
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved figure to {save_path} and {pdf_path}")


def print_summary(args):
    neural = _load_or_none(_cache_key("curve_neural", args))
    twostage = _load_or_none(_cache_key("curve_twostage", args))
    if not (neural and twostage):
        return

    def _stats(runs):
        bests = [max(r) for r in runs]
        finals = [r[-1] for r in runs]
        return (np.mean(bests), np.std(bests),
                np.mean(finals), np.std(finals))

    nb_m, nb_s, nf_m, nf_s = _stats(neural["rewards_per_seed"])
    tb_m, tb_s, tf_m, tf_s = _stats(twostage["rewards_per_seed"])

    print("\n" + "=" * 60)
    print(f"SUMMARY  (n_seeds = {len(args.seeds)})")
    print("=" * 60)
    print(f"  Two-stage:  best = {tb_m:.1f} ± {tb_s:.1f}   "
          f"final = {tf_m:.1f} ± {tf_s:.1f}")
    print(f"  Neural DFL: best = {nb_m:.1f} ± {nb_s:.1f}   "
          f"final = {nf_m:.1f} ± {nf_s:.1f}")
    print(f"  DFL−TS gap (best):  {nb_m - tb_m:+.2f}")
    print(f"  DFL−TS gap (final): {nf_m - tf_m:+.2f}")


# ── CLI ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Figure 1 for offline CartPole DFL")
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[1, 2, 3, 4, 5, 6, 7, 8])
    p.add_argument("--n_workers", type=int, default=8,
                   help="Parallel workers; one process per (algo, seed).")

    # Offline data + behavior policy
    p.add_argument("--behavior", type=str, default="biased",
                   choices=["expert", "noisy", "random",
                            "angle_only", "biased"])
    p.add_argument("--eps", type=float, default=0.5)
    p.add_argument("--td_weight", type=float, default=0.013,
                   help="theta_dot weight in biased heuristic. "
                        "Expert uses 0.5; smaller = more suboptimal. "
                        "0.013 gives behavior-policy reward ~122.")
    p.add_argument("--n_trajectories", type=int, default=100)

    # Per-method pretrain budgets
    p.add_argument("--pretrain_epochs_ts", type=int, default=100,
                   help="CE pretrain epochs for two-stage (let it converge).")
    p.add_argument("--pretrain_epochs_dfl", type=int, default=30,
                   help="CE pretrain epochs for DFL (leave room for LP).")

    # Iteration budget
    p.add_argument("--n_iterations", type=int, default=16)
    p.add_argument("--n_eval_episodes", type=int, default=100)
    p.add_argument("--n_ce_steps", type=int, default=0,
                   help="0 in offline mode to avoid LP-policy relabel collapse")
    p.add_argument("--n_dfl_steps", type=int, default=5)
    p.add_argument("--dfl_lr", type=float, default=None,
                   help="LR for DFL-phase optimizer (Adam over features+head). "
                        "Lower than cfg.lr keeps features close to BC basin and "
                        "stabilizes late-iteration behavior. None = use cfg.lr.")

    # Other
    p.add_argument("--n_features", type=int, default=128)
    p.add_argument("--bins_per_dim", type=int, default=8)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--qp_eta", type=float, default=0.01)
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--rho", type=float, default=10.0)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--eval_policy", type=str, default="ce",
                   choices=["ce", "lp"])

    p.add_argument("--save_path", type=str,
                   default=str(HERE / "figure1_cartpole.pdf"))
    p.add_argument("--show_running_max", action="store_true",
                   help="Plot per-seed best-so-far instead of raw per-iter "
                        "reward (matches what the algorithm actually returns "
                        "via best-so-far restoration).")
    p.add_argument("--plot_only", action="store_true",
                   help="Skip experiments, rebuild plot from cache")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if cache exists")
    p.add_argument("--quick", action="store_true",
                   help="Smoke test: small seeds/iters")
    return p.parse_args()


def main():
    args = parse_args()

    if args.quick:
        args.seeds = [1, 2]
        args.n_iterations = 8
        args.pretrain_epochs_ts = 30
        args.pretrain_epochs_dfl = 15
        args.n_workers = min(args.n_workers, 4)

    if not args.plot_only:
        run_all_curves(args)

    plot_figure1(args, save_path=args.save_path)
    print_summary(args)


if __name__ == "__main__":
    main()
