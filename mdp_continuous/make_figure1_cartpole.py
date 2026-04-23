"""
Generate Figure 1 for the NeurIPS CartPole experiment (continuous-state DFL).

Single-panel learning curve: avg episode reward vs. iteration
(mean +/- std over seeds) for two methods:
    - Neural DFL (ours, CE pretrain + LP-DFL gradient)
    - Two-stage (predict-then-optimize, CE pretrain only)

Usage (run from project root):
    python -m mdp_continuous.make_figure1_cartpole --seeds 1 2 3
    python -m mdp_continuous.make_figure1_cartpole --plot_only   # use cache
    python -m mdp_continuous.make_figure1_cartpole --quick       # fast smoke test
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .configs import CartPoleConfig
from .train_cartpole import run_neural_dfl, run_two_stage, set_seed


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


# ── Experiment runners ──────────────────────────────────────────────────

def make_cfg(args, n_iterations=None):
    return CartPoleConfig(
        n_iterations=n_iterations or args.n_iterations,
        gamma=args.gamma,
        lr=args.lr,
        qp_eta=args.qp_eta,
        reward_type="cos_theta",
        n_features=args.n_features,
        n_trajectories=args.n_trajectories,
        n_epochs=args.n_iterations,
        ridge=1e-4,
        rho=1.0,
        n_eval_episodes=args.n_eval_episodes,
    )


def _cache_key(prefix: str, args) -> str:
    return (f"{prefix}_iters{args.n_iterations}"
            f"_pre{args.pretrain_epochs}_eval{args.eval_policy}")


def run_curve_neural_dfl(args):
    """Neural DFL learning curve across seeds."""
    name = _cache_key("curve_neural", args)
    cached = _load_or_none(name)
    if cached and not args.force:
        return cached

    runs = []
    for seed in args.seeds:
        print(f"[Neural DFL] seed={seed}")
        set_seed(seed)
        cfg = make_cfg(args)
        out = run_neural_dfl(
            cfg, verbose=False,
            bins_per_dim=args.bins_per_dim, tau=args.tau,
            use_dfl_phase=True,
            pretrain_epochs=args.pretrain_epochs,
            eval_policy_type=args.eval_policy,
        )
        rewards = [h["avg_reward"] for h in out["history"]]
        print(f"Rewards: {rewards}")
        print(f"History: {out['history']}")
        runs.append(rewards)

    payload = {"seeds": args.seeds, "rewards_per_seed": runs}
    _save(name, payload)
    return payload


def run_curve_two_stage(args):
    """Two-stage (predict-then-optimize) learning curve across seeds."""
    name = _cache_key("curve_twostage", args)
    cached = _load_or_none(name)
    if cached and not args.force:
        return cached

    runs = []
    for seed in args.seeds:
        print(f"[Two-Stage] seed={seed}")
        set_seed(seed)
        cfg = make_cfg(args)
        out = run_two_stage(
            cfg, verbose=False,
            bins_per_dim=args.bins_per_dim, tau=args.tau,
            pretrain_epochs=args.pretrain_epochs,
            eval_policy_type=args.eval_policy,
        )
        rewards = [h["avg_reward"] for h in out["history"]]
        print(f"Rewards: {rewards}")
        print(f"History: {out['history']}")
        runs.append(rewards)

    payload = {"seeds": args.seeds, "rewards_per_seed": runs}
    _save(name, payload)
    return payload


# ── Plotting ────────────────────────────────────────────────────────────

COLORS = {
    "neural_dfl": "#1f77b4",   # blue
    "two_stage":  "#ff7f0e",   # orange
}


def _stack_curves(runs):
    """Pad ragged seed runs to common length (min) and stack."""
    if not runs:
        return None, None
    L = min(len(r) for r in runs)
    arr = np.array([r[:L] for r in runs], dtype=float)
    return arr.mean(axis=0), arr.std(axis=0)


def plot_figure1(args, save_path: str):
    neural = _load_or_none(_cache_key("curve_neural", args))
    twostage = _load_or_none(_cache_key("curve_twostage", args))

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 4.0))

    if neural:
        mu, sd = _stack_curves(neural["rewards_per_seed"])
        xs = np.arange(len(mu))
        ax.plot(xs, mu, color=COLORS["neural_dfl"], lw=2.0,
                label="Neural DFL (ours)")
        ax.fill_between(xs, mu - sd, mu + sd,
                        color=COLORS["neural_dfl"], alpha=0.20)

    if twostage:
        mu, sd = _stack_curves(twostage["rewards_per_seed"])
        xs = np.arange(len(mu))
        ax.plot(xs, mu, color=COLORS["two_stage"], lw=2.0,
                linestyle="--", label="Two-stage (predict-then-optimize)")
        ax.fill_between(xs, mu - sd, mu + sd,
                        color=COLORS["two_stage"], alpha=0.15)

    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Avg. discounted episode reward")
    ax.set_title("CartPole: Neural DFL vs. two-stage")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved figure to {save_path} and {pdf_path}")


# ── CLI ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Figure 1 for CartPole DFL")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--n_iterations", type=int, default=100)
    p.add_argument("--bins_per_dim", type=int, default=8,
                   help="Discretization used only inside the LP warm-up")
    p.add_argument("--n_eval_episodes", type=int, default=50)
    p.add_argument("--n_features", type=int, default=128)
    p.add_argument("--n_trajectories", type=int, default=1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--qp_eta", type=float, default=0.01)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--pretrain_epochs", type=int, default=10,
                   help="CE pretrain steps before iterative training; "
                        "smaller → curves start lower and diverge more.")
    p.add_argument("--eval_policy", type=str, default="ce",
                   choices=["ce", "lp"],
                   help="Policy used for evaluation at each iteration.")

    p.add_argument("--save_path", type=str,
                   default=str(HERE / "figure1_cartpole.png"))
    p.add_argument("--plot_only", action="store_true",
                   help="Skip experiments, just rebuild plot from cache")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if cache exists")
    p.add_argument("--quick", action="store_true",
                   help="Smoke test: small seeds/iters")
    return p.parse_args()


def main():
    args = parse_args()

    if args.quick:
        args.seeds = [1, 2]
        args.n_iterations = 20
        args.pretrain_epochs = 5

    if not args.plot_only:
        t0 = time.time()
        run_curve_neural_dfl(args)
        run_curve_two_stage(args)
        print(f"All experiments done in {time.time() - t0:.1f}s")

    plot_figure1(args, save_path=args.save_path)


if __name__ == "__main__":
    main()
