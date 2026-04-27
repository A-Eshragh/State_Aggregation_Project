#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from .configs import CartPoleConfig
from .train_cartpole import run_neural_dfl, run_two_stage, set_seed


ALGO_CHOICES = ("neural_dfl", "two_stage")


def _build_cfg(args: argparse.Namespace) -> CartPoleConfig:
    return CartPoleConfig(
        n_iterations=args.n_iterations,
        gamma=args.gamma,
        lr=args.lr,
        qp_eta=args.qp_eta,
        reward_type=args.reward_type,
        n_features=args.n_features,
        n_trajectories=args.n_trajectories,
        n_epochs=args.n_iterations,
        ridge=args.ridge,
        rho=args.rho,
        n_eval_episodes=args.n_eval_episodes,
    )


def _save_dir(args: argparse.Namespace) -> Path:
    root = Path(args.results_root)
    out = root / args.algo
    if args.tag:
        out = out / f"tag_{args.tag}"
    out = out / f"n_iter_{args.n_iterations}" / f"pre_{args.pretrain_epochs}_eval_{args.eval_policy}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _history_to_arrays(history: List[Dict]) -> Dict[str, List[float]]:
    """Flatten the per-iteration history dicts into column-oriented lists."""
    if not history:
        return {"iteration": [], "avg_reward": [], "avg_length": []}
    cols: Dict[str, List[float]] = {}
    for row in history:
        for k, v in row.items():
            cols.setdefault(k, []).append(v)
    return cols


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-run CartPole DFL experiment")
    p.add_argument("--algo", type=str, required=True, choices=ALGO_CHOICES,
                   help="neural_dfl (CE+LP-DFL) or two_stage (CE only)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--tag", type=str, default="",
                   help="Optional subfolder tag (e.g. seed_1) for output dir")

    # Learning-curve size
    p.add_argument("--n_iterations", type=int, default=100,
                   help="Outer training iterations (each is one eval point)")
    p.add_argument("--n_eval_episodes", type=int, default=50)

    # Features / pretrain
    p.add_argument("--n_features", type=int, default=128)
    p.add_argument("--n_trajectories", type=int, default=500)
    p.add_argument("--pretrain_epochs", type=int, default=10,
                   help="CE/DAgger supervised pretrain steps before iterative phase")
    p.add_argument("--eval_policy", type=str, default="ce", choices=["ce", "lp"],
                   help="Policy used for evaluation at each iteration")
    p.add_argument("--n_ce_steps", type=int, default=2,
                   help="In-loop DAgger CE steps per iteration. Set to 0 in "
                        "offline mode to avoid LP-policy relabel collapse.")
    p.add_argument("--n_dfl_steps", type=int, default=5,
                   help="LP-DFL gradient steps per iteration "
                        "(neural_dfl only).")
    p.add_argument("--dfl_lr", type=float, default=None,
                   help="LR for the DFL-phase optimizer. None = use --lr. "
                        "A smaller value (e.g. 3e-4) keeps features near the "
                        "BC basin and stabilizes late iterations.")

    # LP / shaping knobs
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--qp_eta", type=float, default=0.01)
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--rho", type=float, default=10.0)
    p.add_argument("--reward_type", type=str, default="cos_theta",
                   choices=["cos_theta", "constant"])

    # Internal (matches run_neural_dfl signature)
    p.add_argument("--bins_per_dim", type=int, default=8)
    p.add_argument("--tau", type=float, default=0.5)

    p.add_argument("--results_root", type=str,
                   default="mdp_continuous/results_cartpole",
                   help="Root directory to save results")
    p.add_argument("--verbose", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(args.seed)
    cfg = _build_cfg(args)

    runner = run_neural_dfl if args.algo == "neural_dfl" else run_two_stage

    print("=" * 60)
    print(f"[run_cartpole] algo={args.algo} seed={args.seed} "
          f"n_iter={args.n_iterations} pretrain={args.pretrain_epochs} "
          f"eval={args.eval_policy}")
    print("=" * 60)

    t0 = time.time()
    out = runner(
        cfg,
        verbose=args.verbose,
        bins_per_dim=args.bins_per_dim,
        tau=args.tau,
        pretrain_epochs=args.pretrain_epochs,
        eval_policy_type=args.eval_policy,
        n_ce_steps=args.n_ce_steps,
        n_dfl_steps=args.n_dfl_steps,
        dfl_lr=args.dfl_lr,
    )
    total_time = time.time() - t0

    history = out.get("history", [])
    cols = _history_to_arrays(history)
    rewards = cols.get("avg_reward", [])
    lengths = cols.get("avg_length", [])

    results_payload = {
        # Identifiers (mirrors mdp_lp schema so plot script stays familiar)
        "algo": args.algo,
        "seed": int(args.seed),
        "tag": args.tag,
        "n_iterations": int(args.n_iterations),
        "pretrain_epochs": int(args.pretrain_epochs),
        "eval_policy": args.eval_policy,
        # Learning curve
        "avg_reward_per_iter": list(map(float, rewards)),
        "avg_length_per_iter": list(map(float, lengths)),
        "final_avg_reward": float(rewards[-1]) if rewards else float("nan"),
        "best_avg_reward": float(np.max(rewards)) if rewards else float("nan"),
        # Wall-clock (mirrors mdp_lp's train_time field so plotters can reuse)
        "total_train_time": float(total_time),
        "train_time_per_iter": (
            [float(total_time) / len(rewards)] * len(rewards) if rewards else []
        ),
        "status": "complete",
    }

    save_dir = _save_dir(args)
    out_file = save_dir / "results.json"
    out_file.write_text(json.dumps(results_payload, indent=2))

    print(f"[run_cartpole] final_reward={results_payload['final_avg_reward']:.2f} "
          f"best={results_payload['best_avg_reward']:.2f} "
          f"time={total_time:.1f}s")
    print(f"[run_cartpole] saved -> {out_file}")


if __name__ == "__main__":
    main()
