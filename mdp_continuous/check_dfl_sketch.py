"""
Quick smoke test: does sketch-and-solve DFL work, and does it match or
beat dense-LP DFL on CartPole?

Runs the existing run_neural_dfl loop twice on shared data + world model,
once with the dense LP and once with the row-sketched LP. The sketched
variant is injected by monkey-patching the symbol that train_cartpole
imported, so we don't have to touch the training code.
"""

from __future__ import annotations

import argparse
import time

import torch

from . import train_cartpole as tc
from .cartpole_env import CartPoleDynamics, collect_trajectories, heuristic_policy
from .configs import CartPoleConfig
from .reduced_solver import differentiable_reduced_lp
from .sketched_solver import differentiable_sketched_lp
from .train_cartpole import run_neural_dfl, set_seed
from .world_model import train_world_model


def _make_sketched_lp_fn(n_sketches: int, sketch_ratio: float, sketch_mode: str,
                         sketch_seed):
    def _fn(M, c, b):
        return differentiable_sketched_lp(
            M, c, b,
            n_sketches=n_sketches,
            sketch_ratio=sketch_ratio,
            sketch_mode=sketch_mode,
            sketch_seed=sketch_seed,
        )
    return _fn


def _run_one(label: str, cfg: CartPoleConfig, *, data, world_model, seed: int,
             use_sketch: bool, sketch_kwargs: dict | None,
             pretrain_epochs: int, n_dfl_steps: int, n_ce_steps: int):
    print(f"\n=== {label} ===")
    set_seed(seed)

    if use_sketch:
        # train_cartpole imported the symbol directly; rebinding the
        # module attribute is what matters for the dispatch in
        # run_neural_dfl.
        original_lp = tc.differentiable_reduced_lp
        tc.differentiable_reduced_lp = _make_sketched_lp_fn(**sketch_kwargs)

    t0 = time.time()
    try:
        out = run_neural_dfl(
            cfg, verbose=False,
            pretrain_epochs=pretrain_epochs,
            eval_policy_type="lp",
            n_ce_steps=n_ce_steps,
            n_dfl_steps=n_dfl_steps,
            data=data,
            world_model=world_model,
            solver="lp",  # dispatches to the (possibly patched) symbol
        )
    finally:
        if use_sketch:
            tc.differentiable_reduced_lp = original_lp

    wall = time.time() - t0
    rewards = [h["avg_reward"] for h in out["history"]]
    final = rewards[-1] if rewards else float("nan")
    print(f"  best_reward = {out['best_reward']:.2f}    "
          f"final_reward = {final:.2f}    "
          f"wall = {wall:.1f}s    "
          f"solver_fwd = {out['solver_forward_time']:.1f}s    "
          f"solver_bwd = {out['backward_time']:.1f}s")
    return {
        "label": label,
        "best": float(out["best_reward"]),
        "final": float(final),
        "rewards": rewards,
        "wall": wall,
        "solver_forward": float(out["solver_forward_time"]),
        "solver_backward": float(out["backward_time"]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n_iterations", type=int, default=15)
    p.add_argument("--pretrain_epochs", type=int, default=30)
    p.add_argument("--n_dfl_steps", type=int, default=5)
    p.add_argument("--n_ce_steps", type=int, default=1)
    p.add_argument("--n_features", type=int, default=128)
    p.add_argument("--n_trajectories", type=int, default=200)
    p.add_argument("--world_epochs", type=int, default=200)
    p.add_argument("--n_eval_episodes", type=int, default=30)
    p.add_argument("--n_sketches", type=int, default=4)
    p.add_argument("--sketch_ratio", type=float, default=0.7)
    p.add_argument("--sketch_mode", type=str, default="gaussian",
                   choices=["gaussian", "rademacher", "subsample"])
    p.add_argument("--only", type=str, default="both",
                   choices=["both", "dense", "sketch"])
    args = p.parse_args()

    set_seed(args.seed)
    cfg = CartPoleConfig(
        n_iterations=args.n_iterations,
        n_features=args.n_features,
        n_trajectories=args.n_trajectories,
        n_eval_episodes=args.n_eval_episodes,
        rho=10.0,
    )

    device = torch.device("cpu")
    dynamics = CartPoleDynamics(cfg)

    print(f"Collecting {cfg.n_trajectories} trajectories...")
    buf = collect_trajectories(
        dynamics, heuristic_policy, cfg.n_trajectories, cfg.max_steps, device)
    print(f"  {buf['states'].shape[0]} transitions, "
          f"{buf['init_states'].shape[0]} init states")

    print(f"Fitting world model ({args.world_epochs} epochs)...")
    wm = train_world_model(buf, n_epochs=args.world_epochs, lr=1e-3,
                           verbose=False)

    common = dict(
        cfg=cfg, data=buf, world_model=wm, seed=args.seed,
        pretrain_epochs=args.pretrain_epochs,
        n_dfl_steps=args.n_dfl_steps,
        n_ce_steps=args.n_ce_steps,
    )

    res_dense = None
    res_sketch = None

    if args.only in ("both", "dense"):
        res_dense = _run_one(
            "DFL (dense LP)", use_sketch=False, sketch_kwargs=None, **common)

    if args.only in ("both", "sketch"):
        sketch_kwargs = dict(
            n_sketches=args.n_sketches,
            sketch_ratio=args.sketch_ratio,
            sketch_mode=args.sketch_mode,
            # Vary sketches across DFL steps -> resampled regularization.
            sketch_seed=None,
        )
        res_sketch = _run_one(
            f"DFL (sketch K={args.n_sketches}, "
            f"ratio={args.sketch_ratio}, mode={args.sketch_mode})",
            use_sketch=True, sketch_kwargs=sketch_kwargs, **common)

    print("\n=== Summary ===")
    if res_dense is not None:
        print(f"  dense  best={res_dense['best']:6.2f}  "
              f"final={res_dense['final']:6.2f}  "
              f"solver={res_dense['solver_forward'] + res_dense['solver_backward']:5.1f}s")
    if res_sketch is not None:
        print(f"  sketch best={res_sketch['best']:6.2f}  "
              f"final={res_sketch['final']:6.2f}  "
              f"solver={res_sketch['solver_forward'] + res_sketch['solver_backward']:5.1f}s")
    if res_dense is not None and res_sketch is not None:
        delta = res_sketch["best"] - res_dense["best"]
        verdict = "BETTER" if delta > 0 else ("WORSE" if delta < 0 else "TIE")
        print(f"  sketch - dense (best reward) = {delta:+.2f}   [{verdict}]")


if __name__ == "__main__":
    main()
