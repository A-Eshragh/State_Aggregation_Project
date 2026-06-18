"""
Predict-then-optimize / DFL on CartPole with a learned world model.

Pipeline:
  1. Collect a real offline buffer with the behavior policy (same as
     train_cartpole.py — no synthetic trajectories, no env access during
     training).
  2. Fit a deterministic world model (delta-dynamics + reward) on it.
  3. Hand the buffer + world model to run_neural_dfl. Inside the trainer the
     LP estimator's (r, s') are silently swapped for (r̂, ŝ'). The CE pretrain
     still uses the real expert actions.
  4. Compare DFL vs two-stage. Both methods use the same buffer and the same
     frozen world model — the only thing that differs is whether the LP
     gradient flows back into the features.

Why this is delicate (and why DFL can lose if you're not careful):
  - With predicted (r, s') the LP solves for the optimum of the *learned*
    MDP. DFL trains features to maximise that optimum, so any model bias
    is a free lunch the optimiser will eat. Two-stage doesn't optimise the
    LP value, only imitates the LP-policy, so it's less exposed.
  - Mitigations: (a) train the world model carefully (early-stop on val);
    (b) use deterministic predictions, not samples; (c) keep the DFL phase
    gentle so features can't drift far chasing model artefacts.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from .cartpole_env import (
    CartPoleDynamics,
    collect_trajectories,
    epsilon_greedy_policy,
    heuristic_policy,
)
from .configs import CartPoleConfig
from .train_cartpole import run_neural_dfl, set_seed
from .world_model import train_world_model


def collect_and_fit(cfg: CartPoleConfig,
                    n_real_episodes: int = 30,
                    world_epochs: int = 400,
                    world_lr: float = 1e-3,
                    behavior_eps: float = 0.5,
                    verbose: bool = True,
                    device=None):
    """Collect real buffer + fit world model on it. Used once per seed."""
    if device is None:
        device = torch.device("cpu")
    dynamics = CartPoleDynamics(cfg)
    behavior_policy = epsilon_greedy_policy(heuristic_policy, epsilon=behavior_eps)

    if verbose:
        print(f"[MB] Collecting {n_real_episodes} real episodes "
              f"(epsilon={behavior_eps})...")
    real = collect_trajectories(dynamics, behavior_policy,
                                n_real_episodes, cfg.max_steps, device)
    if verbose:
        print(f"[MB] Real buffer: {real['states'].shape[0]} transitions, "
              f"{real['init_states'].shape[0]} init states")

    if verbose:
        print(f"[MB] Training world model ({world_epochs} epochs)...")
    world_model = train_world_model(real, n_epochs=world_epochs, lr=world_lr,
                                    verbose=verbose)
    return real, world_model


def run_pto(cfg: CartPoleConfig, use_dfl: bool, real_buffer: dict,
            world_model, verbose: bool = True, **dfl_kwargs):
    """Predict-then-optimize: real (s, a) buffer + predicted (r̂, ŝ') in LP."""
    if verbose:
        tag = "DFL" if use_dfl else "Two-Stage"
        print(f"\n[PtO] Running {tag} on real buffer with predicted (r, s')...")
    out = run_neural_dfl(cfg, verbose=verbose,
                         use_dfl_phase=use_dfl,
                         data=real_buffer,
                         world_model=world_model,
                         **dfl_kwargs)

    # Re-evaluate the snapshot-restored policy in the true env (apples to
    # apples with two-stage — both methods restore best snapshot).
    if "features" in out and "w_head" in out:
        from .policy import evaluate_policy as _eval

        dynamics = CartPoleDynamics(cfg)
        feats = out["features"]
        w_head = out["w_head"]

        def deployed_policy(states):
            with torch.no_grad():
                f_s = feats._state_features(states)
                return (f_s @ w_head).argmax(dim=1)

        deployed = _eval(dynamics, deployed_policy,
                         n_episodes=cfg.n_eval_episodes,
                         max_steps=cfg.max_steps, gamma=cfg.gamma)
        out["deployed_eval"] = deployed
        if verbose:
            print(f"  [PtO] Deployed (best-snapshot) reward: "
                  f"{deployed['avg_reward']:.2f}")
    return out


def _summarize(history, label, threshold=180.0):
    rewards = [h["avg_reward"] for h in history]
    iters_to_threshold = next(
        (i for i, r in enumerate(rewards) if r >= threshold), None)
    iter_str = f"{iters_to_threshold}" if iters_to_threshold is not None else "—"
    running_best = np.maximum.accumulate(rewards)
    last5 = rewards[-5:] if len(rewards) >= 5 else rewards
    print(f"  {label:32s} | best={max(rewards):7.2f} | "
          f"final-5-mean={np.mean(last5):7.2f} | "
          f"area-under-best={running_best.mean():7.2f} | "
          f"iter≥{int(threshold)}={iter_str}")


def _run_one_seed(cfg, args, seed):
    set_seed(seed)
    real, wm = collect_and_fit(
        cfg,
        n_real_episodes=args.n_real_episodes,
        world_epochs=args.world_epochs,
        world_lr=args.world_lr,
        behavior_eps=args.behavior_eps,
    )

    print("\n" + "-" * 60)
    print(f"[seed={seed}] Predict-then-optimize: DFL")
    print("-" * 60)
    set_seed(seed)
    out_dfl = run_pto(cfg, use_dfl=True, real_buffer=real, world_model=wm,
                      pretrain_epochs=args.pretrain_epochs,
                      dfl_lr=args.dfl_lr,
                      n_dfl_steps=args.n_dfl_steps,
                      n_ce_steps=args.n_ce_steps)

    print("\n" + "-" * 60)
    print(f"[seed={seed}] Predict-then-optimize: Two-Stage")
    print("-" * 60)
    set_seed(seed)
    out_ts = run_pto(cfg, use_dfl=False, real_buffer=real, world_model=wm,
                     pretrain_epochs=args.pretrain_epochs,
                     dfl_lr=args.dfl_lr,
                     n_dfl_steps=args.n_dfl_steps,
                     n_ce_steps=args.n_ce_steps)
    return out_dfl, out_ts


def parse_args():
    p = argparse.ArgumentParser(
        description="Predict-then-optimize DFL on CartPole (learned world model)")
    p.add_argument("--n_iterations", type=int, default=30)
    p.add_argument("--n_real_episodes", type=int, default=30)
    p.add_argument("--world_epochs", type=int, default=400)
    p.add_argument("--world_lr", type=float, default=1e-3)
    p.add_argument("--n_features", type=int, default=128)
    p.add_argument("--pretrain_epochs", type=int, default=70)
    p.add_argument("--n_eval_episodes", type=int, default=50)
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--save_plot", type=str, default=None)
    # Slightly gentler LP-gradient updates so DFL doesn't chase model
    # artefacts as aggressively. The defaults are tuned for stability.
    p.add_argument("--dfl_lr", type=float, default=3e-4)
    p.add_argument("--n_dfl_steps", type=int, default=3)
    p.add_argument("--n_ce_steps", type=int, default=2)
    p.add_argument("--behavior_eps", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = CartPoleConfig(
        n_iterations=args.n_iterations,
        n_features=args.n_features,
        n_eval_episodes=args.n_eval_episodes,
        rho=10.0,
    )

    all_dfl, all_ts = [], []
    deployed_dfl, deployed_ts = [], []

    for seed in args.seeds:
        out_dfl, out_ts = _run_one_seed(cfg, args, seed)
        all_dfl.append(out_dfl)
        all_ts.append(out_ts)
        deployed_dfl.append(out_dfl.get("deployed_eval", {}).get("avg_reward", float("nan")))
        deployed_ts.append(out_ts.get("deployed_eval", {}).get("avg_reward", float("nan")))

    print("\n" + "=" * 60)
    print(f"Per-seed comparison ({len(args.seeds)} seeds):")
    print("=" * 60)
    for s, dfl, ts, dd, dt in zip(args.seeds, all_dfl, all_ts,
                                  deployed_dfl, deployed_ts):
        print(f"\n  seed={s}:")
        _summarize(dfl["history"], "    DFL (predict-then-opt)")
        _summarize(ts["history"], "    Two-Stage (predict-then-opt)")
        print(f"    Deployed (best snapshot, fresh eval) — "
              f"DFL={dd:.2f}  vs  Two-Stage={dt:.2f}  "
              f"(Δ={dd-dt:+.2f})")

    if len(args.seeds) > 1:
        print("\n" + "=" * 60)
        print("Across-seed summary (deployed best-snapshot reward):")
        print("=" * 60)
        print(f"  DFL:       mean={np.nanmean(deployed_dfl):7.2f}  "
              f"std={np.nanstd(deployed_dfl):6.2f}  "
              f"min={np.nanmin(deployed_dfl):7.2f}  "
              f"max={np.nanmax(deployed_dfl):7.2f}")
        print(f"  Two-Stage: mean={np.nanmean(deployed_ts):7.2f}  "
              f"std={np.nanstd(deployed_ts):6.2f}  "
              f"min={np.nanmin(deployed_ts):7.2f}  "
              f"max={np.nanmax(deployed_ts):7.2f}")
        wins = sum(1 for d, t in zip(deployed_dfl, deployed_ts) if d > t)
        ties = sum(1 for d, t in zip(deployed_dfl, deployed_ts) if abs(d - t) < 1e-6)
        print(f"  DFL wins on {wins}/{len(args.seeds)} seeds, ties on {ties}.")

    out_dfl, out_ts = all_dfl[0], all_ts[0]
    dfl_rewards = np.array([h["avg_reward"] for h in out_dfl["history"]])
    ts_rewards = np.array([h["avg_reward"] for h in out_ts["history"]])
    dfl_best = np.maximum.accumulate(dfl_rewards)
    ts_best = np.maximum.accumulate(ts_rewards)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    xs = list(range(len(dfl_rewards)))

    ax = axes[0]
    ax.plot(xs, dfl_rewards, label="DFL", marker="o", markersize=3, alpha=0.85)
    ax.plot(xs, ts_rewards, label="Two-Stage", marker="x", markersize=3, alpha=0.85)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Avg Episode Reward (true env)")
    ax.set_title("Per-iteration eval")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(xs, dfl_best, label="DFL (best-so-far)", linewidth=2)
    ax.plot(xs, ts_best, label="Two-Stage (best-so-far)", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Avg Reward So Far")
    ax.set_title("Running best (deployable model)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Predict-then-optimize CartPole: DFL vs Two-Stage",
                 fontsize=13)
    plt.tight_layout()
    if args.save_plot:
        plt.savefig(args.save_plot, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {args.save_plot}")


if __name__ == "__main__":
    main()
