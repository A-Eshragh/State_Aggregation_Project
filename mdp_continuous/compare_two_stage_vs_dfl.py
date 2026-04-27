"""Offline comparison: two-stage (CE only) vs DFL (CE+LP) on CartPole.

Goal: a regime where two-stage cannot reach optimal reward but DFL can.

The key insight: BC on noisy expert data still converges to the expert mode
(noise averages out), so noise alone does NOT cap two-stage. To genuinely
cap BC, the behavior policy must be *systematically suboptimal* — so that
"perfectly imitating it" still falls short of optimal.

This script supports several deterministic-but-suboptimal behavior policies
(weak / biased heuristics) and per-method pretrain budgets so two-stage can
fully converge while DFL is held back to leave room for the LP-gradient
phase to add value.
"""

import argparse
import torch

from .train_cartpole import set_seed, run_neural_dfl, run_two_stage
from .configs import CartPoleConfig
from .cartpole_env import (
    CartPoleDynamics,
    heuristic_policy,
    random_policy,
    epsilon_greedy_policy,
)
from .policy import evaluate_policy


def angle_only_policy(states: torch.Tensor) -> torch.Tensor:
    """Suboptimal: ignores theta_dot. Cannot react to angular velocity."""
    return (states[:, 2] > 0).long()


def make_biased_policy(theta_dot_weight: float):
    """Biased PD controller: underweights theta_dot vs the expert's 0.5."""
    def policy(states: torch.Tensor) -> torch.Tensor:
        return ((states[:, 2] + theta_dot_weight * states[:, 3]) > 0).long()
    return policy


def make_behavior_policy(kind: str, eps: float, td_weight: float):
    if kind == "expert":
        return heuristic_policy
    if kind == "noisy":
        return epsilon_greedy_policy(heuristic_policy, epsilon=eps)
    if kind == "random":
        return random_policy
    if kind == "angle_only":
        return angle_only_policy
    if kind == "biased":
        return make_biased_policy(td_weight)
    raise ValueError(kind)


def summarize(history):
    rewards = [h["avg_reward"] for h in history]
    final = sum(rewards[-5:]) / min(5, len(rewards))
    best = max(rewards)
    return {"final_avg5": final, "best": best, "all": rewards}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n_iterations", type=int, default=20)
    # Per-method pretrain budgets — TS gets full convergence, DFL is held back.
    p.add_argument("--pretrain_epochs_ts", type=int, default=100,
                   help="CE pretrain epochs for two-stage (let it converge).")
    p.add_argument("--pretrain_epochs_dfl", type=int, default=30,
                   help="CE pretrain epochs for DFL (leave room for LP phase).")
    p.add_argument("--n_trajectories", type=int, default=100)
    p.add_argument("--n_features", type=int, default=128)
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--rho", type=float, default=10.0)
    p.add_argument("--behavior", type=str, default="biased",
                   choices=["expert", "noisy", "random",
                            "angle_only", "biased"])
    p.add_argument("--eps", type=float, default=0.5,
                   help="epsilon for noisy behavior policy")
    p.add_argument("--td_weight", type=float, default=0.1,
                   help="theta_dot weight for biased heuristic "
                        "(expert uses 0.5; smaller = more suboptimal)")
    p.add_argument("--eval_policy_type", type=str, default="ce",
                   choices=["ce", "lp"])
    p.add_argument("--n_ce_steps", type=int, default=0,
                   help="DAgger CE steps per iteration. Set to 0 in offline "
                        "mode to prevent collapse from LP-policy relabeling.")
    p.add_argument("--n_dfl_steps", type=int, default=5,
                   help="LP-DFL gradient steps per iteration.")
    p.add_argument("--measure_behavior", action="store_true",
                   help="Just evaluate the behavior policy and exit.")
    args = p.parse_args()

    cfg = CartPoleConfig(
        n_iterations=args.n_iterations,
        n_trajectories=args.n_trajectories,
        n_features=args.n_features,
        ridge=args.ridge,
        rho=args.rho,
    )

    if args.measure_behavior:
        set_seed(args.seed)
        dyn = CartPoleDynamics(cfg)
        behavior = make_behavior_policy(args.behavior, args.eps, args.td_weight)
        res = evaluate_policy(dyn, behavior, cfg.n_eval_episodes,
                              cfg.max_steps, cfg.gamma, torch.device("cpu"))
        print(f"[behavior eval] {args.behavior} "
              f"(eps={args.eps}, td_weight={args.td_weight}): "
              f"avg_reward={res['avg_reward']:.2f} "
              f"avg_length={res['avg_length']:.1f}")
        return

    print("\n" + "#" * 60)
    print(f"Behavior: {args.behavior}  eps={args.eps}  td_weight={args.td_weight}  "
          f"n_traj={args.n_trajectories}  n_feat={args.n_features}  "
          f"iters={args.n_iterations}  seed={args.seed}")
    print(f"Pretrain: TS={args.pretrain_epochs_ts}  DFL={args.pretrain_epochs_dfl}")
    print("#" * 60)

    # --- TWO-STAGE: full pretrain convergence ---
    set_seed(args.seed)
    behavior = make_behavior_policy(args.behavior, args.eps, args.td_weight)
    out_ts = run_two_stage(
        cfg, verbose=True,
        pretrain_epochs=args.pretrain_epochs_ts,
        eval_policy_type=args.eval_policy_type,
        behavior_policy=behavior,
        n_ce_steps=args.n_ce_steps,
        n_dfl_steps=args.n_dfl_steps,
    )
    ts = summarize(out_ts["history"])

    # --- DFL: shorter pretrain so the LP phase has room to refine ---
    set_seed(args.seed)  # identical init + identical offline buffer
    behavior = make_behavior_policy(args.behavior, args.eps, args.td_weight)
    out_dfl = run_neural_dfl(
        cfg, verbose=True,
        pretrain_epochs=args.pretrain_epochs_dfl,
        eval_policy_type=args.eval_policy_type,
        behavior_policy=behavior,
        n_ce_steps=args.n_ce_steps,
        n_dfl_steps=args.n_dfl_steps,
    )
    df = summarize(out_dfl["history"])

    print("\n" + "=" * 60)
    print("RESULT SUMMARY")
    print("=" * 60)
    print(f"Two-Stage:  final(avg of last 5) = {ts['final_avg5']:.2f}   "
          f"best = {ts['best']:.2f}")
    print(f"DFL:        final(avg of last 5) = {df['final_avg5']:.2f}   "
          f"best = {df['best']:.2f}")
    print(f"DFL - TwoStage gap (best):  {df['best'] - ts['best']:+.2f}")
    print(f"DFL - TwoStage gap (final): {df['final_avg5'] - ts['final_avg5']:+.2f}")


if __name__ == "__main__":
    main()
