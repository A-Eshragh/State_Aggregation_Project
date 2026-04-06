"""
Main training pipeline for continuous-state LP on CartPole.

Algorithms:
    1. RFF + single-shot LP
    2. RFF + iterative refinement (on-policy data collection)
    3. Neural features + DFL (end-to-end through differentiable LP)
"""

import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from .configs import CartPoleConfig
from .cartpole_env import CartPoleDynamics, collect_trajectories, random_policy, epsilon_greedy_policy
from .features import RandomFourierFeatures, NeuralFeatures
from .lp_matrices import estimate_lp_matrices, estimate_lp_matrices_batched
from .reduced_solver import solve_reduced_lp_gurobi, differentiable_reduced_lp
from .policy import (
    occupancy_to_policy_fn,
    occupancy_to_stochastic_policy_fn,
    evaluate_policy,
)


# ═══════════════════════════════════════════════════════════════════
# Algorithm 1: RFF + Single-Shot LP
# ═══════════════════════════════════════════════════════════════════

def run_rff_single_shot(cfg: CartPoleConfig, verbose: bool = True):
    """Collect data with random policy, solve LP once, evaluate.

    Steps:
        1. Sample initial states from rho
        2. Collect trajectories with random policy
        3. Build RFF features
        4. Estimate M, c, b
        5. Solve reduced LP
        6. Extract and evaluate policy
    """
    device = torch.device("cpu")
    dynamics = CartPoleDynamics(cfg)

    if verbose:
        print("=" * 60)
        print("Algorithm 1: RFF + Single-Shot LP")
        print("=" * 60)

    # ── Step 1-2: Collect data ──
    if verbose:
        print(f"Collecting {cfg.n_trajectories} episodes with random policy...")
    data = collect_trajectories(dynamics, random_policy, cfg.n_trajectories,
                                cfg.max_steps, device)
    N = data["states"].shape[0]
    if verbose:
        print(f"  Collected {N} transitions")

    # ── Step 3: Build features ──
    features = RandomFourierFeatures(
        state_dim=4, n_actions=2,
        n_features=cfg.n_features, bandwidth=cfg.rff_bandwidth,
    ).to(device)

    # ── Step 4: Estimate LP matrices ──
    if verbose:
        print("Estimating LP matrices...")
    with torch.no_grad():
        M, c, b = estimate_lp_matrices(features, data, gamma=cfg.gamma, ridge=cfg.ridge)
    if verbose:
        print(f"  M shape: {M.shape}, cond(M): {torch.linalg.cond(M):.2f}")

    # ── Step 5: Solve LP ──
    if verbose:
        print("Solving reduced LP...")
    w_star, basis, dual = solve_reduced_lp_gurobi(M, c, b)
    lp_value = (c @ w_star).item()
    if verbose:
        print(f"  LP value: {lp_value:.4f}")
        print(f"  w* stats: min={w_star.min():.4f}, max={w_star.max():.4f}, "
              f"nnz={int((w_star > 1e-8).sum())}/{cfg.n_features}")

    # ── Step 6: Evaluate ──
    policy_fn = occupancy_to_policy_fn(features, w_star)
    results = evaluate_policy(dynamics, policy_fn, cfg.n_eval_episodes,
                              cfg.max_steps, cfg.gamma, device)
    if verbose:
        print(f"  Avg reward:    {results['avg_reward']:.2f}")
        print(f"  Avg length:    {results['avg_length']:.1f}")
        print(f"  Avg disc. ret: {results['avg_discounted_return']:.4f}")

    # Random baseline
    rand_results = evaluate_policy(dynamics, random_policy, cfg.n_eval_episodes,
                                   cfg.max_steps, cfg.gamma, device)
    if verbose:
        print(f"  Random baseline reward: {rand_results['avg_reward']:.2f}, "
              f"length: {rand_results['avg_length']:.1f}")

    return {
        "w_star": w_star, "features": features,
        "results": results, "random_results": rand_results,
        "M": M, "c": c, "b": b,
    }


# ═══════════════════════════════════════════════════════════════════
# Algorithm 2: RFF + Iterative Refinement
# ═══════════════════════════════════════════════════════════════════

def run_rff_iterative(cfg: CartPoleConfig, verbose: bool = True):
    """Iteratively: collect on-policy data -> solve LP -> update policy.

    Each iteration:
        1. Collect data with current policy (eps-greedy)
        2. Re-estimate M, c, b
        3. Solve LP
        4. Update policy from w*
        5. Evaluate
    """
    device = torch.device("cpu")
    dynamics = CartPoleDynamics(cfg)

    if verbose:
        print("=" * 60)
        print("Algorithm 2: RFF + Iterative Refinement")
        print("=" * 60)

    features = RandomFourierFeatures(
        state_dim=4, n_actions=2,
        n_features=cfg.n_features, bandwidth=cfg.rff_bandwidth,
    ).to(device)

    # Start with random policy; maintain a replay buffer across iterations
    current_policy = random_policy
    history = []
    buffer = None

    for it in range(cfg.n_iterations):
        t0 = time.time()

        # ── Collect data with epsilon-greedy exploration ──
        explore_policy = epsilon_greedy_policy(current_policy, cfg.exploration_eps)
        new_data = collect_trajectories(dynamics, explore_policy, cfg.n_trajectories,
                                        cfg.max_steps, device)

        # Merge into replay buffer (keep bounded)
        if buffer is None:
            buffer = new_data
        else:
            for key in buffer:
                combined = torch.cat([buffer[key], new_data[key]], dim=0)
                buffer[key] = combined[-cfg.buffer_size:]

        # ── Estimate and solve ──
        with torch.no_grad():
            M, c, b = estimate_lp_matrices(features, buffer, gamma=cfg.gamma,
                                           ridge=cfg.ridge)

        try:
            w_star, _, _ = solve_reduced_lp_gurobi(M, c, b)
        except RuntimeError as e:
            if verbose:
                print(f"  Iter {it}: LP failed ({e}), keeping current policy")
            history.append(history[-1] if history else {"avg_reward": 0, "avg_length": 0})
            continue

        # ── Update policy ──
        current_policy = occupancy_to_policy_fn(features, w_star)

        # ── Evaluate ──
        results = evaluate_policy(dynamics, current_policy, cfg.n_eval_episodes,
                                  cfg.max_steps, cfg.gamma, device)
        elapsed = time.time() - t0
        history.append(results)

        if verbose:
            print(f"  Iter {it:3d} | reward={results['avg_reward']:.2f} | "
                  f"length={results['avg_length']:.1f} | "
                  f"cond(M)={torch.linalg.cond(M):.1f} | "
                  f"time={elapsed:.1f}s")

    return {"history": history, "w_star": w_star, "features": features}


# ═══════════════════════════════════════════════════════════════════
# Algorithm 3: Neural Features + DFL
# ═══════════════════════════════════════════════════════════════════

def run_neural_dfl(cfg: CartPoleConfig, verbose: bool = True):
    """End-to-end DFL: learn features xi by differentiating through LP.

    The self-consistent DFL loss from soft_state_agg_convergence.tex:
        L = -r_true^T y*(xi) + lambda/2 * ||flow_residual||^2

    where y(s,a) = sum_j w_j* phi_xi(s,a) and w* solves the reduced LP.

    Since LP gradient through c is zero (dead r-channel), the gradient
    flows only through M and b (which depend on xi). The self-consistent
    formulation provides the direct gradient term -dr_theta/dtheta^T y*
    that is continuous across basis transitions.
    """
    device = torch.device("cpu")
    dynamics = CartPoleDynamics(cfg)

    if verbose:
        print("=" * 60)
        print("Algorithm 3: Neural Features + DFL")
        print("=" * 60)

    features = NeuralFeatures(
        state_dim=4, n_actions=2,
        n_features=cfg.n_features,
        hidden_dims=cfg.neural_hidden,
    ).to(device)

    optimizer = torch.optim.Adam(features.parameters(), lr=cfg.lr)

    # ── Initial data collection ──
    if verbose:
        print("Collecting initial data with random policy...")
    data = collect_trajectories(dynamics, random_policy, cfg.n_trajectories,
                                cfg.max_steps, device)
    if verbose:
        print(f"  Collected {data['states'].shape[0]} transitions")

    history = []

    for epoch in range(cfg.n_epochs):
        t0 = time.time()
        optimizer.zero_grad()

        # ── Estimate LP matrices (differentiable w.r.t. xi) ──
        M, c, b = estimate_lp_matrices(features, data, gamma=cfg.gamma, ridge=cfg.ridge)

        # ── Solve LP (differentiable backward via basis sensitivity) ──
        try:
            w_star = differentiable_reduced_lp(M, c, b)  # (d,)
        except RuntimeError as e:
            if verbose:
                print(f"  Epoch {epoch}: LP failed ({e}), skipping")
            continue

        # ── Compute DFL loss ──
        # Occupancy at data points: y(s_t, a_t) = phi(s_t, a_t) @ w*
        states = data["states"]
        actions = data["actions"]
        rewards = data["rewards"]
        next_states = data["next_states"]
        dones = data["dones"]

        phi_sa = features.phi(states, actions)  # (N, d)
        y_at_data = phi_sa @ w_star  # (N,)

        # Self-consistent loss: -r_true^T y  (sample average)
        # r_true^T y ≈ (1/N) sum_t r_t * y(s_t, a_t)
        reward_value = (rewards * y_at_data).mean()
        loss_reward = -reward_value

        # Flow residual penalty: for each sampled s', check flow balance
        # flow(s') = sum_a y(s', a) - gamma * y(s, a) * p(s'|s,a)
        # Approximate via: psi(s') @ (M w* - b)  should be zero
        psi_b = features.psi(data["init_states"])  # (N_0, m)
        flow_residual = M @ w_star - b  # (m,)
        loss_flow = 0.5 * cfg.dfl_lambda * (flow_residual ** 2).sum()

        loss = loss_reward + loss_flow
        loss.backward()
        optimizer.step()

        elapsed = time.time() - t0

        # ── Evaluate periodically ──
        if epoch % 5 == 0 or epoch == cfg.n_epochs - 1:
            with torch.no_grad():
                policy_fn = occupancy_to_policy_fn(features, w_star.detach())
                results = evaluate_policy(dynamics, policy_fn, cfg.n_eval_episodes,
                                          cfg.max_steps, cfg.gamma, device)
            history.append({"epoch": epoch, **results})

            if verbose:
                print(f"  Epoch {epoch:3d} | loss={loss.item():.4f} "
                      f"(reward={loss_reward.item():.4f}, flow={loss_flow.item():.4f}) | "
                      f"eval_reward={results['avg_reward']:.2f} | "
                      f"length={results['avg_length']:.1f} | "
                      f"time={elapsed:.1f}s")

        # ── Periodically refresh data with current policy ──
        if (epoch + 1) % 20 == 0 and epoch < cfg.n_epochs - 1:
            with torch.no_grad():
                policy_fn = occupancy_to_policy_fn(features, w_star.detach())
            explore = epsilon_greedy_policy(policy_fn, cfg.exploration_eps)
            new_data = collect_trajectories(dynamics, explore,
                                            cfg.n_trajectories // 2,
                                            cfg.max_steps, device)
            # Merge with old data (keep buffer bounded)
            for key in data:
                combined = torch.cat([data[key], new_data[key]], dim=0)
                data[key] = combined[-cfg.buffer_size:]
            if verbose:
                print(f"  [Data refresh] Buffer size: {data['states'].shape[0]}")

    return {"history": history, "features": features}


# ═══════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════

def plot_learning_curves(results_dict: dict, save_path: str = None):
    """Plot learning curves for all algorithms.

    Args:
        results_dict: {name: list of eval dicts with 'avg_reward'}
        save_path: optional file path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, history in results_dict.items():
        if isinstance(history, list) and len(history) > 0:
            if "epoch" in history[0]:
                xs = [h["epoch"] for h in history]
            else:
                xs = list(range(len(history)))
            rewards = [h["avg_reward"] for h in history]
            lengths = [h["avg_length"] for h in history]
            ax1.plot(xs, rewards, label=name, marker="o", markersize=3)
            ax2.plot(xs, lengths, label=name, marker="o", markersize=3)

    ax1.set_xlabel("Iteration / Epoch")
    ax1.set_ylabel("Average Episode Reward")
    ax1.set_title("Learning Curves — Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Iteration / Epoch")
    ax2.set_ylabel("Average Episode Length")
    ax2.set_title("Learning Curves — Episode Length")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Continuous MDP LP on CartPole")
    p.add_argument("--algo", type=str, default="all",
                   choices=["rff_single", "rff_iter", "neural_dfl", "all"])
    p.add_argument("--n_features", type=int, default=128)
    p.add_argument("--n_trajectories", type=int, default=500)
    p.add_argument("--n_iterations", type=int, default=20)
    p.add_argument("--n_epochs", type=int, default=100)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--bandwidth", type=float, default=0.5)
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dfl_lambda", type=float, default=1.0)
    p.add_argument("--reward_type", type=str, default="cos_theta",
                   choices=["cos_theta", "constant"])
    p.add_argument("--save_plot", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    cfg = CartPoleConfig(
        n_features=args.n_features,
        n_trajectories=args.n_trajectories,
        n_iterations=args.n_iterations,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        rff_bandwidth=args.bandwidth,
        ridge=args.ridge,
        lr=args.lr,
        dfl_lambda=args.dfl_lambda,
        reward_type=args.reward_type,
    )

    all_results = {}

    if args.algo in ("rff_single", "all"):
        out = run_rff_single_shot(cfg)
        all_results["RFF Single-Shot"] = [out["results"]]

    if args.algo in ("rff_iter", "all"):
        out = run_rff_iterative(cfg)
        all_results["RFF Iterative"] = out["history"]

    if args.algo in ("neural_dfl", "all"):
        out = run_neural_dfl(cfg)
        all_results["Neural DFL"] = out["history"]

    if len(all_results) > 1:
        plot_learning_curves(all_results, save_path=args.save_plot)


if __name__ == "__main__":
    main()
