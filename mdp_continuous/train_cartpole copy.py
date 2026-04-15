"""
Main training pipeline for continuous-state LP on CartPole.

Algorithms:
    1. RFF + single-shot LP
    2. RFF + iterative refinement (on-policy data collection)
    3. Neural features + DFL (end-to-end through differentiable LP)
"""

import argparse
import copy
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from .configs import CartPoleConfig
from .cartpole_env import CartPoleDynamics, collect_trajectories, random_policy, epsilon_greedy_policy
from .features import RandomFourierFeatures, NeuralFeatures, ActionIndexedFeatures
from .lp_matrices import estimate_lp_matrices, estimate_lp_matrices_batched
from .reduced_solver import (
    solve_reduced_lp_gurobi, differentiable_reduced_lp,
    solve_reduced_qp_gurobi, differentiable_reduced_qp,
)
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
        n_test_features=cfg.n_test_features,
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
        n_test_features=cfg.n_test_features,
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
    """Neural features trained with fitted Q-iteration + LP policy extraction.

    The LP/QP with learned features faces two problems:
      1. Too few features (128) → LP can't approximate optimal occupancy
         (fixed 1024 RFF features give reward 180, but 128 gives ~17).
      2. DFL loss inflates values without improving policy.

    This algorithm learns features via fitted Q-iteration (Bellman residual)
    which directly teaches the features to represent Q-values.  The
    action-indexed structure  φ(s,a=0)=[f(s);0], φ(s,a=1)=[0;f(s)]  gives
    Q(s,a) = f(s)^T w_a,  preserving the LP structure.  After training,
    the LP/QP can optionally extract an occupancy-constrained policy.

    Q-learning uses semi-gradient TD: targets are stop-gradient.
    """
    device = torch.device("cpu")
    dynamics = CartPoleDynamics(cfg)

    if verbose:
        print("=" * 60)
        print("Algorithm 3: Neural Features + DFL")
        print("=" * 60)

    features = ActionIndexedFeatures(
        state_dim=4, n_actions=2,
        n_features=cfg.n_features,
        hidden_dims=cfg.neural_hidden,
    ).to(device)

    # Learnable Q-weights — asymmetric init to break action symmetry
    w_init = torch.randn(cfg.n_features) * 0.1
    w_param = torch.nn.Parameter(w_init)
    optimizer = torch.optim.Adam(
        list(features.parameters()) + [w_param], lr=cfg.lr,
    )

    # Target network — periodic hard copy during training
    target_features = copy.deepcopy(features)
    target_features.eval()
    target_w = w_param.detach().clone()

    # ── Initial data collection with random policy ──
    if verbose:
        print("Collecting initial data with random policy...")
    data = collect_trajectories(dynamics, random_policy, cfg.n_trajectories,
                                cfg.max_steps, device)
    buffer = {k_: v.clone() for k_, v in data.items()}

    if verbose:
        print(f"  Collected {buffer['states'].shape[0]} transitions")
        print(f"  Feature dims: d={cfg.n_features}, m={features.n_test_features} "
              f"(LP has {cfg.n_features - features.n_test_features} degrees of freedom)")

    current_policy = random_policy
    history = []
    n_inner = max(200, cfg.n_epochs)   # many mini-batch gradient steps per iter
    n_actions = 2
    batch_size = 256
    target_update_period = max(50, n_inner // 4)
    best_reward = -float("inf")
    best_state = None

    for iteration in range(cfg.n_iterations):
        t0 = time.time()

        # ── Phase 1: Collect on-policy data ──
        if iteration > 0:
            explore_policy = epsilon_greedy_policy(current_policy, cfg.exploration_eps)
            new_data = collect_trajectories(
                dynamics, explore_policy,
                max(cfg.n_trajectories // 4, 50),
                cfg.max_steps, device,
            )
            for key in buffer:
                combined = torch.cat([buffer[key], new_data[key]], dim=0)
                buffer[key] = combined[-cfg.buffer_size:]

        # ── Phase 2: Fitted Q-iteration (mini-batch DQN) ──
        all_s = buffer["states"]
        all_a = buffer["actions"]
        all_r = buffer["rewards"]
        all_ns = buffer["next_states"]
        all_d = buffer["dones"]
        N = all_s.shape[0]
        bs = min(batch_size, N)

        last_loss = float("nan")
        for inner in range(n_inner):
            idx = torch.randint(0, N, (bs,))
            optimizer.zero_grad()

            Q_pred = features.phi(all_s[idx], all_a[idx]) @ w_param

            with torch.no_grad():
                Q_next = []
                for a in range(n_actions):
                    a_vec = torch.full((bs,), a, device=device, dtype=torch.long)
                    Q_next.append(target_features.phi(all_ns[idx], a_vec) @ target_w)
                Q_next_max = torch.stack(Q_next, dim=1).max(dim=1)[0]
                not_done = (~all_d[idx]).float()
                targets = all_r[idx] + cfg.gamma * not_done * Q_next_max

            loss = torch.nn.functional.smooth_l1_loss(Q_pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(features.parameters()) + [w_param], max_norm=1.0,
            )
            optimizer.step()
            last_loss = loss.item()

            # Periodic hard-copy target network
            if (inner + 1) % target_update_period == 0:
                target_features.load_state_dict(features.state_dict())
                target_w = w_param.detach().clone()

        # ── Phase 3: Sync target network ──
        target_features.load_state_dict(features.state_dict())
        target_w = w_param.detach().clone()

        # ── Phase 4: Extract policy ──
        # Use the learned Q-weights directly (greedy policy from Q-function)
        current_policy = occupancy_to_policy_fn(features, w_param.detach())

        # ── Phase 5: Evaluate and checkpoint best model ──
        results = evaluate_policy(
            dynamics, current_policy, cfg.n_eval_episodes,
            cfg.max_steps, cfg.gamma, device,
        )
        elapsed = time.time() - t0
        history.append({"iteration": iteration, **results})

        if results["avg_reward"] > best_reward:
            best_reward = results["avg_reward"]
            best_state = {
                "features": copy.deepcopy(features.state_dict()),
                "w": w_param.detach().clone(),
            }
        elif (best_state is not None
              and best_reward > 30
              and results["avg_reward"] < best_reward * 0.3):
            # Restore best model when performance degrades significantly
            features.load_state_dict(best_state["features"])
            w_param.data.copy_(best_state["w"])
            target_features.load_state_dict(features.state_dict())
            target_w = w_param.detach().clone()
            current_policy = occupancy_to_policy_fn(features, w_param.detach())
            # Reduce learning rate to stabilize around best policy
            for pg in optimizer.param_groups:
                pg["lr"] *= 0.5
            if verbose:
                new_lr = optimizer.param_groups[0]["lr"]
                print(f"         ↳ restored best (reward={best_reward:.1f}), "
                      f"lr→{new_lr:.1e}")

        if verbose:
            print(
                f"  Iter {iteration:3d} | reward={results['avg_reward']:.2f} | "
                f"length={results['avg_length']:.1f} | "
                f"bellman={last_loss:.4f} | "
                f"buf={buffer['states'].shape[0]} | time={elapsed:.1f}s"
            )

    return {"history": history, "features": features, "w": w_param.detach()}


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
    p.add_argument("--n_test_features", type=int, default=None,
                   help="Galerkin test dim m < d (default: n_features // 2)")
    p.add_argument("--n_trajectories", type=int, default=500)
    p.add_argument("--n_iterations", type=int, default=20)
    p.add_argument("--n_epochs", type=int, default=100)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--bandwidth", type=float, default=0.5)
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dfl_lambda", type=float, default=0.1)
    p.add_argument("--qp_eta", type=float, default=0.01,
                   help="QP regularization (eta/2)||w||^2 to spread LP weight")
    p.add_argument("--reward_type", type=str, default="cos_theta",
                   choices=["cos_theta", "constant"])
    p.add_argument("--save_plot", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    cfg = CartPoleConfig(
        n_features=args.n_features,
        n_test_features=args.n_test_features,
        n_trajectories=args.n_trajectories,
        n_iterations=args.n_iterations,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        rff_bandwidth=args.bandwidth,
        ridge=args.ridge,
        lr=args.lr,
        dfl_lambda=args.dfl_lambda,
        qp_eta=args.qp_eta,
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
