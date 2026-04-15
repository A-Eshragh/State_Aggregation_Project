"""
Main training pipeline for continuous-state LP on CartPole.

Neural features + DFL (end-to-end through differentiable LP).
"""

import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from .configs import CartPoleConfig
from .cartpole_env import CartPoleDynamics, collect_trajectories, random_policy, epsilon_greedy_policy
from .features import ActionIndexedFeatures
from .lp_matrices import estimate_lp_matrices
from .reduced_solver import (
    solve_reduced_qp_gurobi, differentiable_reduced_qp,
)
from .policy import (
    occupancy_to_policy_fn,
    evaluate_policy,
)


# ═══════════════════════════════════════════════════════════════════
# Pretraining (warm-start for DFL)
# ═══════════════════════════════════════════════════════════════════

def pretrain_features(features, buffer, cfg, pretrain_epochs=50, verbose=True):
    """Pretrain features by minimizing the LP objective with high QP regularization.

    Runs the same DFL objective (c^T w*) as the main loop but with a large
    fixed eta, giving a stable warm-start that is fully consistent with the
    LP formulation.  No Bellman bootstrapping.
    """
    lr = 1e-3
    eta_pretrain = 1.0  # moderate eta for good gradient flow during pretrain

    optimizer = torch.optim.Adam(features.parameters(), lr=lr)

    batch_size = 4096
    N = buffer["states"].shape[0]

    for epoch in range(pretrain_epochs):
        optimizer.zero_grad()

        # Sub-sample for speed when buffer is large
        if N > batch_size:
            idx = torch.randperm(N)[:batch_size]
            batch = {
                "states": buffer["states"][idx],
                "actions": buffer["actions"][idx],
                "rewards": buffer["rewards"][idx],
                "next_states": buffer["next_states"][idx],
                "dones": buffer["dones"][idx],
                "init_states": buffer["init_states"],
            }
        else:
            batch = buffer

        M, c, b = estimate_lp_matrices(
            features, batch, gamma=cfg.gamma, ridge=cfg.ridge)

        w_star = differentiable_reduced_qp(M, c, b, eta=eta_pretrain)

        loss = c @ w_star
        loss.backward()
        torch.nn.utils.clip_grad_norm_(features.parameters(), max_norm=1.0)
        optimizer.step()

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  [Pretrain] Epoch {epoch+1}/{pretrain_epochs}, "
                  f"lp_obj: {loss.item():.4f}")

    return features


# ═══════════════════════════════════════════════════════════════════
# Neural Features + DFL
# ═══════════════════════════════════════════════════════════════════

def run_neural_dfl(cfg: CartPoleConfig, verbose: bool = True):
    """Neural features + DFL (end-to-end through differentiable QP).

    Proper DFL bilevel formulation (continuous_mdp.pdf, eq 23):

      Inner:  w*_θ = argmin_{w≥0} { c_θ^T w + (η/2)||w||² : M_θ w = b_θ }
      Outer:  F(θ) = c_true^T w*_θ  +  λ ||M_true w*_θ - b_true||²

    The inner LP uses PREDICTED matrices (from current features θ).
    The outer loss evaluates w* on FIXED "true" matrices (from buffer, detached)
    — reward fidelity + constraint violation.  Matches the discrete DFL in
    mdp_lp/ where the LP is solved with predicted params but loss evaluates
    on the true MDP.
    """
    device = torch.device("cpu")
    dynamics = CartPoleDynamics(cfg)

    if verbose:
        print("=" * 60)
        print("Neural Features + DFL")
        print("=" * 60)

    features = ActionIndexedFeatures(
        state_dim=4, n_actions=2,
        n_features=cfg.n_features,
        hidden_dims=cfg.neural_hidden,
    ).to(device)

    # ── Initial data collection with random policy ──
    if verbose:
        print("Collecting initial data with random policy...")
    data = collect_trajectories(dynamics, random_policy, cfg.n_trajectories,
                                cfg.max_steps, device)
    buffer = {k: v.clone() for k, v in data.items()}

    if verbose:
        print(f"  Collected {buffer['states'].shape[0]} transitions")
        print(f"  Feature dims: d={cfg.n_features}, m={features.n_test_features} "
              f"(LP has {cfg.n_features - features.n_test_features} degrees of freedom)")

    # ── Pretrain features (warm-start before DFL) ──
    if verbose:
        print("Pretraining features...")
    pretrain_features(features, buffer, cfg, pretrain_epochs=10, verbose=verbose)

    optimizer = torch.optim.Adam(features.parameters(), lr=cfg.lr)

    current_policy = random_policy
    history = []
    batch_size = 4096
    dfl_lambda = cfg.dfl_lambda  # penalty weight for constraint violation

    for iteration in range(cfg.n_iterations):
        t0 = time.time()

        # ── Step 1: Collect on-policy data ──
        if iteration > 0:
            explore_policy = epsilon_greedy_policy(current_policy, cfg.exploration_eps)
            new_data = collect_trajectories(
                dynamics, explore_policy,
                max(cfg.n_trajectories // 4, 50),
                cfg.max_steps, device,
            )
            for key in buffer:
                if key in new_data:
                    combined = torch.cat([buffer[key], new_data[key]], dim=0)
                    buffer[key] = combined[-cfg.buffer_size:]

        N = buffer["states"].shape[0]
        eta = max(cfg.qp_eta, 1.0 * (0.95 ** iteration))
        last_loss = float("nan")

        # ── Step 2: DFL training ──
        for epoch in range(cfg.n_epochs):
            optimizer.zero_grad()

            # Train/eval split: two independent batches from buffer
            perm = torch.randperm(N)
            half = min(batch_size, N // 2)
            idx_train = perm[:half]
            idx_eval = perm[half:half * 2]

            def make_batch(indices):
                return {
                    "states": buffer["states"][indices],
                    "actions": buffer["actions"][indices],
                    "rewards": buffer["rewards"][indices],
                    "next_states": buffer["next_states"][indices],
                    "dones": buffer["dones"][indices],
                    "init_states": buffer["init_states"],
                }

            # Inner: solve LP with train batch (predicted)
            M_pred, c_pred, b_pred = estimate_lp_matrices(
                features, make_batch(idx_train),
                gamma=cfg.gamma, ridge=cfg.ridge)
            w_star = differentiable_reduced_qp(M_pred, c_pred, b_pred, eta=eta)

            # Outer: evaluate w* on eval batch (with gradients through features)
            M_eval, c_eval, b_eval = estimate_lp_matrices(
                features, make_batch(idx_eval),
                gamma=cfg.gamma, ridge=cfg.ridge)
            reward_loss = c_eval @ w_star
            violation = M_eval @ w_star - b_eval
            loss = reward_loss + dfl_lambda * violation.pow(2).sum()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(features.parameters(), max_norm=1.0)
            optimizer.step()
            last_loss = loss.item()

        # ── Step 3: Policy extraction and evaluation ──
        with torch.no_grad():
            M_eval, c_eval, b_eval = estimate_lp_matrices(
                features, buffer, gamma=cfg.gamma, ridge=cfg.ridge)
        try:
            w_lp, _, _ = solve_reduced_qp_gurobi(
                M_eval, c_eval, b_eval, eta=eta)
        except RuntimeError:
            w_lp = torch.zeros(cfg.n_features)

        current_policy = occupancy_to_policy_fn(features, w_lp)

        results = evaluate_policy(
            dynamics, current_policy, cfg.n_eval_episodes,
            cfg.max_steps, cfg.gamma, device)
        elapsed = time.time() - t0
        history.append({"iteration": iteration, **results})

        if verbose:
            w_nnz = int((w_lp > 1e-8).sum())
            print(
                f"  Iter {iteration:3d} | reward={results['avg_reward']:.2f} | "
                f"length={results['avg_length']:.1f} | "
                f"lp_obj={last_loss:.4f} | "
                f"nnz={w_nnz}/{cfg.n_features} | "
                f"buf={N} | time={elapsed:.1f}s"
            )

    return {"history": history, "features": features}


# ═══════════════════════════════════════════════════════════════════
# Soft State Aggregation with True Dynamics
# ═══════════════════════════════════════════════════════════════════

def run_soft_agg_dfl(cfg: CartPoleConfig, n_clusters: int = 32,
                     bins_per_dim: int = 8, n_iterations: int = 100,
                     lr: float = 1e-3, tau: float = 0.5,
                     verbose: bool = True):
    """Soft state aggregation for CartPole using TRUE dynamics.

    Mirrors mdp_lp/soft_state_aggregation.py:
      1. Discretize state space into a grid of S = bins_per_dim^4 states
      2. Compute TRUE P[s,a,s'] and R[s,a] on the grid (nothing estimated)
      3. Learn soft memberships W[s,m] via NN (softmax over M clusters)
      4. Aggregate P, R, gamma into an M-cluster MDP
      5. Solve aggregated LP differentiably (M*A variables, M constraints)
      6. DFL loss = c^T y* → backprop to membership NN
      7. Policy: for any continuous state s, use W(s) to weight y_agg
    """
    from .baselines import DiscretizedCartPole

    device = torch.device("cpu")
    dynamics = CartPoleDynamics(cfg)
    A = 2
    M = n_clusters
    gamma_disc = cfg.gamma

    if verbose:
        print("=" * 60)
        print(f"Soft State Aggregation (M={M} clusters, "
              f"{bins_per_dim} bins/dim = {bins_per_dim**4} grid states)")
        print("=" * 60)

    # ── Step 1: Discretize and compute TRUE dynamics ──
    if verbose:
        print("Building true P and R on grid...")
    disc = DiscretizedCartPole(cfg, bins_per_dim)
    S = disc.n_states
    P_np, R_np = disc.build_transition_and_reward()
    P = torch.tensor(P_np, dtype=torch.float32, device=device)  # (S, A, S)
    R = torch.tensor(R_np, dtype=torch.float32, device=device)  # (S, A)

    # All grid states as tensor
    all_states = torch.zeros(S, 4, device=device)
    for s_idx in range(S):
        all_states[s_idx] = torch.tensor(disc.index_to_state(s_idx),
                                         dtype=torch.float32)

    # Zero out reward for absorbing states (self-loop in P).
    # Without this, the LP exploits absorbing states for infinite reward.
    is_absorbing = torch.zeros(S, dtype=torch.bool, device=device)
    for a in range(A):
        is_absorbing |= (P[torch.arange(S), a, torch.arange(S)] > 0.99)
    R[is_absorbing] = 0.0

    # State scales for normalization
    state_scales = torch.tensor(
        [cfg.x_max, cfg.x_dot_max, cfg.theta_max, cfg.theta_dot_max],
        device=device)

    # Initial distribution: Gaussian centered at origin
    norm_states = all_states / state_scales
    dists = (norm_states ** 2).sum(dim=1)  # squared distance from origin
    gamma_init = torch.exp(-dists / 0.1)   # concentrated near origin
    gamma_init = gamma_init / gamma_init.sum()

    if verbose:
        n_init = int((gamma_init > 1e-4).sum())
        print(f"  Grid: {S} states, ~{n_init} effective initial states, "
              f"{int(is_absorbing.sum())} absorbing")

    # ── Step 2: Membership network ──
    membership_net = torch.nn.Sequential(
        torch.nn.Linear(4, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, M),
    ).to(device)

    optimizer = torch.optim.Adam(membership_net.parameters(), lr=lr)
    history = []

    for iteration in range(n_iterations):
        t0 = time.time()
        optimizer.zero_grad()

        # ── Step 3: Compute memberships ──
        norm_states = all_states / state_scales
        logits = membership_net(norm_states)          # (S, M)
        W = torch.softmax(logits / tau, dim=1)        # (S, M)
        mass = W.sum(dim=0).clamp_min(1e-8)           # (M,)

        # ── Step 4: Aggregate using TRUE dynamics ──
        # r_agg[m, a] = sum_s W[s,m] * R[s,a] / mass[m]
        r_agg = torch.einsum('sm,sa->ma', W, R) / mass.unsqueeze(-1)

        # right[s, a, m'] = sum_{s'} P[s,a,s'] * W[s',m']
        right = torch.einsum('sap,pm->sam', P, W)     # (S, A, M)
        # P_agg[m, a, m'] = sum_s W[s,m] * right[s,a,m'] / mass[m]
        P_agg = torch.einsum('sm,san->man', W, right) / mass[:, None, None]

        # Normalize rows for numerical stability
        P_agg = P_agg.clamp_min(0.0)
        P_agg = P_agg / P_agg.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        # gamma_agg[m] = sum_s gamma[s] * W[s,m]
        gamma_agg = gamma_init @ W                     # (M,)
        gamma_agg = gamma_agg / gamma_agg.sum().clamp_min(1e-8)

        # ── Step 5: Build and solve aggregated LP ──
        # H[m, m'*A + a] = delta[m,m'] - gamma * P_agg[m', a, m]
        H = torch.zeros(M, M * A, device=device)
        for a in range(A):
            # Identity part: H[m, m*A+a] += 1
            H[torch.arange(M), torch.arange(M) * A + a] += 1.0
            # Transition part: H[m, m'*A+a] -= gamma * P_agg[m', a, m]
            H[:, a::A] -= gamma_disc * P_agg[:, a, :].T

        c_agg = -r_agg.reshape(-1)  # (M*A,) negative reward for minimization

        # Solve aggregated QP (inner problem with predicted/aggregated params)
        eta = max(cfg.qp_eta, 1.0 * (0.95 ** iteration))
        y_agg = differentiable_reduced_qp(H, c_agg, gamma_agg, eta=eta)

        # ── Outer DFL loss: evaluate y_agg on TRUE (non-aggregated) MDP ──
        # Disaggregate: y_orig[s, a] = sum_m W[s, m] * y_agg[m, a]
        y_agg_2d = y_agg.reshape(M, A)                # (M, A)
        y_orig = W @ y_agg_2d                          # (S, A)

        # True reward: -sum_{s,a} R[s,a] * y_orig[s,a]  (minimize negative reward)
        reward_loss = -(R * y_orig).sum()

        # True flow constraint violation:
        #   sum_a y(s,a) - gamma * sum_{s',a} P[s',a,s] * y(s',a) = rho(s)
        outflow = y_orig.sum(dim=1)                    # (S,)
        inflow = torch.zeros(S, device=device)
        for a in range(A):
            inflow += P[:, a, :].T @ y_orig[:, a]      # (S,)
        violation = outflow - gamma_disc * inflow - gamma_init
        constraint_loss = violation.pow(2).sum()

        loss = reward_loss + cfg.dfl_lambda * constraint_loss
        loss.backward()
        grad_norm = sum(p.grad.norm().item() for p in membership_net.parameters()
                        if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(membership_net.parameters(), max_norm=1.0)
        optimizer.step()

        # ── Step 6: Extract policy and evaluate ──
        with torch.no_grad():
            y_agg_2d_eval = y_agg.detach().reshape(M, A)  # (M, A)

        def make_policy(net, scales, y2d, t):
            """Create a policy function that uses memberships + LP solution."""
            _net = net
            _scales = scales
            _y2d = y2d.clone()
            _tau = t
            def policy_fn(states):
                with torch.no_grad():
                    ns = states / _scales
                    w = torch.softmax(_net(ns) / _tau, dim=1)  # (B, M)
                    y_per_action = w @ _y2d                     # (B, A)
                    return y_per_action.argmax(dim=1)
            return policy_fn

        current_policy = make_policy(membership_net, state_scales, y_agg_2d_eval, tau)
        results = evaluate_policy(
            dynamics, current_policy, cfg.n_eval_episodes,
            cfg.max_steps, cfg.gamma, device)
        elapsed = time.time() - t0
        history.append({"iteration": iteration, **results})

        if verbose:
            y_nnz = int((y_agg > 1e-8).sum())
            print(
                f"  Iter {iteration:3d} | reward={results['avg_reward']:.2f} | "
                f"length={results['avg_length']:.1f} | "
                f"lp_obj={loss.item():.4f} | "
                f"nnz={y_nnz}/{M*A} | "
                f"grad={grad_norm:.4f} | "
                f"time={elapsed:.1f}s"
            )

    return {"history": history, "membership_net": membership_net}


# ═══════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════

def plot_learning_curves(history: list, save_path: str = None):
    """Plot learning curves.

    Args:
        history: list of eval dicts with 'avg_reward' and 'avg_length'
        save_path: optional file path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    xs = list(range(len(history)))
    rewards = [h["avg_reward"] for h in history]
    lengths = [h["avg_length"] for h in history]
    ax1.plot(xs, rewards, marker="o", markersize=3)
    ax2.plot(xs, lengths, marker="o", markersize=3)

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Average Episode Reward")
    ax1.set_title("Learning Curve — Reward")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Average Episode Length")
    ax2.set_title("Learning Curve — Episode Length")
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
    p = argparse.ArgumentParser(description="Continuous MDP DFL on CartPole")
    p.add_argument("--method", type=str, default="soft_agg",
                   choices=["neural_dfl", "soft_agg"],
                   help="Training method")
    # Shared
    p.add_argument("--n_iterations", type=int, default=100)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--qp_eta", type=float, default=0.01)
    p.add_argument("--reward_type", type=str, default="cos_theta",
                   choices=["cos_theta", "constant"])
    p.add_argument("--save_plot", type=str, default=None)
    # Neural DFL
    p.add_argument("--n_features", type=int, default=128)
    p.add_argument("--n_trajectories", type=int, default=500)
    p.add_argument("--n_epochs", type=int, default=100)
    p.add_argument("--ridge", type=float, default=1e-4)
    # Soft aggregation
    p.add_argument("--n_clusters", type=int, default=32)
    p.add_argument("--bins_per_dim", type=int, default=8)
    p.add_argument("--tau", type=float, default=0.5,
                   help="Softmax temperature for membership")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = CartPoleConfig(
        n_iterations=args.n_iterations,
        gamma=args.gamma,
        lr=args.lr,
        qp_eta=args.qp_eta,
        reward_type=args.reward_type,
        n_features=args.n_features,
        n_trajectories=args.n_trajectories,
        n_epochs=args.n_epochs,
        ridge=args.ridge,
    )

    if args.method == "soft_agg":
        out = run_soft_agg_dfl(
            cfg, n_clusters=args.n_clusters,
            bins_per_dim=args.bins_per_dim,
            n_iterations=args.n_iterations,
            lr=args.lr, tau=args.tau)
    else:
        out = run_neural_dfl(cfg)

    plot_learning_curves(out["history"], save_path=args.save_plot)


if __name__ == "__main__":
    main()
