"""
DFL training for Swimmer with continuous 2D actions.

LP occupancy:  min c^T w  s.t. Mw = b, w >= 0
Policy:        a(s) = f(s)^T w_reg  (BC regressor head, refined by DFL)
DFL loss:      c_true^T w* + rho * ||b - M_true @ w*||^2
"""

import argparse
import random
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from .features import ContinuousActionFeatures
from .lp_matrices_multidim import estimate_lp_matrices_multidim
from .reduced_solver import differentiable_reduced_lp, differentiable_reduced_qp
from .swimmer_env import (
    SwimmerDynamics, collect_trajectories, evaluate_policy,
    random_policy, oscillating_policy,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def exploration_policy(base_policy_fn, noise_std=0.3, a_low=-1.0, a_high=1.0):
    def policy_fn(states):
        a = base_policy_fn(states)
        noise = torch.randn_like(a) * noise_std
        return (a + noise).clamp(a_low, a_high)
    return policy_fn


def run_swimmer_dfl(
    n_features: int = 256,
    n_action_bases_per_dim: int = 5,  # 5x5 = 25 bases for 2D action
    n_iterations: int = 30,
    n_trajectories: int = 50,
    pretrain_epochs: int = 300,
    max_steps: int = 1000,
    gamma: float = 0.99,
    lr: float = 1e-3,
    ridge: float = 1e-3,
    rho: float = 1.0,
    tau: float = 0.5,
    n_quad_per_dim: int = 6,
    buffer_size: int = 50000,
    dfl_weight: float = 0.01,
    qp_eta: float = 0.1,
    verbose: bool = True,
):
    device = torch.device("cpu")
    dynamics = SwimmerDynamics()

    # For 2D action, n_action_bases becomes n_per_dim^2
    n_action_bases_total = n_action_bases_per_dim ** 2
    d_per_action = max(n_features // n_action_bases_total, 4)

    if verbose:
        print("=" * 60)
        print(f"Swimmer DFL (n_bases={n_action_bases_total} ({n_action_bases_per_dim}x{n_action_bases_per_dim}), "
              f"d_per_action={d_per_action})")
        print("=" * 60)

    # Softmax features: bounded (sum to 1) → LP stays bounded.
    # Tradeoff vs softplus: less expressive for linear controls, but safer.
    features = ContinuousActionFeatures(
        state_dim=dynamics.state_dim,
        action_dim=dynamics.action_dim,
        n_action_bases=n_action_bases_total,
        d_per_action=d_per_action,
        hidden_dims=[128, 128],
        state_scales=dynamics.state_scales,
        action_low=dynamics.action_low,
        action_high=dynamics.action_high,
        tau_s=tau,
        state_features_type="softmax",
    ).to(device)

    # After tensor-product grid, actual n_bases may change slightly
    actual_n_bases = features.n_action_bases
    k = features.d_per_action
    d_actual = features.n_features

    if verbose:
        print(f"  Actual features: {actual_n_bases} bases x {k} per-action = {d_actual}")

    # ── Phase 1: Collect exploration data + pretrain ──
    if verbose:
        print("Phase 1: Collecting exploration data + BC pretraining...")

    # Perturbed expert + relabel (DAgger pattern from InvertedPendulum).
    # Oscillating expert reaches ~200 reward; perturbation gives diverse
    # states; relabel with deterministic expert for clean BC targets.
    perturbed_pol = exploration_policy(
        oscillating_policy, noise_std=0.3,
        a_low=dynamics.action_low, a_high=dynamics.action_high)
    raw_data = collect_trajectories(
        dynamics, perturbed_pol, n_trajectories, max_steps, device)
    with torch.no_grad():
        raw_data["actions"] = oscillating_policy(raw_data["states"])

    # Add pure expert data for LP cost estimation
    expert_data = collect_trajectories(
        dynamics, oscillating_policy, n_trajectories // 2, max_steps, device)

    buffer = {
        key: torch.cat([raw_data[key], expert_data[key]], dim=0)
        for key in raw_data if key != "init_states"
    }
    buffer["init_states"] = torch.cat(
        [raw_data["init_states"], expert_data["init_states"]], dim=0)

    pretrain_buffer = {kk: v.clone() for kk, v in buffer.items()}
    N = pretrain_buffer["states"].shape[0]

    if verbose:
        print(f"  Total transitions: {N}")
        print(f"  Reward stats: mean={buffer['rewards'].mean():.3f}, "
              f"std={buffer['rewards'].std():.3f}, "
              f"range=[{buffer['rewards'].min():.3f}, {buffer['rewards'].max():.3f}]")

    # BC pretraining: predict expert actions from states.
    w_reg = (torch.randn(k, dynamics.action_dim, device=device) * 0.01
             ).requires_grad_(True)
    feat_optimizer = torch.optim.Adam(
        list(features.parameters()) + [w_reg], lr=lr)

    batch_size = 2048

    for epoch in range(pretrain_epochs):
        feat_optimizer.zero_grad()
        idx = torch.randperm(N)[:batch_size]
        f_s = features._state_features(pretrain_buffer["states"][idx])
        a_pred = f_s @ w_reg
        loss = (a_pred - pretrain_buffer["actions"][idx]).pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(features.parameters(), max_norm=1.0)
        feat_optimizer.step()

        if verbose and (epoch + 1) % 100 == 0:
            print(f"  [Pretrain] Epoch {epoch+1}/{pretrain_epochs}, mse={loss.item():.4f}")

    # ── Phase 2: DFL iterations ──
    if verbose:
        print("Phase 2: DFL iterations...")

    joint_optimizer = torch.optim.Adam(
        list(features.parameters()) + [w_reg], lr=lr * 0.5)

    def bc_policy(states, f=features, w=w_reg,
                  a_lo=dynamics.action_low, a_hi=dynamics.action_high):
        with torch.no_grad():
            f_s = f._state_features(states)
            a = f_s @ w
            return a.clamp(a_lo, a_hi)

    current_policy = bc_policy

    ws = evaluate_policy(dynamics, current_policy, 20, max_steps, gamma, device)
    if verbose:
        print(f"  Warm-start reward={ws['avg_reward']:.2f}")

    best_reward = -float("inf")
    best_state = {
        "features": {k_: v.clone() for k_, v in features.state_dict().items()},
        "w_reg": w_reg.detach().clone(),
    }

    history = []
    for iteration in range(n_iterations):
        t0 = time.time()

        # Evaluate
        results = evaluate_policy(
            dynamics, current_policy, 30, max_steps, gamma, device)

        # DAgger: on-policy states, RELABEL with expert for pretrain_buffer
        explore_pol = exploration_policy(
            current_policy, noise_std=0.15,
            a_low=dynamics.action_low, a_high=dynamics.action_high)
        new_data = collect_trajectories(
            dynamics, explore_pol, max(n_trajectories // 5, 10), max_steps, device)

        # Create relabeled copy for pretrain buffer (BC gets expert targets)
        relabeled = {k: v.clone() for k, v in new_data.items()}
        with torch.no_grad():
            relabeled["actions"] = oscillating_policy(new_data["states"])

        for key in ["states", "actions", "rewards", "next_states", "dones"]:
            buffer[key] = torch.cat(
                [buffer[key], new_data[key]], dim=0)[-buffer_size:]
            pretrain_buffer[key] = torch.cat(
                [pretrain_buffer[key], relabeled[key]], dim=0)[-buffer_size:]
        buffer["init_states"] = torch.cat(
            [buffer["init_states"], new_data["init_states"]], dim=0)[-buffer_size:]
        pretrain_buffer["init_states"] = torch.cat(
            [pretrain_buffer["init_states"], new_data["init_states"]], dim=0)[-buffer_size:]
        N = buffer["states"].shape[0]
        N_pretrain = pretrain_buffer["states"].shape[0]

        # Joint BC + DFL updates (same pattern as InvertedPendulum)
        for _ in range(30):
            joint_optimizer.zero_grad()

            # BC on relabeled expert data (keeps policy near expert)
            idx_bc = torch.randperm(N_pretrain)[:batch_size]
            f_s = features._state_features(pretrain_buffer["states"][idx_bc])
            a_pred = f_s @ w_reg
            loss_bc = (a_pred - pretrain_buffer["actions"][idx_bc]).pow(2).mean()

            # DFL via LP occupancy
            idx_dfl = torch.randperm(N)[:batch_size]
            batch = {
                "states": buffer["states"][idx_dfl],
                "actions": buffer["actions"][idx_dfl],
                "rewards": buffer["rewards"][idx_dfl],
                "next_states": buffer["next_states"][idx_dfl],
                "dones": buffer["dones"][idx_dfl],
                "init_states": buffer["init_states"],
            }
            M, c, b = estimate_lp_matrices_multidim(
                features, batch, gamma=gamma,
                action_low=dynamics.action_low, action_high=dynamics.action_high,
                n_quad_per_dim=n_quad_per_dim, ridge=ridge)
            w_star = differentiable_reduced_qp(M, c, b, eta=qp_eta)

            lp_obj = c @ w_star
            violation = b - M @ w_star
            aug_lag = rho * violation.pow(2).mean()
            dfl_loss = lp_obj + aug_lag

            loss = loss_bc + dfl_weight * dfl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(features.parameters(), max_norm=1.0)
            joint_optimizer.step()

        current_policy = bc_policy  # re-binds to updated features/w_reg

        if results["avg_reward"] > best_reward:
            best_reward = results["avg_reward"]
            best_state = {
                "features": {k_: v.clone() for k_, v in features.state_dict().items()},
                "w_reg": w_reg.detach().clone(),
            }

        elapsed = time.time() - t0
        history.append({"iteration": iteration, **results, "best_reward": best_reward})

        if verbose and (iteration % 5 == 0 or iteration == n_iterations - 1):
            with torch.no_grad():
                w_nnz = int((w_star.detach() > 1e-8).sum())
            print(f"  Iter {iteration:3d} | reward={results['avg_reward']:7.2f} | "
                  f"best={best_reward:7.2f} | bc={loss_bc.item():.4f} | "
                  f"dfl={dfl_loss.item():.4f} | lp_nnz={w_nnz}/{d_actual} | "
                  f"time={elapsed:.1f}s")

    # Restore best
    features.load_state_dict(best_state["features"])
    with torch.no_grad():
        w_reg.copy_(best_state["w_reg"])

    if verbose:
        print(f"\n  Best policy reward: {best_reward:.2f}")

    return {"history": history, "features": features, "w_reg": w_reg,
            "best_reward": best_reward}


def plot_learning_curves(history, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    xs = list(range(len(history)))
    rewards = [h["avg_reward"] for h in history]
    best = [h["best_reward"] for h in history]
    ax1.plot(xs, rewards, marker="o", markersize=3, label="current")
    ax1.plot(xs, best, marker="s", markersize=3, label="best")
    ax1.legend()
    ax2.plot(xs, [h["avg_length"] for h in history], marker="o", markersize=3)
    ax1.set(xlabel="Iteration", ylabel="Reward", title="Learning — Reward")
    ax2.set(xlabel="Iteration", ylabel="Episode Length", title="Learning — Length")
    ax1.grid(True, alpha=0.3); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_features", type=int, default=256)
    p.add_argument("--n_action_bases_per_dim", type=int, default=5)
    p.add_argument("--n_iterations", type=int, default=30)
    p.add_argument("--n_trajectories", type=int, default=50)
    p.add_argument("--pretrain_epochs", type=int, default=300)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--n_quad_per_dim", type=int, default=6)
    p.add_argument("--dfl_weight", type=float, default=0.01)
    p.add_argument("--qp_eta", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_plot", type=str, default=None)
    args = p.parse_args()

    set_seed(args.seed)
    out = run_swimmer_dfl(
        n_features=args.n_features,
        n_action_bases_per_dim=args.n_action_bases_per_dim,
        n_iterations=args.n_iterations,
        n_trajectories=args.n_trajectories,
        pretrain_epochs=args.pretrain_epochs,
        max_steps=args.max_steps,
        lr=args.lr,
        ridge=args.ridge,
        rho=args.rho,
        tau=args.tau,
        n_quad_per_dim=args.n_quad_per_dim,
        dfl_weight=args.dfl_weight,
        qp_eta=args.qp_eta,
    )
    plot_learning_curves(out["history"], save_path=args.save_plot)


if __name__ == "__main__":
    main()
