"""
DFL training for Hopper (3D continuous action, 11D state).

Uses the same framework as Swimmer — phi = g(a) ⊗ f(s), LP occupancy + DFL.
Tensor-product RBF grid over 3D action space.
"""

import argparse
import random
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from .features import ContinuousActionFeatures
from .lp_matrices_multidim import estimate_lp_matrices_multidim
from .reduced_solver import differentiable_reduced_qp
from .hopper_env import (
    HopperDynamics, collect_trajectories, evaluate_policy,
    random_policy, pd_policy,
)


def load_sac_expert(model_path: str):
    """Load a trained SAC model and wrap as a policy_fn(states) -> actions."""
    from stable_baselines3 import SAC
    model = SAC.load(model_path)

    def expert_fn(states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            obs = states.cpu().numpy()
            if obs.ndim == 1:
                obs = obs[None, :]
            actions, _ = model.predict(obs, deterministic=True)
            return torch.from_numpy(actions).float().to(states.device)

    return expert_fn


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def exploration_policy(base_fn, noise_std=0.2, a_low=-1.0, a_high=1.0):
    def pol(states):
        a = base_fn(states)
        return (a + torch.randn_like(a) * noise_std).clamp(a_low, a_high)
    return pol


def run_hopper_dfl(
    n_features: int = 512,
    n_action_bases_per_dim: int = 4,   # 4^3 = 64 bases for 3D action
    n_iterations: int = 30,
    n_trajectories: int = 50,
    pretrain_epochs: int = 1000,
    max_steps: int = 1000,
    gamma: float = 0.99,
    lr: float = 1e-3,
    ridge: float = 0.01,
    rho: float = 1.0,
    tau: float = 0.5,
    n_quad_per_dim: int = 4,
    buffer_size: int = 100000,
    dfl_weight: float = 0.01,
    qp_eta: float = 0.1,
    expert_path: str = None,   # if None, use simple pd_policy
    verbose: bool = True,
):
    device = torch.device("cpu")
    dynamics = HopperDynamics()

    # Select expert: SAC checkpoint (if provided) or simple PD
    if expert_path is not None:
        expert_fn = load_sac_expert(expert_path)
        if verbose:
            print(f"Loaded SAC expert from {expert_path}")
    else:
        expert_fn = pd_policy
        if verbose:
            print("Using simple PD expert")

    n_action_bases_total = n_action_bases_per_dim ** dynamics.action_dim
    d_per_action = max(n_features // n_action_bases_total, 4)

    if verbose:
        print("=" * 60)
        print(f"Hopper DFL (n_bases={n_action_bases_total} "
              f"({n_action_bases_per_dim}^{dynamics.action_dim}), "
              f"d_per_action={d_per_action})")
        print("=" * 60)

    # Softplus features with LARGER state MLP for Hopper's complex gaits
    features = ContinuousActionFeatures(
        state_dim=dynamics.state_dim,
        action_dim=dynamics.action_dim,
        n_action_bases=n_action_bases_total,
        d_per_action=d_per_action,
        hidden_dims=[400, 300],
        state_scales=dynamics.state_scales,
        action_low=dynamics.action_low,
        action_high=dynamics.action_high,
        tau_s=tau,
        state_features_type="softplus",
    ).to(device)

    actual_n_bases = features.n_action_bases
    k = features.d_per_action
    d_actual = features.n_features

    if verbose:
        print(f"  Actual: {actual_n_bases} bases x {k}/action = {d_actual} features")

    # ── Phase 1: Data collection + BC pretrain ──
    if verbose:
        print("Phase 1: Perturbed expert data + DAgger relabel + BC...")

    perturbed = exploration_policy(
        expert_fn, noise_std=0.15,
        a_low=dynamics.action_low, a_high=dynamics.action_high)
    raw_data = collect_trajectories(dynamics, perturbed, n_trajectories, max_steps, device)
    with torch.no_grad():
        raw_data["actions"] = expert_fn(raw_data["states"])

    expert_data = collect_trajectories(
        dynamics, expert_fn, n_trajectories // 2, max_steps, device)

    buffer = {
        key: torch.cat([raw_data[key], expert_data[key]], dim=0)
        for key in raw_data if key != "init_states"
    }
    buffer["init_states"] = torch.cat(
        [raw_data["init_states"], expert_data["init_states"]], dim=0)
    pretrain_buffer = {kk: v.clone() for kk, v in buffer.items()}
    N = pretrain_buffer["states"].shape[0]

    if verbose:
        print(f"  Transitions: {N}, avg_len: {N/(n_trajectories + n_trajectories//2):.0f}")
        print(f"  Reward stats: mean={buffer['rewards'].mean():.3f}, "
              f"range=[{buffer['rewards'].min():.3f}, {buffer['rewards'].max():.3f}]")

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
            return (f_s @ w).clamp(a_lo, a_hi)

    # LP-stochastic policy: sample actions from the occupancy density.
    # y(s, a) = phi(s, a)^T w_star = sum_j g_j(a) * (f(s)^T w_j)
    # Pick basis j ~ clamp(f(s)^T w_j, 0), then sample a ~ N(c_j, sigma).
    def lp_stochastic_policy_fn(w_star_tensor):
        centers = features.centers       # (n_bases, action_dim)
        sigma = features.sigma
        n_bases = features.n_action_bases
        k_dim = features.d_per_action
        a_lo = dynamics.action_low
        a_hi = dynamics.action_high

        def policy(states):
            B = states.shape[0]
            with torch.no_grad():
                f_s = features._state_features(states)  # (B, k)
                alphas = torch.zeros(B, n_bases, device=states.device)
                for j in range(n_bases):
                    alphas[:, j] = f_s @ w_star_tensor[j * k_dim:(j + 1) * k_dim]
                alphas = alphas.clamp(min=0.0) + 1e-10
                alphas = alphas / alphas.sum(dim=1, keepdim=True)
                comp = torch.multinomial(alphas, num_samples=1).squeeze(1)  # (B,)
                mu = centers[comp]  # (B, action_dim)
                noise = sigma * torch.randn_like(mu)
                return (mu + noise).clamp(a_lo, a_hi)
        return policy

    current_policy = bc_policy
    ws = evaluate_policy(dynamics, current_policy, 20, max_steps, gamma, device)
    if verbose:
        print(f"  Warm-start reward={ws['avg_reward']:.2f}, len={ws['avg_length']:.0f}")

    best_reward = -float("inf")
    best_state = {
        "features": {k_: v.clone() for k_, v in features.state_dict().items()},
        "w_reg": w_reg.detach().clone(),
    }

    # For LP-stochastic exploration we need a current w_star
    with torch.no_grad():
        M0, c0, b0 = estimate_lp_matrices_multidim(
            features, buffer, gamma=gamma,
            action_low=dynamics.action_low, action_high=dynamics.action_high,
            n_quad_per_dim=n_quad_per_dim, ridge=ridge)
        current_w_star = differentiable_reduced_qp(M0, c0, b0, eta=qp_eta).detach()

    history = []
    for iteration in range(n_iterations):
        t0 = time.time()

        # BC weight anneals from 1.0 (all BC) to 0.05 (DFL-dominated)
        frac = iteration / max(1, n_iterations - 1)
        bc_schedule = max(0.05, 1.0 - frac * 0.95)

        results = evaluate_policy(
            dynamics, current_policy, 30, max_steps, gamma, device)

        # Mix of two exploration sources:
        #  (1) BC + Gaussian noise around current policy (stable)
        #  (2) LP-stochastic sampling from occupancy density (discovery)
        explore_pol_bc = exploration_policy(
            current_policy, noise_std=0.1,
            a_low=dynamics.action_low, a_high=dynamics.action_high)
        explore_pol_lp = lp_stochastic_policy_fn(current_w_star)

        # Mix 50/50 when BC dominates; more LP exploration as BC weight drops
        lp_frac = 0.3 + 0.4 * frac  # 0.3 → 0.7 over training
        n_traj_lp = int(max(n_trajectories // 5, 10) * lp_frac)
        n_traj_bc = max(n_trajectories // 5, 10) - n_traj_lp

        data_bc = collect_trajectories(dynamics, explore_pol_bc, n_traj_bc, max_steps, device)
        data_lp = collect_trajectories(dynamics, explore_pol_lp, n_traj_lp, max_steps, device)
        new_data = {key: torch.cat([data_bc[key], data_lp[key]], dim=0)
                    for key in data_bc}

        # Only relabel BC targets — not full buffer
        relabeled = {k: v.clone() for k, v in new_data.items()}
        with torch.no_grad():
            relabeled["actions"] = expert_fn(new_data["states"])

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

        # Joint BC + DFL updates with annealed BC weight
        for _ in range(20):
            joint_optimizer.zero_grad()

            idx_bc = torch.randperm(N_pretrain)[:batch_size]
            f_s = features._state_features(pretrain_buffer["states"][idx_bc])
            a_pred = f_s @ w_reg
            loss_bc = (a_pred - pretrain_buffer["actions"][idx_bc]).pow(2).mean()

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

            # Annealed BC + scaled DFL
            loss = bc_schedule * loss_bc + dfl_weight * dfl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(features.parameters(), max_norm=1.0)
            joint_optimizer.step()

        # Update current_w_star for next iteration's LP-stochastic exploration
        with torch.no_grad():
            Mf, cf, bf = estimate_lp_matrices_multidim(
                features, buffer, gamma=gamma,
                action_low=dynamics.action_low, action_high=dynamics.action_high,
                n_quad_per_dim=n_quad_per_dim, ridge=ridge)
            current_w_star = differentiable_reduced_qp(
                Mf, cf, bf, eta=qp_eta).detach()

        current_policy = bc_policy

        if results["avg_reward"] > best_reward:
            best_reward = results["avg_reward"]
            best_state = {
                "features": {k_: v.clone() for k_, v in features.state_dict().items()},
                "w_reg": w_reg.detach().clone(),
            }

        elapsed = time.time() - t0
        history.append({"iteration": iteration, **results,
                        "best_reward": best_reward})

        if verbose and (iteration % 2 == 0 or iteration == n_iterations - 1):
            print(f"  Iter {iteration:3d} | reward={results['avg_reward']:7.1f} | "
                  f"best={best_reward:7.1f} | bc_wt={bc_schedule:.2f} | "
                  f"lp_frac={lp_frac:.2f} | bc={loss_bc.item():.4f} | "
                  f"dfl={dfl_loss.item():.3f} | time={elapsed:.1f}s")

    features.load_state_dict(best_state["features"])
    with torch.no_grad():
        w_reg.copy_(best_state["w_reg"])

    if verbose:
        print(f"\n  Best policy reward: {best_reward:.1f}")

    return {"history": history, "features": features, "w_reg": w_reg,
            "best_reward": best_reward}


def plot_learning_curves(history, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    xs = list(range(len(history)))
    ax1.plot(xs, [h["avg_reward"] for h in history], marker="o", markersize=3, label="current")
    ax1.plot(xs, [h["best_reward"] for h in history], marker="s", markersize=3, label="best")
    ax1.legend()
    ax2.plot(xs, [h["avg_length"] for h in history], marker="o", markersize=3)
    ax1.set(xlabel="Iteration", ylabel="Reward", title="Hopper — Reward")
    ax2.set(xlabel="Iteration", ylabel="Episode Length", title="Hopper — Length")
    ax1.grid(True, alpha=0.3); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_features", type=int, default=512)
    p.add_argument("--n_action_bases_per_dim", type=int, default=4)
    p.add_argument("--n_iterations", type=int, default=30)
    p.add_argument("--n_trajectories", type=int, default=50)
    p.add_argument("--pretrain_epochs", type=int, default=1000)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ridge", type=float, default=0.01)
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--n_quad_per_dim", type=int, default=4)
    p.add_argument("--dfl_weight", type=float, default=0.01)
    p.add_argument("--qp_eta", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_plot", type=str, default=None)
    p.add_argument("--expert_path", type=str, default=None,
                   help="Path to SAC model (.zip). If None, uses simple PD controller.")
    args = p.parse_args()

    set_seed(args.seed)
    out = run_hopper_dfl(
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
        expert_path=args.expert_path,
    )
    plot_learning_curves(out["history"], save_path=args.save_plot)


if __name__ == "__main__":
    main()
