"""
DFL training for InvertedPendulum with continuous actions.

Occupancy LP:  min c^T w  s.t.  Mw = b, w >= 0
Policy:        pi(a|s) ∝ y(s,a) = phi(s,a)^T w  (mixture of RBFs)
DFL loss:      c_true^T w* + rho * ||b - M_true @ w*||^2
"""

import argparse
import random
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from .features import ContinuousActionFeatures
from .lp_matrices_continuous import estimate_lp_matrices_continuous
from .reduced_solver import differentiable_reduced_lp
from .invpend_env import (
    InvertedPendulumDynamics,
    collect_trajectories,
    evaluate_policy,
    random_policy,
    pd_policy,
    weak_pd_policy,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════════════
# Stochastic policy from occupancy measure
# ═══════════════════════════════════════════════════════════════════

def occupancy_to_continuous_policy_fn(feature_model, w_star):
    """Return a stochastic policy that samples from the occupancy density.

    y(s, a) = phi(s, a)^T w = sum_j g_j(a) * f(s)^T w_j

    This is a mixture of RBFs in action space with state-dependent weights.
    Sample: pick component j ~ alpha_j(s), then a ~ N(c_j, sigma^2).
    """
    centers = feature_model.centers.squeeze(-1)      # (n_bases,)
    sigma = feature_model.sigma
    n_bases = feature_model.n_action_bases
    k = feature_model.d_per_action
    a_low = feature_model.centers.min().item() - 2 * sigma
    a_high = feature_model.centers.max().item() + 2 * sigma

    def policy_fn(states):
        B = states.shape[0]
        device = states.device

        with torch.no_grad():
            f_s = feature_model._state_features(states)  # (B, k)

            # Mixing weights: alpha_j(s) = f(s)^T w_j  for each RBF center
            # w_star has blocks [w_0, w_1, ..., w_{n_bases-1}], each of size k
            alphas = torch.zeros(B, n_bases, device=device)
            for j in range(n_bases):
                w_j = w_star[j * k : (j + 1) * k]
                alphas[:, j] = f_s @ w_j  # (B,)

            # Clamp to non-negative and normalize
            alphas = alphas.clamp(min=0.0) + 1e-10
            alphas = alphas / alphas.sum(dim=1, keepdim=True)

            # Sample component
            comp = torch.multinomial(alphas, num_samples=1).squeeze(1)  # (B,)

            # Sample action from Gaussian at selected center
            mu = centers[comp]  # (B,)
            actions = mu + sigma * torch.randn(B, device=device)
            actions = actions.clamp(a_low, a_high)

        return actions  # (B,)

    return policy_fn


def occupancy_to_mean_policy_fn(feature_model, w_star):
    """Deterministic policy: weighted mean of RBF centers.

    a(s) = sum_j alpha_j(s) * c_j  where alpha_j = clamp(f(s)^T w_j, 0)
    """
    centers = feature_model.centers.squeeze(-1)
    n_bases = feature_model.n_action_bases
    k = feature_model.d_per_action

    def policy_fn(states):
        with torch.no_grad():
            f_s = feature_model._state_features(states)
            alphas = torch.zeros(states.shape[0], n_bases, device=states.device)
            for j in range(n_bases):
                w_j = w_star[j * k : (j + 1) * k]
                alphas[:, j] = f_s @ w_j
            alphas = alphas.clamp(min=0.0) + 1e-10
            alphas = alphas / alphas.sum(dim=1, keepdim=True)
            actions = (alphas * centers.unsqueeze(0)).sum(dim=1)
        return actions

    return policy_fn


def occupancy_to_argmax_policy_fn(feature_model, w_star, n_grid: int = 64):
    """Deterministic policy: argmax_a y(s, a) on a dense action grid.

    y(s, a) = phi(s, a)^T w is a smooth function of a (sum of weighted RBFs).
    Evaluate on a grid and pick the best action.
    """
    a_low = feature_model.centers.min().item()
    a_high = feature_model.centers.max().item()

    def policy_fn(states):
        B = states.shape[0]
        with torch.no_grad():
            a_grid = torch.linspace(a_low, a_high, n_grid, device=states.device)
            best_a = torch.zeros(B, device=states.device)
            best_y = torch.full((B,), -float("inf"), device=states.device)

            for a_val in a_grid:
                a_vec = torch.full((B,), a_val.item(), device=states.device)
                y_sa = feature_model.phi(states, a_vec) @ w_star
                improved = y_sa > best_y
                best_y = torch.where(improved, y_sa, best_y)
                best_a = torch.where(improved, a_vec, best_a)
        return best_a

    return policy_fn


def exploration_policy(base_policy_fn, noise_std: float = 0.5,
                       a_low: float = -3.0, a_high: float = 3.0):
    """Add Gaussian exploration noise to a base policy."""
    def policy_fn(states):
        actions = base_policy_fn(states)
        noise = torch.randn_like(actions) * noise_std
        return (actions + noise).clamp(a_low, a_high)
    return policy_fn


# ═══════════════════════════════════════════════════════════════════
# DFL training
# ═══════════════════════════════════════════════════════════════════

def run_invpend_dfl(
    n_features: int = 128,
    n_action_bases: int = 8,
    n_iterations: int = 50,
    n_trajectories: int = 200,
    pretrain_epochs: int = 200,
    max_steps: int = 1000,
    gamma: float = 0.99,
    lr: float = 1e-3,
    ridge: float = 1e-4,
    rho: float = 1.0,
    tau: float = 0.5,
    n_quadrature: int = 32,
    buffer_size: int = 100000,
    expert: str = "strong",  # "strong" (1000) or "weak" (~265)
    dfl_weight: float = 0.01,
    verbose: bool = True,
):
    device = torch.device("cpu")
    dynamics = InvertedPendulumDynamics()

    # Pick expert based on option
    if expert == "strong":
        expert_policy_fn = pd_policy
    elif expert == "weak":
        expert_policy_fn = weak_pd_policy
    else:
        raise ValueError(f"Unknown expert: {expert}")

    d_per_action = n_features // n_action_bases

    if verbose:
        print("=" * 60)
        print(f"InvertedPendulum DFL (n_bases={n_action_bases}, "
              f"d_per_action={d_per_action}, d={n_features})")
        print("=" * 60)

    # Softplus state features (not softmax) so f(s) can represent smooth
    # linear functions without being bottlenecked by the probability simplex.
    # Full [-3, 3] action range since perturbed expert uses the whole range.
    features = ContinuousActionFeatures(
        state_dim=4,
        action_dim=1,
        n_action_bases=n_action_bases,
        d_per_action=d_per_action,
        hidden_dims=[128, 128],
        state_scales=dynamics.state_scales,
        action_low=dynamics.action_low,
        action_high=dynamics.action_high,
        tau_s=tau,
        state_features_type="softplus",
    ).to(device)

    k = features.d_per_action

    # ── Phase 1: Collect PERTURBED expert data ──
    # Pure LQR expert on unperturbed env uses nearly constant action (std~0.004),
    # giving BC no signal to handle off-distribution states. We perturb actions
    # during collection, then RELABEL states with the deterministic expert.
    # This is DAgger-style expert labeling.
    if verbose:
        print("Phase 1: Collecting perturbed expert data + relabeling...")

    # Perturb + relabel works for both experts. With a weak expert, the
    # relabels only give pole-correction (ignoring cart), so BC ends up at
    # the weak expert's level (~265). DFL then drives improvement via LP
    # reward signal.
    perturbed_pol = exploration_policy(
        expert_policy_fn, noise_std=1.0,
        a_low=dynamics.action_low, a_high=dynamics.action_high)
    raw_data = collect_trajectories(
        dynamics, perturbed_pol, n_trajectories, max_steps, device)
    with torch.no_grad():
        raw_data["actions"] = expert_policy_fn(raw_data["states"])

    expert_data = collect_trajectories(
        dynamics, expert_policy_fn, n_trajectories // 2, max_steps, device)

    # Pretrain buffer: perturbed+relabeled (diverse states, expert actions)
    pretrain_buffer = {kk: v.clone() for kk, v in raw_data.items()}
    # LP buffer: mix of perturbed and pure expert (both have expert action labels)
    buffer = {kk: torch.cat([raw_data[kk], expert_data[kk]], dim=0)
              for kk in raw_data if kk != "init_states"}
    buffer["init_states"] = torch.cat(
        [raw_data["init_states"], expert_data["init_states"]], dim=0)
    N = pretrain_buffer["states"].shape[0]

    if verbose:
        print(f"  Expert: {pretrain_buffer['states'].shape[0]} transitions "
              f"({pretrain_buffer['states'].shape[0] / n_trajectories:.0f} avg steps)")
        print(f"  Mixed:  {buffer['states'].shape[0]} transitions "
              f"({buffer['states'].shape[0] / n_trajectories:.0f} avg steps)")

    # Pretrain via regression: f(s) @ w_reg -> expert action
    w_reg = torch.randn(k, 1, device=device, requires_grad=True)
    feat_optimizer = torch.optim.Adam(
        list(features.parameters()) + [w_reg], lr=lr)

    batch_size = 4096
    N_pretrain = pretrain_buffer["states"].shape[0]

    if verbose:
        print("  Pre-training features via action regression...")

    for epoch in range(pretrain_epochs):
        feat_optimizer.zero_grad()
        idx = torch.randperm(N_pretrain)[:batch_size]
        f_s = features._state_features(pretrain_buffer["states"][idx])
        a_pred = (f_s @ w_reg).squeeze(-1)
        loss = (a_pred - pretrain_buffer["actions"][idx]).pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(features.parameters(), max_norm=1.0)
        feat_optimizer.step()

        if verbose and (epoch + 1) % 50 == 0:
            with torch.no_grad():
                mae = (a_pred - pretrain_buffer["actions"][idx]).abs().mean()
            print(f"  [Pretrain] Epoch {epoch+1}/{pretrain_epochs}, "
                  f"mse={loss.item():.4f}, mae={mae.item():.3f}")

    # ── Phase 2: DFL iterations (LP + constraint penalty) ──
    # Reward is 1.0 per step (action-independent) → LP cost can't break action
    # symmetry. Shape reward with expert bonus: actions close to the pretrained
    # regressor's prediction get higher reward.
    if verbose:
        print("Phase 2: DFL iterations (LP + constraint penalty)...")

    alpha_shape = 2.0

    def compute_shaped_rewards(states, actions, rewards):
        with torch.no_grad():
            f_s = feature_model._state_features(states)
            a_pred = (f_s @ w_reg).squeeze(-1)  # (B,)
            # Bonus: Gaussian centered on expert action prediction
            bonus = torch.exp(-((actions - a_pred) ** 2) / (2.0 * 1.0 ** 2))
        return rewards + alpha_shape * bonus

    # Alias for the closure
    feature_model = features

    # Warm-start policy: BC regressor from pretrain (LP will refine features)
    def _warmstart_policy(states, f=features, w=w_reg,
                          a_lo=dynamics.action_low, a_hi=dynamics.action_high):
        with torch.no_grad():
            f_s = f._state_features(states)
            return (f_s @ w).squeeze(-1).clamp(a_lo, a_hi)
    current_policy = _warmstart_policy

    if verbose:
        ws_results = evaluate_policy(
            dynamics, current_policy, 50, max_steps, gamma, device)
        print(f"  Warm-start reward={ws_results['avg_reward']:.1f}, "
              f"length={ws_results['avg_length']:.1f}")

    # Combined BC + DFL optimizer — BC preserves feature quality for the
    # policy, DFL refines via LP occupancy gradient.
    joint_optimizer = torch.optim.Adam(
        list(features.parameters()) + [w_reg], lr=lr * 0.5)

    # Best-policy tracking (DFL is noisy; save best features + w_reg seen)
    best_reward = -float("inf")
    best_state = {
        "features": {k: v.clone() for k, v in features.state_dict().items()},
        "w_reg": w_reg.detach().clone(),
    }

    history = []
    for iteration in range(n_iterations):
        t0 = time.time()

        # ── Evaluate current policy ──
        results = evaluate_policy(
            dynamics, current_policy, 100, max_steps, gamma, device)

        # ── DAgger: collect states under current policy, RELABEL with expert ──
        # This is the key fix for BC distribution shift. States visited by the
        # policy get labeled with the LQR expert's optimal action, so features
        # stay accurate on the policy's actual state distribution.
        explore_pol = exploration_policy(
            current_policy, noise_std=0.1,
            a_low=dynamics.action_low, a_high=dynamics.action_high)
        new_data = collect_trajectories(
            dynamics, explore_pol,
            max(n_trajectories // 5, 20), max_steps, device)
        # Relabel with expert
        with torch.no_grad():
            expert_relabels = expert_policy_fn(new_data["states"])
        new_data["actions"] = expert_relabels

        for key in ["states", "actions", "rewards", "next_states", "dones"]:
            buffer[key] = torch.cat(
                [buffer[key], new_data[key]], dim=0)[-buffer_size:]
            pretrain_buffer[key] = torch.cat(
                [pretrain_buffer[key], new_data[key]], dim=0)[-buffer_size:]
        buffer["init_states"] = torch.cat(
            [buffer["init_states"], new_data["init_states"]], dim=0)[-buffer_size:]
        pretrain_buffer["init_states"] = torch.cat(
            [pretrain_buffer["init_states"], new_data["init_states"]], dim=0)[-buffer_size:]
        N = buffer["states"].shape[0]
        N_pretrain = pretrain_buffer["states"].shape[0]

        # ── Joint BC + DFL training ──
        # Multiple gradient steps: BC keeps features aligned with expert policy,
        # DFL (LP objective + constraint penalty) refines via occupancy.
        for _ in range(50):
            joint_optimizer.zero_grad()

            # BC loss on DAgger-relabeled data
            idx_bc = torch.randperm(N_pretrain)[:batch_size]
            f_s = features._state_features(pretrain_buffer["states"][idx_bc])
            a_pred = (f_s @ w_reg).squeeze(-1)
            loss_bc = (a_pred - pretrain_buffer["actions"][idx_bc]).pow(2).mean()

            # DFL loss via LP occupancy
            idx_dfl = torch.randperm(N)[:batch_size]
            batch = {
                "states": buffer["states"][idx_dfl],
                "actions": buffer["actions"][idx_dfl],
                "rewards": buffer["rewards"][idx_dfl],
                "next_states": buffer["next_states"][idx_dfl],
                "dones": buffer["dones"][idx_dfl],
                "init_states": buffer["init_states"],
            }
            shaped_batch = {kk: v for kk, v in batch.items()}
            shaped_batch["rewards"] = compute_shaped_rewards(
                batch["states"], batch["actions"], batch["rewards"])

            M_shaped, c_shaped, b_shaped = estimate_lp_matrices_continuous(
                features, shaped_batch, gamma=gamma,
                action_low=dynamics.action_low, action_high=dynamics.action_high,
                n_quadrature=n_quadrature, ridge=ridge)
            w_star = differentiable_reduced_lp(M_shaped, c_shaped, b_shaped)

            M_true, c_true, b_true = estimate_lp_matrices_continuous(
                features, batch, gamma=gamma,
                action_low=dynamics.action_low, action_high=dynamics.action_high,
                n_quadrature=n_quadrature, ridge=ridge)
            lp_obj = c_true @ w_star
            violation = b_true - M_true @ w_star
            aug_lag = rho * violation.pow(2).mean()
            dfl_loss = lp_obj + aug_lag

            # Combined loss
            loss = loss_bc + dfl_weight * dfl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(features.parameters(), max_norm=1.0)
            joint_optimizer.step()

        # ── Extract policy from BC regressor (features learned via LP+BC) ──
        # LP solution is too sparse for continuous argmax, so we use the
        # linear policy a(s) = f(s)^T w_reg learned jointly with LP structure.
        # The LP+DFL has refined f(s) to satisfy occupancy consistency.
        w_eval = w_reg  # for logging
        def _policy_fn(states, f=features, w=w_reg, a_lo=dynamics.action_low,
                       a_hi=dynamics.action_high):
            with torch.no_grad():
                f_s = f._state_features(states)
                return (f_s @ w).squeeze(-1).clamp(a_lo, a_hi)
        current_policy = _policy_fn

        # Re-solve LP for diagnostic
        with torch.no_grad():
            shaped_full = {kk: v for kk, v in buffer.items()}
            shaped_full["rewards"] = compute_shaped_rewards(
                buffer["states"], buffer["actions"], buffer["rewards"])
            M_eval, c_eval, b_eval = estimate_lp_matrices_continuous(
                features, shaped_full, gamma=gamma,
                action_low=dynamics.action_low, action_high=dynamics.action_high,
                n_quadrature=n_quadrature, ridge=ridge)
            w_lp = differentiable_reduced_lp(M_eval, c_eval, b_eval).detach()

        # Save best-so-far
        if results["avg_reward"] > best_reward:
            best_reward = results["avg_reward"]
            best_state = {
                "features": {k: v.clone() for k, v in features.state_dict().items()},
                "w_reg": w_reg.detach().clone(),
            }

        elapsed = time.time() - t0
        history.append({"iteration": iteration, **results,
                        "best_reward": best_reward})

        if verbose and (iteration % 5 == 0 or iteration == n_iterations - 1):
            w_nnz = int((w_lp > 1e-8).sum())
            print(
                f"  Iter {iteration:3d} | reward={results['avg_reward']:.1f} | "
                f"best={best_reward:.1f} | "
                f"bc={loss_bc.item():.4f} | dfl={dfl_loss.item():.4f} | "
                f"lp_nnz={w_nnz}/{n_features} | "
                f"time={elapsed:.1f}s"
            )

    # Restore best policy
    features.load_state_dict(best_state["features"])
    with torch.no_grad():
        w_reg.copy_(best_state["w_reg"])

    if verbose:
        print(f"\n  Best policy reward: {best_reward:.1f}")

    return {"history": history, "features": features, "w_reg": w_reg,
            "best_reward": best_reward}


# ═══════════════════════════════════════════════════════════════════
# Plotting & CLI
# ═══════════════════════════════════════════════════════════════════

def plot_learning_curves(history, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    xs = list(range(len(history)))
    rewards = [h["avg_reward"] for h in history]
    lengths = [h["avg_length"] for h in history]
    ax1.plot(xs, rewards, marker="o", markersize=3)
    ax2.plot(xs, lengths, marker="o", markersize=3)
    ax1.set(xlabel="Iteration", ylabel="Average Episode Reward",
            title="Learning Curve - Reward")
    ax2.set(xlabel="Iteration", ylabel="Average Episode Length",
            title="Learning Curve - Length")
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    plt.show()


def main():
    p = argparse.ArgumentParser(description="InvertedPendulum DFL")
    p.add_argument("--n_features", type=int, default=128)
    p.add_argument("--n_action_bases", type=int, default=8)
    p.add_argument("--n_iterations", type=int, default=50)
    p.add_argument("--n_trajectories", type=int, default=200)
    p.add_argument("--pretrain_epochs", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--n_quadrature", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_plot", type=str, default=None)
    p.add_argument("--expert", type=str, default="strong",
                   choices=["strong", "weak"],
                   help="'strong' = LQR (1000 reward), 'weak' = simple PD (~265 reward)")
    p.add_argument("--dfl_weight", type=float, default=0.01,
                   help="Weight of DFL loss vs BC loss. Use larger (~1.0) with weak expert.")
    args = p.parse_args()

    set_seed(args.seed)

    out = run_invpend_dfl(
        n_features=args.n_features,
        n_action_bases=args.n_action_bases,
        n_iterations=args.n_iterations,
        n_trajectories=args.n_trajectories,
        pretrain_epochs=args.pretrain_epochs,
        max_steps=args.max_steps,
        gamma=args.gamma,
        lr=args.lr,
        ridge=args.ridge,
        rho=args.rho,
        tau=args.tau,
        n_quadrature=args.n_quadrature,
        expert=args.expert,
        dfl_weight=args.dfl_weight,
    )

    plot_learning_curves(out["history"], save_path=args.save_plot)


if __name__ == "__main__":
    main()
