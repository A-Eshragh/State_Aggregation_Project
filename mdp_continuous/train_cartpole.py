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
from .cartpole_env import CartPoleDynamics, collect_trajectories, random_policy, heuristic_policy, epsilon_greedy_policy
from .features import ActionIndexedFeatures, ContinuousActionFeatures
from .lp_matrices import estimate_lp_matrices
from .reduced_solver import (
    solve_reduced_qp_gurobi, differentiable_reduced_lp, differentiable_reduced_qp,
    differentiable_unconstrained_qp,
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

def _solve_full_lp_on_grid(P, R, gamma_init, gamma, device):
    """Solve the full occupancy LP on the discretized grid.

    min  c^T y   s.t.  H y = rho,  y >= 0
    where y(s,a) is the occupancy, c = -R (maximise reward),
    H encodes the flow constraint: sum_a y(s,a) - gamma * sum_{s',a} P(s|s',a) y(s',a) = rho(s).

    Returns:
        y_full: (S, A) optimal occupancy
        pi_full: (S,) optimal deterministic policy
    """
    import gurobipy as gp
    from gurobipy import GRB

    S, A, _ = P.shape
    n_vars = S * A

    # Build constraint matrix H: (S, S*A)
    # H[s, s'*A+a] = delta(s,s') - gamma * P[s',a,s]
    H_np = np.zeros((S, n_vars), dtype=np.float64)
    for a in range(A):
        for s in range(S):
            H_np[s, s * A + a] += 1.0  # outflow: y(s,a) contributes to state s
        for sp in range(S):
            for s in range(S):
                H_np[s, sp * A + a] -= gamma * P[sp, a, s].item()  # inflow

    c_np = np.zeros(n_vars, dtype=np.float64)
    for s in range(S):
        for a in range(A):
            c_np[s * A + a] = -R[s, a].item()

    rho_np = gamma_init.cpu().numpy().astype(np.float64)

    model = gp.Model()
    model.Params.OutputFlag = 0
    y_vars = model.addMVar(shape=n_vars, lb=0.0, vtype=GRB.CONTINUOUS)
    model.setObjective(c_np @ y_vars, GRB.MINIMIZE)
    model.addMConstr(H_np, y_vars, "=", rho_np)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        raise RuntimeError(f"Full LP failed with status {model.status}")

    y_flat = torch.tensor(y_vars.X, dtype=torch.float32, device=device)
    y_full = y_flat.reshape(S, A)
    pi_full = y_full.argmax(dim=1)

    return y_full, pi_full


def run_neural_dfl(cfg: CartPoleConfig, verbose: bool = True,
                   bins_per_dim: int = 8, tau: float = 0.5):
    """Neural features + DFL on CartPole.

    Two-phase approach:
      Phase 1: Pre-train features f(s) via supervised learning on heuristic
               expert actions, giving the state network meaningful spatial
               discrimination before DFL begins.
      Phase 2: DFL loop — estimate LP matrices from trajectory data, solve
               the QP differentiably, backprop LP objective to update features,
               and re-collect data with the improved policy.
    """
    device = torch.device("cpu")
    dynamics = CartPoleDynamics(cfg)
    A = 2

    # Derive ContinuousActionFeatures params from cfg.n_features
    n_action_bases = 4
    d_per_action = cfg.n_features // n_action_bases

    if verbose:
        print("=" * 60)
        print(f"Neural DFL (n_action_bases={n_action_bases}, "
              f"d_per_action={d_per_action}, total d={cfg.n_features})")
        print("=" * 60)

    features = ContinuousActionFeatures(
        state_dim=4, action_dim=1,
        n_action_bases=n_action_bases,
        d_per_action=d_per_action,
        hidden_dims=cfg.neural_hidden,
        action_low=0.0, action_high=1.0,
        tau_s=tau,
    ).to(device)

    k = features.d_per_action  # state feature dimension

    # ── Phase 1: Pre-train features to discriminate expert actions ──
    if verbose:
        print("Phase 1: Pre-training features on expert actions...")

    data = collect_trajectories(dynamics, heuristic_policy,
                                cfg.n_trajectories, cfg.max_steps, device)
    buffer = {k_: v.clone() for k_, v in data.items()}

    if verbose:
        print(f"  Collected {buffer['states'].shape[0]} expert transitions")

    # Train features + linear head to predict expert actions
    w_head = torch.randn(k, A, device=device, requires_grad=True)
    feat_optimizer = torch.optim.Adam(
        list(features.parameters()) + [w_head], lr=cfg.lr)

    batch_size = 4096
    N = buffer["states"].shape[0]
    ce_loss = torch.nn.CrossEntropyLoss()

    n_pretrain = 200
    for epoch in range(n_pretrain):
        feat_optimizer.zero_grad()

        idx = torch.randperm(N)[:batch_size]
        states_batch = buffer["states"][idx]
        expert_actions = buffer["actions"][idx].long()

        f_s = features._state_features(states_batch)  # (B, k)
        logits = f_s @ w_head                          # (B, A)
        loss = ce_loss(logits, expert_actions)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(features.parameters(), max_norm=1.0)
        feat_optimizer.step()

        if verbose and (epoch + 1) % 50 == 0:
            with torch.no_grad():
                acc = (logits.argmax(1) == expert_actions).float().mean()
            print(f"  [Pretrain] Epoch {epoch+1}/{n_pretrain}, "
                  f"loss={loss.item():.4f}, acc={acc.item():.3f}")

    # ── Phase 2: DFL iterations ──
    # R(s,a) = cos(theta) is action-independent, so the LP cost c is symmetric
    # across actions → the LP degenerates to one action.  Fix: augment the
    # reward with a small expert-action bonus (reward shaping) so the cost
    # vector c can break the symmetry. The dynamics constraint M is already
    # action-dependent; the bonus just nudges the LP toward the expert action.
    if verbose:
        print("Phase 2: DFL iterations (LP solve with shaped reward)...")

    dfl_optimizer = torch.optim.Adam(
        list(features.parameters()) + [w_head], lr=cfg.lr * 0.1)
    alpha_shape = 2.0  # reward shaping strength

    # Compute shaped rewards: r_shaped = r + alpha * expert_bonus
    # expert_bonus = softmax(f(s) @ w_head)[a_taken] — high when the taken
    # action matches the pretrained classifier's preference.
    def compute_shaped_rewards(states, actions, rewards):
        with torch.no_grad():
            f_s = features._state_features(states)
            logits = f_s @ w_head
            probs = torch.softmax(logits, dim=1)
            # Bonus: probability the classifier assigns to the taken action
            bonus = probs.gather(1, actions.long().unsqueeze(1)).squeeze(1)
        return rewards + alpha_shape * bonus

    # Initial LP solve for warm-start policy
    with torch.no_grad():
        shaped_buffer = {kk: v.clone() for kk, v in buffer.items()}
        shaped_buffer["rewards"] = compute_shaped_rewards(
            buffer["states"], buffer["actions"], buffer["rewards"])
        M, c, b = estimate_lp_matrices(
            features, shaped_buffer, gamma=cfg.gamma, ridge=cfg.ridge)
        w_star = differentiable_reduced_qp(M, c, b, eta=cfg.qp_eta).detach()
    current_policy = occupancy_to_policy_fn(features, w_star)

    if verbose:
        # Diagnostic: check warm-start policy
        test_s = torch.tensor([
            [0.0, 0.0, 0.05, 0.0],   # leaning right
            [0.0, 0.0, -0.05, 0.0],   # leaning left
        ], device=device)
        acts = current_policy(test_s)
        expert_acts = heuristic_policy(test_s)
        ws_results = evaluate_policy(
            dynamics, current_policy, 50, cfg.max_steps, cfg.gamma, device)
        print(f"  Warm-start: acts={acts.tolist()} (expert={expert_acts.tolist()}), "
              f"reward={ws_results['avg_reward']:.1f}")

    history = []
    for iteration in range(cfg.n_iterations):
        t0 = time.time()

        # ── Evaluate current policy BEFORE any updates ──
        results = evaluate_policy(
            dynamics, current_policy, cfg.n_eval_episodes,
            cfg.max_steps, cfg.gamma, device)

        # ── Collect on-policy data ──
        explore_policy = epsilon_greedy_policy(current_policy,
                                               epsilon=cfg.exploration_eps)
        new_data = collect_trajectories(
            dynamics, explore_policy,
            max(cfg.n_trajectories // 5, 50), cfg.max_steps, device)
        for key in ["states", "actions", "rewards", "next_states", "dones"]:
            buffer[key] = torch.cat([buffer[key], new_data[key]], dim=0)[-cfg.buffer_size:]
        buffer["init_states"] = torch.cat(
            [buffer["init_states"], new_data["init_states"]], dim=0)[-cfg.buffer_size:]
        N = buffer["states"].shape[0]

        # ── DFL gradient step ──
        dfl_optimizer.zero_grad()

        # Sub-sample
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

        # Shaped rewards: break LP action degeneracy using pretrained classifier
        shaped_batch = {kk: v for kk, v in batch.items()}
        shaped_batch["rewards"] = compute_shaped_rewards(
            batch["states"], batch["actions"], batch["rewards"])

        # LP objective: estimate matrices with shaped reward, solve QP
        M, c, b = estimate_lp_matrices(
            features, shaped_batch, gamma=cfg.gamma, ridge=cfg.ridge)
        eta = max(cfg.qp_eta, 1.0 * (0.95 ** iteration))
        w_star = differentiable_reduced_qp(M, c, b, eta=eta)
        lp_loss = c @ w_star

        # CE term: co-train the classifier head alongside features
        ce_idx = torch.randperm(N)[:batch_size]
        f_s = features._state_features(buffer["states"][ce_idx])
        logits = f_s @ w_head
        ce_reg = ce_loss(logits, buffer["actions"][ce_idx].long())

        loss = lp_loss + 10.0 * ce_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(features.parameters()) + [w_head], max_norm=1.0)
        dfl_optimizer.step()

        # ── Extract policy from LP solution ──
        with torch.no_grad():
            shaped_full = {kk: v for kk, v in buffer.items()}
            shaped_full["rewards"] = compute_shaped_rewards(
                buffer["states"], buffer["actions"], buffer["rewards"])
            M_eval, c_eval, b_eval = estimate_lp_matrices(
                features, shaped_full, gamma=cfg.gamma, ridge=cfg.ridge)
            w_eval = differentiable_reduced_qp(
                M_eval, c_eval, b_eval, eta=cfg.qp_eta).detach()
        current_policy = occupancy_to_policy_fn(features, w_eval)

        elapsed = time.time() - t0
        history.append({"iteration": iteration, **results})

        if verbose and (iteration % 5 == 0 or iteration == cfg.n_iterations - 1):
            w_nnz = int((w_eval > 1e-8).sum())
            print(
                f"  Iter {iteration:3d} | reward={results['avg_reward']:.2f} | "
                f"length={results['avg_length']:.1f} | "
                f"lp={lp_loss.item():.4f} | ce={ce_reg.item():.4f} | "
                f"eta={eta:.4f} | nnz={w_nnz}/{features.n_features} | "
                f"time={elapsed:.1f}s"
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
        out = run_neural_dfl(cfg, bins_per_dim=args.bins_per_dim, tau=args.tau)

    plot_learning_curves(out["history"], save_path=args.save_plot)


if __name__ == "__main__":
    main()
