"""
Main training pipeline for continuous-state LP on CartPole.

Neural features + DFL (end-to-end through differentiable LP).
"""

import argparse
import random
import time
import torch
import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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


def run_neural_dfl(cfg: CartPoleConfig, verbose: bool = True,
                   bins_per_dim: int = 8, tau: float = 0.5,
                   use_dfl_phase: bool = True,
                   pretrain_epochs: int = 70,
                   eval_policy_type: str = "ce",
                   behavior_policy=None,
                   n_ce_steps: int = 2,
                   n_dfl_steps: int = 5,
                   dfl_lr: float = None):
    assert eval_policy_type in ("ce", "lp")
    device = torch.device("cpu")
    dynamics = CartPoleDynamics(cfg)
    A = 2
    if behavior_policy is None:
        behavior_policy = heuristic_policy
    # Separate LR for the DFL phase keeps the LP-gradient updates gentle
    # so features don't drift far from the BC basin (avoids late-iteration
    # collapse). If unspecified, falls back to cfg.lr.
    if dfl_lr is None:
        dfl_lr = cfg.lr

    if verbose:
        print("=" * 60)
        tag = "Neural DFL (CE+LP)" if use_dfl_phase else "Two-Stage (CE only)"
        print(f"Running Continuous State DFL — {tag}")
        print("=" * 60)

    # Discrete Actions
    features = ActionIndexedFeatures(
        state_dim=4, n_actions=2,
        n_features=cfg.n_features,
        hidden_dims=cfg.neural_hidden,
    ).to(device)

    # Continuous Actions
    # n_action_bases = 4
    # d_per_action = cfg.n_features // n_action_bases

    # features = ContinuousActionFeatures(
    #     state_dim=4, action_dim=1,
    #     n_action_bases=n_action_bases,
    #     d_per_action=d_per_action,
    #     hidden_dims=cfg.neural_hidden,
    #     action_low=0.0, action_high=1.0,
    #     tau_s=tau,
    # ).to(device)

    k = features.d_per_action  # state feature dimension

    # Pre-train features to discriminate expert actions
    if verbose:
        print("Phase 1: Pre-training features on expert actions...")

    data = collect_trajectories(dynamics, behavior_policy,
                                cfg.n_trajectories, cfg.max_steps, device)
    buffer = {k_: v.clone() for k_, v in data.items()}

    if verbose:
        print(f"  Collected {buffer['states'].shape[0]} expert transitions")

    w_head = torch.randn(k, A, device=device, requires_grad=True)
    feat_optimizer = torch.optim.Adam(
        list(features.parameters()) + [w_head], lr=cfg.lr)

    batch_size = 4096
    N = buffer["states"].shape[0]
    ce_loss = torch.nn.CrossEntropyLoss()

    for epoch in range(pretrain_epochs):
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
            print(f"  [Pretrain] Epoch {epoch+1}/{pretrain_epochs}, "
                  f"loss={loss.item():.4f}, acc={acc.item():.3f}")

    if verbose:
        phase_desc = ("Phase 2: DFL iterations (LP + constraint penalty)..."
                      if use_dfl_phase else
                      "Phase 2: CE-only (two-stage) iterations...")
        print(phase_desc)

    # A small warm-up keeps M full-rank when n_trajectories is tiny.
    warmup = collect_trajectories(dynamics, behavior_policy,
                                  5, cfg.max_steps, device)
    for key in ["states", "actions", "rewards", "next_states", "dones"]:
        buffer[key] = torch.cat([buffer[key], warmup[key]], dim=0)
    buffer["init_states"] = torch.cat(
        [buffer["init_states"], warmup["init_states"]], dim=0)
    N = buffer["states"].shape[0]
    if verbose:
        print(f"  Offline dataset frozen: {N} transitions, "
              f"{buffer['init_states'].shape[0]} initial states")

    # dfl_optimizer = torch.optim.Adam(features.parameters(), lr=cfg.lr)
    dfl_optimizer = torch.optim.Adam(list(features.parameters()) + [w_head], lr=dfl_lr)
    alpha_shape = 2.0
    rho = cfg.rho
    lp_weight = 1

    def solve_lp(M, c, b):
        return differentiable_reduced_lp(M, c, b)

    def compute_shaped_rewards(states, actions, rewards):
        with torch.no_grad():
            f_s = features._state_features(states)
            logits = f_s @ w_head
            probs = torch.softmax(logits, dim=1)
            bonus = probs.gather(1, actions.long().unsqueeze(1)).squeeze(1)
        return rewards + alpha_shape * bonus

    with torch.no_grad():
        shaped_buffer = {kk: v.clone() for kk, v in buffer.items()}
        shaped_buffer["rewards"] = compute_shaped_rewards(
            buffer["states"], buffer["actions"], buffer["rewards"])
        M, c, b = estimate_lp_matrices(
            features, shaped_buffer, gamma=cfg.gamma, ridge=cfg.ridge)
        w_star = solve_lp(M, c, b).detach()
    current_policy = occupancy_to_policy_fn(features, w_star)

    def ce_policy(states):
        with torch.no_grad():
            f_s = features._state_features(states)
            return (f_s @ w_head).argmax(dim=1)

    eval_policy = ce_policy if eval_policy_type == "ce" else current_policy

    if verbose:
        test_s = torch.tensor([
            [0.0, 0.0, 0.05, 0.0],
            [0.0, 0.0, -0.05, 0.0],
        ], device=device)
        acts = current_policy(test_s)
        expert_acts = heuristic_policy(test_s)
        ws_results = evaluate_policy(
            dynamics, eval_policy, 50, cfg.max_steps, cfg.gamma, device)
        print(f"  Warm-start: acts={acts.tolist()} (expert={expert_acts.tolist()}), "
              f"reward={ws_results['avg_reward']:.1f}")

    def snapshot():
        return (
            {kk: v.detach().clone() for kk, v in features.state_dict().items()},
            w_star.detach().clone(),
        )

    best_reward = ws_results["avg_reward"] if verbose else -float("inf")
    best_state, best_w = snapshot()

    history = []
    for iteration in range(cfg.n_iterations):
        t0 = time.time()

        if eval_policy_type == "lp":
            eval_policy = occupancy_to_policy_fn(features, w_star)

        results = evaluate_policy(
            dynamics, eval_policy, cfg.n_eval_episodes,
            cfg.max_steps, cfg.gamma, device)
        if results["avg_reward"] > best_reward:
            best_reward = results["avg_reward"]
            best_state, best_w = snapshot()

        with torch.no_grad():
            dagger_actions = current_policy(buffer["states"]).long()
        dagger_states = buffer["states"]

        # Phase A: CE training (DAgger) — drives features to expert-accuracy on
        # the growing state distribution. Required for the LP-derived policy to
        # be non-pathological.
        ce_log = 0.0
        for _ in range(n_ce_steps):
            dfl_optimizer.zero_grad()
            ce_idx = torch.randperm(dagger_states.shape[0])[:batch_size]
            f_s = features._state_features(dagger_states[ce_idx])
            logits = f_s @ w_head
            ce_aux = ce_loss(logits, dagger_actions[ce_idx])
            ce_aux.backward()
            torch.nn.utils.clip_grad_norm_(features.parameters(), max_norm=1.0)
            dfl_optimizer.step()
            ce_log = ce_aux.item()

        # Phase B: DFL — LP-objective gradient on top of the strong features.
        # Skipped when `use_dfl_phase=False` (two-stage baseline).
        lp_obj_log = 0.0
        aug_lag_log = 0.0
        if use_dfl_phase:
            for _ in range(n_dfl_steps):
                dfl_optimizer.zero_grad()

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

                shaped_batch = {kk: v for kk, v in batch.items()}
                shaped_batch["rewards"] = compute_shaped_rewards(
                    batch["states"], batch["actions"], batch["rewards"])
                M_shaped, c_shaped, b_shaped = estimate_lp_matrices(
                    features, shaped_batch, gamma=cfg.gamma, ridge=cfg.ridge)
                w_star_inner = solve_lp(M_shaped, c_shaped, b_shaped)

                M_true, c_true, b_true = estimate_lp_matrices(
                    features, batch, gamma=cfg.gamma, ridge=cfg.ridge)

                lp_obj = c_true @ w_star_inner
                violation = b_true - M_true @ w_star_inner
                aug_lag = rho * violation.pow(2).sum()

                loss = (lp_obj + aug_lag) * lp_weight
                loss.backward()
                torch.nn.utils.clip_grad_norm_(features.parameters(), max_norm=1.0)
                dfl_optimizer.step()

                lp_obj_log = lp_obj.item()
                aug_lag_log = aug_lag.item()

        with torch.no_grad():
            shaped_full = {kk: v for kk, v in buffer.items()}
            shaped_full["rewards"] = compute_shaped_rewards(
                buffer["states"], buffer["actions"], buffer["rewards"])
            M_eval, c_eval, b_eval = estimate_lp_matrices(
                features, shaped_full, gamma=cfg.gamma, ridge=cfg.ridge)
            w_star = solve_lp(M_eval, c_eval, b_eval).detach()
        current_policy = occupancy_to_policy_fn(features, w_star)

        elapsed = time.time() - t0
        history.append({"iteration": iteration, **results})

        if verbose:
            w_nnz = int((w_star > 1e-8).sum())
            print(
                f"  Iter {iteration:3d} | reward={results['avg_reward']:.2f} | "
                f"length={results['avg_length']:.1f} | "
                f"lp_obj={lp_obj_log:.4f} | aug_lag={aug_lag_log:.4f} | "
                f"ce={ce_log:.4f} | nnz={w_nnz}/{features.n_features} | "
                f"time={elapsed:.1f}s"
            )

    # Final eval + restore best-so-far if the last iterate regressed.
    if eval_policy_type == "lp":
        eval_policy = occupancy_to_policy_fn(features, w_star)
    final_results = evaluate_policy(
        dynamics, eval_policy, cfg.n_eval_episodes,
        cfg.max_steps, cfg.gamma, device)
    if final_results["avg_reward"] > best_reward:
        best_reward = final_results["avg_reward"]
    else:
        features.load_state_dict(best_state)
        w_star = best_w
        current_policy = occupancy_to_policy_fn(features, w_star)
    if verbose:
        print(f"  Best reward over run: {best_reward:.2f}")

    return {"history": history, "features": features}


def run_two_stage(cfg: CartPoleConfig, verbose: bool = True,
                  bins_per_dim: int = 8, tau: float = 0.5,
                  pretrain_epochs: int = 70,
                  eval_policy_type: str = "ce",
                  behavior_policy=None,
                  n_ce_steps: int = 2,
                  n_dfl_steps: int = 5,
                  dfl_lr: float = None):
    return run_neural_dfl(
        cfg, verbose=verbose, bins_per_dim=bins_per_dim, tau=tau,
        use_dfl_phase=False,
        pretrain_epochs=pretrain_epochs,
        eval_policy_type=eval_policy_type,
        behavior_policy=behavior_policy,
        n_ce_steps=n_ce_steps,
        n_dfl_steps=n_dfl_steps,
        dfl_lr=dfl_lr,
    )

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

        # Solve aggregated LP (inner problem with predicted/aggregated params)
        y_agg = differentiable_reduced_lp(H, c_agg, gamma_agg)

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

        loss = reward_loss + cfg.rho * constraint_loss
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
    p.add_argument("--rho", type=float, default=1.0)
    # Soft aggregation
    p.add_argument("--n_clusters", type=int, default=32)
    p.add_argument("--bins_per_dim", type=int, default=8)
    p.add_argument("--tau", type=float, default=0.5,
                   help="Softmax temperature for membership")
    p.add_argument("--seed", type=int, default=1,
                   help="Random seed for reproducibility")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

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
        rho=10.0
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