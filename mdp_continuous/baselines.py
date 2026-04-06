"""
Baseline methods for CartPole comparison.

1. Discretize + Value Iteration
2. Discretize + Adaptive State Aggregation (Chen et al., 2107.11053)
"""

import torch
import numpy as np
from .cartpole_env import CartPoleDynamics
from .configs import CartPoleConfig
from .policy import evaluate_policy


# ═══════════════════════════════════════════════════════════════════
# Discretization utilities
# ═══════════════════════════════════════════════════════════════════

class DiscretizedCartPole:
    """Discretize CartPole into a grid of (bins_per_dim)^4 states.

    Each continuous dimension is divided into equidistant bins within bounds.
    """

    def __init__(self, cfg: CartPoleConfig, bins_per_dim: int = 20):
        self.cfg = cfg
        self.bins_per_dim = bins_per_dim
        self.n_actions = 2
        self.dynamics = CartPoleDynamics(cfg)

        # Bounds for discretization
        self.bounds = [
            (-cfg.x_max, cfg.x_max),
            (-cfg.x_dot_max, cfg.x_dot_max),
            (-cfg.theta_max, cfg.theta_max),
            (-cfg.theta_dot_max, cfg.theta_dot_max),
        ]

        # Bin edges per dimension
        self.edges = [
            np.linspace(lo, hi, bins_per_dim + 1)
            for lo, hi in self.bounds
        ]

        self.n_states = bins_per_dim ** 4
        self.state_dim = 4

    def state_to_index(self, state: np.ndarray) -> int:
        """Map continuous state to discrete index.

        Args:
            state: (4,) array

        Returns:
            index: int in [0, n_states)
        """
        b = self.bins_per_dim
        indices = []
        for dim in range(4):
            idx = np.searchsorted(self.edges[dim], state[dim]) - 1
            idx = np.clip(idx, 0, b - 1)
            indices.append(idx)
        return indices[0] * b**3 + indices[1] * b**2 + indices[2] * b + indices[3]

    def index_to_state(self, idx: int) -> np.ndarray:
        """Map discrete index to bin center (continuous state)."""
        b = self.bins_per_dim
        i3 = idx % b
        idx //= b
        i2 = idx % b
        idx //= b
        i1 = idx % b
        i0 = idx // b

        state = np.zeros(4)
        for dim, i in enumerate([i0, i1, i2, i3]):
            lo, hi = self.bounds[dim]
            bin_width = (hi - lo) / b
            state[dim] = lo + (i + 0.5) * bin_width
        return state

    def build_transition_and_reward(self):
        """Build P[s, a, s'] and R[s, a] by simulating transitions.

        Returns:
            P: (n_states, n_actions, n_states) transition probabilities
            R: (n_states, n_actions) rewards
        """
        S = self.n_states
        A = self.n_actions
        P = np.zeros((S, A, S))
        R = np.zeros((S, A))

        for s_idx in range(S):
            s_cont = self.index_to_state(s_idx)
            s_tensor = torch.tensor(s_cont, dtype=torch.float32).unsqueeze(0)

            for a in range(A):
                a_tensor = torch.tensor([a], dtype=torch.long)
                s_next = self.dynamics.step(s_tensor, a_tensor).squeeze(0).numpy()
                r = self.dynamics.reward(s_tensor, a_tensor).item()

                # Check terminal
                if (abs(s_next[0]) > self.cfg.x_max or
                        abs(s_next[2]) > self.cfg.theta_max):
                    # Terminal: reward is given, but no next state transition
                    # Absorb into a "terminal" state (self-loop with zero reward)
                    P[s_idx, a, s_idx] = 1.0  # absorbing
                    R[s_idx, a] = r
                else:
                    s_next_idx = self.state_to_index(s_next)
                    P[s_idx, a, s_next_idx] = 1.0
                    R[s_idx, a] = r

        return P, R


# ═══════════════════════════════════════════════════════════════════
# Baseline 1: Value Iteration
# ═══════════════════════════════════════════════════════════════════

def run_value_iteration(cfg: CartPoleConfig, bins_per_dim: int = None,
                        verbose: bool = True):
    """Discretize CartPole and run value iteration.

    Returns policy function that maps continuous states to actions.
    """
    if bins_per_dim is None:
        bins_per_dim = cfg.vi_bins_per_dim

    if verbose:
        print("=" * 60)
        print(f"Baseline: Value Iteration ({bins_per_dim} bins/dim, "
              f"{bins_per_dim**4} total states)")
        print("=" * 60)

    disc = DiscretizedCartPole(cfg, bins_per_dim)

    if verbose:
        print("Building transition and reward matrices...")
    P, R = disc.build_transition_and_reward()
    S = disc.n_states
    A = disc.n_actions
    gamma = cfg.gamma

    # Value iteration
    V = np.zeros(S)
    if verbose:
        print(f"Running VI for {cfg.vi_n_iters} iterations...")

    for it in range(cfg.vi_n_iters):
        Q = np.zeros((S, A))
        for a in range(A):
            Q[:, a] = R[:, a] + gamma * P[:, a, :] @ V
        V_new = Q.max(axis=1)

        delta = np.abs(V_new - V).max()
        V = V_new

        if verbose and (it % 100 == 0 or delta < 1e-6):
            print(f"  Iter {it:4d} | delta={delta:.6f}")

        if delta < 1e-6:
            if verbose:
                print(f"  Converged at iteration {it}")
            break

    # Extract greedy policy
    Q_final = np.zeros((S, A))
    for a in range(A):
        Q_final[:, a] = R[:, a] + gamma * P[:, a, :] @ V
    pi = Q_final.argmax(axis=1)  # (S,)

    # Wrap as continuous policy function
    def policy_fn(states):
        """states: (B, 4) tensor -> actions: (B,) tensor"""
        B = states.shape[0]
        actions = torch.zeros(B, dtype=torch.long, device=states.device)
        for i in range(B):
            s_np = states[i].cpu().numpy()
            s_idx = disc.state_to_index(s_np)
            actions[i] = int(pi[s_idx])
        return actions

    # Evaluate
    dynamics = CartPoleDynamics(cfg)
    results = evaluate_policy(dynamics, policy_fn, cfg.n_eval_episodes,
                              cfg.max_steps, cfg.gamma)
    if verbose:
        print(f"  VI policy: reward={results['avg_reward']:.2f}, "
              f"length={results['avg_length']:.1f}")

    return {"policy_fn": policy_fn, "V": V, "pi": pi, "results": results,
            "discretized": disc}


# ═══════════════════════════════════════════════════════════════════
# Baseline 2: Adaptive State Aggregation (Chen et al., 2107.11053)
# ═══════════════════════════════════════════════════════════════════

def aggregate_states_by_value(V: np.ndarray, epsilon: float):
    """Group states into mega-states by value similarity.

    States with V values within epsilon of each other are grouped.
    Algorithm 2 from Chen et al.

    Args:
        V: (S,) value vector
        epsilon: aggregation width

    Returns:
        assignments: (S,) int — mega-state index for each state
        n_mega: number of mega-states
    """
    v_min, v_max = V.min(), V.max()
    n_mega = max(1, int(np.ceil((v_max - v_min) / epsilon)))
    assignments = np.clip(
        ((V - v_min) / max(epsilon, 1e-10)).astype(int),
        0, n_mega - 1,
    )
    return assignments, n_mega


def run_adaptive_aggregation(cfg: CartPoleConfig, bins_per_dim: int = None,
                             verbose: bool = True):
    """Value iteration with adaptive state aggregation.

    Algorithm 3 from Chen et al.: alternates between global VI updates
    and aggregate updates where similar-value states are grouped.
    """
    if bins_per_dim is None:
        bins_per_dim = cfg.vi_bins_per_dim
    epsilon = cfg.agg_epsilon
    gamma = cfg.gamma

    if verbose:
        print("=" * 60)
        print(f"Baseline: Adaptive Aggregation (eps={epsilon}, "
              f"{bins_per_dim} bins/dim)")
        print("=" * 60)

    disc = DiscretizedCartPole(cfg, bins_per_dim)

    if verbose:
        print("Building transition and reward matrices...")
    P, R = disc.build_transition_and_reward()
    S = disc.n_states
    A = disc.n_actions

    V = np.zeros(S)

    # Phase structure: alternate global and aggregate phases
    n_global = 5    # global VI iterations per cycle
    n_agg = 2       # aggregate iterations per cycle
    n_cycles = cfg.vi_n_iters // (n_global + n_agg)
    alpha_base = 1.0
    step_count = 0

    history = []

    for cycle in range(n_cycles):
        # ── Global phase: standard VI ──
        for _ in range(n_global):
            Q = np.zeros((S, A))
            for a in range(A):
                Q[:, a] = R[:, a] + gamma * P[:, a, :] @ V
            V = Q.max(axis=1)
            step_count += 1

        # ── Aggregation: group states by value ──
        assignments, n_mega = aggregate_states_by_value(V, epsilon)

        # ── Aggregate phase: update mega-state representatives ──
        for _ in range(n_agg):
            step_count += 1
            alpha = alpha_base / np.sqrt(step_count)

            # For each mega-state, sample a representative state and update
            for m in range(n_mega):
                members = np.where(assignments == m)[0]
                if len(members) == 0:
                    continue
                s_rep = np.random.choice(members)

                # Bellman update for representative
                Q_rep = np.zeros(A)
                for a in range(A):
                    Q_rep[a] = R[s_rep, a] + gamma * P[s_rep, a, :] @ V
                v_new = Q_rep.max()

                # Update all members with blended value
                for s in members:
                    V[s] = (1 - alpha) * V[s] + alpha * v_new

        delta = 0
        Q_check = np.zeros((S, A))
        for a in range(A):
            Q_check[:, a] = R[:, a] + gamma * P[:, a, :] @ V
        delta = np.abs(Q_check.max(axis=1) - V).max()

        if verbose and (cycle % 10 == 0 or cycle == n_cycles - 1):
            print(f"  Cycle {cycle:3d} | n_mega={n_mega:5d} | "
                  f"delta={delta:.6f} | steps={step_count}")

        history.append({"cycle": cycle, "n_mega": n_mega, "delta": delta})

        if delta < 1e-6:
            if verbose:
                print(f"  Converged at cycle {cycle}")
            break

    # Extract policy
    Q_final = np.zeros((S, A))
    for a in range(A):
        Q_final[:, a] = R[:, a] + gamma * P[:, a, :] @ V
    pi = Q_final.argmax(axis=1)

    def policy_fn(states):
        B = states.shape[0]
        actions = torch.zeros(B, dtype=torch.long, device=states.device)
        for i in range(B):
            s_np = states[i].cpu().numpy()
            s_idx = disc.state_to_index(s_np)
            actions[i] = int(pi[s_idx])
        return actions

    dynamics = CartPoleDynamics(cfg)
    results = evaluate_policy(dynamics, policy_fn, cfg.n_eval_episodes,
                              cfg.max_steps, cfg.gamma)
    if verbose:
        print(f"  Adaptive agg policy: reward={results['avg_reward']:.2f}, "
              f"length={results['avg_length']:.1f}")

    return {"policy_fn": policy_fn, "V": V, "pi": pi, "results": results,
            "history": history}
