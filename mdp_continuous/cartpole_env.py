"""
Analytical CartPole dynamics, reward functions, and trajectory collection.

State: (x, x_dot, theta, theta_dot) in R^4
Actions: {0=left, 1=right} -> force in {-force_mag, +force_mag}
"""

import torch
import math
from .configs import CartPoleConfig


class CartPoleDynamics:
    """Vectorized CartPole dynamics with analytical equations of motion."""

    def __init__(self, cfg: CartPoleConfig = None):
        if cfg is None:
            cfg = CartPoleConfig()
        self.cfg = cfg
        self.state_dim = 4
        self.n_actions = 2

    # ── Physics step ──────────────────────────────────────────────
    def step(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Euler-integrated CartPole dynamics.

        Args:
            state:  (B, 4)  — (x, x_dot, theta, theta_dot)
            action: (B,) int — 0 (left) or 1 (right)

        Returns:
            next_state: (B, 4)
        """
        cfg = self.cfg
        g = cfg.gravity
        mc = cfg.m_cart
        mp = cfg.m_pole
        l = cfg.pole_length
        dt = cfg.dt
        total_mass = mc + mp
        polemass_length = mp * l

        force = torch.where(
            action == 1,
            torch.tensor(cfg.force_mag, device=state.device, dtype=state.dtype),
            torch.tensor(-cfg.force_mag, device=state.device, dtype=state.dtype),
        )

        x, x_dot, theta, theta_dot = state[:, 0], state[:, 1], state[:, 2], state[:, 3]

        cos_th = torch.cos(theta)
        sin_th = torch.sin(theta)

        # Standard CartPole equations of motion
        temp = (force + polemass_length * theta_dot ** 2 * sin_th) / total_mass
        theta_acc = (g * sin_th - cos_th * temp) / (
            l * (4.0 / 3.0 - mp * cos_th ** 2 / total_mass)
        )
        x_acc = temp - polemass_length * theta_acc * cos_th / total_mass

        # Euler integration
        x_new = x + dt * x_dot
        x_dot_new = x_dot + dt * x_acc
        theta_new = theta + dt * theta_dot
        theta_dot_new = theta_dot + dt * theta_acc

        next_state = torch.stack([x_new, x_dot_new, theta_new, theta_dot_new], dim=1)

        # Add stochastic noise if configured
        if cfg.noise_std > 0:
            noise = torch.randn_like(next_state) * cfg.noise_std
            next_state = next_state + noise

        return next_state

    # ── Reward ────────────────────────────────────────────────────
    def reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute reward for (state, action) pairs.

        Args:
            state:  (B, 4)
            action: (B,)

        Returns:
            rewards: (B,)
        """
        if self.cfg.reward_type == "cos_theta":
            return torch.cos(state[:, 2])
        elif self.cfg.reward_type == "constant":
            return torch.ones(state.shape[0], device=state.device, dtype=state.dtype)
        else:
            raise ValueError(f"Unknown reward_type: {self.cfg.reward_type}")

    # ── Terminal check ────────────────────────────────────────────
    def is_terminal(self, state: torch.Tensor) -> torch.Tensor:
        """Check if states are terminal.

        Args:
            state: (B, 4)

        Returns:
            done: (B,) bool
        """
        cfg = self.cfg
        x_out = state[:, 0].abs() > cfg.x_max
        theta_out = state[:, 2].abs() > cfg.theta_max
        return x_out | theta_out

    # ── Initial state sampling ────────────────────────────────────
    def sample_initial_states(self, n: int, device: torch.device = None) -> torch.Tensor:
        """Sample from initial distribution rho: uniform in [-0.05, 0.05]^4."""
        if device is None:
            device = torch.device("cpu")
        return (torch.rand(n, 4, device=device) - 0.5) * 0.1

    # ── State normalization ───────────────────────────────────────
    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Scale state dimensions to roughly [-1, 1]."""
        cfg = self.cfg
        scales = torch.tensor(
            [cfg.x_max, cfg.x_dot_max, cfg.theta_max, cfg.theta_dot_max],
            device=state.device, dtype=state.dtype,
        )
        return state / scales


# ── Trajectory collection ─────────────────────────────────────────

def collect_trajectories(
    dynamics: CartPoleDynamics,
    policy_fn,
    n_episodes: int,
    max_steps: int = 200,
    device: torch.device = None,
):
    """Collect trajectory data by rolling out a policy.

    Args:
        dynamics: CartPoleDynamics instance
        policy_fn: callable  state (B, 4) -> action (B,) int
        n_episodes: number of episodes to collect
        max_steps: maximum steps per episode
        device: torch device

    Returns:
        dict with tensors:
            states:      (N, 4)
            actions:     (N,)
            rewards:     (N,)
            next_states: (N, 4)
            dones:       (N,) bool
            init_states: (n_episodes, 4)  initial states sampled from rho
    """
    if device is None:
        device = torch.device("cpu")

    all_states = []
    all_actions = []
    all_rewards = []
    all_next_states = []
    all_dones = []
    all_returns = []
    init_states = []

    gamma = dynamics.cfg.gamma

    for _ in range(n_episodes):
        s = dynamics.sample_initial_states(1, device=device)  # (1, 4)
        init_states.append(s.squeeze(0))

        ep_states, ep_actions, ep_rewards = [], [], []
        ep_next_states, ep_dones = [], []

        for t in range(max_steps):
            a = policy_fn(s)  # (1,)
            r = dynamics.reward(s, a)  # (1,)
            s_next = dynamics.step(s, a)  # (1, 4)
            done = dynamics.is_terminal(s_next)  # (1,)

            ep_states.append(s.squeeze(0))
            ep_actions.append(a.squeeze(0))
            ep_rewards.append(r.squeeze(0))
            ep_next_states.append(s_next.squeeze(0))
            ep_dones.append(done.squeeze(0))

            if done.item():
                break
            s = s_next

        # Compute discounted returns (reward-to-go)
        G = torch.tensor(0.0, device=device)
        ep_returns = []
        for r in reversed(ep_rewards):
            G = r + gamma * G
            ep_returns.insert(0, G)

        all_states.extend(ep_states)
        all_actions.extend(ep_actions)
        all_rewards.extend(ep_rewards)
        all_next_states.extend(ep_next_states)
        all_dones.extend(ep_dones)
        all_returns.extend(ep_returns)

    return {
        "states": torch.stack(all_states),
        "actions": torch.stack(all_actions),
        "rewards": torch.stack(all_rewards),
        "returns": torch.stack(all_returns),
        "next_states": torch.stack(all_next_states),
        "dones": torch.stack(all_dones),
        "init_states": torch.stack(init_states),
    }


def random_policy(states: torch.Tensor) -> torch.Tensor:
    """Uniform random over {0, 1}."""
    return torch.randint(0, 2, (states.shape[0],), device=states.device)


def heuristic_policy(states: torch.Tensor) -> torch.Tensor:
    """Simple PD controller: push in the direction the pole is falling."""
    theta = states[:, 2]
    theta_dot = states[:, 3]
    return ((theta + 0.5 * theta_dot) > 0).long()


def epsilon_greedy_policy(base_policy_fn, epsilon: float = 0.1):
    """Wrap a policy with epsilon-greedy exploration."""
    def policy(states):
        actions = base_policy_fn(states)
        mask = torch.rand(states.shape[0], device=states.device) < epsilon
        random_actions = torch.randint(0, 2, (states.shape[0],), device=states.device)
        return torch.where(mask, random_actions, actions)
    return policy
