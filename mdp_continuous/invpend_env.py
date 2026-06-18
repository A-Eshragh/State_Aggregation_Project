"""
Gymnasium InvertedPendulum wrapper matching the CartPole dynamics interface.

State: (x, theta, x_dot, theta_dot) in R^4
Action: continuous in [-3, 3]
Reward: 1.0 per step (survival)
"""

import numpy as np
import torch
import gymnasium as gym


class InvertedPendulumDynamics:
    """Wrapper around gymnasium InvertedPendulum-v5."""

    def __init__(self, env_id: str = "InvertedPendulum-v5"):
        self.env_id = env_id
        self.state_dim = 4
        self.action_dim = 1
        self.action_low = -3.0
        self.action_high = 3.0

        # State scales for normalization (empirical ranges)
        self._state_scales = torch.tensor([1.0, 1.0, 5.0, 10.0])

    @property
    def state_scales(self) -> torch.Tensor:
        return self._state_scales

    def _make_env(self, seed=None):
        env = gym.make(self.env_id)
        if seed is not None:
            env.reset(seed=seed)
        return env

    def sample_initial_states(self, n: int, device=None):
        """Sample initial states by resetting the env n times."""
        if device is None:
            device = torch.device("cpu")
        states = []
        env = self._make_env()
        for i in range(n):
            obs, _ = env.reset()
            states.append(torch.tensor(obs, dtype=torch.float32))
        env.close()
        return torch.stack(states).to(device)


def collect_trajectories(
    dynamics: InvertedPendulumDynamics,
    policy_fn,
    n_episodes: int,
    max_steps: int = 1000,
    device=None,
):
    """Collect trajectory data from InvertedPendulum.

    Args:
        dynamics: InvertedPendulumDynamics instance
        policy_fn: callable  state (B, 4) -> action (B,) or (B, 1) float
        n_episodes: number of episodes
        max_steps: max steps per episode
        device: torch device

    Returns:
        dict with tensors: states, actions, rewards, next_states, dones, init_states
    """
    if device is None:
        device = torch.device("cpu")

    all_states, all_actions, all_rewards = [], [], []
    all_next_states, all_dones = [], []
    init_states = []

    env = gym.make(dynamics.env_id)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(np.random.randint(0, 2**31)))
        s = torch.tensor(obs, dtype=torch.float32, device=device)
        init_states.append(s.clone())

        for t in range(max_steps):
            with torch.no_grad():
                a = policy_fn(s.unsqueeze(0))  # (1,) or (1, 1)
                if a.dim() > 1:
                    a = a.squeeze(-1)
                a_val = a.squeeze(0)

            # Clip to action range
            a_np = a_val.cpu().numpy().clip(dynamics.action_low, dynamics.action_high)
            obs_next, r, terminated, truncated, _ = env.step([float(a_np)])
            done = terminated or truncated

            s_next = torch.tensor(obs_next, dtype=torch.float32, device=device)

            all_states.append(s)
            all_actions.append(a_val)
            all_rewards.append(torch.tensor(r, dtype=torch.float32, device=device))
            all_next_states.append(s_next)
            all_dones.append(torch.tensor(done, device=device))

            if done:
                break
            s = s_next

    env.close()

    return {
        "states": torch.stack(all_states),
        "actions": torch.stack(all_actions),
        "rewards": torch.stack(all_rewards),
        "next_states": torch.stack(all_next_states),
        "dones": torch.stack(all_dones),
        "init_states": torch.stack(init_states),
    }


def random_policy(states: torch.Tensor) -> torch.Tensor:
    """Uniform random action in [-3, 3]."""
    B = states.shape[0]
    return torch.rand(B, device=states.device) * 6.0 - 3.0


def pd_policy(states: torch.Tensor) -> torch.Tensor:
    """LQR-style controller for inverted pendulum — reaches 1000 reward.

    Obs: (x, theta, x_dot, theta_dot)
    Stabilizes BOTH cart position and pole angle (not just pole).
    Gains tuned by grid search.
    """
    if states.dim() == 1:
        states = states.unsqueeze(0)
    x = states[:, 0]
    theta = states[:, 1]
    x_dot = states[:, 2]
    theta_dot = states[:, 3]
    action = 0.5 * x + 20.0 * theta + 1.0 * x_dot + 2.0 * theta_dot
    return action.clamp(-3.0, 3.0)


def weak_pd_policy(states: torch.Tensor) -> torch.Tensor:
    """Suboptimal pole-only controller — only reaches ~230 reward.

    Uses only pole angle feedback (ignores cart AND angular velocity).
    The cart drifts over time AND the pole oscillates, giving a mediocre
    baseline that DFL can improve upon via LP reward signal.
    """
    if states.dim() == 1:
        states = states.unsqueeze(0)
    theta = states[:, 1]
    action = 5.0 * theta
    return action.clamp(-3.0, 3.0)


def evaluate_policy(
    dynamics: InvertedPendulumDynamics,
    policy_fn,
    n_episodes: int = 100,
    max_steps: int = 1000,
    gamma: float = 0.99,
    device=None,
):
    """Evaluate a policy on InvertedPendulum.

    Returns dict with avg_reward, avg_discounted_return, avg_length.
    """
    if device is None:
        device = torch.device("cpu")

    env = gym.make(dynamics.env_id)
    episode_rewards, episode_disc_returns, episode_lengths = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(np.random.randint(0, 2**31)))
        s = torch.tensor(obs, dtype=torch.float32, device=device)
        total_reward, disc_return, discount = 0.0, 0.0, 1.0

        for t in range(max_steps):
            with torch.no_grad():
                a = policy_fn(s.unsqueeze(0))
                if a.dim() > 1:
                    a = a.squeeze(-1)
                a_val = a.squeeze(0)

            a_np = a_val.cpu().numpy().clip(dynamics.action_low, dynamics.action_high)
            obs_next, r, terminated, truncated, _ = env.step([float(a_np)])
            done = terminated or truncated

            total_reward += r
            disc_return += discount * r
            discount *= gamma

            if done:
                break
            s = torch.tensor(obs_next, dtype=torch.float32, device=device)

        episode_rewards.append(total_reward)
        episode_disc_returns.append(disc_return)
        episode_lengths.append(t + 1)

    env.close()

    rewards_t = torch.tensor(episode_rewards)
    lengths_t = torch.tensor(episode_lengths, dtype=torch.float32)

    return {
        "avg_reward": rewards_t.mean().item(),
        "avg_discounted_return": torch.tensor(episode_disc_returns).mean().item(),
        "avg_length": lengths_t.mean().item(),
        "rewards": rewards_t,
        "lengths": lengths_t,
    }
