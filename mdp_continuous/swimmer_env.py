"""
Gymnasium Swimmer-v5 wrapper. Continuous 2D action, 8D observation.
No termination; runs full 1000 steps. Reward = forward velocity - control cost.
"""

import numpy as np
import torch
import gymnasium as gym


class SwimmerDynamics:
    """Wrapper around Swimmer-v5."""

    def __init__(self, env_id: str = "Swimmer-v5"):
        self.env_id = env_id
        self.state_dim = 8
        self.action_dim = 2
        self.action_low = -1.0
        self.action_high = 1.0
        # Empirical state scales for normalization
        self._state_scales = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0])

    @property
    def state_scales(self) -> torch.Tensor:
        return self._state_scales


def collect_trajectories(
    dynamics: SwimmerDynamics,
    policy_fn,
    n_episodes: int,
    max_steps: int = 1000,
    device=None,
):
    """Collect trajectories. policy_fn: (B, state_dim) -> (B, action_dim)."""
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
                a = policy_fn(s.unsqueeze(0))  # (1, action_dim)
            a_np = a.squeeze(0).cpu().numpy().clip(
                dynamics.action_low, dynamics.action_high)
            obs_next, r, terminated, truncated, _ = env.step(a_np.astype(np.float32))
            done = terminated or truncated

            s_next = torch.tensor(obs_next, dtype=torch.float32, device=device)

            all_states.append(s)
            all_actions.append(a.squeeze(0).cpu())
            all_rewards.append(torch.tensor(r, dtype=torch.float32, device=device))
            all_next_states.append(s_next)
            all_dones.append(torch.tensor(done, device=device))

            if done:
                break
            s = s_next

    env.close()

    return {
        "states": torch.stack(all_states),
        "actions": torch.stack(all_actions),     # (N, 2)
        "rewards": torch.stack(all_rewards),
        "next_states": torch.stack(all_next_states),
        "dones": torch.stack(all_dones),
        "init_states": torch.stack(init_states),
    }


def random_policy(states: torch.Tensor) -> torch.Tensor:
    B = states.shape[0]
    return torch.rand(B, 2, device=states.device) * 2.0 - 1.0


def oscillating_policy(states: torch.Tensor) -> torch.Tensor:
    """Hand-tuned linear controller — reaches ~200 reward on Swimmer-v5.

    Linear feedback on joint angles (obs[0], obs[1]) found by grid search.
    Not optimal (~350) but a strong baseline for DFL to improve upon.
    """
    if states.dim() == 1:
        states = states.unsqueeze(0)
    theta1 = states[:, 0]
    theta2 = states[:, 1]
    a1 = 1.5 * theta1 + 1.0 * theta2
    a2 = 1.5 * theta1 + 1.5 * theta2
    return torch.stack([a1, a2], dim=1).clamp(-1.0, 1.0)


def evaluate_policy(dynamics, policy_fn, n_episodes=50, max_steps=1000,
                    gamma=0.99, device=None):
    if device is None:
        device = torch.device("cpu")
    env = gym.make(dynamics.env_id)
    rewards, disc_rets, lengths = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(np.random.randint(0, 2**31)))
        s = torch.tensor(obs, dtype=torch.float32, device=device)
        total_r, disc_r, discount = 0.0, 0.0, 1.0
        for t in range(max_steps):
            with torch.no_grad():
                a = policy_fn(s.unsqueeze(0))
            a_np = a.squeeze(0).cpu().numpy().clip(
                dynamics.action_low, dynamics.action_high)
            obs_next, r, terminated, truncated, _ = env.step(a_np.astype(np.float32))
            total_r += r
            disc_r += discount * r
            discount *= gamma
            if terminated or truncated:
                break
            s = torch.tensor(obs_next, dtype=torch.float32, device=device)
        rewards.append(total_r)
        disc_rets.append(disc_r)
        lengths.append(t + 1)
    env.close()
    return {
        "avg_reward": float(np.mean(rewards)),
        "avg_discounted_return": float(np.mean(disc_rets)),
        "avg_length": float(np.mean(lengths)),
        "rewards": torch.tensor(rewards),
        "lengths": torch.tensor(lengths, dtype=torch.float32),
    }
