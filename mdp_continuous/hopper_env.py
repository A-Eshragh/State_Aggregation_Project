"""
Gymnasium Hopper-v5 wrapper. 3D continuous action, 11D observation.
Terminates if torso falls (height < 0.7, angle too large).
Reward = healthy_reward + forward_velocity - small_ctrl_cost.
"""

import numpy as np
import torch
import gymnasium as gym


class HopperDynamics:
    def __init__(self, env_id: str = "Hopper-v5"):
        self.env_id = env_id
        self.state_dim = 11
        self.action_dim = 3
        self.action_low = -1.0
        self.action_high = 1.0
        # Empirical ranges for Hopper obs
        self._state_scales = torch.tensor(
            [1.5, 0.5, 0.5, 0.5, 0.5, 3.0, 3.0, 3.0, 5.0, 5.0, 5.0])

    @property
    def state_scales(self) -> torch.Tensor:
        return self._state_scales


def collect_trajectories(dynamics, policy_fn, n_episodes, max_steps=1000, device=None):
    if device is None:
        device = torch.device("cpu")

    all_s, all_a, all_r, all_ns, all_d = [], [], [], [], []
    init_states = []
    env = gym.make(dynamics.env_id)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(np.random.randint(0, 2**31)))
        s = torch.tensor(obs, dtype=torch.float32, device=device)
        init_states.append(s.clone())

        for t in range(max_steps):
            with torch.no_grad():
                a = policy_fn(s.unsqueeze(0))
            a_np = a.squeeze(0).cpu().numpy().clip(
                dynamics.action_low, dynamics.action_high)
            obs_next, r, term, trunc, _ = env.step(a_np.astype(np.float32))
            done = term or trunc
            s_next = torch.tensor(obs_next, dtype=torch.float32, device=device)

            all_s.append(s)
            all_a.append(a.squeeze(0).cpu())
            all_r.append(torch.tensor(r, dtype=torch.float32, device=device))
            all_ns.append(s_next)
            all_d.append(torch.tensor(done, device=device))

            if done:
                break
            s = s_next

    env.close()
    return {
        "states": torch.stack(all_s),
        "actions": torch.stack(all_a),
        "rewards": torch.stack(all_r),
        "next_states": torch.stack(all_ns),
        "dones": torch.stack(all_d),
        "init_states": torch.stack(init_states),
    }


def random_policy(states: torch.Tensor) -> torch.Tensor:
    return torch.rand(states.shape[0], 3, device=states.device) * 2.0 - 1.0


def pd_policy(states: torch.Tensor) -> torch.Tensor:
    """Simple PD on joint angles — reaches ~998 reward by standing stable.

    Obs indices: [z, theta_torso, theta_thigh, theta_leg, theta_foot,
                  vx, vz, vtheta_torso, vtheta_thigh, vtheta_leg, vtheta_foot]
    """
    if states.dim() == 1:
        states = states.unsqueeze(0)
    theta_thigh = states[:, 2]
    theta_leg = states[:, 3]
    theta_foot = states[:, 4]
    v_thigh = states[:, 8]
    v_leg = states[:, 9]
    v_foot = states[:, 10]
    # -2*theta - 1*theta_dot (PD toward zero joint angle)
    a1 = -2.0 * theta_thigh - 1.0 * v_thigh
    a2 = -2.0 * theta_leg - 1.0 * v_leg
    a3 = -2.0 * theta_foot - 1.0 * v_foot
    return torch.stack([a1, a2, a3], dim=1).clamp(-1.0, 1.0)


def evaluate_policy(dynamics, policy_fn, n_episodes=30, max_steps=1000,
                    gamma=0.99, device=None):
    if device is None:
        device = torch.device("cpu")
    env = gym.make(dynamics.env_id)
    rewards, disc, lens = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(np.random.randint(0, 2**31)))
        s = torch.tensor(obs, dtype=torch.float32, device=device)
        total_r, disc_r, discount = 0.0, 0.0, 1.0
        for t in range(max_steps):
            with torch.no_grad():
                a = policy_fn(s.unsqueeze(0))
            a_np = a.squeeze(0).cpu().numpy().clip(
                dynamics.action_low, dynamics.action_high)
            obs_next, r, term, trunc, _ = env.step(a_np.astype(np.float32))
            total_r += r
            disc_r += discount * r
            discount *= gamma
            if term or trunc:
                break
            s = torch.tensor(obs_next, dtype=torch.float32, device=device)
        rewards.append(total_r)
        disc.append(disc_r)
        lens.append(t + 1)
    env.close()
    return {
        "avg_reward": float(np.mean(rewards)),
        "avg_discounted_return": float(np.mean(disc)),
        "avg_length": float(np.mean(lens)),
        "rewards": torch.tensor(rewards),
        "lengths": torch.tensor(lens, dtype=torch.float32),
    }
