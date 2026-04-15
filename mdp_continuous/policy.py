"""
Policy extraction from LP solution and episode-based evaluation.

Given w* from the reduced LP, the occupancy measure is:
    y(s, a) = sum_j w_j* phi_j(s, a)

The policy is:
    pi(a | s) = max(y(s,a), 0) / sum_{a'} max(y(s,a'), 0)

If all y(s, a) <= 0 for some s, fall back to uniform.
"""

import torch
from .cartpole_env import CartPoleDynamics


def occupancy_to_policy_fn(feature_model, w_star, n_actions=2):
    """Return a policy function from the LP solution.

    Args:
        feature_model: has .phi(states, actions) -> (B, d)
        w_star: (d,) optimal weights from reduced LP
        n_actions: 2 for CartPole

    Returns:
        policy_fn: callable  states (B, 4) -> actions (B,) int
    """
    def policy_fn(states):
        B = states.shape[0]
        device = states.device

        # Compute y(s, a) for each action
        ys = []
        for a in range(n_actions):
            a_vec = torch.full((B,), a, device=device, dtype=torch.long)
            phi_sa = feature_model.phi(states, a_vec)  # (B, d)
            y_sa = phi_sa @ w_star                     # (B,)
            ys.append(y_sa)
        ys = torch.stack(ys, dim=1)  # (B, n_actions)

        # Pick action with highest occupancy.
        # With non-negative features + w >= 0, y >= 0 so argmax is well-defined.
        # For signed features (RFF), relative ordering still encodes LP preference.
        actions = torch.argmax(ys, dim=1)
        return actions

    return policy_fn


def occupancy_to_stochastic_policy_fn(feature_model, w_star, n_actions=2):
    """Return a stochastic policy (samples proportional to occupancy).

    Args:
        feature_model, w_star: as above
        n_actions: 2

    Returns:
        policy_fn: callable  states (B, 4) -> actions (B,) int
    """
    def policy_fn(states):
        B = states.shape[0]
        device = states.device

        ys = []
        for a in range(n_actions):
            a_vec = torch.full((B,), a, device=device, dtype=torch.long)
            phi_sa = feature_model.phi(states, a_vec)
            y_sa = phi_sa @ w_star
            ys.append(y_sa)
        ys = torch.stack(ys, dim=1)  # (B, n_actions)

        # Softmax over occupancies for stochastic sampling
        ys_pos = torch.clamp(ys, min=0.0) + 1e-10
        probs = ys_pos / ys_pos.sum(dim=1, keepdim=True)
        actions = torch.multinomial(probs, num_samples=1).squeeze(1)
        return actions

    return policy_fn


def evaluate_policy(
    dynamics: CartPoleDynamics,
    policy_fn,
    n_episodes: int = 100,
    max_steps: int = 200,
    gamma: float = 0.99,
    device: torch.device = None,
):
    """Evaluate a policy by rolling out episodes in CartPole.

    Returns:
        dict with:
            avg_reward:           mean undiscounted episode reward
            avg_discounted_return: mean discounted return
            avg_length:           mean episode length
            rewards:              (n_episodes,) per-episode total rewards
            lengths:              (n_episodes,) per-episode lengths
    """
    if device is None:
        device = torch.device("cpu")

    episode_rewards = []
    episode_disc_returns = []
    episode_lengths = []

    for _ in range(n_episodes):
        s = dynamics.sample_initial_states(1, device=device)
        total_reward = 0.0
        disc_return = 0.0
        discount = 1.0

        for t in range(max_steps):
            with torch.no_grad():
                a = policy_fn(s)
                r = dynamics.reward(s, a)

            total_reward += r.item()
            disc_return += discount * r.item()
            discount *= gamma

            s_next = dynamics.step(s, a)
            done = dynamics.is_terminal(s_next)
            if done.item():
                break
            s = s_next

        episode_rewards.append(total_reward)
        episode_disc_returns.append(disc_return)
        episode_lengths.append(t + 1)

    rewards_t = torch.tensor(episode_rewards)
    lengths_t = torch.tensor(episode_lengths, dtype=torch.float32)

    return {
        "avg_reward": rewards_t.mean().item(),
        "avg_discounted_return": torch.tensor(episode_disc_returns).mean().item(),
        "avg_length": lengths_t.mean().item(),
        "rewards": rewards_t,
        "lengths": lengths_t,
    }
