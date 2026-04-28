"""
World-model utilities for model-based DFL on CartPole.

Pipeline:
  1. Train a Gaussian dynamics model (delta-prediction) and a deterministic
     reward model on a small *real* buffer.
  2. Use the model to roll out short branches from observed states, producing
     a synthetic buffer that augments the real one for the LP-feature trainer.

Termination is delegated to the analytic boundary check `dynamics.is_terminal`.
This is "knowledge of the task" (what counts as failure), not knowledge of
the dynamics — we still never query the true env for next-state predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WorldModel(nn.Module):
    """Predict (next-state delta, reward) from (state, one-hot action).

    Diagonal Gaussian transition with a learned per-dim log-variance.
    Reward head is deterministic (MSE).
    """

    def __init__(self, state_dim: int = 4, n_actions: int = 2,
                 hidden=(128, 128)):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions

        in_dim = state_dim + n_actions
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.SiLU()]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.delta_head = nn.Linear(prev, state_dim)
        self.reward_head = nn.Linear(prev, 1)
        self.log_var = nn.Parameter(torch.full((state_dim,), -4.0))

    def _encode(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        a_oh = F.one_hot(actions.long(), self.n_actions).to(states.dtype)
        return torch.cat([states, a_oh], dim=1)

    def predict(self, states: torch.Tensor, actions: torch.Tensor):
        h = self.trunk(self._encode(states, actions))
        delta = self.delta_head(h)
        reward = self.reward_head(h).squeeze(-1)
        return delta, reward

    def sample_next(self, states: torch.Tensor, actions: torch.Tensor,
                    deterministic: bool = False):
        delta, reward = self.predict(states, actions)
        if deterministic:
            return states + delta, reward
        std = torch.exp(0.5 * self.log_var.clamp(-10.0, 2.0))
        noise = torch.randn_like(delta) * std
        return states + delta + noise, reward


def train_world_model(buffer: dict, n_epochs: int = 400, batch_size: int = 256,
                      lr: float = 1e-3, weight_decay: float = 1e-5,
                      val_frac: float = 0.1, verbose: bool = True,
                      log_every: int = 50):
    """Fit dynamics + reward via Gaussian NLL + MSE."""
    states = buffer["states"]
    actions = buffer["actions"]
    rewards = buffer["rewards"]
    next_states = buffer["next_states"]
    dones = buffer["dones"]

    # Drop terminal transitions: their next_state is post-failure and may not
    # follow the smooth dynamics. Including them poisons the dynamics fit.
    keep = ~dones
    states = states[keep]
    actions = actions[keep]
    rewards = rewards[keep]
    next_states = next_states[keep]

    target_delta = next_states - states
    N = states.shape[0]
    n_val = max(int(N * val_frac), 1)
    perm = torch.randperm(N)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    N_tr = tr_idx.shape[0]

    model = WorldModel()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val, best_state = float("inf"), None

    for epoch in range(n_epochs):
        idx = tr_idx[torch.randperm(N_tr)[:batch_size]]
        delta_pred, r_pred = model.predict(states[idx], actions[idx])
        log_var = model.log_var.clamp(-10.0, 2.0)
        var = torch.exp(log_var)
        nll_dyn = (((delta_pred - target_delta[idx]) ** 2) / (2.0 * var)
                   + 0.5 * log_var).sum(dim=-1).mean()
        mse_r = ((r_pred - rewards[idx]) ** 2).mean()
        loss = nll_dyn + mse_r

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        if (epoch + 1) % log_every == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                d_val, r_val = model.predict(states[val_idx], actions[val_idx])
                val_dyn_mse = ((d_val - target_delta[val_idx]) ** 2).mean().item()
                val_r_mse = ((r_val - rewards[val_idx]) ** 2).mean().item()
                val_loss = val_dyn_mse + val_r_mse
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if verbose:
                print(f"  [WM] epoch={epoch+1:4d} train_nll={loss.item():+.4f} "
                      f"val_dyn_mse={val_dyn_mse:.5f} val_r_mse={val_r_mse:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def generate_synthetic_buffer(world_model: WorldModel, real_buffer: dict,
                              n_total: int, terminal_fn,
                              rollout_length: int = 1,
                              action_policy=None, branch_per_state: int = 1,
                              device=None):
    """Branch short rollouts from real states using the world model.

    Args:
        world_model: trained WorldModel
        real_buffer: dict with at least "states"
        n_total: target total number of synthetic transitions
        terminal_fn: callable next_state -> bool (analytic boundary check)
        rollout_length: steps per branch (k=1 keeps compounding error tiny)
        action_policy: callable states -> actions; default = uniform random
        branch_per_state: how many independent rollouts to start per sampled state
    """
    if device is None:
        device = real_buffer["states"].device

    states_real = real_buffer["states"]
    N_real = states_real.shape[0]
    n_actions = world_model.n_actions

    n_branches = max(n_total // rollout_length, 1)
    syn = {k: [] for k in ["states", "actions", "rewards", "next_states", "dones"]}

    start_idx = torch.randint(0, N_real, (n_branches,), device=device)
    s = states_real[start_idx].clone()
    alive = torch.ones(n_branches, dtype=torch.bool, device=device)

    with torch.no_grad():
        for t in range(rollout_length):
            if not alive.any():
                break
            if action_policy is None:
                a = torch.randint(0, n_actions, (s.shape[0],), device=device)
            else:
                a = action_policy(s).long()
            s_next, r = world_model.sample_next(s, a)
            done = terminal_fn(s_next)

            syn["states"].append(s[alive])
            syn["actions"].append(a[alive])
            syn["rewards"].append(r[alive])
            syn["next_states"].append(s_next[alive])
            syn["dones"].append(done[alive])

            alive = alive & (~done)
            s = s_next

    out = {k: torch.cat(v, dim=0) if len(v) > 0 else torch.empty(0)
           for k, v in syn.items()}
    out["returns"] = out["rewards"].clone()
    return out


def merge_buffers(real: dict, synthetic: dict) -> dict:
    """Concatenate transitions; init_states stay real-only (LP RHS uses rho)."""
    keys = ["states", "actions", "rewards", "next_states", "dones"]
    merged = {}
    for k in keys:
        if k in synthetic and synthetic[k].numel() > 0:
            merged[k] = torch.cat([real[k], synthetic[k]], dim=0)
        else:
            merged[k] = real[k].clone()
    merged["init_states"] = real["init_states"].clone()
    if "returns" in real:
        if "returns" in synthetic and synthetic["returns"].numel() > 0:
            merged["returns"] = torch.cat([real["returns"], synthetic["returns"]], dim=0)
        else:
            merged["returns"] = real["returns"].clone()
    return merged
