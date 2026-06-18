"""
Sanity test: verify our CliffWalking geometry matches gymnasium's
CliffWalking-v0. We compare the deterministic (intended-only) transition
table produced by our `build_P_r_from_slip` against gym's `env.P`.

If this ever fails, someone has broken the grid (state indexing, cliff
list, wall clipping, reward scheme, or action ordering).

Run:
    python -m mdp_cw.test_geometry
"""

from __future__ import annotations

import sys

import torch

try:
    import gymnasium as gym
except ImportError:
    print("gymnasium is not installed. `pip install gymnasium` to run this test.")
    sys.exit(2)

from mdp_cw.cliff_mdp import N_ACTIONS, N_STATES, N_SLIP, build_P_r_from_slip


from mdp_cw.cliff_mdp import GOAL_STATE


def gym_reference_P_r():
    """Load gym's CliffWalking-v0 and flatten env.P into (S,A,S) and (S,A) tensors.

    For the goal state, we replace gym's normal transitions with an absorbing
    self-loop (semantically identical under gym's done=True flag, but needed
    for an infinite-horizon LP formulation).
    """
    env = gym.make("CliffWalking-v0").unwrapped
    P = torch.zeros(N_STATES, N_ACTIONS, N_STATES)
    r = torch.zeros(N_STATES, N_ACTIONS)
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            if s == GOAL_STATE:
                # gym marks these transitions done=True; we model as self-loop
                P[s, a, s] = 1.0
                r[s, a] = 0.0
                continue
            for prob, ns, rew, _done in env.P[s][a]:
                P[s, a, ns] += prob
                r[s, a] += prob * rew
    return P, r


def ours_deterministic_P_r():
    """Our build_P_r_from_slip with slip_dist = one-hot on 'intended' (kind 0)."""
    slip = torch.zeros(1, N_ACTIONS, N_SLIP)
    slip[:, :, 0] = 1.0
    P, r = build_P_r_from_slip(slip)
    return P.squeeze(0), r.squeeze(0)


def main():
    Pg, rg = gym_reference_P_r()
    Po, ro = ours_deterministic_P_r()

    p_diff = (Pg - Po).abs().max().item()
    r_diff = (rg - ro).abs().max().item()

    print(f"max |P_gym - P_ours| = {p_diff:.6f}")
    print(f"max |r_gym - r_ours| = {r_diff:.6f}")

    ok = True
    if p_diff > 1e-6:
        print("MISMATCH in transitions. Per-state where they differ:")
        where = (Pg - Po).abs().sum(dim=-1) > 1e-6
        bad = where.nonzero().tolist()
        for s, a in bad[:10]:
            gym_targets = (Pg[s, a] > 0).nonzero().flatten().tolist()
            our_targets = (Po[s, a] > 0).nonzero().flatten().tolist()
            print(f"  s={s} a={a}  gym -> {gym_targets}  ours -> {our_targets}")
        ok = False

    if r_diff > 1e-6:
        print("MISMATCH in rewards.")
        ok = False

    if ok:
        print("OK: our geometry matches gymnasium CliffWalking-v0 exactly.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
