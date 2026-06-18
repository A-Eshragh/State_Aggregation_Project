"""
Cliff-Walking MDP (Gymnasium-style 4x12 grid) wired into the LP formulation.

Grid layout (4 rows x 12 cols, 48 states):

    row 0: 0  1  2  3  4  5  6  7  8  9  10 11
    row 1: 12 13 14 ... ... ... ... ... ... 23
    row 2: 24 25 ... ... ... ... ... ... ... 35
    row 3: 36 C  C  C  C  C  C  C  C  C  C  47
             ^start                           ^goal
    (C = cliff)

Actions (canonical gym order): 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT.

Rewards: -1 per step, -100 on falling into cliff (and teleport to start),
0 at the goal (terminal -> self loop).

Stochasticity is introduced by weather: each weather regime defines a slip
distribution over K=5 outcomes per action:
    0: intended
    1: slip LEFT  (rotate -90° from intended heading)
    2: slip RIGHT (rotate +90° from intended heading)
    3: slip BACK  (rotate 180° from intended heading)
    4: NO MOVE    (stay in place)

The slip distribution lives in R^{A x K} and is shared across states (a
whole-floor weather effect). Transitions are then built deterministically
from the slip distribution.
"""

from __future__ import annotations

import torch

from mdp_lp.mdp import MDP


# ---------------------------------------------------------------------------
# Grid constants
# ---------------------------------------------------------------------------
N_ROWS = 4
N_COLS = 12
N_STATES = N_ROWS * N_COLS  # 48
N_ACTIONS = 4
N_SLIP = 5  # intended, left, right, back, no-move

START_ROW, START_COL = 3, 0
GOAL_ROW, GOAL_COL = 3, 11
CLIFF_COLS = list(range(1, 11))  # columns 1..10 of row 3

STEP_REWARD = -1.0
# CLIFF_REWARD is the penalty applied when the agent lands on a cliff cell
# (and gets teleported back to start). The canonical Gymnasium CliffWalking-v0
# value is -100. For experiments where we want a softer cliff (so the reward
# landscape is less catastrophically asymmetric and DFL gradients are not
# dominated by rare cliff events) we allow this to be overridden at runtime
# via `set_cliff_penalty(...)`. Both the transition tables and the geometry
# test are rebuilt on the fly when this value changes.
CLIFF_REWARD = -100.0
GOAL_REWARD = 0.0

# Action deltas (dr, dc): 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
ACTION_DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def state_idx(row: int, col: int) -> int:
    return row * N_COLS + col


START_STATE = state_idx(START_ROW, START_COL)
GOAL_STATE = state_idx(GOAL_ROW, GOAL_COL)
CLIFF_STATES = [state_idx(3, c) for c in CLIFF_COLS]


def is_cliff(s: int) -> bool:
    return s in CLIFF_STATES


def is_goal(s: int) -> bool:
    return s == GOAL_STATE


# ---------------------------------------------------------------------------
# Slip -> actual action
# ---------------------------------------------------------------------------
# Rotation mapping for the slip kind (in the (dr, dc) plane):
#   kind 0 (intended): no rotation
#   kind 1 (LEFT slip):  -90°   (dr, dc) -> (-dc, dr)
#   kind 2 (RIGHT slip): +90°   (dr, dc) -> (dc, -dr)
#   kind 3 (BACK slip): 180°    (dr, dc) -> (-dr, -dc)
#   kind 4 (NO MOVE):   zero
def _slip_delta(intended_a: int, slip_kind: int) -> tuple[int, int]:
    if slip_kind == 4:
        return (0, 0)
    dr, dc = ACTION_DELTAS[intended_a]
    if slip_kind == 0:
        return (dr, dc)
    if slip_kind == 1:
        return (-dc, dr)
    if slip_kind == 2:
        return (dc, -dr)
    if slip_kind == 3:
        return (-dr, -dc)
    raise ValueError(f"Invalid slip kind {slip_kind}")


# ---------------------------------------------------------------------------
# Precomputed (next_state, reward) tables for every (s, a, k) triple.
# These are fixed by the environment geometry, independent of weather.
# ---------------------------------------------------------------------------
def _build_next_and_reward_tables():
    next_idx = torch.zeros(N_STATES, N_ACTIONS, N_SLIP, dtype=torch.long)
    rew = torch.zeros(N_STATES, N_ACTIONS, N_SLIP, dtype=torch.float32)

    for s in range(N_STATES):
        if is_goal(s):
            # Goal is a zero-reward absorbing self loop.
            next_idx[s, :, :] = s
            rew[s, :, :] = GOAL_REWARD
            continue
        # Note: cliff states are treated like any other non-terminal cell here
        # (matching gymnasium CliffWalking-v0). They are unreachable because
        # landing on a cliff from any adjacent state teleports to start
        # (handled below by the is_cliff(ns) branch), and gamma puts zero
        # mass on them, so their occupancy is always zero.

        row, col = s // N_COLS, s % N_COLS
        for a in range(N_ACTIONS):
            for k in range(N_SLIP):
                dr, dc = _slip_delta(a, k)
                nr = max(0, min(N_ROWS - 1, row + dr))
                nc = max(0, min(N_COLS - 1, col + dc))
                ns = state_idx(nr, nc)
                if is_cliff(ns):
                    next_idx[s, a, k] = START_STATE
                    rew[s, a, k] = CLIFF_REWARD
                else:
                    # Gym semantics: every step costs -1, including the step
                    # that reaches the goal. The goal itself (as a source
                    # state) is handled above as an absorbing self-loop with
                    # reward 0.
                    next_idx[s, a, k] = ns
                    rew[s, a, k] = STEP_REWARD
    return next_idx, rew


# Cache tables at import time.
NEXT_IDX, REW_PER_SLIP = _build_next_and_reward_tables()  # (S, A, K), (S, A, K)


def set_cliff_penalty(penalty: float) -> None:
    """Override the cliff-reward and rebuild the cached reward table.

    The gym reference value is -100. Setting this to a less extreme value
    (e.g. -10 or -5) yields a "soft cliff" variant of CliffWalking with a
    much smoother reward landscape -- useful for DFL experiments because
    small slip mispredictions no longer produce wild swings in policy
    value. This variant is NOT gym-equivalent; the geometry test
    (`test_geometry.py`) only passes at the default -100.
    """
    global CLIFF_REWARD, REW_PER_SLIP, NEXT_IDX
    CLIFF_REWARD = float(penalty)
    NEXT_IDX, REW_PER_SLIP = _build_next_and_reward_tables()


def get_cliff_penalty() -> float:
    return CLIFF_REWARD


# ---------------------------------------------------------------------------
# Build P, r from a slip distribution
# ---------------------------------------------------------------------------
def build_P_r_from_slip(slip_dist: torch.Tensor):
    """
    Construct transition kernel P (B, S, A, S) and per-state reward r (B, S, A)
    from a per-sample slip distribution.

    Args:
        slip_dist: either
            * (B, A, K) — state-independent slip (v1/v2 regimes), or
            * (B, S, A, K) — state-dependent slip (v3 with volatile cells).
        In both cases rows-sum-to-one along the last dim.

    Returns:
        P: (B, S, A, S), rows-sum-to-one along the last dim.
        r: (B, S, A).
    """
    if slip_dist.dim() == 3:
        if slip_dist.shape[1] != N_ACTIONS or slip_dist.shape[2] != N_SLIP:
            raise ValueError(
                f"slip_dist (B,A,K) must have A={N_ACTIONS}, K={N_SLIP}; got {tuple(slip_dist.shape)}"
            )
        B = slip_dist.shape[0]
        state_dependent = False
    elif slip_dist.dim() == 4:
        if (slip_dist.shape[1] != N_STATES or slip_dist.shape[2] != N_ACTIONS
                or slip_dist.shape[3] != N_SLIP):
            raise ValueError(
                f"slip_dist (B,S,A,K) must have S={N_STATES}, A={N_ACTIONS}, K={N_SLIP}; "
                f"got {tuple(slip_dist.shape)}"
            )
        B = slip_dist.shape[0]
        state_dependent = True
    else:
        raise ValueError(
            f"slip_dist must have 3 or 4 dims; got {slip_dist.dim()}"
        )

    device = slip_dist.device
    dtype = slip_dist.dtype

    next_idx = NEXT_IDX.to(device)                                     # (S, A, K)
    rew_per_slip = REW_PER_SLIP.to(device=device, dtype=dtype)         # (S, A, K)

    if state_dependent:
        slip_expanded = slip_dist                                      # (B, S, A, K)
        r = torch.einsum("bsak,sak->bsa", slip_dist, rew_per_slip)
    else:
        slip_expanded = slip_dist.unsqueeze(1).expand(B, N_STATES, N_ACTIONS, N_SLIP)
        r = torch.einsum("bak,sak->bsa", slip_dist, rew_per_slip)

    idx_expanded = next_idx.unsqueeze(0).expand(B, N_STATES, N_ACTIONS, N_SLIP)
    P = torch.zeros(B, N_STATES, N_ACTIONS, N_STATES, device=device, dtype=dtype)
    P.scatter_add_(dim=3, index=idx_expanded, src=slip_expanded)

    return P, r


# ---------------------------------------------------------------------------
# MDP class
# ---------------------------------------------------------------------------
class CliffWalkingLP(MDP):
    """Cliff-Walking MDP in the occupancy-LP format."""

    def __init__(self, P: torch.Tensor, r: torch.Tensor, beta: float,
                 gamma: torch.Tensor | None = None, start_bias: float = 0.99):
        """
        Args:
            P: (B, S, A, S) transition tensor.
            r: (B, S, A) reward tensor.
            beta: discount factor.
            gamma: optional (B, S) initial-state distribution. If None, a
                mostly-start distribution with small positive mass on all
                states is used (non-degeneracy for the LP).
            start_bias: fraction of initial mass placed on the start state
                when building the default gamma.
        """
        B, S, A, _ = P.shape
        assert S == N_STATES and A == N_ACTIONS, (
            f"Unexpected (S, A)=({S},{A}); expected ({N_STATES},{N_ACTIONS})"
        )
        self.P = P
        self.r = r
        self.n_states = S
        self.n_actions = A

        if gamma is None:
            # Put most mass on start, small positive mass on every other
            # non-cliff state (for LP non-degeneracy). Cliff states get 0.
            n_nonstart_nonclif = S - 1 - len(CLIFF_STATES)
            base = (1.0 - start_bias) / max(1, n_nonstart_nonclif)
            g = torch.full((B, S), base, device=P.device, dtype=P.dtype)
            g[:, START_STATE] = start_bias
            for c in CLIFF_STATES:
                g[:, c] = 0.0
            g = g / g.sum(dim=1, keepdim=True)
            gamma = g
        super().__init__(beta, gamma=gamma)


# Convenience exports
__all__ = [
    "N_ROWS", "N_COLS", "N_STATES", "N_ACTIONS", "N_SLIP",
    "START_STATE", "GOAL_STATE", "CLIFF_STATES",
    "ACTION_DELTAS", "NEXT_IDX", "REW_PER_SLIP",
    "CLIFF_REWARD", "STEP_REWARD", "GOAL_REWARD",
    "set_cliff_penalty", "get_cliff_penalty",
    "build_P_r_from_slip", "CliffWalkingLP",
    "state_idx", "is_cliff", "is_goal",
]
