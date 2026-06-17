"""
Sanity check: solve the oracle LP for each regime and report the optimal
action at cells along row 2 (the cliff-adjacent row). For regime_set=v2
we expect the SAFE family to pick RIGHT (hug row 2) while the DANGER
family should pick UP (evacuate to row 1 or row 0).

Usage:
    python -m mdp_cw.verify_policies --regime_set v2
"""

from __future__ import annotations

import argparse

import torch

from mdp_cw.cliff_mdp import CliffWalkingLP, N_COLS, N_STATES, set_cliff_penalty, state_idx
from mdp_cw import weather_dynamics as wd
from mdp_cw.weather_dynamics import (
    ground_truth_mdp_tensors,
    n_regimes,
    set_regime_set,
)
from mdp_lp.solvers.dense_lp_solver import DenseLPSolver
from mdp_lp.utils import solve_lp


ACTION_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]


def solve_for_all_regimes(beta: float = 0.95):
    solver = DenseLPSolver()

    def lp_solver(H, rr, gamma):
        return solver.solve(H, rr, gamma)

    ids = torch.arange(n_regimes())
    _, P, r = ground_truth_mdp_tensors(ids)       # (Z, S, A, S), (Z, S, A)

    results = []
    for i in range(n_regimes()):
        mdp = CliffWalkingLP(P=P[i:i+1], r=r[i:i+1], beta=beta)
        y, obj, _, _ = solve_lp(mdp, lp_solver)
        y = y.squeeze(0).view(N_STATES, -1)       # (S, A)
        pi = y.argmax(dim=1).tolist()             # greedy action per state (ok for deterministic-ish LPs)
        results.append((i, wd.REGIME_NAMES[i], float(obj.squeeze()), pi))
    return results


def summarise(results):
    print()
    print(f"{'id':>2}  {'regime':<20}  {'V*':>9}  row2 path (cols 0..11)")
    print("-" * 72)
    for i, name, v, pi in results:
        row2 = [state_idx(2, c) for c in range(N_COLS)]
        row2_actions = [ACTION_NAMES[pi[s]][0] for s in row2]
        path = " ".join(row2_actions)
        print(f"{i:>2}  {name:<20}  {v:>9.3f}   {path}")

    # row-choice summary for each regime: from col 0, is the optimal policy to
    # stay on row 2 (first action = RIGHT) or to evacuate upward (first = UP)?
    print()
    print("Optimal first-move from (2, 0):")
    for i, name, v, pi in results:
        first = ACTION_NAMES[pi[state_idx(2, 0)]]
        print(f"  regime {i} ({name}): {first}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime_set", choices=["v1", "v2", "v3", "v4", "v5", "v6"], default="v2")
    ap.add_argument("--beta", type=float, default=0.95)
    ap.add_argument("--cliff_penalty", type=float, default=-100.0)
    args = ap.parse_args()
    set_regime_set(args.regime_set)
    set_cliff_penalty(args.cliff_penalty)
    print(f"Regime set = {args.regime_set}  ({n_regimes()} regimes)  cliff_penalty={args.cliff_penalty}")
    res = solve_for_all_regimes(beta=args.beta)
    summarise(res)


if __name__ == "__main__":
    main()
