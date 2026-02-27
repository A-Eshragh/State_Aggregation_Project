#!/usr/bin/env python3
"""
Inspect MDP policy from LP occupancy for sanity checking.

Loads a run's results.json (or explicit MDP params), builds the inventory MDP
for a few (lambda, p) scenarios, solves the LP to get y*, and derives the
policy π(a|s) = y*(s,a) / sum_a y*(s,a). Prints a summary so you can verify
order-up-to behavior (e.g. order more when inventory is low, less when high).

Usage:
  python -m mdp_lp.inspect_policy mdp_lp/exp_b/high_margin/dfl/tag_default_high/epochs_100
  python -m mdp_lp.inspect_policy path/to/results.json --n_samples 5
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# Add project root for imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from mdp_lp.inventory_elasticity import create_inventory_from_lambda
from mdp_lp.solvers.dense_lp_solver import DenseLPSolver
from mdp_lp.utils import solve_lp


def load_config_from_run(run_path: Path) -> dict:
    """Load experiment config from a run directory or results.json path."""
    path = Path(run_path)
    if path.is_dir():
        json_path = path / "results.json"
    else:
        json_path = path
    if not json_path.exists():
        raise FileNotFoundError(f"Not found: {json_path}")
    data = json.loads(json_path.read_text())
    return data


def occupancy_to_policy(y_star: torch.Tensor, S: int, A: int) -> torch.Tensor:
    """
    Convert occupancy y* (batch, S*A) to policy π(a|s): (batch, S, A).
    π(a|s) = y*(s,a) / sum_a y*(s,a). Normalize so each row sums to 1.
    """
    batch = y_star.shape[0]
    y = y_star.view(batch, S, A)
    state_mass = y.sum(dim=2, keepdim=True).clamp(min=1e-12)
    pi = y / state_mass
    return pi


def main():
    parser = argparse.ArgumentParser(description="Inspect MDP policy from LP occupancy")
    parser.add_argument("run_path", type=str, help="Path to run dir or results.json")
    parser.add_argument("--n_samples", type=int, default=3,
                        help="Number of (lambda, p) scenarios to inspect")
    parser.add_argument("--out", type=str, default=None,
                        help="Optional path to save policy summary (text)")
    args = parser.parse_args()

    run_path = Path(args.run_path)
    config = load_config_from_run(run_path)

    b_max = int(config.get("b_max", 50))
    s_max = int(config.get("s_max", 100))
    q_max = int(config.get("q_max", 20))
    d_max = int(config.get("d_max", 30))
    foc = float(config.get("foc", 1.0))
    voc = float(config.get("voc", 0.5))
    hc = float(config.get("hc", 0.1))
    bc = float(config.get("bc", 0.2))
    beta = float(config.get("beta", 0.95))
    p_mean = float(config.get("p_lognormal_mean", 10.0))
    p_std = float(config.get("p_lognormal_std", 0.5))
    beta_0 = float(config.get("beta_0", 2.0))
    beta_1 = float(config.get("beta_1", -1.5))

    S = b_max + s_max + 1
    A = q_max + 1

    # Sample a few (lambda, p) for exp b style: p lognormal, lambda = exp(beta_0 + beta_1*log(p))
    torch.manual_seed(42)
    log_p = torch.randn(args.n_samples) * p_std + torch.log(torch.tensor(p_mean))
    p = torch.exp(log_p)
    # True elasticity: lambda = exp(beta_0 + beta_1 * log(p))
    lambda_true = torch.exp(beta_0 + beta_1 * log_p)

    lp_solver = DenseLPSolver()
    def solver(H, r, gamma):
        y, basis, _ = lp_solver.solve(H, r, gamma)
        return y, basis

    lines = []
    def log(s):
        print(s)
        lines.append(s)

    log(f"Inspecting MDP policy (S={S}, A={A}, b_max={b_max}, s_max={s_max})")
    log(f"Costs: foc={foc}, voc={voc}, hc={hc}, bc={bc}, beta={beta}")
    log("")

    for i in range(args.n_samples):
        lam = lambda_true[i:i+1]
        price = p[i:i+1]
        mdp = create_inventory_from_lambda(
            lam, b_max, s_max, q_max, d_max,
            foc, voc, hc, bc, beta, price
        )
        y_star, obj, _ = solve_lp(mdp, solver)
        pi = occupancy_to_policy(y_star, S, A)  # (1, S, A)
        pi = pi[0]  # (S, A)

        best_actions = pi.argmax(dim=1)  # (S,) order quantity per state
        inv_levels = torch.arange(S, dtype=torch.float32) - b_max  # inventory = s - b_max

        log(f"--- Scenario {i+1}: λ={lam.item():.3f}, p={price.item():.3f}, v*={obj.item():.2f} ---")
        # Print a subset of states (e.g. every 10th, plus boundaries) so output is readable
        step = max(1, S // 20)
        indices = list(range(0, S, step)) + ([S-1] if S-1 not in range(0, S, step) else [])
        indices = sorted(set(indices))
        for s in indices:
            inv = int(inv_levels[s].item())
            a_star = int(best_actions[s].item())
            prob_a = pi[s, a_star].item()
            log(f"  inv={inv:4d}  best_order={a_star:2d}  π(a*|s)={prob_a:.3f}")
        log("")

    if args.out:
        Path(args.out).write_text("\n".join(lines))
        print(f"Wrote summary to {args.out}")


if __name__ == "__main__":
    main()
