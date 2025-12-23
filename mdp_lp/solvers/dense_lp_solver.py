# lp/dense_lp_solver.py

import torch
import gurobipy as gp
from gurobipy import GRB


class DenseLPSolver:
    """
    Dense LP solver using Gurobi.
    Solves the discounted MDP LP for each batch element:

        maximize     r^T y
        subject to   H y = gamma
                     y >= 0

    Inputs:
        H:      (batch, S, SA)
        r:      (batch, SA)
        gamma:  (batch, S)

    Outputs:
        y_star: (batch, SA)
        basis:  (batch, S)   # basis column indices
    """

    def __init__(self):
        pass

    def solve_single(self, H_i, r_i, gamma_i):
        """
        Solve ONE LP using Gurobi.

        H_i:     (S, SA)
        r_i:     (SA,)
        gamma_i: (S,)
        """

        S, SA = H_i.shape

        model = gp.Model()
        model.Params.OutputFlag = 0  # silence logs

        # Variables y_j >= 0
        y = model.addMVar(shape=SA, lb=0.0, vtype=GRB.CONTINUOUS)

        # Objective: maximize r^T y
        model.setObjective(r_i @ y, GRB.MAXIMIZE)

        # Equality constraints: H y = gamma
        model.addMConstr(H_i, y, '=', gamma_i)

        model.optimize()

        if model.status != GRB.OPTIMAL:
            raise RuntimeError("Gurobi failed to solve LP.")

        # Extract y*
        y_star = y.X  # numpy array

        # Extract basis:
        # VBasis[j] == 0  → basic variable
        # VBasis[j] != 0 → nonbasic
        vb = y.VBasis

        basis_idx = torch.tensor(
            [j for j in range(SA) if vb[j] == 0][:S],
            dtype=torch.long
        )

        return torch.tensor(y_star, dtype=torch.float32), basis_idx

    def solve(self, H, r, gamma):
        """
        Solve LP for the whole batch.

        Returns:
            y_star: (batch, SA)
            basis:  (batch, S)
        """
        batch, S, SA = H.shape

        ys = []
        bases = []

        for b in range(batch):
            y_b, B_b = self.solve_single(H[b].detach().cpu().numpy(),
                                         r[b].detach().cpu().numpy(),
                                         gamma[b].detach().cpu().numpy())
            ys.append(y_b)
            bases.append(B_b)

        return torch.stack(ys, dim=0), torch.stack(bases, dim=0)


def solve_lp_dense(H, r, gamma):
    """
    Solve LP for single instance (not batched).
    
    H:     (S, SA) or (batch, S, SA) - if batched, solves first element
    r:     (SA,) or (batch, SA)
    gamma: (S,) or (batch, S)
    
    Returns:
        y_star: (SA,)
        basis:  (S,)
    """
    solver = DenseLPSolver()
    # Handle both single and batched inputs
    if H.dim() == 3:
        # Batched - solve first element
        y_star, basis = solver.solve_single(H[0].detach().cpu().numpy(),
                                           r[0].detach().cpu().numpy(),
                                           gamma[0].detach().cpu().numpy())
    else:
        # Single instance
        y_star, basis = solver.solve_single(H.detach().cpu().numpy(),
                                           r.detach().cpu().numpy(),
                                           gamma.detach().cpu().numpy())
    return y_star, basis


def solve_lp_gurobi(H, r, gamma):
    """
    Alias for solve_lp_dense for backward compatibility.
    Solves batched LP and returns only y_star (no basis).
    
    H:     (batch, S, SA)
    r:     (batch, SA)
    gamma: (batch, S)
    
    Returns:
        y_star: (batch, SA)
    """
    solver = DenseLPSolver()
    y_star, basis = solver.solve(H, r, gamma)
    return y_star, basis


