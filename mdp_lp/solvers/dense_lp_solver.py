# lp/dense_lp_solver.py

import numpy as np
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
        dual:   (batch, S)   # dual values (lambda) for H y = gamma
    """

    def __init__(self):
        pass

    def solve_single(self, H_i, r_i, gamma_i):
        """
        Solve ONE LP using Gurobi.

        H_i:     (S, SA)
        r_i:     (SA,)
        gamma_i: (S,)

        Returns:
            y_star: (SA,) primal solution
            basis_idx: (S,) basis column indices
            dual: (S,) dual values for H y = gamma (lambda)
        """

        S, SA = H_i.shape

        model = gp.Model()
        model.Params.OutputFlag = 0

        y = model.addMVar(shape=SA, lb=0.0, vtype=GRB.CONTINUOUS)
        model.setObjective(r_i @ y, GRB.MAXIMIZE)
        model.addMConstr(H_i, y, '=', gamma_i)

        model.optimize()

        if model.status != GRB.OPTIMAL:
            raise RuntimeError("Gurobi failed to solve LP.")

        y_star = y.X
        vb = y.VBasis

        basis_idx = torch.tensor(
            [j for j in range(SA) if vb[j] == 0][:S],
            dtype=torch.long
        )

        # Dual values (Pi) for equality constraints H y = gamma
        # addMConstr yields one MConstr with Pi as array (S,); addConstr yields S constrs with scalar Pi
        constrs = model.getConstrs()
        pi_vals = []
        for c in constrs:
            p = c.Pi
            if hasattr(p, "__len__") and not isinstance(p, str):
                pi_vals.extend(np.atleast_1d(p).flatten().tolist())
            else:
                pi_vals.append(float(p))
        dual = torch.tensor(pi_vals, dtype=torch.float32)

        return torch.tensor(y_star, dtype=torch.float32), basis_idx, dual

    def solve(self, H, r, gamma):
        """
        Solve LP for the whole batch.

        Returns:
            y_star: (batch, SA)
            basis:  (batch, S)
            dual:   (batch, S)  dual values for H y = gamma
        """
        batch, S, SA = H.shape

        ys = []
        bases = []
        duals = []

        for b in range(batch):
            y_b, B_b, dual_b = self.solve_single(H[b].detach().cpu().numpy(),
                                                 r[b].detach().cpu().numpy(),
                                                 gamma[b].detach().cpu().numpy())
            ys.append(y_b)
            bases.append(B_b)
            duals.append(dual_b)

        return torch.stack(ys, dim=0), torch.stack(bases, dim=0), torch.stack(duals, dim=0)


def solve_lp_dense(H, r, gamma):
    """
    Solve LP for single instance (not batched).

    H:     (S, SA) or (batch, S, SA) - if batched, solves first element
    r:     (SA,) or (batch, SA)
    gamma: (S,) or (batch, S)

    Returns:
        y_star: (SA,)
        basis:  (S,)
        dual:   (S,)  dual values for H y = gamma
    """
    solver = DenseLPSolver()
    if H.dim() == 3:
        y_star, basis, dual = solver.solve_single(H[0].detach().cpu().numpy(),
                                                 r[0].detach().cpu().numpy(),
                                                 gamma[0].detach().cpu().numpy())
    else:
        y_star, basis, dual = solver.solve_single(H.detach().cpu().numpy(),
                                                 r.detach().cpu().numpy(),
                                                 gamma.detach().cpu().numpy())
    return y_star, basis, dual


def solve_lp_gurobi(H, r, gamma):
    """Solve batched LP and return y_star, basis, and dual."""
    solver = DenseLPSolver()
    y_star, basis, dual = solver.solve(H, r, gamma)
    return y_star, basis, dual


