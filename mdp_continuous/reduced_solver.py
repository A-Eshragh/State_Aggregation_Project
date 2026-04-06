"""
LP solver for the reduced continuous-MDP system.

Solves:
    minimize   c^T w
    subject to Mw = b
               w >= 0

where M in R^{m x d}, c in R^d, b in R^m.

This is the projected occupancy LP:  w* gives y(s,a) = sum_j w_j phi_j(s,a).

Uses Gurobi for forward solve and basis-restricted sensitivity for backward
(same approach as mdp_lp/solvers/differentiable_lp.py).
"""

import torch
import torch.autograd as autograd
import numpy as np
import gurobipy as gp
from gurobipy import GRB


# ── Forward solver (Gurobi) ──────────────────────────────────────

def solve_reduced_lp_gurobi(M: torch.Tensor, c: torch.Tensor, b: torch.Tensor):
    """Solve a single reduced LP with Gurobi.

    minimize   c^T w
    subject to Mw = b,  w >= 0

    If the exact LP is infeasible (common with sample-estimated M),
    falls back to a penalized formulation with slack variables:
        minimize c^T w + penalty * 1^T (s+ + s-)
        subject to Mw + s+ - s- = b,  w, s+, s- >= 0

    Args:
        M: (m, d) constraint matrix
        c: (d,)   cost vector
        b: (m,)   RHS

    Returns:
        w_star:    (d,) optimal primal solution
        basis_idx: (m,) indices of basis columns
        dual:      (m,) dual values
    """
    M_np = M.detach().cpu().numpy().astype(np.float64)
    c_np = c.detach().cpu().numpy().astype(np.float64)
    b_np = b.detach().cpu().numpy().astype(np.float64)
    m, d = M_np.shape

    # ── Attempt 1: exact LP ──
    model = gp.Model()
    model.Params.OutputFlag = 0
    model.Params.FeasibilityTol = 1e-6
    model.Params.OptimalityTol = 1e-6

    w = model.addMVar(shape=d, lb=0.0, vtype=GRB.CONTINUOUS)
    model.setObjective(c_np @ w, GRB.MINIMIZE)
    model.addMConstr(M_np, w, "=", b_np)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        # ── Attempt 2: penalized slack ──
        penalty = 1e3
        model2 = gp.Model()
        model2.Params.OutputFlag = 0

        w2 = model2.addMVar(shape=d, lb=0.0, vtype=GRB.CONTINUOUS)
        sp = model2.addMVar(shape=m, lb=0.0, vtype=GRB.CONTINUOUS)
        sn = model2.addMVar(shape=m, lb=0.0, vtype=GRB.CONTINUOUS)

        # Build augmented system: [M | I | -I] [w; s+; s-] = b
        M_aug = np.hstack([M_np, np.eye(m), -np.eye(m)])
        c_aug = np.concatenate([c_np, penalty * np.ones(m), penalty * np.ones(m)])

        all_vars = model2.addMVar(shape=d + 2 * m, lb=0.0, vtype=GRB.CONTINUOUS)
        model2.remove(w2)
        model2.remove(sp)
        model2.remove(sn)
        model2.update()

        model2.setObjective(c_aug @ all_vars, GRB.MINIMIZE)
        model2.addMConstr(M_aug, all_vars, "=", b_np)
        model2.optimize()

        if model2.status != GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi failed even with slack (status={model2.status}).")

        w_star = torch.tensor(all_vars.X[:d], dtype=torch.float32)

        # Extract basis from first d variables
        vb = all_vars.VBasis
        basis_idx = torch.tensor(
            [j for j in range(d) if vb[j] == 0][:m],
            dtype=torch.long,
        )

        constrs = model2.getConstrs()
        pi_vals = [float(ci.Pi) for ci in constrs]
        dual = torch.tensor(pi_vals[:m], dtype=torch.float32)

        return w_star, _pad_basis(basis_idx, m, d, w_star), dual

    w_star = torch.tensor(w.X, dtype=torch.float32)

    # Extract basis from variable basis status
    vb = w.VBasis
    basis_idx = torch.tensor(
        [j for j in range(d) if vb[j] == 0][:m],
        dtype=torch.long,
    )

    constrs = model.getConstrs()
    pi_vals = [float(ci.Pi) for ci in constrs]
    dual = torch.tensor(pi_vals[:m], dtype=torch.float32)

    return w_star, _pad_basis(basis_idx, m, d, w_star), dual


def _pad_basis(basis_idx, m, d, w_star):
    """Ensure basis has exactly m elements by padding with largest-w columns."""
    if basis_idx.numel() >= m:
        return basis_idx[:m]

    existing = set(basis_idx.tolist())
    remaining = [(j, w_star[j].item()) for j in range(d) if j not in existing]
    remaining.sort(key=lambda x: -x[1])
    for j, _ in remaining:
        if basis_idx.numel() >= m:
            break
        basis_idx = torch.cat([basis_idx, torch.tensor([j], dtype=torch.long)])
    return basis_idx


def solve_reduced_lp_batch(M: torch.Tensor, c: torch.Tensor, b: torch.Tensor):
    """Batched version: M (B, m, d), c (B, d), b (B, m)."""
    if M.dim() == 2:
        w, basis, dual = solve_reduced_lp_gurobi(M, c, b)
        return w.unsqueeze(0), basis.unsqueeze(0), dual.unsqueeze(0)

    B = M.shape[0]
    ws, bases, duals = [], [], []
    for i in range(B):
        w_i, b_i, d_i = solve_reduced_lp_gurobi(M[i], c[i], b[i])
        ws.append(w_i)
        bases.append(b_i)
        duals.append(d_i)
    return torch.stack(ws), torch.stack(bases), torch.stack(duals)


# ── Differentiable LP layer ──────────────────────────────────────

class DifferentiableReducedLP(autograd.Function):
    """Differentiable LP for the reduced continuous-MDP system.

    Forward:  Gurobi solves  min c^T w  s.t.  Mw = b, w >= 0
    Backward: Basis-restricted sensitivity (same as differentiable_lp.py):
              At optimum, w_B = M_B^{-1} b, w_N = 0.
              dL/db   = M_B^{-T} (dL/dw_B)
              dL/dM_B = -(M_B^{-T} dL/dw_B) w_B^T
              dL/dc   = 0   (LP vertex doesn't depend on cost)
    """

    @staticmethod
    def forward(ctx, M, c, b):
        """
        Args:
            M: (m, d) or (B, m, d)
            c: (d,)   or (B, d)
            b: (m,)   or (B, m)

        Returns:
            w_star: (d,) or (B, d)
        """
        batched = M.dim() == 3

        with torch.no_grad():
            w_star, basis, dual = solve_reduced_lp_batch(M, c, b)

        ctx.batched = batched
        ctx.save_for_backward(M, c, b, w_star, basis)

        return w_star.squeeze(0) if not batched else w_star

    @staticmethod
    def backward(ctx, grad_w):
        M, c, b, w_star, basis = ctx.saved_tensors

        if not ctx.batched:
            M = M.unsqueeze(0)
            c = c.unsqueeze(0)
            b = b.unsqueeze(0)
            w_star = w_star.unsqueeze(0)
            grad_w = grad_w.unsqueeze(0)

        B, m_dim, d = M.shape
        device = M.device
        dtype = M.dtype

        grad_M = torch.zeros_like(M)
        grad_c = torch.zeros_like(c)  # always zero for LP
        grad_b = torch.zeros_like(b)

        for i in range(B):
            Bidx = basis[i]                     # (m,)
            M_B = M[i, :, Bidx]                 # (m, m)
            w_B = w_star[i, Bidx]               # (m,)
            g_B = grad_w[i, Bidx]               # (m,)

            # dL/db = M_B^{-T} g_B
            x = torch.linalg.solve(M_B.T, g_B)  # (m,)
            grad_b[i] = x

            # dL/dM_B = -x w_B^T
            grad_M_B = -torch.outer(x, w_B)     # (m, m)
            for j_local, j_global in enumerate(Bidx):
                grad_M[i, :, j_global] = grad_M_B[:, j_local]

        if not ctx.batched:
            grad_M = grad_M.squeeze(0)
            grad_c = grad_c.squeeze(0)
            grad_b = grad_b.squeeze(0)

        return grad_M, grad_c, grad_b


def differentiable_reduced_lp(M, c, b):
    """User-facing differentiable reduced LP call."""
    return DifferentiableReducedLP.apply(M, c, b)
