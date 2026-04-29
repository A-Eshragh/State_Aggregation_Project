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
            # `solve_reduced_lp_batch` returns (1, d) and (1, m) even for
            # unbatched inputs; don't add an extra leading dimension.
            if w_star.dim() == 1:
                w_star = w_star.unsqueeze(0)
            if basis.dim() == 1:
                basis = basis.unsqueeze(0)
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
            grad_M[i].index_copy_(1, Bidx, grad_M_B)

        if not ctx.batched:
            grad_M = grad_M.squeeze(0)
            grad_c = grad_c.squeeze(0)
            grad_b = grad_b.squeeze(0)

        return grad_M, grad_c, grad_b


def differentiable_reduced_lp(M, c, b):
    """User-facing differentiable reduced LP call."""
    return DifferentiableReducedLP.apply(M, c, b)


# ── QP solver (LP + quadratic regularisation) ──────────────────

def solve_reduced_qp_gurobi(M: torch.Tensor, c: torch.Tensor,
                             b: torch.Tensor, eta: float = 0.01):
    """Solve a reduced QP with Gurobi.

    minimize   c^T w  +  (eta/2) ||w||^2
    subject to Mw = b,  w >= 0

    The quadratic term spreads weight across more variables than
    a pure LP (which always sits at a sparse vertex).

    Returns:
        w_star:  (d,)
        active:  (d,)  bool — which variables are positive
        dual:    (m,)  dual values (lambda)
    """
    M_np = M.detach().cpu().numpy().astype(np.float64)
    c_np = c.detach().cpu().numpy().astype(np.float64)
    b_np = b.detach().cpu().numpy().astype(np.float64)
    m, d = M_np.shape

    model = gp.Model()
    model.Params.OutputFlag = 0
    model.Params.FeasibilityTol = 1e-6
    model.Params.OptimalityTol = 1e-6

    w = model.addMVar(shape=d, lb=0.0, vtype=GRB.CONTINUOUS)

    # Objective:  c^T w  +  (eta/2) w^T w
    Q_half = (eta / 2.0) * np.eye(d)   # Gurobi uses x^T Q x (no extra 1/2)
    model.setMObjective(Q_half, c_np, 0.0, xc=w, sense=GRB.MINIMIZE)
    model.addMConstr(M_np, w, "=", b_np)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        # Fallback: penalised slack
        penalty = 1e3
        d_aug = d + 2 * m
        model2 = gp.Model()
        model2.Params.OutputFlag = 0

        all_v = model2.addMVar(shape=d_aug, lb=0.0, vtype=GRB.CONTINUOUS)
        M_aug = np.hstack([M_np, np.eye(m), -np.eye(m)])
        c_aug = np.concatenate([c_np,
                                penalty * np.ones(m),
                                penalty * np.ones(m)])
        Q_aug = np.zeros((d_aug, d_aug))
        Q_aug[:d, :d] = (eta / 2.0) * np.eye(d)

        model2.setMObjective(Q_aug, c_aug, 0.0, xc=all_v, sense=GRB.MINIMIZE)
        model2.addMConstr(M_aug, all_v, "=", b_np)
        model2.optimize()

        if model2.status != GRB.OPTIMAL:
            raise RuntimeError(
                f"QP failed even with slack (status={model2.status}).")

        w_star = torch.tensor(all_v.X[:d], dtype=torch.float32)
        constrs = model2.getConstrs()
        dual = torch.tensor([float(ci.Pi) for ci in constrs[:m]],
                            dtype=torch.float32)
        active = (w_star > 1e-7)
        return w_star, active, dual

    w_star = torch.tensor(w.X, dtype=torch.float32)
    constrs = model.getConstrs()
    dual = torch.tensor([float(ci.Pi) for ci in constrs],
                        dtype=torch.float32)
    active = (w_star > 1e-7)
    return w_star, active, dual


class DifferentiableReducedQP(autograd.Function):
    """Differentiable QP via KKT-based adjoint backward.

    Forward:  Gurobi solves  min c^T w + (eta/2)||w||^2  s.t. Mw=b, w>=0
    Backward: Implicit differentiation on the KKT system for the active set
              A = {j : w_j > 0}:

              [eta I_A,  M_A^T] [w_A   ]   [-c_A]
              [M_A,      0    ] [lambda ] = [b   ]

              Adjoint:
              v_lam = (M_A M_A^T)^{-1} M_A g_A
              v_w   = (g_A - M_A^T v_lam) / eta

              dL/dc_A  = -v_w
              dL/db    = v_lam
              dL/dM_A  = -(outer(lambda, v_w) + outer(v_lam, w_A))
    """

    @staticmethod
    def forward(ctx, M, c, b, eta):
        with torch.no_grad():
            w_star, active, dual = solve_reduced_qp_gurobi(
                M, c, b, eta if isinstance(eta, float) else eta.item())

        ctx.save_for_backward(M, c, b, w_star, active, dual)
        ctx.eta = eta if isinstance(eta, float) else eta.item()
        return w_star

    @staticmethod
    def backward(ctx, grad_w):
        M, c, b, w_star, active, dual = ctx.saved_tensors
        eta = ctx.eta
        device = M.device
        dtype = M.dtype

        A = active.nonzero(as_tuple=True)[0]
        n_A = len(A)
        m_dim = M.shape[0]

        grad_M = torch.zeros_like(M)
        grad_c = torch.zeros_like(c)
        grad_b = torch.zeros_like(b)

        if n_A == 0:
            return grad_M, grad_c, grad_b, None

        M_A = M[:, A]          # (m, |A|)
        w_A = w_star[A]        # (|A|,)
        g_A = grad_w[A]        # (|A|,)
        lam = dual             # (m,)

        # v_lam = (M_A M_A^T)^{-1} M_A g_A  (use lstsq for robustness)
        MMT = M_A @ M_A.T
        MMT += 1e-3 * torch.eye(m_dim, device=device, dtype=dtype)
        v_lam = torch.linalg.lstsq(MMT, (M_A @ g_A).unsqueeze(1)).solution.squeeze(1)

        # v_w = (g_A - M_A^T v_lam) / eta
        v_w = (g_A - M_A.T @ v_lam) / eta

        grad_c[A] = -v_w
        grad_b = v_lam
        grad_M[:, A] = -(torch.outer(lam, v_w) + torch.outer(v_lam, w_A))

        return grad_M, grad_c, grad_b, None


def differentiable_reduced_qp(M, c, b, eta=0.01):
    """User-facing differentiable reduced QP call (with w >= 0)."""
    return DifferentiableReducedQP.apply(M, c, b, eta)


def differentiable_unconstrained_qp(M, c, b, eta=1.0):
    """Differentiable QP without non-negativity constraint.

    Solves:  min  c^T w + (eta/2)||w||^2   s.t.  Mw = b

    Closed-form via KKT:
        lambda = -(MM^T)^{-1} (eta b + M c)
        w*     = -(c + M^T lambda) / eta

    All torch ops → autograd-differentiable, no custom backward needed.
    Returns a dense w (all d components generally nonzero).
    """
    m = M.shape[0]
    reg = 1e-5 * torch.eye(m, device=M.device, dtype=M.dtype)
    MMT = M @ M.T + reg
    rhs = eta * b + M @ c
    lam = -torch.linalg.solve(MMT, rhs)
    w = -(c + M.T @ lam) / eta
    return w
