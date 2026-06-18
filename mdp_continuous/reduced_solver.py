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
    model.Params.Method = 2
    model.Params.Threads = 8

    w = model.addMVar(shape=d, lb=0.0, vtype=GRB.CONTINUOUS)
    model.setObjective(c_np @ w, GRB.MINIMIZE)
    model.addMConstr(M_np, w, "=", b_np)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        # ── Attempt 2: penalized slack ──
        penalty = 1e3
        model2 = gp.Model()
        model2.Params.OutputFlag = 0
        model2.Params.Threads = 1

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
    have = basis_idx.numel()
    if have >= m:
        return basis_idx[:m]

    # Vectorized: mask out the existing basis, then topk over the rest.
    # Avoids per-element .item() syncs and repeated torch.cat reallocs.
    mask = torch.ones(d, dtype=torch.bool)
    mask[basis_idx] = False
    candidate_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
    candidate_w = w_star.detach()[candidate_idx]
    k = m - have
    if candidate_idx.numel() <= k:
        pad = candidate_idx
    else:
        topk = torch.topk(candidate_w, k=k, largest=True).indices
        pad = candidate_idx[topk]
    return torch.cat([basis_idx, pad.to(basis_idx.dtype)])


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
    M_np = M.detach().cpu().numpy().astype(np.float64)
    c_np = c.detach().cpu().numpy().astype(np.float64)
    b_np = b.detach().cpu().numpy().astype(np.float64)
    m, d = M_np.shape

    model = gp.Model()
    model.Params.OutputFlag = 0
    model.Params.FeasibilityTol = 1e-6
    model.Params.OptimalityTol = 1e-6
    model.Params.Threads = 1  # avoid oversubscription under N-parallel workers

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
        model2.Params.Threads = 1

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



_QPTH_FN = None


def _get_qpth_qpfunction(eps: float = 1e-7, max_iter: int = 50):
    """Lazy singleton — qpth import is heavy and may not be installed."""
    global _QPTH_FN
    if _QPTH_FN is None:
        from qpth.qp import QPFunction, QPSolvers
        _QPTH_FN = QPFunction(
            eps=eps, maxIter=max_iter, solver=QPSolvers.PDIPM_BATCHED,
            verbose=-1,
        )
    return _QPTH_FN


def differentiable_qp_qpth(M, c, b, eta: float = 0.01,
                           feas_tol: float = 1e-3):
    """qpth-based differentiable reduced QP — replaces Gurobi-QP forward+adjoint.

    Solves   min  c^T w + (eta/2) ||w||^2   s.t.  M w = b,  w >= 0

    qpth's PDIPM is GPU-batched and natively differentiable, so we get
    autograd for free (no custom backward). Avoids the per-call Gurobi
    Python overhead that dominates the Gurobi-QP path.

    Non-convergence guard: if qpth's PDIPM hits maxIter without satisfying
    eps, the returned w may violate (Mw=b) or be slightly negative. We
    expose the KKT residual via the second return so the caller can skip
    the optimizer step on bad solves (rather than silently clamping).

    Args:
        M, c, b: same convention as the Gurobi reduced QP — single-instance
            shapes (m, d), (d,), (m,).
        eta: quadratic regularization strength (>0 required by PDIPM).
        feas_tol: max allowed equality + non-negativity residual.

    Returns:
        w_star:  (d,) torch tensor, autograd-tracked through M, c, b.
        kkt_res: (scalar tensor) max(|Mw - b|, max(-w, 0)) — for bad-step
            detection; the caller decides what to do with it.
    """
    if eta <= 0:
        raise ValueError(f"eta must be > 0 for qpth (PDIPM); got {eta}")

    qp_fn = _get_qpth_qpfunction()

    M_b = M.unsqueeze(0)            # (1, m, d)
    c_b = c.unsqueeze(0)            # (1, d)
    b_b = b.unsqueeze(0)            # (1, m)
    d = M.shape[1]
    device = M.device
    dtype = M.dtype

    # qpth: min 0.5 w^T Q w + p^T w  s.t.  Gw <= h, Aw = b
    Q = (eta * torch.eye(d, device=device, dtype=dtype)).unsqueeze(0)  # (1, d, d)
    p = c_b
    G = (-torch.eye(d, device=device, dtype=dtype)).unsqueeze(0)       # (1, d, d)
    h = torch.zeros(1, d, device=device, dtype=dtype)
    A = M_b
    bb = b_b

    w_batched = qp_fn(Q, p, G, h, A, bb)   # (1, d)
    w_star = w_batched.squeeze(0)

    # KKT residual *before* any feasibility patching.
    eq_res = (M @ w_star - b).abs().max()
    nn_res = (-w_star).clamp(min=0.0).max()
    kkt_res = torch.max(eq_res, nn_res)

    return w_star, kkt_res


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
