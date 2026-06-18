"""
qpth-based differentiable QP solver for the cartpole reduced LP triple.

Mirrors mdp_lp/solvers/qpth_qp_solver.py — instead of Gurobi+custom-KKT
(reduced_solver.differentiable_reduced_qp), we use qpth's batched
PDIPM solver, which provides implicit-KKT gradients out of the box.

Problem solved:

    minimize   c^T w + (eta/2) ||w||^2
    subject to M w = b
               w >= 0

Mapped to qpth's   min 0.5 y^T Q y + p^T y   s.t. G y <= h, A y = b   form:

    Q = eta * I        (so 0.5 y^T Q y = (eta/2) ||y||^2)
    p = c
    G = -I, h = 0      (so -w <= 0  <=>  w >= 0)
    A = M, b = b
"""

from __future__ import annotations

import torch
from qpth.qp import QPFunction, QPSolvers


_QP_SOLVER = QPFunction(eps=1e-10, maxIter=20, solver=QPSolvers.PDIPM_BATCHED)


def differentiable_reduced_qpth(M: torch.Tensor,
                                c: torch.Tensor,
                                b: torch.Tensor,
                                eta: float = 0.01) -> torch.Tensor:
    """Differentiable QP solve via qpth.

    Args:
        M: (m, d)
        c: (d,)
        b: (m,)
        eta: nonnegative quadratic regularization coefficient

    Returns:
        w_star: (d,) — clamped to >= 0 to absorb PDIPM's small negative tails.
    """
    if eta < 0:
        raise ValueError(f"eta must be >= 0, got {eta}")

    m, d = M.shape
    device = M.device
    dtype = M.dtype

    Q = float(eta) * torch.eye(d, device=device, dtype=dtype).unsqueeze(0)
    p = c.unsqueeze(0)
    G = -torch.eye(d, device=device, dtype=dtype).unsqueeze(0)
    h = torch.zeros(1, d, device=device, dtype=dtype)
    A = M.unsqueeze(0)
    b_ = b.unsqueeze(0)

    w = _QP_SOLVER(Q, p, G, h, A, b_)
    w = w.squeeze(0).clamp(min=0.0)
    return w
