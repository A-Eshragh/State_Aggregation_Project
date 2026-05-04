"""
qpth-based (QP) solver for the discounted-occupancy formulation.

We solve a *QP-regularized* version of the LP:

    maximize     r^T y  -  reg * ||y||^2
    subject to   H y = gamma
                 y >= 0

qpth solves minimization QPs:

    minimize   0.5 y^T Q y + p^T y
    subject to G y <= h
               A y  = b

So we set:
    Q = 2*reg*I
    p = -r
    G = -I, h = 0
    A = H,  b = gamma
"""

from __future__ import annotations

import torch


def solve_occupancy_qp_qpth(
    H: torch.Tensor,
    r: torch.Tensor,
    gamma: torch.Tensor,
    reg: float,
    qp_solver,
    return_residual: bool = False,
):
    """
    Args:
        H: (B, S, SA)
        r: (B, SA)
        gamma: (B, S)
        reg: nonnegative quadratic regularization coefficient
        qp_solver: qpth.qp.QPFunction instance
        return_residual: if True, also return a per-batch KKT residual so
            the caller can detect non-convergence and skip the optimizer
            step. The previous silent `clamp(min=0)` masked exactly this
            failure mode (PDIPM exhausting its iteration budget without
            satisfying eps), causing transient one-epoch nosedives during
            training (see plan two-issues-i-found-keen-hearth).

    Returns:
        y: (B, SA) solution.
        kkt_res (only if return_residual=True): (B,) tensor with
            max(|H y - gamma|, max(-y, 0)) per batch element.
    """
    if reg < 0:
        raise ValueError(f"reg must be >= 0, got {reg}")
    if H.dim() != 3 or r.dim() != 2 or gamma.dim() != 2:
        raise ValueError(f"Expected H(B,S,SA), r(B,SA), gamma(B,S), got {H.shape}, {r.shape}, {gamma.shape}")

    B, S, SA = H.shape
    if r.shape != (B, SA) or gamma.shape != (B, S):
        raise ValueError(f"Shape mismatch: H={H.shape}, r={r.shape}, gamma={gamma.shape}")

    device = H.device
    dtype = H.dtype

    # QP matrices
    Q = (2.0 * float(reg)) * torch.eye(SA, device=device, dtype=dtype).unsqueeze(0).expand(B, SA, SA)
    p = -r

    G = -torch.eye(SA, device=device, dtype=dtype).unsqueeze(0).expand(B, SA, SA)
    h = torch.zeros(B, SA, device=device, dtype=dtype)

    A = H
    b = gamma

    y = qp_solver(Q, p, G, h, A, b)

    if return_residual:
        # Compute residual *before* any feasibility patching so the caller
        # can see real PDIPM failures and decide whether to skip the step.
        with torch.no_grad():
            eq_res = (torch.bmm(H, y.unsqueeze(-1)).squeeze(-1) - gamma).abs().max(dim=1).values  # (B,)
            nn_res = (-y).clamp(min=0.0).max(dim=1).values                                       # (B,)
            kkt_res = torch.maximum(eq_res, nn_res)
        return y, kkt_res

    # Backward-compat path (no residual requested, e.g. eval-only callers
    # under torch.no_grad): keep the safety clamp so slightly-negative
    # PDIPM iterates don't break downstream non-negativity assumptions.
    # Gradient-bearing training callers MUST pass return_residual=True.
    return y.clamp(min=0.0)

