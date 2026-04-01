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
):
    """
    Args:
        H: (B, S, SA)
        r: (B, SA)
        gamma: (B, S)
        reg: nonnegative quadratic regularization coefficient
        qp_solver: qpth.qp.QPFunction instance

    Returns:
        y: (B, SA) solution
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
    # PDIPM may return slightly negative values; clamp to maintain feasibility
    y = y.clamp(min=0.0)
    return y

