"""
Sketch-and-solve LP solver for the continuous-MDP reduced LP.

Mirrors mdp_lp/solvers/sketched_lp.py in spirit, but adapted to the
single-instance reduced LP triple (M, c, b) used throughout
mdp_continuous (see reduced_solver.differentiable_reduced_lp):

    minimize   c^T w
    subject to M w = b,   w >= 0

A row-sketch S: (k, m) compresses the m equality constraints to k:

    minimize   c^T w
    subject to (S M) w = S b,   w >= 0

For K independent sketches we solve K reduced LPs in a single batched
Gurobi sweep through DifferentiableReducedLP, then average the K
primal solutions. Gradients flow through autograd via (S @ M) and
(S @ b); S itself is a constant tensor.

Usage matches the other mdp_continuous solvers — drop-in replacement
for `differentiable_reduced_lp(M, c, b)`:

    from mdp_continuous.sketched_solver import differentiable_sketched_lp
    w = differentiable_sketched_lp(M, c, b, n_sketches=4, sketch_ratio=0.7)
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from .reduced_solver import (
    differentiable_reduced_lp,
    differentiable_reduced_qp,
    solve_reduced_lp_batch,
)


class _SketchedReducedLP(torch.autograd.Function):
    """Like reduced_solver.DifferentiableReducedLP but with a singularity-
    tolerant backward.

    Sketch-augmented systems (k+1 equality rows where the (k+1)-th is the
    redundant anchor (1^T M) w = 1^T b) frequently produce optimal bases
    M_B that are numerically singular on the basis-restricted KKT solve.
    The dense path uses torch.linalg.solve(M_B.T, .), which raises
    LinAlgError for those bases; here we fall back to a ridge-regularised
    lstsq so the gradient is a small-bias approximation rather than a
    crash.
    """

    @staticmethod
    def forward(ctx, M, c, b):
        batched = M.dim() == 3
        with torch.no_grad():
            w_star, basis, _ = solve_reduced_lp_batch(M, c, b)
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
            if w_star.dim() == 1:
                w_star = w_star.unsqueeze(0)
            if basis.dim() == 1:
                basis = basis.unsqueeze(0)
            grad_w = grad_w.unsqueeze(0)

        B, m_dim, _ = M.shape
        device = M.device
        dtype = M.dtype

        grad_M = torch.zeros_like(M)
        grad_c = torch.zeros_like(c)
        grad_b = torch.zeros_like(b)

        eye_m = torch.eye(m_dim, device=device, dtype=dtype)
        for i in range(B):
            Bidx = basis[i]
            M_B = M[i, :, Bidx]                        # (m, m)
            w_B = w_star[i, Bidx]
            g_B = grad_w[i, Bidx]
            try:
                x = torch.linalg.solve(M_B.T, g_B)
            except torch._C._LinAlgError:
                # Ridge-regularised lstsq fallback for singular bases.
                x = torch.linalg.lstsq(
                    M_B.T + 1e-6 * eye_m, g_B.unsqueeze(-1)
                ).solution.squeeze(-1)
            grad_b[i] = x
            grad_M[i].index_copy_(1, Bidx, -torch.outer(x, w_B))

        if not ctx.batched:
            grad_M = grad_M.squeeze(0)
            grad_c = grad_c.squeeze(0)
            grad_b = grad_b.squeeze(0)
        return grad_M, grad_c, grad_b


def _sketched_reduced_lp(M, c, b):
    return _SketchedReducedLP.apply(M, c, b)


def _sample_sketch(m: int, k: int, mode: str,
                   generator: Optional[torch.Generator],
                   dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Build a single (k, m) sketch matrix.

    'gaussian'   : S_{ij} ~ N(0, 1/k)
    'rademacher' : S_{ij} ~ +/- 1 / sqrt(k)
    'subsample'  : k rows of sqrt(m/k) * e_idx (sampled w/o replacement)
    """
    if mode == "gaussian":
        S = torch.randn(k, m, generator=generator, dtype=dtype, device=device)
        S = S / math.sqrt(float(k))
    elif mode == "rademacher":
        signs = torch.randint(0, 2, (k, m), generator=generator,
                              dtype=torch.int64, device=device).to(dtype)
        S = (signs * 2.0 - 1.0) / math.sqrt(float(k))
    elif mode == "subsample":
        perm = torch.randperm(m, generator=generator, device=device)[:k]
        S = torch.zeros(k, m, dtype=dtype, device=device)
        S[torch.arange(k, device=device), perm] = math.sqrt(float(m) / float(k))
    else:
        raise ValueError(f"Unknown sketch mode {mode!r}")
    return S


def _build_sketched_batch(M: torch.Tensor, c: torch.Tensor, b: torch.Tensor,
                          n_sketches: int, sketch_ratio: float,
                          sketch_mode: str,
                          sketch_seed: Optional[int],
                          use_anchor: bool = True):
    """Stack K row-sketched copies of (M, c, b) into batched tensors.

    When ``use_anchor`` is True and k < m, we mirror mdp_lp's sketched LP:
    augment the equality system with the redundant row

        (1^T M) w = (1^T b)              # sum of the m equality rows

    *before* sketching, then keep that anchor row exactly intact through
    each sketch. In the unsketched LP the anchor adds no information
    (it's a linear combination of M's rows), but after row-sketching the
    random sketch can destroy this direction, leaving an unbounded
    recession ray that drives c^T w to -inf. Preserving the anchor row
    keeps the sketched LP bounded — same trick as mdp_lp's
    1^T y = 1/(1-beta) anchor for the occupancy LP.
    """
    if M.dim() != 2:
        raise ValueError(f"M must be (m, d); got {tuple(M.shape)}")
    m, d = M.shape
    if c.shape != (d,):
        raise ValueError(f"c must be (d,); got {tuple(c.shape)} (d={d})")
    if b.shape != (m,):
        raise ValueError(f"b must be (m,); got {tuple(b.shape)} (m={m})")

    K = int(n_sketches)
    if K < 1:
        raise ValueError("n_sketches must be >= 1")

    k = max(1, int(round(float(sketch_ratio) * m)))
    k = min(k, m)

    dtype = M.dtype
    device = M.device

    # Augment with the redundant total-row-sum anchor when we are actually
    # compressing rows. With k == m the random sketch is full-rank in
    # expectation and the anchor just makes the basis rank-deficient.
    augment = bool(use_anchor) and (k < m)
    if augment:
        anchor_row = M.sum(dim=0, keepdim=True)         # (1, d)
        anchor_rhs = b.sum().unsqueeze(0)               # (1,)
        M_aug = torch.cat([M, anchor_row], dim=0)       # (m+1, d)
        b_aug = torch.cat([b, anchor_rhs], dim=0)       # (m+1,)
        m_aug = m + 1
    else:
        M_aug = M
        b_aug = b
        m_aug = m

    if sketch_seed is not None:
        gen = torch.Generator(device=device).manual_seed(int(sketch_seed))
    else:
        gen = None

    # Build K sketches. Each is (k+1, m+1) when augmented: the top k rows
    # are a random sketch over the original m rows (with a zero in the
    # anchor column so they don't mix), and the bottom row is e_{m+1}
    # which selects the anchor exactly.
    S_list = []
    for _ in range(K):
        S_rand = _sample_sketch(m, k, sketch_mode, gen, dtype, device)  # (k, m)
        if augment:
            S_top = torch.cat(
                [S_rand, torch.zeros(k, 1, dtype=dtype, device=device)],
                dim=1)                                                  # (k, m+1)
            anchor_pick = torch.zeros(1, m + 1, dtype=dtype, device=device)
            anchor_pick[0, m] = 1.0
            S_m = torch.cat([S_top, anchor_pick], dim=0)                # (k+1, m+1)
        else:
            S_m = S_rand                                                # (k, m)
        S_list.append(S_m)
    S_stack = torch.stack(S_list, dim=0)               # (K, k_eff, m_aug)

    # Batched sketched system. S is a constant tensor so autograd routes
    # gradients through M and b cleanly via the chain rule. M_aug carries
    # an autograd edge to M (and same for b_aug -> b), so the anchor is
    # handled correctly in the backward pass.
    M_batch = torch.einsum("kij,jp->kip", S_stack, M_aug)   # (K, k_eff, d)
    b_batch = torch.einsum("kij,j->ki", S_stack, b_aug)     # (K, k_eff)
    c_batch = c.unsqueeze(0).expand(K, d).contiguous()      # (K, d)

    return M_batch, c_batch, b_batch, S_stack.shape[1]


def differentiable_sketched_lp(
    M: torch.Tensor,
    c: torch.Tensor,
    b: torch.Tensor,
    *,
    n_sketches: int = 4,
    sketch_ratio: float = 0.7,
    sketch_mode: str = "gaussian",
    sketch_seed: Optional[int] = None,
) -> torch.Tensor:
    """Sketch-and-solve LP with ensemble averaging over K sketches.

    Solves K independent row-sketched reduced LPs

        min  c^T w   s.t.  (S_i M) w = S_i b,  w >= 0

    via the batched DifferentiableReducedLP path (Gurobi forward,
    basis-sensitivity backward), then returns the mean of the K
    primal solutions. Gradients flow through (M, b) since each
    S_i is constant.

    Args:
        M: (m, d) constraint matrix.
        c: (d,)   cost vector.
        b: (m,)   RHS.
        n_sketches:   K, number of independent sketches to average over.
        sketch_ratio: target k = max(1, round(sketch_ratio * m)) sketched
                      constraints per replica (capped at m).
        sketch_mode:  'gaussian' | 'rademacher' | 'subsample'.
        sketch_seed:  optional generator seed for reproducibility.

    Returns:
        w_avg: (d,) ensemble-averaged primal solution, autograd-tracked
               through (M, b).
    """
    M_batch, c_batch, b_batch, _ = _build_sketched_batch(
        M, c, b, n_sketches, sketch_ratio, sketch_mode, sketch_seed)

    # Batched LP: returns (K, d). _sketched_reduced_lp uses a ridge-
    # tolerant backward to survive the singular bases produced by the
    # anchor-augmented sketched systems.
    w_per_sketch = _sketched_reduced_lp(M_batch, c_batch, b_batch)

    return w_per_sketch.mean(dim=0)


def differentiable_sketched_qp(
    M: torch.Tensor,
    c: torch.Tensor,
    b: torch.Tensor,
    *,
    eta: float = 0.01,
    n_sketches: int = 4,
    sketch_ratio: float = 0.7,
    sketch_mode: str = "gaussian",
    sketch_seed: Optional[int] = None,
) -> torch.Tensor:
    """QP variant: sketch-and-solve with quadratic regularization.

    Solves K independent row-sketched reduced QPs

        min  c^T w + (eta/2) ||w||^2   s.t.  (S_i M) w = S_i b,  w >= 0

    one at a time through the existing single-instance Gurobi QP path
    (DifferentiableReducedQP doesn't take a batch dim), and averages the
    K primal solutions. The quadratic term keeps every sketched problem
    strongly convex even when k < d would otherwise leave the LP
    unbounded along a degenerate direction.

    Args:
        M, c, b: same shapes as the LP variant.
        eta:     quadratic regularization strength (must be > 0).
        n_sketches, sketch_ratio, sketch_mode, sketch_seed: same as the
            LP variant.
    """
    if eta <= 0:
        raise ValueError(f"eta must be > 0 for the QP variant; got {eta}")

    M_batch, c_batch, b_batch, _ = _build_sketched_batch(
        M, c, b, n_sketches, sketch_ratio, sketch_mode, sketch_seed)

    K = M_batch.shape[0]
    ws = []
    for i in range(K):
        ws.append(differentiable_reduced_qp(
            M_batch[i], c_batch[i], b_batch[i], eta=eta))
    return torch.stack(ws, dim=0).mean(dim=0)


__all__ = ["differentiable_sketched_lp", "differentiable_sketched_qp"]
