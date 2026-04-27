"""Stochastic-sketch ensemble LP solver for the (aggregated) occupancy LP.

For each ensemble member m = 1..K we solve

    max  r^T y
    s.t. (S_m H_aug) y = (S_m gamma_aug)
         y >= 0

where H_aug stacks H with the redundant total-mass anchor row 1^T y = 1/(1-beta),
and S_m is a (k+1, M+1) row sketch that draws k random Gaussian / Rademacher /
subsample combinations of the M original rows AND ALWAYS preserves the anchor.
The anchor guarantees the sketched LP stays bounded for any k <= M, while the
random rows replace the rest of H with low-dimensional combinations -- different
across m -- so the optimal vertex flips and the ensemble average y_avg lives
in the interior of the polytope, smoothing y(theta).

The original LP automatically satisfies 1^T y = 1/(1-beta), so adding the anchor
to H is redundant and does not alter the original optimum. After sketching with
k < M, the anchor becomes informative and replaces what would otherwise need a
box constraint.

Forward:
    Pre-sample K sketches, build (S_m H_aug, S_m gamma_aug, r) for each, batch
    them as K*B independent LPs, and call DifferentiableLPSolver once.

Backward:
    Each sketch backprops through its own LP basis sensitivity (handled by
    DifferentiableLPSolver). Averaging is linear, so autograd produces the
    mean of per-sketch gradients automatically.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from .differentiable_lp import DifferentiableLPSolver


def _sample_sketch(M: int, k: int, mode: str, generator: Optional[torch.Generator],
                   dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Build a single (k, M) sketch matrix.

    'gaussian'   : S_{ij} ~ N(0, 1/k)
    'rademacher' : S_{ij} ~ +/- 1 / sqrt(k)
    'subsample'  : k rows of sqrt(M/k) * e_idx (sampled w/o replacement)
    """
    if mode == "gaussian":
        S = torch.randn(k, M, generator=generator, dtype=dtype, device=device)
        S = S / math.sqrt(float(k))
    elif mode == "rademacher":
        signs = torch.randint(0, 2, (k, M), generator=generator, dtype=torch.int64,
                              device=device).to(dtype) * 2.0 - 1.0
        S = signs / math.sqrt(float(k))
    elif mode == "subsample":
        perm = torch.randperm(M, generator=generator, device=device)[:k]
        S = torch.zeros(k, M, dtype=dtype, device=device)
        S[torch.arange(k, device=device), perm] = math.sqrt(float(M) / float(k))
    else:
        raise ValueError(f"Unknown sketch mode {mode!r}")
    return S


def sketched_lp(
    H: torch.Tensor,
    r: torch.Tensor,
    gamma: torch.Tensor,
    *,
    n_sketches: int = 4,
    sketch_ratio: float = 0.7,
    sketch_mode: str = "gaussian",
    sketch_seed: Optional[int] = None,
    beta: float = 0.95,
) -> torch.Tensor:
    """Sketch-and-solve LP solver with ensemble averaging.

    Args:
        H:     (B, M, M*A) equality matrix of the (aggregated) occupancy LP
        r:     (B, M*A)    cost vector (we MAXIMIZE r^T y)
        gamma: (B, M)      RHS, expected to sum to 1 along the M axis
        n_sketches:   K, number of independent sketches to average over
        sketch_ratio: target k = max(1, round(sketch_ratio * M)) random rows
                      per sketch, in addition to the anchor row.
        sketch_mode:  'gaussian' | 'rademacher' | 'subsample'
        sketch_seed:  optional generator seed for reproducibility
        beta:         discount factor (sets the total-mass anchor 1/(1-beta))

    Returns:
        y_avg: (B, M*A) ensemble-averaged primal solution. Differentiable
        w.r.t. (H, r, gamma).
    """
    if H.dim() != 3:
        raise ValueError(f"H must be (B, M, M*A); got {tuple(H.shape)}")
    B, M, MA = H.shape
    if r.shape != (B, MA):
        raise ValueError(f"r must be (B, M*A); got {tuple(r.shape)}")
    if gamma.shape != (B, M):
        raise ValueError(f"gamma must be (B, M); got {tuple(gamma.shape)}")

    K = int(n_sketches)
    if K < 1:
        raise ValueError("n_sketches must be >= 1")
    k = max(1, int(round(float(sketch_ratio) * M)))
    k = min(k, M)

    dtype = H.dtype
    device = H.device

    # Augment with the redundant total-mass anchor: 1^T y = 1/(1-beta).
    # Original LP already satisfies it, so the augmented LP has the same
    # optimum but is robust to dropping rows of H. Skip when k == M, since
    # the sketched H rows are full-rank and the anchor would be a redundant
    # row that makes the basis rank-deficient.
    use_anchor = k < M
    if use_anchor:
        total_mass = 1.0 / max(1e-6, 1.0 - float(beta))
        anchor_row = torch.ones(B, 1, MA, dtype=dtype, device=device)
        anchor_rhs = torch.full((B, 1), total_mass, dtype=dtype, device=device)
        H_aug = torch.cat([H, anchor_row], dim=1)             # (B, M+1, MA)
        gamma_aug = torch.cat([gamma, anchor_rhs], dim=1)     # (B, M+1)
    else:
        H_aug = H                                             # (B, M, MA)
        gamma_aug = gamma                                     # (B, M)

    if sketch_seed is not None:
        gen = torch.Generator(device=device).manual_seed(int(sketch_seed))
    else:
        gen = None

    # Pre-sample K sketches.
    #   With anchor    : S_m is (k+1, M+1), preserving the anchor row.
    #   Without anchor : S_m is (k,   M)  , a plain k x M sketch.
    S_list = []
    for _ in range(K):
        S_rand = _sample_sketch(M, k, sketch_mode, gen, dtype, device)  # (k, M)
        if use_anchor:
            S_top = torch.cat([S_rand, torch.zeros(k, 1, dtype=dtype, device=device)],
                              dim=1)                                     # (k, M+1)
            anchor_pick = torch.zeros(1, M + 1, dtype=dtype, device=device)
            anchor_pick[0, M] = 1.0
            S_m = torch.cat([S_top, anchor_pick], dim=0)                 # (k+1, M+1)
        else:
            S_m = S_rand                                                 # (k, M)
        S_list.append(S_m)
    S_stack = torch.stack(S_list, dim=0)                                 # (K, k_eff, M_aug)
    k_eff = S_stack.shape[1]

    # Apply each sketch to (H_aug, gamma_aug):
    #   SH:  (K, B, k+1, MA)
    #   Sg:  (K, B, k+1)
    SH = torch.einsum("kij,bjp->kbip", S_stack, H_aug)
    Sg = torch.einsum("kij,bj->kbi", S_stack, gamma_aug)

    KB = K * B
    SH_flat = SH.reshape(KB, k_eff, MA)
    Sg_flat = Sg.reshape(KB, k_eff)
    r_rep = r.unsqueeze(0).expand(K, B, MA).reshape(KB, MA).contiguous()

    # All KB LPs share the same y >= 0 cone and identical row count k+1, so
    # DifferentiableLPSolver returns a uniformly shaped basis.
    y_flat = DifferentiableLPSolver.apply(SH_flat, r_rep, Sg_flat)       # (KB, MA)
    y_per_sketch = y_flat.reshape(K, B, MA)                              # (K, B, MA)

    # Ensemble average over the K sketches.
    y_avg = y_per_sketch.mean(dim=0)                                     # (B, MA)
    return y_avg


__all__ = ["sketched_lp"]
