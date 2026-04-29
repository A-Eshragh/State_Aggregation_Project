"""Berthet-style perturbed-cost LP solver with ensemble averaging.

For each ensemble member m = 1..K we solve

    max  (r + sigma * Z_m)^T y
    s.t. H y = gamma
         y >= 0

with Z_m drawn i.i.d. from N(0, I) (gaussian) or {-1, +1}^{MA} (rademacher),
then return the ensemble-averaged primal y_avg = (1/K) sum_m y^(m).

Differentiability comes from Berthet et al. (2020), "Learning with
Differentiable Perturbed Optimizers." The key facts:

    1. H- and gamma-gradients flow through DifferentiableLPSolver as usual
       (basis-sensitivity at each perturbed solve, averaged across K).

    2. The r-gradient is NOT zero on average, even though each individual
       perturbed LP has ∂y/∂r = 0 within its basis. By Stein's identity for
       Gaussian Z,
           ∂E[y_i]/∂r_j = (1/sigma) E[Z_j (y_i - E[y_i])].
       The Monte-Carlo estimator of this Jacobian J (B, MA, MA) is
           J[b, i, j] = (1/(sigma*K)) sum_k Z[k, j] * (y_per[k, b, i] - y_avg[b, i]).
       We inject this gradient into autograd via the identity trick
           y_returned = y_avg_detached_in_r + einsum(J.detach(), r - r.detach())
       which is value-equal to y_avg but has Jacobian J w.r.t. r.

The DifferentiableLPSolver path is fed r.detach() + sigma * Z_m so that the
existing solver-side r-channel (which is hard-zeroed) is bypassed; H- and
gamma-gradients still flow through that path correctly. The Stein term
contributes zero to the value but the right gradient on r.
"""

from __future__ import annotations

from typing import Optional

import torch

from .differentiable_lp import DifferentiableLPSolver


def perturbed_lp(
    H: torch.Tensor,
    r: torch.Tensor,
    gamma: torch.Tensor,
    *,
    n_perturb: int = 4,
    sigma: float = 0.3,
    perturb_mode: str = "gaussian",
    perturb_seed: Optional[int] = None,
) -> torch.Tensor:
    """Perturbed-optimizer LP solver with ensemble averaging.

    Args:
        H:     (B, M, M*A) equality matrix
        r:     (B, M*A)    cost vector (we MAXIMIZE r^T y)
        gamma: (B, M)      RHS, expected to sum to 1 along M
        n_perturb:    K, number of independent cost-noise samples to average
        sigma:        noise scale for the cost perturbation
        perturb_mode: 'gaussian' (Z ~ N(0, I)) or 'rademacher' (Z ~ +/-1)
        perturb_seed: optional generator seed

    Returns:
        y_avg: (B, M*A) ensemble-averaged primal solution. Differentiable
        w.r.t. (H, r, gamma) -- r-gradient comes from Stein's identity, the
        H/gamma gradients from DifferentiableLPSolver basis sensitivity.
    """
    if H.dim() != 3:
        raise ValueError(f"H must be (B, M, M*A); got {tuple(H.shape)}")
    B, M, MA = H.shape
    if r.shape != (B, MA):
        raise ValueError(f"r must be (B, M*A); got {tuple(r.shape)}")
    if gamma.shape != (B, M):
        raise ValueError(f"gamma must be (B, M); got {tuple(gamma.shape)}")

    K = int(n_perturb)
    if K < 1:
        raise ValueError("n_perturb must be >= 1")
    sigma = float(sigma)

    dtype = H.dtype
    device = H.device

    if perturb_seed is not None:
        gen = torch.Generator(device=device).manual_seed(int(perturb_seed))
    else:
        gen = None

    # K independent cost-noise vectors, shape (K, MA).
    if perturb_mode == "gaussian":
        Z = torch.randn(K, MA, generator=gen, dtype=dtype, device=device)
    elif perturb_mode == "rademacher":
        signs = torch.randint(0, 2, (K, MA), generator=gen, dtype=torch.int64,
                              device=device).to(dtype) * 2.0 - 1.0
        Z = signs
    else:
        raise ValueError(f"Unknown perturb_mode {perturb_mode!r}")

    # Replicate H/gamma K times, perturb r with detached r so the LP solver's
    # zero r-gradient doesn't propagate through. H/gamma still see autograd.
    KB = K * B
    H_rep = H.unsqueeze(0).expand(K, B, M, MA).reshape(KB, M, MA)
    gamma_rep = gamma.unsqueeze(0).expand(K, B, M).reshape(KB, M)
    r_pert = r.detach().unsqueeze(0) + sigma * Z.unsqueeze(1)        # (K, B, MA)
    r_flat = r_pert.reshape(KB, MA).contiguous()

    y_flat = DifferentiableLPSolver.apply(H_rep, r_flat, gamma_rep)  # (KB, MA)
    y_per = y_flat.reshape(K, B, MA)                                 # (K, B, MA)
    y_avg = y_per.mean(dim=0)                                        # (B, MA)

    # ----- Stein r-gradient injection -----
    # J[b, i, j] = (1/(sigma*K)) sum_k Z[k, j] (y_per[k, b, i] - y_avg[b, i])
    # The control-variate form (subtract y_avg) reduces variance vs the bare
    # estimator (1/(sigma*K)) sum_k y_per[k, b, i] Z[k, j].
    if sigma > 0:
        diff = (y_per - y_avg.unsqueeze(0)).detach()                 # (K, B, MA)
        J = torch.einsum("kbi,kj->bij", diff, Z) / (sigma * K)       # (B, MA, MA)
        # einsum(J, r - r.detach()) is value-zero with gradient J w.r.t. r.
        stein_term = torch.einsum("bij,bj->bi", J, r - r.detach())   # (B, MA)
        return y_avg + stein_term
    return y_avg


__all__ = ["perturbed_lp"]
