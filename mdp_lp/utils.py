"""
Utility functions for MDP LP experiments.
"""

import time
import torch
import random
import numpy as np
from mdp_lp.inventory_poisson_grad import poisson_pmf


def estimate_penalty_lambda_star(
    dfl_loss: torch.Tensor,
    lambda_pred: torch.Tensor,
    lambda_min: float = 1.0,
    curvature_eps: float = 1e-6,
    max_newton_step: float = 10.0,
):
    """
    Estimate the stationary point lambda* of the penalty Lagrangian loss
    with a local Newton step on lambda_pred:

        lambda* ≈ lambda_pred - grad / h_diag

    where grad = dL/dlambda_pred and h_diag is a curvature proxy obtained from
    second-order autograd on the same batch objective.

    Returns per-sample lists for the current batch, plus compact summaries.
    """
    grad = torch.autograd.grad(
        dfl_loss, lambda_pred, retain_graph=True, create_graph=True, allow_unused=True
    )[0]
    if grad is None:
        return None

    h_diag = torch.autograd.grad(
        grad.sum(), lambda_pred, retain_graph=True, allow_unused=True
    )[0]
    if h_diag is None:
        h_diag = torch.zeros_like(grad)

    safe_h = torch.where(
        h_diag.abs() < float(curvature_eps),
        torch.full_like(h_diag, float(curvature_eps)),
        h_diag,
    )
    step = torch.clamp(grad / safe_h, min=-float(max_newton_step), max=float(max_newton_step))
    lambda_star = lambda_pred - step
    if lambda_min is not None:
        lambda_star = torch.clamp(lambda_star, min=float(lambda_min))

    lambda_star_flat = lambda_star.detach().reshape(-1)
    grad_abs = grad.detach().abs()
    grad_abs_flat = grad_abs.reshape(-1)
    return {
        "lambda_star_list": [float(x) for x in lambda_star_flat.cpu().tolist()],
        "grad_abs_list": [float(x) for x in grad_abs_flat.cpu().tolist()],
        "lambda_star_mean": float(lambda_star_flat.mean().item()),
        "lambda_star_min": float(lambda_star_flat.min().item()),
        "lambda_star_max": float(lambda_star_flat.max().item()),
        "grad_abs_mean": float(grad_abs_flat.mean().item()),
    }


def compute_zero_grad_stats(v_star, lambda_pred, threshold=None, threshold_ratio=0.1):
    """
    Compute zero-gradient statistics: dV*/dλ_pred per sample.

    Measures how sensitive the value function is to changes in λ_pred.
    A high zero-grad fraction means the LP solution is insensitive to λ
    (flat landscape, weak DFL learning signal).

    Args:
        v_star: (batch,) value tensor, must be in the computation graph.
        lambda_pred: (batch,) or (batch,1) predicted lambda, must be in the graph.
        threshold: absolute threshold for "near-zero".
                   If None, uses threshold_ratio * mean|dV/dλ|.
        threshold_ratio: fraction of mean |grad| used when threshold is None.

    Returns:
        zero_grad_frac: float in [0, 1], fraction of samples with |dV/dλ| < threshold
        grad_mean_abs:  float, mean |dV/dλ| across the batch
        threshold_used: float, the threshold that was applied
    """
    grad_lambda = torch.autograd.grad(
        v_star.sum(), lambda_pred, retain_graph=True, allow_unused=True
    )[0]

    if grad_lambda is None:
        return 1.0, 0.0, 0.0

    abs_grad = grad_lambda.detach().abs().flatten()
    mean_abs = abs_grad.mean().item()

    if threshold is None:
        threshold = threshold_ratio * mean_abs if mean_abs > 0 else 1e-8

    zero_frac = (abs_grad < threshold).float().mean().item()
    return zero_frac, mean_abs, threshold


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def solve_lp(mdp, lp_solver):
    """
    Solve LP for an MDP using the provided LP solver.
    
    Args:
        mdp: MDP instance with lp_formulation() method
        lp_solver: Function that takes (H, r, gamma) and returns (y_star, basis)
    
    Returns:
        y_star: (batch, SA) occupancy measure
        obj: (batch,) objective values (r^T y_star)
        run_time: float, time taken to solve
    """
    start_time = time.time()
    
    H, r_flat, gamma = mdp.lp_formulation(mdp.P, mdp.r)
    # Ensure gamma is strictly positive (non-degeneracy assumption)
    # This is already done in build_gamma(), but add extra safety
    gamma = torch.clamp(gamma, min=1e-8)
    gamma = gamma / gamma.sum(dim=1, keepdim=True)  # Renormalize
    
    y_star, basis, dual = lp_solver(H, r_flat, gamma)
    
    # Compute objective: r^T y_star
    obj = torch.sum(r_flat * y_star, dim=1)  # (batch,)
    
    run_time = time.time() - start_time
    
    return y_star, obj, dual, run_time


# ----------------------------------------
# Newsvendor base-stock approximation
def _poisson_cdf_inv_truncated(lambda_param, d_max, target, tol=1e-8):
    """
    Inverse CDF: find smallest s in [0, d_max] such that F(s) >= target.
    F is the Poisson(λ) CDF truncated to 0..d_max (renormalized).

    Args:
        lambda_param: (batch,) or scalar
        target: scalar or (batch,) critical ratio in [0, 1]
    Returns:
        s_star: int (if scalar) or LongTensor (batch,)
    """
    if isinstance(lambda_param, torch.Tensor):
        lam = lambda_param.flatten()
        scalar = lam.numel() == 1
        if scalar:
            lam = lam.unsqueeze(0)
    else:
        lam = torch.tensor([float(lambda_param)])
        scalar = True

    pmf = poisson_pmf(lam, d_max)
    if pmf.dim() == 1:
        pmf = pmf.unsqueeze(0)
    pmf = pmf / (pmf.sum(dim=-1, keepdim=True) + 1e-12)
    cdf = torch.cumsum(pmf, dim=-1)  # (batch, d_max+1)

    if isinstance(target, torch.Tensor):
        target_t = target.flatten().to(device=lam.device, dtype=lam.dtype)
    else:
        target_t = torch.full((lam.shape[0],), float(target), device=lam.device, dtype=lam.dtype)
    if target_t.numel() == 1 and lam.shape[0] > 1:
        target_t = target_t.expand(lam.shape[0])

    # Find smallest s: cdf[:, s] >= target
    s_star = torch.full((lam.shape[0],), d_max, dtype=torch.long, device=lam.device)
    for i in range(lam.shape[0]):
        idx = (cdf[i] >= target_t[i] - tol).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            s_star[i] = idx[0].item()
    return s_star[0].item() if scalar else s_star


def newsvendor_base_stock(p, voc, hc, bc, lambda_param, d_max, b_max, s_max):
    """
    Approximate optimal base-stock level S* via newsvendor critical ratio.

    Critical ratio = (p - voc + bc) / (p - voc + hc + bc)
    S* ≈ F_λ^{-1}(critical ratio), with F_λ the Poisson(λ) CDF truncated to 0..d_max.

    Args:
        p: selling price (scalar or tensor)
        voc: variable ordering cost (scalar)
        hc: holding cost (scalar)
        bc: backlog cost (scalar)
        lambda_param: Poisson rate (scalar or tensor)
        d_max: max demand for truncation
        b_max: max backlog (for clamping S*)
        s_max: max storage (for clamping S*)

    Returns:
        S*: base-stock level (int or LongTensor), clamped to [0, s_max]
    """
    device = lambda_param.device if isinstance(lambda_param, torch.Tensor) else torch.device("cpu")
    dtype = lambda_param.dtype if isinstance(lambda_param, torch.Tensor) else torch.float32

    p_t = torch.as_tensor(p, device=device, dtype=dtype)
    if p_t.dim() == 0:
        p_t = p_t.unsqueeze(0)
    lam_t = torch.as_tensor(lambda_param, device=device, dtype=dtype)
    if lam_t.dim() == 0:
        lam_t = lam_t.unsqueeze(0)
    if p_t.shape[0] == 1 and lam_t.shape[0] > 1:
        p_t = p_t.expand(lam_t.shape[0])
    elif p_t.shape[0] > 1 and lam_t.shape[0] == 1:
        lam_t = lam_t.expand(p_t.shape[0])

    denom = p_t - voc + hc + bc
    num = p_t - voc + bc
    denom = torch.where(denom > 1e-10, denom, torch.ones_like(denom))
    num = torch.clamp(num, min=0.0)
    critical_ratio = (num / denom).clamp(0.0, 1.0)

    s_star = _poisson_cdf_inv_truncated(lam_t, d_max, critical_ratio)
    if isinstance(s_star, torch.Tensor):
        s_star = torch.clamp(s_star, 0, s_max)
        return s_star[0].item() if s_star.numel() == 1 else s_star
    else:
        return min(max(int(s_star), 0), s_max)

def newsvendor_policy_actions(S_star, b_max, s_max, q_max):
    """
    Convert base-stock level S* to per-state optimal actions (order quantities).

    Policy: order min(max(S* - inventory, 0), q_max) in each state.

    Args:
        S_star: base-stock level (int or LongTensor of shape (batch,))
        b_max, s_max, q_max: MDP parameters

    Returns:
        actions: (S,) or (batch, S) tensor of order quantities per state.
                 State s has inventory = s - b_max.
    """
    n_states = b_max + s_max + 1
    inv_levels = torch.arange(n_states, dtype=torch.float32) - b_max  # (S,)

    if isinstance(S_star, torch.Tensor):
        S_star = S_star.float().view(-1, 1)
        order = torch.clamp(S_star - inv_levels.unsqueeze(0), min=0)
        order = torch.clamp(order, max=q_max)
        return order.long()
    else:
        order = torch.clamp(torch.tensor(S_star, dtype=torch.float32) - inv_levels, min=0)
        order = torch.clamp(order, max=q_max)
        return order.long()