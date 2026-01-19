"""
Utility functions for MDP LP experiments.
"""

import time
import torch
import random
import numpy as np


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
    
    y_star, basis = lp_solver(H, r_flat, gamma)
    
    # Compute objective: r^T y_star
    obj = torch.sum(r_flat * y_star, dim=1)  # (batch,)
    
    run_time = time.time() - start_time
    
    return y_star, obj, run_time
