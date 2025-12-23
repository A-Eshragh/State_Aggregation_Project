"""
Analytical gradients for Inventory Optimization with Poisson demand (Appendix B).

This module implements closed-form expressions for gradients without using autograd.
All functions are pure and do not rely on PyTorch's automatic differentiation.
"""

import torch
import math


def poisson_pmf(lambda_param, max_d):
    """
    Compute Poisson probability mass function.
    
    Returns P(D=d) for d = 0, 1, ..., max_d where D ~ Poisson(lambda).
    
    Args:
        lambda_param: scalar or (batch,) tensor of Poisson rate parameters
        max_d: maximum demand value (inclusive)
        
    Returns:
        f: tensor of shape (max_d+1,) or (batch, max_d+1) where f[d] = P(D=d)
        
    Formula: f[d] = lambda^d * exp(-lambda) / d!
    """
    # Handle batch vs scalar
    is_batch = lambda_param.dim() > 0
    if not is_batch:
        lambda_param = lambda_param.unsqueeze(0)
    
    batch_size = lambda_param.shape[0]
    device = lambda_param.device
    dtype = lambda_param.dtype
    
    # Create demand values d = 0, 1, ..., max_d
    d_values = torch.arange(max_d + 1, device=device, dtype=dtype).view(1, -1)  # (1, max_d+1)
    lambda_expanded = lambda_param.view(-1, 1)  # (batch, 1)
    
    # Compute log probabilities: log P(D=d) = d*log(lambda) - lambda - log(d!)
    # Using log-space for numerical stability
    log_lambda = torch.log(lambda_param + 1e-10)  # Add small epsilon to avoid log(0)
    log_probs = d_values * log_lambda.view(-1, 1) - lambda_expanded
    
    # Subtract log(d!) = log(gamma(d+1))
    # Use lgamma for numerical stability: lgamma(d+1) = log(d!)
    log_factorial = torch.lgamma(d_values + 1)  # (1, max_d+1)
    log_probs = log_probs - log_factorial
    
    # Convert back to probability space
    f = torch.exp(log_probs)  # (batch, max_d+1)
    
    # Handle case where lambda is very small (most probability at d=0)
    # Ensure probabilities sum to 1 (with some tolerance for numerical error)
    f = torch.clamp(f, min=0.0, max=1.0)
    
    if not is_batch:
        f = f.squeeze(0)  # (max_d+1,)
    
    return f


def dH_dlambda(lambda_param, b_max, s_max, a_max, beta):
    """
    Compute ∂H/∂λ for Inventory MDP with Poisson demand (Appendix B).
    
    Returns gradient of H matrix with respect to Poisson rate λ.
    H is the constraint matrix in the LP formulation: H y = gamma.
    
    Args:
        lambda_param: scalar or (batch,) tensor of Poisson rate parameters
        b_max: maximum backlog (states go from -b_max to s_max)
        s_max: maximum storage
        a_max: maximum action (order quantity), so A = a_max + 1
        beta: discount factor
        
    Returns:
        dH_dlambda: tensor of shape (S, S*A) or (batch, S, S*A)
                   where S = b_max + s_max + 1
        
    Appendix B formulas:
    - For s' = 0 (lowest inventory state): ∂H/∂λ = −β · f_λ(s + a − 1)
    - For s' = 1..s+a: ∂H/∂λ = −β · (f_λ(s + a − s' − 1) − f_λ(s + a − s'))
    - Else: 0
    
    Note: States are indexed 0..S-1 where state s corresponds to inventory level s - b_max.
          Actions are indexed 0..a_max where action a corresponds to order quantity a.
    """
    # Handle batch vs scalar
    is_batch = lambda_param.dim() > 0
    if not is_batch:
        lambda_param = lambda_param.unsqueeze(0)
    
    batch_size = lambda_param.shape[0]
    S = b_max + s_max + 1  # Total number of states
    A = a_max + 1  # Total number of actions
    device = lambda_param.device
    dtype = lambda_param.dtype
    
    # Initialize output: (batch, S, S*A) or (S, S*A)
    if is_batch:
        dH_dlambda = torch.zeros(batch_size, S, S * A, device=device, dtype=dtype)
    else:
        dH_dlambda = torch.zeros(S, S * A, device=device, dtype=dtype)
    
    # Compute Poisson PMF for all possible demand values
    # Maximum demand needed: when s=0 (inventory = -b_max), a=a_max, we need d up to b_max + a_max
    max_demand_needed = b_max + a_max + 1  # Add 1 for safety
    f_lambda = poisson_pmf(lambda_param, max_demand_needed)  # (batch, max_d+1) or (max_d+1,)
    
    # Convert to batch format for easier indexing
    if not is_batch:
        f_lambda = f_lambda.unsqueeze(0)  # (1, max_d+1)
    
    # Iterate over all state-action pairs (s, a)
    for s in range(S):
        inv_s = s - b_max  # Inventory level at state s
        for a in range(A):
            col_idx = s * A + a  # Column index in H matrix for (s, a)
            
            # Compute s + a (in terms of code indices)
            # This represents the inventory level after ordering: inv_s + a
            inv_after_order = inv_s + a
            
            # Iterate over all next states s'
            for s_prime in range(S):
                inv_s_prime = s_prime - b_max  # Inventory level at state s'
                
                # Demand required: d = inv_after_order - inv_s_prime = (s - b_max) + a - (s' - b_max) = s + a - s'
                d_required = inv_after_order - inv_s_prime
                
                # Appendix B formula (notation: s, a, s' refer to inventory levels, not code indices):
                # - For s' = 0: ∂H/∂λ = −β · f_λ(s + a − 1)
                # - For s' = 1..s+a: ∂H/∂λ = −β · (f_λ(s + a − s' − 1) − f_λ(s + a − s'))
                # - Else: 0
                #
                # Conversion to code indices:
                # - State s (code) has inventory inv_s = s - b_max
                # - Action a (code) is order quantity a  
                # - State s' (code) has inventory inv_s_prime = s' - b_max
                # - After ordering: inv_after_order = inv_s + a
                
                if inv_s_prime == 0:
                    # Case: s' = 0 (inventory level 0, corresponds to state index s' = b_max)
                    # Appendix B: ∂H/∂λ = −β · f_λ(s + a − 1)
                    # In inventory terms: f_λ(inv_s + a - 1)
                    d_idx = inv_s + a - 1
                    if 0 <= d_idx < f_lambda.shape[1]:
                        dH_dlambda[:, s_prime, col_idx] = -beta * f_lambda[:, d_idx]
                elif 1 <= inv_s_prime <= inv_after_order:
                    # Case: s' = 1..s+a (inventory levels from 1 to inv_after_order)
                    # Appendix B: ∂H/∂λ = −β · (f_λ(s + a − s' − 1) − f_λ(s + a − s'))
                    # In inventory terms: f_λ(inv_s + a - inv_s_prime - 1) - f_λ(inv_s + a - inv_s_prime)
                    d_idx1 = inv_s + a - inv_s_prime - 1  # demand index for (s + a - s' - 1)
                    d_idx2 = inv_s + a - inv_s_prime       # demand index for (s + a - s')
                    
                    term1 = torch.zeros(batch_size, device=device, dtype=dtype)
                    term2 = torch.zeros(batch_size, device=device, dtype=dtype)
                    
                    if 0 <= d_idx1 < f_lambda.shape[1]:
                        term1 = f_lambda[:, d_idx1]
                    if 0 <= d_idx2 < f_lambda.shape[1]:
                        term2 = f_lambda[:, d_idx2]
                    
                    dH_dlambda[:, s_prime, col_idx] = -beta * (term1 - term2)
                # Else: 0 (already initialized)
    
    if not is_batch:
        dH_dlambda = dH_dlambda.squeeze(0)  # (S, S*A)
    
    return dH_dlambda


def dr_dlambda(lambda_param, b_max, s_max, a_max, p, h):
    """
    Compute ∂r/∂λ for Inventory MDP with Poisson demand (Appendix B).
    
    Returns gradient of reward vector r with respect to Poisson rate λ.
    
    Args:
        lambda_param: scalar or (batch,) tensor of Poisson rate parameters
        b_max: maximum backlog (states go from -b_max to s_max)
        s_max: maximum storage
        a_max: maximum action (order quantity), so A = a_max + 1
        p: price (or penalty coefficient)
        h: holding cost coefficient
        
    Returns:
        dr_dlambda: tensor of shape (S*A,) or (batch, S*A)
        
    Appendix B formula:
    ∂r/∂λ(s,a) = (p + h) · sum_{x=0}^{s+a-1} f_λ(x)
    
    This represents the gradient of reward w.r.t. lambda for state-action pair (s, a).
    The sum is over demand values x from 0 to s+a-1 (inclusive).
    
    Note: States are indexed 0..S-1 where state s corresponds to inventory level s - b_max.
          Actions are indexed 0..a_max where action a corresponds to order quantity a.
    """
    # Handle batch vs scalar
    is_batch = lambda_param.dim() > 0
    if not is_batch:
        lambda_param = lambda_param.unsqueeze(0)
    
    batch_size = lambda_param.shape[0]
    S = b_max + s_max + 1  # Total number of states
    A = a_max + 1  # Total number of actions
    device = lambda_param.device
    dtype = lambda_param.dtype
    
    # Initialize output: (batch, S*A) or (S*A,)
    if is_batch:
        dr_dlambda = torch.zeros(batch_size, S * A, device=device, dtype=dtype)
    else:
        dr_dlambda = torch.zeros(S * A, device=device, dtype=dtype)
    
    # Compute Poisson PMF for all possible demand values
    # Maximum demand needed: when s=S-1 (inventory = s_max), a=a_max, we need d up to s_max + a_max
    max_demand_needed = s_max + a_max + 1  # Add 1 for safety
    f_lambda = poisson_pmf(lambda_param, max_demand_needed)  # (batch, max_d+1) or (max_d+1,)
    
    # Convert to batch format for easier indexing
    if not is_batch:
        f_lambda = f_lambda.unsqueeze(0)  # (1, max_d+1)
    
    # Iterate over all state-action pairs (s, a)
    for s in range(S):
        inv_s = s - b_max  # Inventory level at state s
        for a in range(A):
            idx = s * A + a  # Index in r vector for (s, a)
            
            # Compute s + a (in terms of inventory levels)
            inv_after_order = inv_s + a
            
            # Appendix B formula: sum_{x=0}^{s+a-1} f_λ(x)
            # In terms of inventory: sum over x from 0 to inv_s + a - 1
            # But we need to be careful: the sum is over demand values x
            # x ranges from 0 to (inv_s + a - 1), but we need to ensure x >= 0
            max_x = max(0, inv_after_order - 1)  # s + a - 1 in inventory terms
            
            # Sum f_λ(x) for x = 0, 1, ..., max_x
            if max_x >= 0:
                # Sum probabilities from x=0 to x=max_x (inclusive)
                # f_lambda[:, 0:max_x+1] gives P(D=0), P(D=1), ..., P(D=max_x)
                dr_dlambda[:, idx] = (p + h) * torch.sum(f_lambda[:, :max_x + 1], dim=1)
            else:
                # If max_x < 0, sum is empty (0)
                dr_dlambda[:, idx] = 0.0
    
    if not is_batch:
        dr_dlambda = dr_dlambda.squeeze(0)  # (S*A,)
    
    return dr_dlambda


