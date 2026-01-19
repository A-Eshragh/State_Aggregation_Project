"""
Inventory MDP with elasticity-based demand (Poisson with price-dependent lambda).
"""

import torch
from mdp_lp.mdp_derived import InventoryLP
from mdp_lp.inventory_poisson_grad import poisson_pmf


def create_inventory_from_lambda(lambda_param, b_max, s_max, q_max, d_max, 
                                  foc, voc, hc, bc, beta, selling_price):
    """
    Create InventoryLP MDP from lambda values (Poisson demand) with selling price.
    
    Args:
        lambda_param: (batch,) tensor of Poisson rate parameters
        b_max, s_max, q_max, d_max: inventory MDP parameters
        foc, voc, hc, bc: cost parameters
        beta: discount factor
        selling_price: (batch,) tensor of selling prices
        
    Returns:
        InventoryLP instance with modified rewards
    """
    batch = lambda_param.shape[0]
    
    # Convert lambda to demand probabilities using Poisson PMF
    prob_demand = poisson_pmf(lambda_param, d_max)  # (batch, d_max+1)
    
    # InventoryLP applies softmax internally, so pass logits
    epsilon = 1e-10
    prob_demand_clamped = torch.clamp(prob_demand, min=epsilon)
    prob_demand_normalized = prob_demand_clamped / prob_demand_clamped.sum(dim=1, keepdim=True)
    logits_demand = torch.log(prob_demand_normalized + epsilon)
    
    mdp = InventoryLP(b_max, s_max, q_max, d_max, foc, voc, hc, bc, beta, logits_demand)
    
    # Use mdp.prob_demand (after softmax) for consistency
    prob_demand_final = mdp.prob_demand
    
    # Compute reward according to the formula:
    # r_λ(s, a) = p · E_λ[Number of Sales] - ca - h · E_λ[Number of Leftover Items]
    # where:
    #   E_λ[Number of Sales] = Σ_{x=1}^{s+a} x·f_λ(x) + (s+a)·Σ_{x=s+a+1}^{∞} f_λ(x)
    #   E_λ[Number of Leftover Items] = Σ_{x=0}^{s+a} (s+a-x)·f_λ(x)
    
    n_states = mdp.n_states
    n_actions = mdp.n_actions
    n_demands = mdp.n_demands
    
    # State indices: s ∈ [0, n_states-1] where inventory = s - b_max
    # Action indices: a ∈ [0, n_actions-1] where order quantity = a
    begin_inv = torch.arange(n_states, dtype=torch.float32).view(-1, 1, 1) - b_max  # (S, 1, 1)
    orders = torch.arange(n_actions, dtype=torch.float32).view(1, -1, 1)  # (1, A, 1)
    demands = torch.arange(n_demands, dtype=torch.float32).view(1, 1, -1)  # (1, 1, D)
    
    # Available inventory after ordering: s + a (in inventory units)
    available = begin_inv + orders  # (S, A, 1)
    available_clamped = torch.clamp(available, 0, None)  # Can't be negative
    
    # Expand for batch dimension
    prob_demand_expanded = prob_demand_final.view(batch, 1, 1, n_demands)  # (batch, 1, 1, D)
    selling_price_expanded = selling_price.view(batch, 1, 1, 1)  # (batch, 1, 1, 1)
    
    # Compute E_λ[Number of Sales]
    # For x = 1 to s+a: sales = x with probability f_λ(x)
    # For x > s+a: sales = s+a with probability f_λ(x)
    # So: E[sales] = Σ_{x=1}^{s+a} x·f_λ(x) + (s+a)·Σ_{x=s+a+1}^{∞} f_λ(x)
    
    # Create mask for demands <= available
    demand_mask = demands <= available_clamped  # (S, A, D)
    
    # Sales when demand <= available: sales = demand
    sales_low = demands * demand_mask  # (S, A, D)
    
    # Sales when demand > available: sales = available
    sales_high = available_clamped * (~demand_mask)  # (S, A, D)
    
    sales = sales_low + sales_high  # (S, A, D)
    
    # Expected sales: Σ_{x} sales(x) · f_λ(x)
    expected_sales = (prob_demand_expanded * sales.unsqueeze(0)).sum(dim=3)  # (batch, S, A)
    
    # Revenue = p · E[sales]
    expected_revenue = selling_price_expanded.squeeze(-1) * expected_sales  # (batch, S, A)
    
    # Compute E_λ[Number of Leftover Items] = Σ_{x=0}^{s+a} (s+a-x)·f_λ(x)
    # Leftover = max(0, s+a - demand) for demand <= s+a
    leftover = torch.clamp(available_clamped - demands, 0, None)  # (S, A, D)
    # Only count leftover when demand <= available (otherwise sold out)
    leftover = leftover * demand_mask  # (S, A, D)
    expected_leftover = (prob_demand_expanded * leftover.unsqueeze(0)).sum(dim=3)  # (batch, S, A)
    
    # Ordering cost: ca = foc * I(a>0) + voc * a
    orders_tensor = torch.arange(n_actions, dtype=torch.float32).view(1, 1, -1)  # (1, 1, A)
    order_cost = foc * (orders_tensor > 0).float() + voc * orders_tensor  # (1, 1, A)
    order_cost = order_cost.expand(batch, n_states, -1)  # (batch, S, A)
    
    # Reward = p · E[sales] - ca - h · E[leftover]
    # Note: h = hc (holding cost)
    mdp.r = expected_revenue - order_cost - hc * expected_leftover
    
    return mdp





