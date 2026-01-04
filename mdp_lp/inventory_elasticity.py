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
    
    # Modify rewards to include selling price revenue
    n_states = mdp.n_states
    n_actions = mdp.n_actions
    n_demands = mdp.n_demands
    
    begin_inv = torch.arange(n_states).view(-1, 1, 1) - b_max
    orders = torch.arange(n_actions).view(1, -1, 1)
    demands = torch.arange(n_demands).view(1, 1, -1)
    
    available = begin_inv + orders
    available_clamped = torch.clamp(available, 0, None)
    sales = torch.min(demands, available_clamped)
    
    selling_price_expanded = selling_price.view(batch, 1, 1, 1)
    revenue = selling_price_expanded * sales.unsqueeze(0)
    
    prob_demand_expanded = prob_demand_final.view(batch, 1, 1, n_demands)
    expected_revenue = (prob_demand_expanded * revenue).sum(dim=3)
    
    overage = torch.clamp(mdp.end_inv, 0, s_max).unsqueeze(0)
    underage = torch.clamp(mdp.end_inv, -b_max, 0).unsqueeze(0)
    inventory_cost = (prob_demand_expanded * (hc * overage - bc * underage)).sum(dim=3)
    orders_tensor = torch.arange(n_actions).view(1, 1, -1)
    order_cost = foc * (orders_tensor > 0).float() + voc * orders_tensor
    
    mdp.r = -(expected_revenue - inventory_cost - order_cost)
    
    return mdp





