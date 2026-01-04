import torch
import numpy as np
from mdp_lp.mdp import MDP


def sparse_to_dense_transitions(sparse_trans_probs, n_samples, n_states, n_actions):
    """
    Convert sparse COO transition probabilities to dense format (batch, S, A, S).
    
    Args:
        sparse_trans_probs: torch.sparse_coo_tensor of shape (n_samples, n_states*n_actions, n_states)
        n_samples: number of samples
        n_states: number of states
        n_actions: number of actions
    
    Returns:
        P: (n_samples, n_states, n_actions, n_states) dense tensor
    """
    P = torch.zeros(n_samples, n_states, n_actions, n_states, 
                    device=sparse_trans_probs.device, dtype=sparse_trans_probs.dtype)
    
    sp = sparse_trans_probs.coalesce()
    idx = sp.indices()  # (3, nnz): [sample, row, col]
    val = sp.values()   # (nnz,)
    
    # row is linearized (state, action) pair: row = s * n_actions + a
    sample_idx = idx[0]
    row_idx = idx[1]
    col_idx = idx[2]
    
    # Convert linearized row to (state, action)
    state_idx = row_idx // n_actions
    action_idx = row_idx % n_actions
    
    # Scatter values into dense tensor
    P[sample_idx, state_idx, action_idx, col_idx] = val
    
    return P


class InventoryLP(MDP):
    """
    Inventory MDP adapted for LP formulation.
    Based on the QP version but converts to dense (batch, S, A, S) format for LP.
    """
    def __init__(self, b_max, s_max, q_max, d_max, foc, voc, hc, bc, beta, mdp_params):
        self.b_max = b_max
        self.s_max = s_max
        self.q_max = q_max
        self.d_max = d_max
        self.foc = foc
        self.voc = voc
        self.hc = hc
        self.bc = bc
        self.beta = beta

        self.n_states = b_max + s_max + 1
        self.n_actions = q_max + 1
        self.n_demands = d_max + 1
        self.n_samples = mdp_params.shape[0]

        self.end_inv = self._get_end_inv()
        self.prob_demand = torch.softmax(mdp_params, dim=1)
        
        trans_probs = self.gen_trans_probs()
        P_dense = sparse_to_dense_transitions(trans_probs, self.n_samples, self.n_states, self.n_actions)
        obj_fun = self.gen_obj_fun()
        r = -obj_fun
        
        self.P = P_dense
        self.r = r
        super().__init__(beta, gamma=None)

    def _get_end_inv(self):
        begin_inv = torch.arange(self.n_states).view(-1, 1, 1) - self.b_max
        orders = torch.arange(self.n_actions).view(1, -1, 1)
        demands = torch.arange(self.d_max + 1).view(1, 1, -1)
        end_inv = torch.clamp(begin_inv + orders - demands, -self.b_max, self.s_max)

        return end_inv

    def gen_trans_probs(self):
        """Generate sparse transition probabilities."""
        end_inv_idx = (self.end_inv + self.b_max).unsqueeze(0).expand(self.n_samples, -1, -1, -1)
        prob_demand = (self.prob_demand.view(self.n_samples, 1, 1, self.n_demands).
                       expand(-1, self.n_states, self.n_actions, -1))
        val = prob_demand.reshape(-1)

        row_base = torch.arange(self.n_states * self.n_actions).view(1, self.n_states, self.n_actions, 1)
        rows = row_base.expand(self.n_samples, -1, -1, self.n_demands).reshape(-1)
        cols = end_inv_idx.reshape(-1)
        sample_ids = (torch.arange(self.n_samples).view(-1, 1, 1, 1).
                      expand(-1, self.n_states, self.n_actions, self.n_demands).reshape(-1))
        idx = torch.stack([sample_ids, rows, cols])

        p_batch = torch.sparse_coo_tensor(
            idx, val,
            size=(self.n_samples, self.n_states * self.n_actions, self.n_states),
            requires_grad=True
        ).coalesce()

        return p_batch

    def gen_obj_fun(self):
        """Generate objective function."""
        overage = torch.clamp(self.end_inv, 0, self.s_max).unsqueeze(0)
        underage = torch.clamp(self.end_inv, -self.b_max, 0).unsqueeze(0)
        prob_demand = self.prob_demand.view(self.n_samples, 1, 1, self.n_demands)
        inventory_cost = (prob_demand * (self.hc * overage - self.bc * underage)).sum(dim=3)
        orders = torch.arange(self.n_actions).view(1, 1, -1)
        order_cost = self.foc * (orders > 0).float() + self.voc * orders

        return inventory_cost + order_cost
