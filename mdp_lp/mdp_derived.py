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


class BirthDeathLP(MDP):
    """
    Birth-and-death MDP adapted for LP formulation.
    Based on the QP version but converts to dense (batch, S, A, S) format for LP.
    """
    def __init__(self, n_states, n_actions, n_neighbors, two_way, pow1, pow2, beta, mdp_params):
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_neighbors = n_neighbors
        self.two_way = two_way
        self.pow1 = pow1
        self.pow2 = pow2
        self.n_samples = mdp_params.shape[0]
        
        # Generate sparse transitions (same as QP version)
        trans_probs = self.gen_trans_probs(mdp_params)
        
        # Convert sparse to dense for LP formulation
        P_dense = sparse_to_dense_transitions(trans_probs, self.n_samples, n_states, n_actions)
        
        # Generate rewards (negative of objective function for LP maximization)
        obj_fun = self.gen_obj_fun()
        r = -obj_fun  # LP maximizes, so negate costs
        
        # Store for LP formulation
        self.P = P_dense  # (batch, S, A, S)
        self.r = r  # (batch, S, A)
        
        # Initialize parent MDP class
        super().__init__(beta, gamma=None)

    def gen_trans_probs(self, mdp_params):
        """Generate sparse transition probabilities (same as QP version)."""
        logits = mdp_params.view(self.n_samples, self.n_states, self.n_actions, -1)
        origin_states = torch.arange(self.n_states).view(1, -1, 1, 1)  # shape: (1, n, 1, 1)
        actions = torch.arange(self.n_actions).view(1, 1, -1, 1)  # shape: (1, 1, k, 1)
        offsets = torch.arange(-self.n_neighbors, self.n_neighbors + 1) \
            if self.two_way else torch.arange(1, self.n_neighbors + 1)
        dest_states = origin_states + offsets.view(1, 1, 1, -1)  # shape: (1, n, 1, n_offset)
        if not self.two_way:
            # all entries in column 0 are positive; append them to the last dimension of the destination indices
            dest_states = torch.cat([torch.zeros_like(origin_states), dest_states], dim=-1)
        dest_states = dest_states.expand_as(logits)
        valid = (dest_states >= 0) & (dest_states < self.n_states)
        masked_theta = torch.where(valid, logits, float('-inf'))
        probs = torch.softmax(masked_theta, dim=-1)  # softmax over the last dimension, i.e., offset dimension

        # build sparse transition probability in COO format
        # linearized (state, action) pair expanded to match the dimension of dest_states
        row_id = (origin_states * self.n_actions + actions).expand_as(dest_states)
        col_id = torch.clamp(dest_states, 0, self.n_states - 1)
        sample_id: torch.Tensor = torch.arange(self.n_samples).view(-1, 1, 1, 1)
        sample_id = sample_id.expand_as(dest_states)
        idx = torch.stack([sample_id.reshape(-1), row_id.reshape(-1), col_id.reshape(-1)])
        val = probs.reshape(-1)

        # val at duplicate (row, col) combinations will be summed up by "coalesce"
        # they are zero probs corresponding to the invalid indices
        p_batch = torch.sparse_coo_tensor(
            idx, val,
            size=(self.n_samples, self.n_states * self.n_actions, self.n_states),
            requires_grad=True
        ).coalesce()

        return p_batch

    def gen_obj_fun(self):
        """Generate objective function (same as QP version)."""
        obj_fun = ((torch.arange(1, self.n_actions + 1).reshape(1, -1)) ** self.pow1 /
                   (torch.arange(1, self.n_states + 1).reshape(-1, 1)) ** self.pow2)
        return obj_fun.unsqueeze(0).expand(self.n_samples, -1, -1)


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
        
        # Generate sparse transitions (same as QP version)
        trans_probs = self.gen_trans_probs()
        
        # Convert sparse to dense for LP formulation
        P_dense = sparse_to_dense_transitions(trans_probs, self.n_samples, self.n_states, self.n_actions)
        
        # Generate rewards (negative of objective function for LP maximization)
        obj_fun = self.gen_obj_fun()
        r = -obj_fun  # LP maximizes, so negate costs
        
        # Store for LP formulation
        self.P = P_dense  # (batch, S, A, S)
        self.r = r  # (batch, S, A)
        
        # Initialize parent MDP class
        super().__init__(beta, gamma=None)

    def _get_end_inv(self):
        # states are indexed from 0, whereas the smallest inventory is -b_max
        begin_inv = torch.arange(self.n_states).view(-1, 1, 1) - self.b_max
        orders = torch.arange(self.n_actions).view(1, -1, 1)
        demands = torch.arange(self.d_max + 1).view(1, 1, -1)
        end_inv = torch.clamp(begin_inv + orders - demands, -self.b_max, self.s_max)

        return end_inv

    def gen_trans_probs(self):
        """Generate sparse transition probabilities (same as QP version)."""
        # get back the indices for the ending inventory
        end_inv_idx = (self.end_inv + self.b_max).unsqueeze(0).expand(self.n_samples, -1, -1, -1)
        prob_demand = (self.prob_demand.view(self.n_samples, 1, 1, self.n_demands).
                       expand(-1, self.n_states, self.n_actions, -1))
        val = prob_demand.reshape(-1)

        row_base = torch.arange(self.n_states * self.n_actions).view(1, self.n_states, self.n_actions, 1)
        rows = row_base.expand(self.n_samples, -1, -1, self.n_demands).reshape(-1)
        cols = end_inv_idx.reshape(-1)
        sample_ids = (torch.arange(self.n_samples).view(-1, 1, 1, 1).
                      expand(-1, self.n_states, self.n_actions, self.n_demands).reshape(-1))
        idx = torch.stack([sample_ids, rows, cols])  # shape: (3, n_samples * n_states * n_actions * n_demands)

        # probability values corresponding to the same (i, a, j) will be summed up
        p_batch = torch.sparse_coo_tensor(
            idx, val,
            size=(self.n_samples, self.n_states * self.n_actions, self.n_states),
            requires_grad=True
        ).coalesce()

        return p_batch

    def gen_obj_fun(self):
        """Generate objective function (same as QP version)."""
        overage = torch.clamp(self.end_inv, 0, self.s_max).unsqueeze(0)  # shape: (1, n_states, n_actions, n_demands)
        underage = torch.clamp(self.end_inv, -self.b_max, 0).unsqueeze(0)  # shape: (1, n_states, n_actions, n_demands)
        prob_demand = self.prob_demand.view(self.n_samples, 1, 1, self.n_demands)
        inventory_cost = (prob_demand * (self.hc * overage - self.bc * underage)).sum(dim=3)
        orders = torch.arange(self.n_actions).view(1, 1, -1)
        order_cost = self.foc * (orders > 0).float() + self.voc * orders  # shape: (n_samples, n_states, n_actions)

        return inventory_cost + order_cost
