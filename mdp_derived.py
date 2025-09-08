import torch
import numpy as np
from mdp_base import MDP


class BirthDeath(MDP):
    def __init__(self, n_states, n_actions, n_neighbors, two_way, pow1, pow2, beta, mdp_params):
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_neighbors = n_neighbors
        self.two_way = two_way
        self.pow1 = pow1
        self.pow2 = pow2
        self.n_samples = mdp_params.shape[0]
        trans_probs = self.gen_trans_probs(mdp_params)
        obj_fun = self.gen_obj_fun()
        gamma = 1 / n_states * torch.ones(n_states)
        super().__init__(np.arange(n_states), np.arange(n_actions), trans_probs, obj_fun, gamma, beta)

    def gen_trans_probs(self, mdp_params):
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

        # # verify row-stochasticity
        # row_sums = torch.sparse.sum(p_batch, dim=2).to_dense()
        # torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

        return p_batch

    def gen_obj_fun(self):
        obj_fun = ((torch.arange(1, self.n_actions + 1).reshape(1, -1)) ** self.pow1 /
                   (torch.arange(1, self.n_states + 1).reshape(-1, 1)) ** self.pow2)
        return obj_fun.unsqueeze(0).expand(self.n_samples, -1, -1)


class Inventory(MDP):
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
        obj_fun = self.gen_obj_fun()
        gamma = 1 / self.n_states * torch.ones(self.n_states)
        super().__init__(np.arange(self.n_states), np.arange(self.n_actions), trans_probs, obj_fun, gamma, beta)

    def _get_end_inv(self):
        # states are indexed from 0, whereas the smallest inventory is -b_max
        begin_inv = torch.arange(self.n_states).view(-1, 1, 1) - self.b_max
        orders = torch.arange(self.n_actions).view(1, -1, 1)
        demands = torch.arange(self.d_max + 1).view(1, 1, -1)
        end_inv = torch.clamp(begin_inv + orders - demands, -self.b_max, self.s_max)

        return end_inv

    def gen_trans_probs(self):
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

        # # verify row-stochasticity
        # row_sums = torch.sparse.sum(p_batch, dim=2).to_dense()
        # torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

        return p_batch

    def gen_obj_fun(self):
        overage = torch.clamp(self.end_inv, 0, self.s_max).unsqueeze(0)  # shape: (1, n_states, n_actions, n_demands)
        underage = torch.clamp(self.end_inv, -self.b_max, 0).unsqueeze(0)  # shape: (1, n_states, n_actions, n_demands)
        prob_demand = self.prob_demand.view(self.n_samples, 1, 1, self.n_demands)
        inventory_cost = (prob_demand * (self.hc * overage - self.bc * underage)).sum(dim=3)
        orders = torch.arange(self.n_actions).view(1, 1, -1)
        order_cost = self.foc * (orders > 0).float() + self.voc * orders  # shape: (n_samples, n_states, n_actions)

        return inventory_cost + order_cost
