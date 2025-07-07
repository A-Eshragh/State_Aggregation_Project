import torch
import numpy as np
import torch.nn as nn
from mdp_base import MDP


class BirthDeath(MDP):
    def __init__(self, n_states, n_actions, n_neighbors, two_way, mdp_params, pow1, pow2, beta):
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_neighbors = n_neighbors
        self.two_way = two_way
        self.pow1 = pow1
        self.pow2 = pow2
        states = np.arange(n_states)
        actions = np.arange(n_actions)
        trans_probs = self.gen_trans_probs(mdp_params)
        obj_fun = self.gen_obj_fun(n_states, n_actions)
        gamma = 1 / n_states * torch.ones(n_states)
        super().__init__(states, actions, trans_probs, obj_fun, gamma, beta)

    def compute_successor_mask(self):
        base = torch.arange(self.n_states).unsqueeze(1)
        zero_col = torch.zeros(self.n_states, 1, dtype=torch.long)
        true_col = torch.ones(self.n_states, 1, dtype=torch.bool)
        offsets = torch.arange(-self.n_neighbors, self.n_neighbors + 1).unsqueeze(0) if self.two_way \
            else torch.arange(1, self.n_neighbors + 1).unsqueeze(0)
        neighbors = base + offsets
        # ensure (n-1, n-1) is included when two_way is False
        neighbors[-1, 0] = neighbors[-1, 0] if self.two_way else self.n_states - 1
        valid_mask = (neighbors > 0) & (neighbors < self.n_states)
        successors = torch.cat([zero_col, neighbors], dim=1)
        final_mask = torch.cat([true_col, valid_mask], dim=1)
        # ensure (0, 0) is excluded when two_way is False
        final_mask[0, 0] = self.two_way

        return successors, final_mask

    def gen_trans_probs(self, mdp_params):
        """
        Vectorized, construction of sparse birth-death transition probability
        Each state transitions to at most m neighbors to the left and right (within bounds)

        Returns:
            row_indices: (nnz, ) LongTensor
            col_indices: (nnz, ) LongTensor
            probs: (n_samples, nnz, n_actions)
        """

        successors, final_mask = self.compute_successor_mask()
        row_ids = torch.arange(self.n_states).unsqueeze(1).expand_as(successors)
        row_indices = row_ids.flatten()[final_mask.flatten()]
        col_indices = successors.flatten()[final_mask.flatten()]

        # construct transition probabilities from mdp_params
        row_sizes = final_mask.sum(dim=1)
        max_nnz_per_row = row_sizes.max().item()
        assert mdp_params.shape[1] == self.n_states * max_nnz_per_row * self.n_actions  # output dim
        logits = mdp_params.view(mdp_params.shape[0], self.n_states, max_nnz_per_row, self.n_actions)
        logits_mask = torch.arange(max_nnz_per_row).unsqueeze(0) < row_sizes.unsqueeze(1)
        logits_mask = logits_mask.unsqueeze(0).unsqueeze(-1)  # shape: (1, n_states, max_nnz_per_row, 1)
        masked_logits = logits.masked_fill(~logits_mask, float('-inf'))
        # apply softmax to each row of the transition probability
        probs_full = nn.functional.softmax(masked_logits, dim=2)
        probs = probs_full.reshape(logits.shape[0], -1, self.n_actions)[:, logits_mask.view(-1), :]

        return row_indices, col_indices, probs

    def gen_obj_fun(self, n_states, n_actions):
        obj_fun = ((torch.arange(1, n_actions + 1).reshape(1, -1)) ** self.pow1 /
                   (torch.arange(1, n_states + 1).reshape(-1, 1)) ** self.pow2)
        return obj_fun
