import random
import time
import torch
import numpy as np
from mdp_sparse import MDP


def gen_cluster_set(mdp, d, rnd=True, initial_s=None, policy=None):
    n = len(mdp.states)
    k = len(mdp.actions)
    row_indices, col_indices, probs = mdp.trans_probs
    row_indices: torch.Tensor
    col_indices: torch.Tensor
    probs: dict[int, torch.Tensor]
    n_samples = len(probs)

    # initialize a policy randomly if not given
    if policy is None:
        policy = np.random.randint(0, k, size=n)

    initial_s = set() if initial_s is None else set(initial_s)
    assert len(initial_s) <= d

    states_d = [[] for _ in range(n_samples)]  # cluster set
    for s in range(n_samples):
        initial_s_iterable = iter(initial_s)
        candidate_pairs = {}
        for i in range(d):
            next_state = next(initial_s_iterable, -1)  # default -1 indicating no next elements in initial_s
            if next_state == -1:
                if len(candidate_pairs) > 0:
                    if rnd:
                        # select a state pair from the candidate pairs dictionary according to the weights
                        # ([0] extracts the element from the one-element list returned by random.choices)
                        state_pair = random.choices(list(candidate_pairs.keys()), list(candidate_pairs.values()))[0]
                    else:
                        state_pair = max(candidate_pairs, key=candidate_pairs.get)
                    # pick the second element of the state pair as the next state and add it to the cluster set
                    next_state = state_pair[1]
                else:
                    next_state = torch.multinomial(mdp.init_dist, num_samples=1).item()

            states_d[s].append(next_state)
            # update the candidate pairs dictionary
            mask = (row_indices == next_state)
            matching_indices = torch.nonzero(mask).squeeze(-1)  # indices of the states in row_indices matched with next_state
            new_pairs = {(next_state, col_indices[idx].item()): probs[s][idx, policy[next_state]]
                         for idx in matching_indices if col_indices[idx].item() not in states_d[s]}
            candidate_pairs.update(new_pairs)
            # delete all pairs whose second componenet is the next state just added to the cluster set
            for state in states_d[s]:
                if candidate_pairs.get((state, next_state), -1) > 0:
                    candidate_pairs.pop((state, next_state))

    return torch.tensor(states_d, dtype=torch.long)


def state_agg(mdp, d, rnd=True, initial_s=None, policy=None):
    start_time = time.time()

    n = len(mdp.states)
    k = len(mdp.actions)
    row_indices, col_indices, probs = mdp.trans_probs
    n_samples = len(probs)
    obj_fun = mdp.obj_fun
    gamma = mdp.init_dist
    beta = mdp.disc_factor

    # generate the cluster set
    states_d = gen_cluster_set(mdp, d, rnd, initial_s, policy)

    # transition probability
    # aggregated transition probs are in dense format, assuming d << n
    trans_probs_d = torch.zeros((n_samples, d + 1, d + 1, k))
    packed_sparse = row_indices * n + col_indices  # linearized indexes for the pairs of the nonzero entries
    for s in range(n_samples):
        # get the indices of the cluster states in the set of all states
        # the states in the complement set of the cluster set are indexed by -1
        state_to_cluster_idx = - torch.ones(n, dtype=torch.long)
        state_to_cluster_idx[states_d[s]] = torch.arange(d)

        # ---top-left block of the aggregated transition probability: cluster-to-cluster transitions---
        # create all (i, j) pairs in cluster_set × cluster_set
        ii, jj = torch.meshgrid(states_d[s], states_d[s], indexing='ij')
        top_left_pairs = ii.ravel() * n + jj.ravel()  # linearized indexes for the top-left pairs (cluster pairs)

        # get the boolean mask for the sparse pairs that are in the top-left cluster pairs
        # If both input arrays are unique in values, setting assume_unique=True can speed up the calculation.
        match_mask_cc = torch.isin(packed_sparse, top_left_pairs, assume_unique=True)
        matched_i = row_indices[match_mask_cc]
        matched_j = col_indices[match_mask_cc]
        matched_vals = probs[s][match_mask_cc]  # shape (n_matched_cluster_pairs, n_actions)
        # get the positions of the matched sparse states in the cluster set
        i_idx = state_to_cluster_idx[matched_i]  # shape: n_matched_cluster_pairs
        j_idx = state_to_cluster_idx[matched_j]  # shape: n_matched_cluster_pairs

        # vectorized accumulation at each cluster (i, j) pair for each action
        expand_i = i_idx[:, None].expand(-1, k)
        expand_j = j_idx[:, None].expand(-1, k)
        expand_a = torch.arange(k).expand(len(i_idx), k)

        trans_probs_d[s].index_put_(
            (expand_i.reshape(-1), expand_j.reshape(-1), expand_a.reshape(-1)),
            matched_vals.reshape(-1),
            accumulate=True
        )

        # ---last row of the aggregated transition probability: complement-to-cluster average---
        complement_set = torch.from_numpy(np.setdiff1d(np.arange(n), states_d[s], assume_unique=True))
        ci, cj = torch.meshgrid(complement_set, states_d[s], indexing='ij')
        bottom_row_pairs = ci.ravel() * n + cj.ravel()

        match_mask_nc = torch.isin(packed_sparse, bottom_row_pairs, assume_unique=True)
        matched_j_nc = col_indices[match_mask_nc]
        matched_vals_nc = probs[s][match_mask_nc]  # shape (n_matched_complement_cluster_pairs, n_actions)
        # note: all the elements in matched_j_nc must be in the cluster set by the construction of the meshgrid
        j_idx_nc = state_to_cluster_idx[matched_j_nc]  # shape: n_matched_complement_cluster_pairs
        # vectorized accumulation at each complement-cluster (i, j) pair for each action
        expand_i_nc = torch.full((len(j_idx_nc), k), d, dtype=torch.long)
        expand_j_nc = j_idx_nc[:, None].expand(-1, k)
        expand_a_nc = torch.arange(k).expand(len(j_idx_nc), k)

        trans_probs_d[s].index_put_(
            (expand_i_nc.reshape(-1), expand_j_nc.reshape(-1), expand_a_nc.reshape(-1)),
            matched_vals_nc.reshape(-1),
            accumulate=True
        )

        # --- Last row: divide by (n - d)
        trans_probs_d[s, -1] /= (n - d)
        # --- Last column: 1 - row sum ---
        trans_probs_d[s, :, -1, :] = 1.0 - torch.clamp(trans_probs_d[s, :, :d, :].sum(dim=1), max=1)

    # cost function
    # use advanced indexing to gather the rows corresponding to the cluster set
    # for row s of states_d, get the rows of obj_fun corresponding states_d[s]
    obj_top_d = torch.gather(obj_fun.expand(n_samples, -1, -1), 1,
                             states_d.unsqueeze(-1).expand(-1, -1, k))   # shape: (n_samples, d, n_actions)
    # compute sum over all states once
    total_sum = obj_fun.sum(dim=0)  # shape: (n_actions,)
    # compute per-sample cluster sum and count
    cluster_sum = obj_top_d.sum(dim=1)  # shape: (n_samples, n_actions)
    # compute complement sum = total sum - cluster sum, and divide by n-d
    avg_complement = (total_sum.unsqueeze(0) - cluster_sum) / (n - d)  # (n_samples, n_actions)
    # stack to get (n_samples, d+1, n_actions)
    obj_fun_d = torch.cat([obj_top_d, avg_complement.unsqueeze(1)], dim=1)

    # initial distribution
    gamma_top_d = torch.gather(gamma.expand(n_samples, -1), 1, states_d)  # shape: (n_samples, d)
    gamma_last = 1.0 - gamma_top_d.sum(dim=1, keepdim=True)  # shape: (n_samples, 1)
    gamma_d = torch.cat([gamma_top_d, gamma_last], dim=1)  # shape: (n_samples, d+1)

    mdp_d = MDP(states_d, mdp.actions, trans_probs_d, obj_fun_d, gamma_d, beta)

    run_time = time.time() - start_time

    return mdp_d, run_time
