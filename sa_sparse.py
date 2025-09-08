import random
import time
import torch
import numpy as np
from mdp_base import MDP
from torch import nn


class SoftAgg(nn.Module):
    """
    Soft state aggregation where memberships W are learned from the transition probability P

    """

    def __init__(self, n_states: int, n_actions: int, m_clusters: int, r_embed: int = 8):
        """
        Args
        ----
        n_states  : n
        n_actions : k
        m_clusters: m  (usually n_agg + 1)
        r_embed   : r  destination embedding dimension
        """

        super().__init__()
        self.n = n_states
        self.k = n_actions
        self.m = m_clusters
        self.r = r_embed

        # (1) Learnable destination-state embedding (n-by-r)
        self.dest_embed = nn.Embedding(num_embeddings=self.n, embedding_dim=self.r)

        # (2) Linear map from concatenated per-action features (k*r) -> logits (m)
        self.to_logits = nn.Linear(self.k * self.r, self.m)

    def forward(self, mdp, tau: float = 0.5):
        """
        Returns
        -------
        mdp_d   : aggregated dense MDP with
                  trans_probs: (S, m*k, m), obj_fun: (S, m, k), init_dist: (S, m)
        run_sec : wall-clock seconds
        """

        start_time = time.time()

        n_samples = mdp.trans_probs.size(0)
        n, k, m, r = self.n, self.k, self.m, self.r

        # === Sparse COO view ===
        sp = mdp.trans_probs.coalesce()
        idx = sp.indices()  # (3, nnz): [s, row, col]
        val = sp.values()  # (nnz,)
        s_idx = idx[0]  # (nnz,)
        row_idx = idx[1]  # (nnz,) in [0, n*k)
        col_idx = idx[2]  # (nnz,) in [0, n)

        # === (A) Build W from P for ALL samples at once ===
        # (A1) Expected dest embeddings per (i, a)
        embed = self.dest_embed.weight  # (n, r)
        # Gather embed[col] for each nonzero, scale by P value, scatter-add to (S*nk, r)
        embed_cols = embed[col_idx, :]  # (nnz, r)
        contrib_phi = embed_cols * val.unsqueeze(1)  # P @ E: (nnz, r)
        phi_flat = torch.zeros(n_samples * n * k, r)  # (S*nk, r)
        keys_sia = s_idx * n * k + row_idx  # linear index of (s, ia): (nnz,)
        phi_flat.index_add_(0, keys_sia, contrib_phi)  # sum over the destination states
        phi = phi_flat.view(n_samples, n, k * r)  # (S, n, k*r)

        # (A2) Logits and soft memberships: W
        logits = self.to_logits(phi)  # (S, n, m)
        weight = torch.softmax(logits / tau, dim=2)  # (S, n, m)
        mass = weight.sum(dim=1) + 1e-6  # (S, m) weight sum for each clustered state

        # === (B) Aggregate costs using W ===
        # obj_fun_d[s,c,a] = sum_i W[s,i,c] * obj_fun[s,i,a]
        obj_fun_d = torch.einsum('snm,snk->smk', weight, mdp.obj_fun)  # (S, m, k)
        obj_fun_d = obj_fun_d / mass.unsqueeze(-1)  # per-cluster averages

        # === (C) Aggregate transitions, vectorized (all samples) ===
        # (C1) Aggregate destinations: Right = P @ W
        # Gather W for each nonzero, scale by P value, scatter-add to (S*nk, m)
        weight_flat = weight.reshape(n_samples * n, m)  # (S*n, m)
        gather_idx = s_idx * n + col_idx  # (nnz,)
        weight_cols = weight_flat.gather(0, gather_idx.unsqueeze(1).expand(-1, m))  # (nnz, m)
        contrib_right = weight_cols * val.unsqueeze(1)  # (nnz, m)

        right_flat = torch.zeros(n_samples * n * k, m)  # (S*nk, m)
        right_flat.index_add_(0, keys_sia, contrib_right)
        right = right_flat.view(n_samples, n, k, m)

        # (C2) Aggregate origins: P_hat[s,c,a,m] = sum_n W[s,n,c] * right[s,n,a,m]
        p_hat = torch.einsum('snc,snkm->sckm', weight, right)  # (S, m, k, m)

        # (C3) Normalize rows (origin clusters) by D^{-1}, ensure row stochasticity
        p_agg = p_hat / mass[:, :, None, None]  # (S, m, k, m)
        p_agg = p_agg / p_agg.sum(dim=-1, keepdim=True)

        # Pack to (S, m*k, m)
        trans_probs_d = p_agg.reshape(n_samples, m * k, m)

        # === (D) Aggregate initial distribution ===
        gamma = mdp.init_dist.expand(n_samples, -1)  # (S, n)
        gamma_d = torch.einsum('sn,snm->sm', gamma, weight)  # (S, m)
        gamma_d = gamma_d / gamma_d.sum(dim=1, keepdim=True)

        # === (E) Assemble aggregated MDP ===
        mdp_d = MDP(
            states=torch.arange(m),
            actions=mdp.actions,
            trans_probs=trans_probs_d,  # (S, m*k, m)
            obj_fun=obj_fun_d,  # (S, m, k)
            init_dist=gamma_d,  # (S, m)
            disc_factor=mdp.disc_factor
        )

        return mdp_d, time.time() - start_time


def gen_cluster_set(mdp, n_agg, rnd=True, initial_s=None, policy=None):
    trans_probs = mdp.trans_probs
    idx = trans_probs.indices()  # shape: (3, nnz), nnz indices for samples, rows, and columns
    val = trans_probs.values()  # shape: (nnz, )
    n_samples = len(trans_probs)

    # initialize a policy randomly if not given
    if policy is None:
        policy = np.random.randint(0, mdp.n_actions, size=mdp.n_states)

    initial_s = set() if initial_s is None else set(initial_s)
    assert len(initial_s) <= n_agg

    states_d = [[] for _ in range(n_samples)]  # cluster set
    for s in range(n_samples):
        initial_s_iterable = iter(initial_s)
        candidate_pairs = {}
        for i in range(n_agg):
            current_state = next(initial_s_iterable, -1)  # default -1 indicating no next elements in initial_s
            if current_state == -1:
                if len(candidate_pairs) > 0:
                    if rnd:
                        # select a state pair from the candidate pairs dictionary according to the weights
                        # ([0] extracts the element from the one-element list returned by random.choices)
                        state_pair = random.choices(list(candidate_pairs.keys()), list(candidate_pairs.values()))[0]
                    else:
                        state_pair = max(candidate_pairs, key=candidate_pairs.get)
                    # pick the second element of the state pair as the current state and add it to the cluster set
                    current_state = state_pair[1]
                else:
                    current_state = torch.multinomial(mdp.init_dist, num_samples=1).item()

            states_d[s].append(current_state)
            # linearized row index for (current_state, pi(current_state))
            sa_idx = current_state * mdp.n_actions + policy[current_state]
            mask = (idx[0] == s) & (idx[1] == sa_idx)
            # get the column indices and the values of the nnz
            new_pairs = {(current_state, next_state): prob
                         for next_state, prob in zip(idx[2, mask].tolist(), val[mask].tolist())
                         if next_state not in states_d[s]}
            candidate_pairs.update(new_pairs)
            # delete all pairs whose second componenet is the current state just added to the cluster set
            for state in states_d[s]:
                if candidate_pairs.get((state, current_state), -1) > 0:
                    candidate_pairs.pop((state, current_state))

    return torch.tensor(states_d, dtype=torch.long)


def state_agg(mdp, n_agg, rnd=True, initial_s=None, policy=None):
    start_time = time.time()

    n = mdp.n_states
    k = mdp.n_actions
    trans_probs = mdp.trans_probs
    idx = trans_probs.indices()  # shape: (3, nnz), nnz indices for samples, rows, and columns
    val = trans_probs.values()  # shape: (nnz, )
    n_samples = len(trans_probs)
    obj_fun = mdp.obj_fun
    gamma = mdp.init_dist
    beta = mdp.disc_factor

    # generate the cluster set
    states_d = gen_cluster_set(mdp, n_agg, rnd, initial_s, policy)

    # transition probability
    # aggregated transition probs are in dense format, assuming n_agg << n
    trans_probs_d = torch.zeros((n_samples, (n_agg + 1) * k, n_agg + 1))

    for s in range(n_samples):
        # filter out entries for sample s
        mask_s = (idx[0] == s)
        ia_s = idx[1, mask_s]  # (origin state, action) pair
        i_s = ia_s // k  # origin state
        a_s = ia_s % k  # action
        j_s = idx[2, mask_s]  # dest state
        v_s = val[mask_s]  # transition values

        # map each state to its cluster index (0...n_agg-1), or -1 if in the complement
        state_to_cluster_idx = -torch.ones(n, dtype=torch.long)
        state_to_cluster_idx[states_d[s]] = torch.arange(n_agg)

        i_cluster = state_to_cluster_idx[i_s]
        j_cluster = state_to_cluster_idx[j_s]

        # cluster-to-cluster transitions
        mask_cc = (i_cluster >= 0) & (j_cluster >= 0)
        flat_i_cc = i_cluster[mask_cc] * k + a_s[mask_cc]

        trans_probs_d[s].index_put_(
            (flat_i_cc, j_cluster[mask_cc]),
            v_s[mask_cc],
            accumulate=True
        )

        # complement-to-cluster transitions (last row block)
        mask_nc = (i_cluster == -1) & (j_cluster >= 0)
        flat_i_nc = (n_agg * k) + a_s[mask_nc]

        trans_probs_d[s].index_put_(
            (flat_i_nc, j_cluster[mask_nc]),
            v_s[mask_nc],
            accumulate=True
        )

        # normalize the last row across complement
        trans_probs_d[s, n_agg * k: (n_agg + 1) * k, :] /= (n - n_agg)

        # residual: last column
        row_sum = trans_probs_d[s, :, :n_agg].sum(dim=1)
        trans_probs_d[s, :, -1] = 1.0 - row_sum

    # cost function
    # advanced indexing to gather the rows corresponding to the cluster set
    # for row s of states_d, get the rows of obj_fun corresponding states_d[s]
    obj_top_d = torch.gather(obj_fun, dim=1, index=states_d.unsqueeze(-1).expand(-1, -1, k))
    # compute the sum over all states for each sample
    total_sum = obj_fun.sum(dim=1)  # shape: (n_samples, n_actions)
    # compute per-sample cluster sum and count
    cluster_sum = obj_top_d.sum(dim=1)  # shape: (n_samples, n_actions)
    # compute complement sum = total sum - cluster sum, and divide by n-n_agg
    avg_complement = (total_sum - cluster_sum) / (n - n_agg)  # (n_samples, n_actions)
    # stack to get (n_samples, n_agg+1, n_actions)
    obj_fun_d = torch.cat([obj_top_d, avg_complement.unsqueeze(1)], dim=1)

    # initial distribution
    gamma_top_d = torch.gather(gamma.expand(n_samples, -1), 1, states_d)  # shape: (n_samples, n_agg)
    gamma_last = 1.0 - gamma_top_d.sum(dim=1, keepdim=True)  # shape: (n_samples, 1)
    gamma_d = torch.cat([gamma_top_d, gamma_last], dim=1)  # shape: (n_samples, n_agg+1)

    mdp_d = MDP(states_d, mdp.actions, trans_probs_d, obj_fun_d, gamma_d, beta)

    run_time = time.time() - start_time

    return mdp_d, run_time
