"""
Soft (learned) state aggregation for the dense inventory MDP used in mdp_lp.

This mirrors the idea in the top-level `sa_sparse.SoftAgg`, but operates on the
dense transition tensor used by `mdp_lp.mdp_derived.InventoryLP`:

    P[b, s, a, s'] = Pr(s' | s, a)
    r[b, s, a]     = reward
    gamma[b, s]    = initial state distribution

We learn a soft membership matrix W[b, s, m] (row-softmax over m clusters) and
build an aggregated MDP with m clusters:

    r_agg[b, m, a]       = average reward in cluster m under action a
    P_agg[b, m, a, m']   = aggregated transition probabilities
    gamma_agg[b, m]      = aggregated initial distribution
"""

from __future__ import annotations

import time
import torch
from torch import nn


class SoftAggDense(nn.Module):
    """
    Learn soft memberships W and aggregate (P, r, gamma).

    Directly learns a dense logit matrix W_logits (S x M) and applies
    softmax(W_logits / tau) to obtain the membership matrix W.
    """

    def __init__(self, n_states: int, n_actions: int, n_clusters: int, **kwargs):
        super().__init__()
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.n_clusters = int(n_clusters)

        
        # NOTE:Method 1: W depends on P
        # self.W_logits = nn.Linear(n_states * n_actions, n_clusters)

        # NOTE:Method 2: W doesn't depend on P and r
        self.W_logits = nn.Parameter(torch.randn(self.n_states, self.n_clusters))

        # NOTE:Method 3: Original method
        # self.dest_embed = nn.Embedding(num_embeddings=self.n_states, embedding_dim=self.r_embed)
        # self.to_logits = nn.Linear(self.n_actions * self.r_embed, self.n_clusters)

    def forward(
        self,
        P: torch.Tensor,
        r: torch.Tensor,
        gamma: torch.Tensor,
        tau: float = 0.5,
        eps: float = 1e-8,
    ):
        """
        Args:
            P: (B, S, A, S)
            r: (B, S, A)
            gamma: (B, S)
            tau: softmax temperature
            eps: numerical stability epsilon

        Returns:
            P_agg:     (B, M, A, M)
            r_agg:     (B, M, A)
            gamma_agg: (B, M)
            info: dict with 'W' and 'runtime_sec'
        """
        start = time.time()

        if P.dim() != 4:
            raise ValueError(f"P must have shape (B,S,A,S), got {tuple(P.shape)}")
        if r.dim() != 3:
            raise ValueError(f"r must have shape (B,S,A), got {tuple(r.shape)}")
        if gamma.dim() != 2:
            raise ValueError(f"gamma must have shape (B,S), got {tuple(gamma.shape)}")

        B, S, A, S2 = P.shape
        if S2 != S or r.shape != (B, S, A) or gamma.shape != (B, S):
            raise ValueError(
                f"Shape mismatch: P={tuple(P.shape)}, r={tuple(r.shape)}, gamma={tuple(gamma.shape)}"
            )
        if S != self.n_states or A != self.n_actions:
            raise ValueError(
                f"SoftAggDense was created for S={self.n_states},A={self.n_actions} "
                f"but got S={S},A={A}"
            )

        # === (1) Build memberships W[b,s,m] from learned logits ===
        # NOTE: Method 1: W depends on P
        # P_flat = P.reshape(B, S, A * S)
        # W_logits = self.W_logits(P_flat)          # (B, S, M)
        # W = torch.softmax(W_logits / tau, dim=-1)

        # NOTE: Method 2: W doesn't depend on P and r
        W = torch.softmax(self.W_logits / float(tau), dim=1)  # (S, M)
        W = W.unsqueeze(0).expand(B, -1, -1)  # (B, S, M)

        # NOTE: Method 3: Original method
        # E = self.dest_embed.weight  # (S, r)
        # # phi[b,s,a,r] = sum_{s'} P[b,s,a,s'] * E[s',r]
        # phi = torch.einsum("bsaj,jr->bsar", P, E)  # (B,S,A,r)
        # phi_cat = phi.reshape(B, S, A * self.r_embed)  # (B,S,A*r)
        # logits = self.to_logits(phi_cat)  # (B,S,M)
        # W = torch.softmax(logits / float(tau), dim=2)  # (B,S,M)

        # Cluster masses: mass[b,m] = sum_s W[b,s,m]
        mass = W.sum(dim=1).clamp_min(eps)  # (B,M)

        # === (2) Aggregate rewards ===
        # r_agg[b,m,a] = sum_s W[b,s,m] r[b,s,a] / mass[b,m]
        r_agg = torch.einsum("bsm,bsa->bma", W, r) / mass.unsqueeze(-1)  # (B,M,A)

        # === (3) Aggregate transitions ===
        # Right[b,s,a,m'] = sum_{s'} P[b,s,a,s'] W[b,s',m']
        right = torch.einsum("bsaj,bjm->bsam", P, W)  # (B,S,A,M)
        # p_hat[b,m,a,m'] = sum_s W[b,s,m] Right[b,s,a,m']
        # NOTE: we must use distinct symbols for source-cluster (m) and dest-cluster (n),
        # otherwise einsum sees repeated indices incorrectly.
        # p_hat[b,m,a,n] = sum_s W[b,s,m] * right[b,s,a,n]
        p_hat = torch.einsum("bsm,bsan->bman", W, right)  # (B,M,A,M)
        P_agg = p_hat / mass[:, :, None, None]  # (B,M,A,M)

        # Enforce row-stochasticity for numerical stability
        P_agg = P_agg.clamp_min(0.0)
        P_agg = P_agg / P_agg.sum(dim=-1, keepdim=True).clamp_min(eps)

        # === (4) Aggregate initial distribution ===
        gamma_agg = torch.einsum("bs,bsm->bm", gamma, W)
        gamma_agg = gamma_agg.clamp_min(eps)
        gamma_agg = gamma_agg / gamma_agg.sum(dim=1, keepdim=True).clamp_min(eps)

        return P_agg, r_agg, gamma_agg, {"W": W, "runtime_sec": time.time() - start}

    @staticmethod
    def disaggregate_y(y_agg: torch.Tensor, W: torch.Tensor, n_actions: int) -> torch.Tensor:
        """Map aggregated occupancy back to the original state space.

        Args:
            y_agg:  (B, M*A) flat occupancy in the aggregated MDP
            W:      (B, S, M) soft membership matrix from forward()
            n_actions: number of actions A

        Returns:
            y_orig: (B, S*A) occupancy in the original state space
        """
        B = y_agg.shape[0]
        S = W.shape[1]
        M = W.shape[2]
        A = n_actions

        y_agg_3d = y_agg.view(B, M, A)                       # (B, M, A)
        # y_orig[b, s, a] = sum_m W[b, s, m] * y_agg[b, m, a]
        y_orig = torch.einsum("bsm,bma->bsa", W, y_agg_3d)   # (B, S, A)
        return y_orig.reshape(B, S * A)