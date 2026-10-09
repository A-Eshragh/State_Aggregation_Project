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

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        n_clusters: int,
        r_embed: int = 8,
        fixed_W: bool = False,
        fixed_W_seed: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.n_clusters = int(n_clusters)
        self.r_embed = int(r_embed)
        self.fixed_W = bool(fixed_W)

        if self.fixed_W:
            gen = torch.Generator()
            gen.manual_seed(int(fixed_W_seed))
            W_logits = torch.randn(self.n_states, self.n_clusters, generator=gen)
            W_fixed = torch.softmax(W_logits, dim=1)  # (S, M)
            self.register_buffer("W_fixed", W_fixed)
        else:
            # NOTE:Method 1: W depends on P
            # self.W_logits = nn.Linear(n_states * n_actions, n_clusters)

            # NOTE:Method 2: W doesn't depend on P and r
            # self.W_logits = nn.Parameter(torch.randn(self.n_states, self.n_clusters))

            # NOTE:Method 3: Original method
            # self.dest_embed = nn.Embedding(num_embeddings=self.n_states, embedding_dim=self.r_embed)
            # self.to_logits = nn.Linear(self.n_actions * self.r_embed, self.n_clusters)
        
            # NOTE:Method 4: Original method with r_bias
            self.dest_embed = nn.Embedding(num_embeddings=self.n_states, embedding_dim=self.r_embed)
            self.to_logits = nn.Linear(self.n_actions * self.r_embed, self.n_clusters)
            self.r_bias = nn.Linear(self.n_actions, self.n_clusters, bias=False)
            nn.init.normal_(self.r_bias.weight, std=1e-3)

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

        W = self.compute_W(P, r, tau)

        P_agg, r_agg, gamma_agg = aggregate_with_W(P, r, gamma, W, eps=eps)

        return P_agg, r_agg, gamma_agg, {"W": W, "runtime_sec": time.time() - start}

    def compute_W(self, P: torch.Tensor, r: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
        """Build soft memberships W[b,s,m] from learned logits (or fixed buffer)."""
        B, S, A, _ = P.shape

        if self.fixed_W:
            # Method 5: fixed random W (buffer), broadcast to batch.
            return self.W_fixed.unsqueeze(0).expand(B, -1, -1)  # (B,S,M)

        # NOTE: Method 1: W depends on P
        # P_flat = P.reshape(B, S, A * S)
        # W_logits = self.W_logits(P_flat)          # (B, S, M)
        # W = torch.softmax(W_logits / tau, dim=-1)

        # NOTE: Method 2: W doesn't depend on P and r
        # W = torch.softmax(self.W_logits / float(tau), dim=1)  # (S, M)
        # W = W.unsqueeze(0).expand(B, -1, -1)  # (B, S, M)

        # NOTE: Method 3: Original method (correct einsum)
        # E = self.dest_embed.weight  # (S, r)
        # # phi[b,s,a,r] = sum_{s'} P[b,s,a,s'] * E[s',r]
        # phi = torch.einsum("bsaj,jr->bsar", P, E)  # (B,S,A,r)
        # phi_cat = phi.reshape(B, S, A * self.r_embed)  # (B,S,A*r)
        # logits = self.to_logits(phi_cat)  # (B,S,M)
        # W = torch.softmax(logits / float(tau), dim=2)  # (B,S,M)

        # NOTE: Method 3.2: Original method (wrong einsum)
        # E = self.dest_embed.weight  # (S, r)
        # # phi[b,s,a,r] = sum_{s'} P[b,s,a,s'] * E[s',r]
        # phi = torch.einsum("bsas,sr->bsar", P, E)  # (B,S,A,r)
        # phi_cat = phi.reshape(B, S, A * self.r_embed)  # (B,S,A*r)
        # logits = self.to_logits(phi_cat)  # (B,S,M)
        # W = torch.softmax(logits / float(tau), dim=2)  # (B,S,M)

        # NOTE: Method 4: Original method with r_bias
        E = self.dest_embed.weight  # (S, r_embed)
        # phi[b,s,a,r] = sum_{s'} P[b,s,a,s'] * E[s',r]
        phi = torch.einsum("bsaj,jr->bsar", P, E)  # (B,S,A,r_embed)
        phi_cat = phi.reshape(B, S, A * self.r_embed)  # (B,S,A*r_embed)
        # NOTE: Only change
        logits = self.to_logits(phi_cat) + self.r_bias(r)  # (B,S,M)
        W = torch.softmax(logits / float(tau), dim=2)  # (B,S,M)
        return W

    def forward_galerkin(
        self,
        P: torch.Tensor,
        r: torch.Tensor,
        gamma: torch.Tensor,
        beta: float,
        tau: float = 0.5,
    ):
        """Exact Galerkin projected LP (LP-2 with Eq.(6) features), WITHOUT the
        diagonal projection: the B-block is the full Gram sum_s W_si W_sm.

        Returns (A, r_tilde, b, info) for  max r_tilde^T w  s.t.  A w = b, w >= 0,
        directly consumable by DifferentiableLPSolver / DenseLPSolver.
        The occupancy lift is y(s,a) = sum_m W_sm w_(m,a) = disaggregate_y(w, W, A).
        """
        start = time.time()
        W = self.compute_W(P, r, tau)
        A_mat, r_tilde, b = galerkin_lp_coefficients(P, r, gamma, W, beta)
        return A_mat, r_tilde, b, {"W": W, "runtime_sec": time.time() - start}

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


def aggregate_with_W(
    P: torch.Tensor,
    r: torch.Tensor,
    gamma: torch.Tensor,
    W: torch.Tensor,
    eps: float = 1e-8,
):
    """Aggregate (P, r, gamma) with a given membership W[b,s,m].

    This is the 'projected MDP' path (diagonal projection of the Galerkin
    B-block): returns a genuine aggregated MDP (P_agg, r_agg, gamma_agg)
    whose occupancy LP is  (I - beta*P_agg) y = gamma_agg.
    """
    # Cluster masses: mass[b,m] = sum_s W[b,s,m]
    mass = W.sum(dim=1).clamp_min(eps)  # (B,M)

    # === (2) Aggregate rewards ===
    # r_agg[b,m,a] = sum_s W[b,s,m] r[b,s,a] / mass[b,m]
    r_agg = torch.einsum("bsm,bsa->bma", W, r) / mass.unsqueeze(-1)  # (B,M,A)

    # === (3) Aggregate transitions ===
    # Right[b,s,a,m'] = sum_{s'} P[b,s,a,s'] W[b,s',m']
    # NOTE: Method 4: Original method with wrong einsum
    # right = torch.einsum("bsas,bsm->bsam", P, W)  # (B,S,A,M)

    right = torch.einsum("bsaj,bjm->bsam", P, W)  # (B,S,A,M)
    p_hat = torch.einsum("bsm,bsan->bman", W, right)  # (B,M,A,M)
    P_agg = p_hat / mass[:, :, None, None]  # (B,M,A,M)

    # Enforce row-stochasticity for numerical stability
    P_agg = P_agg.clamp_min(0.0)
    P_agg = P_agg / P_agg.sum(dim=-1, keepdim=True).clamp_min(eps)

    # === (4) Aggregate initial distribution ===
    gamma_agg = torch.einsum("bs,bsm->bm", gamma, W)
    gamma_agg = gamma_agg.clamp_min(eps)
    gamma_agg = gamma_agg / gamma_agg.sum(dim=1, keepdim=True).clamp_min(eps)

    return P_agg, r_agg, gamma_agg


def galerkin_lp_coefficients(
    P: torch.Tensor,
    r: torch.Tensor,
    gamma: torch.Tensor,
    W: torch.Tensor,
    beta: float,
):
    """Exact Galerkin projected-LP coefficients for a given membership W.

    LP-2 with the Eq.(6) soft-aggregation features, no diagonal projection:

        max  r_tilde^T w   s.t.   A w = b,  w >= 0

        A[b, i, (m,a)]   = (W^T W)[b,i,m] - beta * sum_{s,s'} W[b,s',i] P[b,s,a,s'] W[b,s,m]
        r_tilde[b,(m,a)] = sum_s W[b,s,m] r[b,s,a]          (= mass_m * r_agg, NOT normalized)
        b[b,i]           = sum_s W[b,s,i] gamma[b,s]        (unnormalized W^T gamma)

    Column flattening is (m,a) -> m*A + a, matching MDP.build_H / build_r.
    For hard (one-hot) W this LP is the aggregated-MDP LP up to the variable
    scaling y_agg(m,a) = mass_m * w_(m,a); the lift y(s,a) = sum_m W_sm w_(m,a)
    coincides with the mass-corrected lift of the aggregated occupancy.
    """
    B, S, A, _ = P.shape
    M = W.shape[2]

    # B-block: full Gram, gram[b,i,m] = sum_s W[b,s,i] W[b,s,m]  (indep. of a)
    gram = torch.einsum("bsi,bsm->bim", W, W)  # (B,M,M)

    # P-block: pblk[b,i,m,a] = sum_{s,s'} W[b,s',i] P[b,s,a,s'] W[b,s,m]
    #   (s = source state, s' = destination state)
    right = torch.einsum("bsaj,bji->bsai", P, W)     # (B,S,A,M): dest side
    pblk = torch.einsum("bsm,bsai->bima", W, right)  # (B,M,M,A)

    A_mat = gram.unsqueeze(-1) - float(beta) * pblk  # (B,M,M,A)
    A_mat = A_mat.reshape(B, M, M * A)               # col (m,a) -> m*A + a

    r_tilde = torch.einsum("bsm,bsa->bma", W, r).reshape(B, M * A)  # (B, M*A)
    b_vec = torch.einsum("bsi,bs->bi", W, gamma)                    # (B, M)

    return A_mat, r_tilde, b_vec