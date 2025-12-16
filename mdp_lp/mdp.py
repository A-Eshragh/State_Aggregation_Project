import torch

class MDP:
    """
    Batch discounted MDP → LP formulation builder.

    LP:
        maximize     r^T y
        subject to   H y = γ
                     y ≥ 0

    y ∈ R^{S*A} is the occupancy measure.
    """

    def __init__(self, beta, gamma=None):
        self.beta = beta
        self.gamma = gamma

    # ---------------------------------------------------------------
    # H MATRIX
    # ---------------------------------------------------------------
    def build_H(self, P):
        """
        P[b, s, a, s'] = Pr(s' | s, a)
        H[b, row=s', col=(s,a)] = δ[s'=s] - beta * P[s,a,s'].
        """
        batch, S, A, _ = P.shape
        device = P.device
        dtype = P.dtype

        H = torch.zeros(batch, S, S*A, device=device, dtype=dtype)

        # Identity part
        for s in range(S):
            H[:, s, s*A:(s+1)*A] = 1.0

        # Transition subtraction
        for s in range(S):
            for a in range(A):
                col = s*A + a
                H[:, :, col] -= self.beta * P[:, s, a, :]

        return H

    # ---------------------------------------------------------------
    # REWARD
    # ---------------------------------------------------------------
    def build_r(self, r):
        """
        r[b,s,a] → r[b, s*A + a]
        """
        batch = r.size(0)
        return r.reshape(batch, -1)

    # ---------------------------------------------------------------
    # GAMMA
    # ---------------------------------------------------------------
    def build_gamma(self, batch, S, device):
        if self.gamma is None:
            return torch.ones(batch, S, device=device) / S
        if isinstance(self.gamma, torch.Tensor) and self.gamma.dim() == 1:
            return self.gamma.unsqueeze(0).expand(batch, -1).to(device)
        return self.gamma.to(device)

    # ---------------------------------------------------------------
    # WRAPPER
    # ---------------------------------------------------------------
    def lp_formulation(self, P, r):
        batch, S, A, _ = P.shape
        device = P.device

        return (
            self.build_H(P),
            self.build_r(r),
            self.build_gamma(batch, S, device)
        )
