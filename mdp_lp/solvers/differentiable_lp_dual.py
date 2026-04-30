import torch
import torch.autograd as autograd
from .dense_lp_solver import solve_lp_gurobi


class DifferentiableLPSolverDual(autograd.Function):
    """
    Differentiable LP layer via basis-restricted sensitivity analysis.

    Solves:
        maximize     r^T y
        subject to   H y = gamma
                     y >= 0

    Forward:
        - Uses Gurobi to solve the LP
        - Extracts the optimal basis B

    Backward:
        - Assumes fixed, non-degenerate basis
        - Differentiates y_B = H_B^{-1} gamma
        - Implements standard LP sensitivity formulas
    """

    @staticmethod
    def forward(ctx, H, r, gamma):
        """
        Forward LP solve.

        Inputs:
            H:     (batch, S, SA)
            r:     (batch, SA)
            gamma: (batch, S)

        Returns:
            y_star: (batch, SA)
            dual: (batch, S)
        """
        with torch.no_grad():
            y_star, basis, dual = solve_lp_gurobi(H, r, gamma)
            
        ctx.save_for_backward(H, r, gamma, y_star, basis, dual)

        return y_star, dual

    @staticmethod
    def backward(ctx, grad_y, grad_dual):
        """
        Backward pass using basis sensitivity.

        grad_y: ∂L/∂y*

        Returns:
            ∂L/∂H, ∂L/∂r, ∂L/∂gamma
        """
        H, r, gamma, y_star, basis, dual = ctx.saved_tensors
        batch, S, SA = H.shape

        # We only use r_true in the loss, so direct term is 0; no gradient through r.
        grad_r = torch.zeros_like(r)

        # Batched basis-sensitivity: avoids the per-batch / per-column Python
        # loops that were the dominant CPU cost for the DFL backward.
        idx = basis.unsqueeze(1).expand(-1, S, -1)               # (batch, S, S)
        H_B = torch.gather(H, 2, idx)                            # (batch, S, S)
        y_B = torch.gather(y_star, 1, basis)                     # (batch, S)
        grad_y_B = torch.gather(grad_y, 1, basis)                # (batch, S)

        x = torch.linalg.solve(
            H_B.transpose(1, 2), grad_y_B.unsqueeze(-1)
        ).squeeze(-1)                                            # (batch, S)
        grad_gamma = x

        grad_H_B = -x.unsqueeze(2) * y_B.unsqueeze(1)            # (batch, S, S)
        grad_H = torch.zeros_like(H)
        grad_H.scatter_(2, idx, grad_H_B)

        return grad_H, grad_r, grad_gamma


# Convenience wrapper
def differentiable_lp(H, r, gamma):
    """
    User-facing differentiable LP call.
    """
    return DifferentiableLPSolverDual.apply(H, r, gamma)
