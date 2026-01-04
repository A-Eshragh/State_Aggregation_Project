import torch
import torch.autograd as autograd
from .dense_lp_solver import solve_lp_gurobi


class DifferentiableLPSolver(autograd.Function):
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
        """
        with torch.no_grad():
            y_star, basis = solve_lp_gurobi(H, r, gamma)
            
        ctx.save_for_backward(H, r, gamma, y_star, basis)

        return y_star

    @staticmethod
    def backward(ctx, grad_y):
        """
        Backward pass using basis sensitivity.

        grad_y: ∂L/∂y*

        Returns:
            ∂L/∂H, ∂L/∂r, ∂L/∂gamma
        """
        H, r, gamma, y_star, basis = ctx.saved_tensors
        batch, S, SA = H.shape

        device = H.device
        dtype = H.dtype

        grad_H = torch.zeros_like(H)
        grad_r = torch.zeros_like(r)
        grad_gamma = torch.zeros_like(gamma)

        grad_r = y_star * grad_y
        for b in range(batch):
            B = basis[b]                 # (S,)
            H_B = H[b, :, B]             # (S, S)
            r_B = r[b, B]                # (S,)
            y_B = y_star[b, B]           # (S,)

            x = torch.linalg.solve(H_B.T, r_B)
            grad_gamma[b] = x
            grad_H_B = -torch.outer(x, y_B)
            for i, col_idx in enumerate(B):
                grad_H[b, :, col_idx] = grad_H_B[:, i]

        return grad_H, grad_r, grad_gamma


# Convenience wrapper
def differentiable_lp(H, r, gamma):
    """
    User-facing differentiable LP call.
    """
    return DifferentiableLPSolver.apply(H, r, gamma)
