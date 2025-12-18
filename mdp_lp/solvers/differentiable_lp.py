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

        # Save everything needed for backward
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

        # Initialize gradients
        grad_H = torch.zeros_like(H)
        grad_r = torch.zeros_like(r)
        grad_gamma = torch.zeros_like(gamma)

        # Gradient w.r.t. r is always valid:
        # v = r^T y*  =>  ∂v/∂r = y*
        grad_r = y_star * grad_y

        # Basis-restricted gradients
        for b in range(batch):
            B = basis[b]                 # (S,)
            H_B = H[b, :, B]             # (S, S)
            r_B = r[b, B]                # (S,)
            y_B = y_star[b, B]           # (S,)

            # Solve H_B^T x = r_B   =>   x = H_B^{-T} r_B
            # Never explicitly invert
            x = torch.linalg.solve(H_B.T, r_B)

            # ∂v/∂gamma = H_B^{-T} r_B
            grad_gamma[b] = x

            # ∂v/∂H_B = - (H_B^{-T} r_B) y_B^T
            grad_H_B = -torch.outer(x, y_B)

            # Scatter basis gradients into full H
            for i, col_idx in enumerate(B):
                grad_H[b, :, col_idx] = grad_H_B[:, i]

        return grad_H, grad_r, grad_gamma


# Convenience wrapper
def differentiable_lp(H, r, gamma):
    """
    User-facing differentiable LP call.
    """
    return DifferentiableLPSolver.apply(H, r, gamma)
