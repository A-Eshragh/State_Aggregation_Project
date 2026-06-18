"""
Toy comparison: np.gradient (numerical finite differences on sampled V(λ))
vs the ground-truth analytical gradient from LP sensitivity analysis:

    dy_B/dθ = -H_B^{-1} (∂H_B/∂θ · y_B)

    dV/dθ = (∂r/∂θ)^T y* + r_B^T · dy_B/dθ

We obtain the analytical gradient via PyTorch autograd through the
DifferentiableLPSolver, which implements exactly this basis-sensitivity
formula in its backward pass. This serves as ground truth.

We then compare np.gradient (2nd-order central finite differences on the
sampled V(λ) curve) against it.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import matplotlib.pyplot as plt

from mdp_lp.inventory_elasticity import create_inventory_from_lambda
from mdp_lp.solvers.differentiable_lp import differentiable_lp


def compute_V_and_dV_autograd(lambda_val, b_max, s_max, q_max,
                               foc, voc, hc, bc, beta, selling_price):
    """
    Compute V(λ) and dV/dλ using autograd through the differentiable LP solver.
    The backward pass implements: dy_B/dθ = -H_B^{-1} (∂H_B/∂θ · y_B).
    """
    lam_t = torch.tensor([lambda_val], dtype=torch.float32, requires_grad=True)
    p_t = torch.tensor([selling_price], dtype=torch.float32)

    mdp = create_inventory_from_lambda(lam_t, b_max, s_max, q_max,
                                        foc, voc, hc, bc, beta, p_t)
    P = mdp.P
    r = mdp.r
    H = mdp.build_H(P)
    r_flat = mdp.build_r(r)
    S = H.shape[1]
    gamma = torch.ones(1, S, dtype=torch.float32) / S

    y_star = differentiable_lp(H, r_flat, gamma)
    V = (r_flat[0] * y_star[0]).sum()
    V.backward()

    return V.item(), lam_t.grad.item()


def compute_V(lambda_val, b_max, s_max, q_max,
              foc, voc, hc, bc, beta, selling_price):
    """Compute V(λ) only (no grad)."""
    lam = torch.tensor([lambda_val], dtype=torch.float32)
    p = torch.tensor([selling_price], dtype=torch.float32)

    mdp = create_inventory_from_lambda(lam, b_max, s_max, q_max,
                                        foc, voc, hc, bc, beta, p)
    P = mdp.P
    r = mdp.r
    H = mdp.build_H(P)
    r_flat = mdp.build_r(r)
    S = H.shape[1]
    gamma = torch.ones(1, S, dtype=torch.float32) / S

    from mdp_lp.solvers.dense_lp_solver import solve_lp_gurobi
    y_star, _, _ = solve_lp_gurobi(H, r_flat, gamma)
    return (r_flat[0] @ y_star[0]).item()


def main():
    # --- MDP parameters (small for toy example) ---
    b_max = 3
    s_max = 5
    q_max = 5
    foc = 0.0     # fixed ordering cost
    voc = 1.0     # variable ordering cost
    hc = 0.5      # holding cost
    bc = 0.0      # backlog cost
    beta = 0.9    # discount factor
    selling_price = 5.0

    # --- Range of λ values ---
    N_fine = 200    # fine grid for np.gradient
    N_coarse = 50   # coarse grid for np.gradient
    lambda_fine = np.linspace(1.0, 8.0, N_fine)
    lambda_coarse = np.linspace(1.0, 8.0, N_coarse)

    # --- 1. Compute analytical dV/dλ (autograd = ground truth) on fine grid ---
    print(f"Computing analytical dV/dλ via autograd on {N_fine} points...")
    V_fine = np.zeros(N_fine)
    dV_analytical = np.zeros(N_fine)
    for i, lam in enumerate(lambda_fine):
        V_fine[i], dV_analytical[i] = compute_V_and_dV_autograd(
            lam, b_max, s_max, q_max, foc, voc, hc, bc, beta, selling_price)

    # --- 2. np.gradient on fine grid ---
    dV_npgrad_fine = np.gradient(V_fine, lambda_fine)

    # --- 3. np.gradient on coarse grid ---
    print(f"Computing V(λ) on coarse grid ({N_coarse} points)...")
    V_coarse = np.array([
        compute_V(lam, b_max, s_max, q_max, foc, voc, hc, bc, beta, selling_price)
        for lam in lambda_coarse
    ])
    dV_npgrad_coarse = np.gradient(V_coarse, lambda_coarse)

    # --- 4. Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) V(λ)
    ax = axes[0, 0]
    ax.plot(lambda_fine, V_fine, 'b-', lw=2)
    ax.set_xlabel('λ (Poisson rate)')
    ax.set_ylabel('V(λ)')
    ax.set_title('Optimal LP Value V(λ)')
    ax.grid(True, alpha=0.3)

    # (b) dV/dλ comparison
    ax = axes[0, 1]
    ax.plot(lambda_fine, dV_analytical, 'r-', lw=2,
            label='Analytical (Eq. 3, via autograd)')
    ax.plot(lambda_fine, dV_npgrad_fine, 'b--', lw=1.5,
            label=f'np.gradient (N={N_fine})')
    ax.plot(lambda_coarse, dV_npgrad_coarse, 'g:', lw=2,
            label=f'np.gradient (N={N_coarse})')
    ax.set_xlabel('λ')
    ax.set_ylabel('dV/dλ')
    ax.set_title('Gradient Comparison: Analytical vs np.gradient')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (c) Absolute error
    ax = axes[1, 0]
    err_fine = np.abs(dV_npgrad_fine - dV_analytical)
    ax.plot(lambda_fine, err_fine, 'b-', lw=1.5,
            label=f'np.gradient (N={N_fine})')
    # Interpolate coarse error onto fine grid for comparison
    from scipy.interpolate import interp1d
    dV_analytical_at_coarse = interp1d(lambda_fine, dV_analytical, kind='linear')(lambda_coarse)
    err_coarse = np.abs(dV_npgrad_coarse - dV_analytical_at_coarse)
    ax.plot(lambda_coarse, err_coarse, 'go', ms=4,
            label=f'np.gradient (N={N_coarse})')
    ax.set_xlabel('λ')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Absolute Error vs Analytical (Eq. 3)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (d) Scatter: np.gradient vs analytical
    ax = axes[1, 1]
    ax.scatter(dV_analytical, dV_npgrad_fine, s=10, alpha=0.5, label=f'np.gradient (N={N_fine})')
    ax.scatter(dV_analytical_at_coarse, dV_npgrad_coarse, s=30, alpha=0.7,
               marker='^', label=f'np.gradient (N={N_coarse})')
    lims = [min(dV_analytical.min(), dV_npgrad_fine.min()),
            max(dV_analytical.max(), dV_npgrad_fine.max())]
    ax.plot(lims, lims, 'r--', lw=1, label='y=x (perfect)')
    ax.set_xlabel('Analytical dV/dλ')
    ax.set_ylabel('np.gradient dV/dλ')
    ax.set_title('np.gradient vs Analytical')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), 'plot_results')
    plt.savefig(os.path.join(out_dir, 'gradient_verification.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, 'gradient_verification.pdf'), bbox_inches='tight')
    print("Saved plots to mdp_lp/plot_results/gradient_verification.{png,pdf}")

    # --- 6. Summary statistics ---
    print("\n" + "=" * 70)
    print("SUMMARY: Error of numerical methods vs analytical (Eq. 3)")
    print("=" * 70)
    print(f"{'Metric':<45} {'np.grad fine':>12} {'np.grad coarse':>14}")
    print("-" * 71)
    print(f"{'Mean abs error':<45} {err_fine.mean():>12.4f} {err_coarse.mean():>14.4f}")
    print(f"{'Max abs error':<45} {err_fine.max():>12.4f} {err_coarse.max():>14.4f}")
    rel_fine = err_fine / (np.abs(dV_analytical) + 1e-10)
    rel_coarse = err_coarse / (np.abs(dV_analytical_at_coarse) + 1e-10)
    print(f"{'Mean rel error':<45} {rel_fine.mean():>12.4f} {rel_coarse.mean():>14.4f}")
    print(f"{'Max rel error':<45} {rel_fine.max():>12.4f} {rel_coarse.max():>14.4f}")
    corr_fine = np.corrcoef(dV_npgrad_fine, dV_analytical)[0, 1]
    corr_coarse = np.corrcoef(dV_npgrad_coarse, dV_analytical_at_coarse)[0, 1]
    print(f"{'Pearson correlation':<45} {corr_fine:>12.6f} {corr_coarse:>14.6f}")

    print("\n--- Interpretation ---")
    print("The analytical gradient (Eq. 3) uses basis sensitivity analysis:")
    print("  dy_B/dθ = -H_B^{-1} (∂H_B/∂θ · y_B)")
    print("This gives the exact derivative at each λ assuming the basis is fixed.")
    print("")
    print("np.gradient uses central finite differences on a discrete V(λ) grid.")
    print("Key issues:")
    print("  1. V(λ) is piecewise linear — basis changes create kinks.")
    print("  2. At kinks, np.gradient averages left/right slopes (wrong).")
    print("  3. On coarse grids, kinks are missed entirely.")
    print("  4. Between kinks, np.gradient agrees well with the analytical gradient.")


if __name__ == '__main__':
    main()
