"""
Plot the DFL loss L(λ) = -r_true^T y*(λ_pred) to show:
1. It is piecewise-linear (one value per λ, kinks at basis changes)
2. Contrast: few basis changes (easy) vs many (hard)

Usage:
    python -m mdp_lp.plot_piecewise_linear
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mdp_lp.inventory_elasticity import create_inventory_from_lambda
from mdp_lp.solvers.dense_lp_solver import DenseLPSolver

# ── MDP parameters ──
B_MAX, S_MAX, Q_MAX, D_MAX = 5, 50, 30, 30
FOC, VOC, HC, BC, BETA, PRICE = 0, 0.5, 0.1, 0, 0.95, 10.0
LAMBDA_TRUE = 10.0


def solve_lp_for_lambda(lam_val):
    lam = torch.tensor([lam_val], dtype=torch.float32)
    p = torch.tensor([PRICE], dtype=torch.float32)
    mdp = create_inventory_from_lambda(lam, B_MAX, S_MAX, Q_MAX, D_MAX,
                                       FOC, VOC, HC, BC, BETA, p)
    S = mdp.P.shape[1]
    gamma = mdp.build_gamma(1, S, torch.device("cpu"))
    gamma = torch.clamp(gamma, min=1e-8)
    gamma = gamma / gamma.sum(dim=1, keepdim=True)
    H = mdp.build_H(mdp.P)
    r_flat = mdp.r.reshape(1, -1)
    solver = DenseLPSolver()
    y_star, basis, dual = solver.solve(H, r_flat, gamma)
    return y_star, r_flat, basis


def main():
    # Build true MDP reward
    lam_true_t = torch.tensor([LAMBDA_TRUE], dtype=torch.float32)
    p_t = torch.tensor([PRICE], dtype=torch.float32)
    mdp_true = create_inventory_from_lambda(lam_true_t, B_MAX, S_MAX, Q_MAX, D_MAX,
                                             FOC, VOC, HC, BC, BETA, p_t)
    r_true = mdp_true.r.reshape(1, -1)

    # Sweep λ_pred
    lambda_values = np.linspace(0.5, 50, 500)
    L_values = []
    basis_list = []

    print("Computing L(λ) across λ range...")
    for i, lam in enumerate(lambda_values):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(lambda_values)}")
        y_star, _, basis = solve_lp_for_lambda(lam)
        v = (r_true * y_star).sum().item()
        L_values.append(-v)  # DFL loss = -r_true^T y*
        basis_list.append(tuple(basis[0].tolist()))

    L_values = np.array(L_values)

    # Detect basis changes
    basis_change_idx = [0]
    for i in range(1, len(basis_list)):
        if basis_list[i] != basis_list[i - 1]:
            basis_change_idx.append(i)
    n_basis = len(set(basis_list))
    print(f"Found {n_basis} distinct bases, {len(basis_change_idx)-1} basis changes")

    # ── Figure: full landscape with basis changes marked ──
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

    # Top: L(λ) with basis regions colored
    ax = axes[0]
    # Color each basis region differently
    unique_bases = list(dict.fromkeys(basis_list))  # preserve order
    basis_to_color_idx = {b: i for i, b in enumerate(unique_bases)}
    cmap = plt.cm.tab20

    # Plot segments between basis changes
    change_lams = [lambda_values[i] for i in basis_change_idx] + [lambda_values[-1]]
    for j in range(len(basis_change_idx)):
        start = basis_change_idx[j]
        end = basis_change_idx[j + 1] if j + 1 < len(basis_change_idx) else len(lambda_values)
        color_idx = basis_to_color_idx[basis_list[start]] % 20
        ax.plot(lambda_values[start:end], L_values[start:end],
                color=cmap(color_idx), linewidth=1.5, alpha=0.8)

    # Mark basis changes with vertical lines
    for idx in basis_change_idx[1:]:
        ax.axvline(lambda_values[idx], color='gray', alpha=0.3, linewidth=0.5)

    # Mark λ_true
    ax.axvline(LAMBDA_TRUE, color='red', linestyle='--', linewidth=2, label=f'λ_true = {LAMBDA_TRUE}')

    # Mark the L value at λ_true
    idx_true = np.argmin(np.abs(lambda_values - LAMBDA_TRUE))
    ax.plot(LAMBDA_TRUE, L_values[idx_true], 'r*', markersize=15, zorder=5)

    ax.set_ylabel('DFL Loss  $L(\\lambda) = -r_{true}^T \\, y^*(\\lambda)$', fontsize=13)
    ax.set_title(f'DFL Loss Landscape: Piecewise-Linear with {n_basis} Basis Regions\n'
                 f'(each color = one LP basis; gray lines = basis changes)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Bottom: gradient (finite differences)
    ax2 = axes[1]
    dL_dlam = np.gradient(L_values, lambda_values)

    # Color gradient by basis region too
    for j in range(len(basis_change_idx)):
        start = basis_change_idx[j]
        end = basis_change_idx[j + 1] if j + 1 < len(basis_change_idx) else len(lambda_values)
        color_idx = basis_to_color_idx[basis_list[start]] % 20
        ax2.plot(lambda_values[start:end], dL_dlam[start:end],
                 color=cmap(color_idx), linewidth=1.0, alpha=0.8)

    ax2.axvline(LAMBDA_TRUE, color='red', linestyle='--', linewidth=2)
    ax2.axhline(0, color='black', linewidth=0.5)
    for idx in basis_change_idx[1:]:
        ax2.axvline(lambda_values[idx], color='gray', alpha=0.3, linewidth=0.5)

    ax2.set_xlabel('$\\lambda_{pred}$', fontsize=13)
    ax2.set_ylabel('$\\partial L / \\partial \\lambda$\n(finite diff)', fontsize=11)
    ax2.set_title('Gradient: piecewise-constant within each basis, jumps at basis changes', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mdp_lp/plot_results/piecewise_linear_landscape.png', dpi=150, bbox_inches='tight')
    plt.savefig('mdp_lp/plot_results/piecewise_linear_landscape.pdf', bbox_inches='tight')
    print("Saved to mdp_lp/plot_results/piecewise_linear_landscape.{png,pdf}")

    # ── Figure 2: Zoom in to show linearity within a single piece ──
    fig2, ax3 = plt.subplots(1, 1, figsize=(10, 5))

    # Find a basis region near λ_true that's wide enough to see
    # Pick the region containing λ_true
    region_starts = basis_change_idx + [len(lambda_values)]
    for j in range(len(basis_change_idx)):
        start = basis_change_idx[j]
        end = region_starts[j + 1] if j + 1 < len(region_starts) else len(lambda_values)
        if start < idx_true < end:
            # Found the region containing λ_true
            # Expand a bit to show neighboring regions
            plot_start = max(0, basis_change_idx[max(0, j - 3)])
            plot_end_j = min(len(basis_change_idx) - 1, j + 4)
            plot_end = region_starts[plot_end_j + 1] if plot_end_j + 1 < len(region_starts) else len(lambda_values)

            for k in range(max(0, j - 3), min(len(basis_change_idx), j + 5)):
                s = basis_change_idx[k]
                e = region_starts[k + 1] if k + 1 < len(region_starts) else len(lambda_values)
                if s >= plot_start and e <= plot_end + 1:
                    color_idx = basis_to_color_idx[basis_list[s]] % 20
                    ax3.plot(lambda_values[s:e], L_values[s:e], 'o-',
                             color=cmap(color_idx), linewidth=2, markersize=2, alpha=0.8,
                             label=f'Basis {k}')
                    # Mark basis boundaries
                    if k > max(0, j - 3):
                        ax3.axvline(lambda_values[s], color='gray', linestyle=':', alpha=0.5)
            break

    ax3.axvline(LAMBDA_TRUE, color='red', linestyle='--', linewidth=2, label=f'λ_true = {LAMBDA_TRUE}')
    ax3.plot(LAMBDA_TRUE, L_values[idx_true], 'r*', markersize=15, zorder=5)
    ax3.set_xlabel('$\\lambda_{pred}$', fontsize=13)
    ax3.set_ylabel('$L(\\lambda) = -r_{true}^T y^*(\\lambda)$', fontsize=13)
    ax3.set_title('Zoomed: Each basis region is approximately linear\n'
                  '(slope = constant gradient within that basis)', fontsize=13)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mdp_lp/plot_results/piecewise_linear_zoomed.png', dpi=150, bbox_inches='tight')
    plt.savefig('mdp_lp/plot_results/piecewise_linear_zoomed.pdf', bbox_inches='tight')
    print("Saved to mdp_lp/plot_results/piecewise_linear_zoomed.{png,pdf}")


if __name__ == "__main__":
    main()
