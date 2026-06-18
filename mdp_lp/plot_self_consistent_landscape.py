"""
Plot r_true^T y*(lambda) vs r_theta^T y*(lambda) to show whether using the
predicted reward vector eliminates the discontinuities in the DFL loss landscape.

Key insight: -r_theta^T y*(lambda) is the LP optimal value, which by LP sensitivity
theory is continuous (and concave) in the LP parameters. So basis changes still
exist, but the objective value does NOT jump at them.

Usage:
    python -m mdp_lp.plot_self_consistent_landscape
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mdp_lp.inventory_elasticity import create_inventory_from_lambda
from mdp_lp.solvers.dense_lp_solver import DenseLPSolver

# ── MDP parameters (same as piecewise_linear plot) ──
B_MAX, S_MAX, Q_MAX, D_MAX = 5, 50, 30, 30
FOC, VOC, HC, BC, BETA, PRICE = 0, 0.5, 0.1, 0, 0.95, 10.0
LAMBDA_TRUE = 10.0

N_STATES = B_MAX + S_MAX + 1   # 56
N_ACTIONS = Q_MAX + 1           # 31


def solve_for_lambda(lam_val, solver):
    """Build MDP from lambda, solve LP.

    Returns:
        y_star: (1, S*A) occupancy
        r_pred: (1, S*A) predicted reward vector
        basis:  basis indices
    """
    lam = torch.tensor([lam_val], dtype=torch.float32)
    p = torch.tensor([PRICE], dtype=torch.float32)
    mdp = create_inventory_from_lambda(lam, B_MAX, S_MAX, Q_MAX,
                                       FOC, VOC, HC, BC, BETA, p)
    S = mdp.P.shape[1]
    gamma = mdp.build_gamma(1, S, torch.device("cpu"))
    gamma = torch.clamp(gamma, min=1e-8)
    gamma = gamma / gamma.sum(dim=1, keepdim=True)

    H = mdp.build_H(mdp.P)
    r_flat = mdp.r.reshape(1, -1)
    y_star, basis, _ = solver.solve(H, r_flat, gamma)
    return y_star, r_flat, basis


def main():
    # Build true MDP reward
    lam_true_t = torch.tensor([LAMBDA_TRUE], dtype=torch.float32)
    p_t = torch.tensor([PRICE], dtype=torch.float32)
    mdp_true = create_inventory_from_lambda(lam_true_t, B_MAX, S_MAX, Q_MAX,
                                             FOC, VOC, HC, BC, BETA, p_t)
    r_true = mdp_true.r.reshape(1, -1)

    solver = DenseLPSolver()

    # Sweep lambda_pred
    lambda_values = np.linspace(0.5, 50, 500)
    L_rtrue = []       # -r_true^T y*(lambda)
    L_rtheta = []      # -r_theta^T y*(lambda)
    basis_list = []

    print("Computing landscapes...")
    for i, lam in enumerate(lambda_values):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(lambda_values)}")
        y, r_pred, basis = solve_for_lambda(lam, solver)
        L_rtrue.append(-(r_true * y).sum().item())
        L_rtheta.append(-(r_pred * y).sum().item())
        basis_list.append(tuple(basis[0].tolist()))

    L_rtrue = np.array(L_rtrue)
    L_rtheta = np.array(L_rtheta)

    # Detect basis changes
    basis_changes = [0]
    for i in range(1, len(basis_list)):
        if basis_list[i] != basis_list[i - 1]:
            basis_changes.append(i)
    unique_bases = list(dict.fromkeys(basis_list))
    basis_to_color = {b: i for i, b in enumerate(unique_bases)}
    n_basis = len(unique_bases)
    print(f"Found {n_basis} distinct bases, {len(basis_changes)-1} basis changes")

    idx_true = np.argmin(np.abs(lambda_values - LAMBDA_TRUE))
    cmap = plt.cm.tab20

    # ── Figure: 2x2 comparison ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # --- Top-left: r_true^T y* (standard DFL loss) ---
    ax = axes[0, 0]
    for j in range(len(basis_changes)):
        start = basis_changes[j]
        end = basis_changes[j + 1] if j + 1 < len(basis_changes) else len(lambda_values)
        cidx = basis_to_color[basis_list[start]] % 20
        ax.plot(lambda_values[start:end], L_rtrue[start:end],
                color=cmap(cidx), linewidth=1.5, alpha=0.8, marker='.', markersize=3)
    ax.axvline(LAMBDA_TRUE, color='red', linestyle='--', linewidth=2)
    ax.plot(LAMBDA_TRUE, L_rtrue[idx_true], 'r*', markersize=12, zorder=5)
    ax.set_ylabel('$-r_{true}^T \\, y^*(\\lambda)$', fontsize=13)
    ax.set_title(f'Standard DFL: $-r_{{true}}^T y^*(\\lambda)$\n'
                 f'({n_basis} basis regions, piecewise-linear with kinks)', fontsize=12)
    ax.grid(True, alpha=0.3)

    # --- Top-right: r_theta^T y* (self-consistent) ---
    ax = axes[0, 1]
    for j in range(len(basis_changes)):
        start = basis_changes[j]
        end = basis_changes[j + 1] if j + 1 < len(basis_changes) else len(lambda_values)
        cidx = basis_to_color[basis_list[start]] % 20
        ax.plot(lambda_values[start:end], L_rtheta[start:end],
                color=cmap(cidx), linewidth=1.5, alpha=0.8, marker='.', markersize=3)
    ax.axvline(LAMBDA_TRUE, color='red', linestyle='--', linewidth=2)
    ax.plot(LAMBDA_TRUE, L_rtheta[idx_true], 'r*', markersize=12, zorder=5)
    ax.set_ylabel('$-r_\\theta^T \\, y^*(\\lambda)$', fontsize=13)
    ax.set_title(f'Self-consistent: $-r_\\theta^T y^*(\\lambda)$\n'
                 f'(LP optimal value — continuous by LP theory)', fontsize=12)
    ax.grid(True, alpha=0.3)

    # --- Bottom-left: gradient of r_true^T y* ---
    ax = axes[1, 0]
    dL_rtrue = np.gradient(L_rtrue, lambda_values)
    for j in range(len(basis_changes)):
        start = basis_changes[j]
        end = basis_changes[j + 1] if j + 1 < len(basis_changes) else len(lambda_values)
        cidx = basis_to_color[basis_list[start]] % 20
        ax.plot(lambda_values[start:end], dL_rtrue[start:end],
                color=cmap(cidx), linewidth=1.0, alpha=0.8, marker='.', markersize=3)
    ax.axvline(LAMBDA_TRUE, color='red', linestyle='--', linewidth=2)
    ax.axhline(0, color='black', linewidth=0.5)
    for idx in basis_changes[1:]:
        ax.axvline(lambda_values[idx], color='gray', alpha=0.2, linewidth=0.5)
    ax.set_xlabel('$\\lambda_{pred}$', fontsize=13)
    ax.set_ylabel('$\\partial / \\partial \\lambda$', fontsize=11)
    ax.set_title('Gradient of $-r_{true}^T y^*$: jumps at basis changes', fontsize=12)
    ax.grid(True, alpha=0.3)

    # --- Bottom-right: gradient of r_theta^T y* ---
    ax = axes[1, 1]
    dL_rtheta = np.gradient(L_rtheta, lambda_values)
    for j in range(len(basis_changes)):
        start = basis_changes[j]
        end = basis_changes[j + 1] if j + 1 < len(basis_changes) else len(lambda_values)
        cidx = basis_to_color[basis_list[start]] % 20
        ax.plot(lambda_values[start:end], dL_rtheta[start:end],
                color=cmap(cidx), linewidth=1.0, alpha=0.8, marker='.', markersize=3)
    ax.axvline(LAMBDA_TRUE, color='red', linestyle='--', linewidth=2)
    ax.axhline(0, color='black', linewidth=0.5)
    for idx in basis_changes[1:]:
        ax.axvline(lambda_values[idx], color='gray', alpha=0.2, linewidth=0.5)
    ax.set_xlabel('$\\lambda_{pred}$', fontsize=13)
    ax.set_ylabel('$\\partial / \\partial \\lambda$', fontsize=11)
    ax.set_title('Gradient of $-r_\\theta^T y^*$: smoother?', fontsize=12)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        'Does using $r_\\theta$ instead of $r_{true}$ eliminate discontinuities?\n'
        '$-r_\\theta^T y^*(\\lambda)$ is the LP optimal value — continuous in LP parameters',
        fontsize=14, y=1.02
    )
    plt.tight_layout()
    plt.savefig('mdp_lp/plot_results/self_consistent_landscape.png', dpi=150, bbox_inches='tight')
    plt.savefig('mdp_lp/plot_results/self_consistent_landscape.pdf', bbox_inches='tight')
    print("Saved to mdp_lp/plot_results/self_consistent_landscape.{png,pdf}")


if __name__ == "__main__":
    main()
