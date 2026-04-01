"""
Plot both DFL methods with penalty side-by-side:
  Standard DFL + penalty:     -r_true^T y*(λ) + ρ ||γ_true - H_true y*(λ)||²
  Self-consistent + penalty:  -r_θ^T y*(λ)    + ρ ||γ_true - H_true y*(λ)||²

Shows how the penalty term affects each method's landscape differently:
- Standard DFL: shifted minimum (linear + quadratic)
- Self-consistent: penalty helps center the minimum at λ_true

Usage:
    python -m mdp_lp.plot_both_methods_with_penalty
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


def solve_for_lambda(lam_val, solver):
    """Build MDP from lambda, solve LP. Return y*, r_pred, H_pred, gamma_pred, basis."""
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
    return y_star, r_flat, H, gamma, basis


def main():
    # Build true MDP
    lam_true_t = torch.tensor([LAMBDA_TRUE], dtype=torch.float32)
    p_t = torch.tensor([PRICE], dtype=torch.float32)
    mdp_true = create_inventory_from_lambda(lam_true_t, B_MAX, S_MAX, Q_MAX,
                                             FOC, VOC, HC, BC, BETA, p_t)
    r_true = mdp_true.r.reshape(1, -1)
    H_true = mdp_true.build_H(mdp_true.P)
    S = mdp_true.P.shape[1]
    gamma_true = mdp_true.build_gamma(1, S, torch.device("cpu"))
    gamma_true = torch.clamp(gamma_true, min=1e-8)
    gamma_true = gamma_true / gamma_true.sum(dim=1, keepdim=True)

    solver = DenseLPSolver()

    # Sweep lambda_pred
    lambda_values = np.sort(np.concatenate([
        np.linspace(0.5, 50, 400),
        np.linspace(LAMBDA_TRUE - 5, LAMBDA_TRUE + 5, 200),
    ]))

    L_rtrue = []       # -r_true^T y*(λ)
    L_rtheta = []      # -r_θ^T y*(λ)
    penalty_vals = []   # ||γ_true - H_true y*(λ)||²
    basis_list = []

    print("Computing landscapes...")
    for i, lam in enumerate(lambda_values):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(lambda_values)}")
        y, r_pred, H_pred, gamma_pred, basis = solve_for_lambda(lam, solver)

        # Standard DFL objective
        L_rtrue.append(-(r_true * y).sum().item())
        # Self-consistent objective
        L_rtheta.append(-(r_pred * y).sum().item())
        # Penalty: violation of TRUE constraints
        violation = gamma_true - (H_true @ y.unsqueeze(-1)).squeeze(-1)
        penalty_vals.append(violation.pow(2).sum().item())
        basis_list.append(tuple(basis[0].tolist()))

    L_rtrue = np.array(L_rtrue)
    L_rtheta = np.array(L_rtheta)
    penalty_vals = np.array(penalty_vals)

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

    # ── Figure: 2x3 — top row: loss landscapes, bottom row: gradients ──
    rho = 10000.0
    L_dfl_pen = L_rtrue + rho * penalty_vals
    L_sc_pen = L_rtheta + rho * penalty_vals

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    # --- Top-left: Standard DFL + penalty ---
    ax = axes[0, 0]
    for j in range(len(basis_changes)):
        start = basis_changes[j]
        end = basis_changes[j + 1] if j + 1 < len(basis_changes) else len(lambda_values)
        cidx = basis_to_color[basis_list[start]] % 20
        ax.plot(lambda_values[start:end], L_dfl_pen[start:end],
                color=cmap(cidx), linewidth=1.5, alpha=0.8, marker='.', markersize=3)
    ax.axvline(LAMBDA_TRUE, color='red', linestyle='--', linewidth=2)
    min_idx = np.argmin(L_dfl_pen)
    lam_min_dfl = lambda_values[min_idx]
    ax.plot(LAMBDA_TRUE, L_dfl_pen[idx_true], 'r*', markersize=12, zorder=5)
    ax.plot(lam_min_dfl, L_dfl_pen[min_idx], 'go', markersize=10, zorder=5,
            label=f'min at $\\lambda$={lam_min_dfl:.2f}')
    ax.set_ylabel('Loss', fontsize=13)
    ax.set_title(f'Standard DFL + penalty ($\\rho$={rho})\n'
                 f'$-r_{{true}}^T y^* + \\rho \\|\\gamma_{{true}} - H_{{true}} y^*\\|^2$\n'
                 f'min at $\\lambda^*$={lam_min_dfl:.2f} (shift={lam_min_dfl - LAMBDA_TRUE:.2f})',
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Top-center: Self-consistent + penalty ---
    ax = axes[0, 1]
    for j in range(len(basis_changes)):
        start = basis_changes[j]
        end = basis_changes[j + 1] if j + 1 < len(basis_changes) else len(lambda_values)
        cidx = basis_to_color[basis_list[start]] % 20
        ax.plot(lambda_values[start:end], L_sc_pen[start:end],
                color=cmap(cidx), linewidth=1.5, alpha=0.8, marker='.', markersize=3)
    ax.axvline(LAMBDA_TRUE, color='red', linestyle='--', linewidth=2)
    min_idx_sc = np.argmin(L_sc_pen)
    lam_min_sc = lambda_values[min_idx_sc]
    ax.plot(LAMBDA_TRUE, L_sc_pen[idx_true], 'r*', markersize=12, zorder=5)
    ax.plot(lam_min_sc, L_sc_pen[min_idx_sc], 'go', markersize=10, zorder=5,
            label=f'min at $\\lambda$={lam_min_sc:.2f}')
    ax.set_title(f'Self-consistent + penalty ($\\rho$={rho})\n'
                 f'$-r_\\theta^T y^* + \\rho \\|\\gamma_{{true}} - H_{{true}} y^*\\|^2$\n'
                 f'min at $\\lambda^*$={lam_min_sc:.2f} (shift={lam_min_sc - LAMBDA_TRUE:.2f})',
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Top-right: Overlay both (zoomed near λ_true) ---
    ax = axes[0, 2]
    zoom = (lambda_values >= LAMBDA_TRUE - 5) & (lambda_values <= LAMBDA_TRUE + 5)
    zoom_lam = lambda_values[zoom]
    # Normalize both to same scale for comparison
    ax.plot(zoom_lam, L_dfl_pen[zoom], 'b-', linewidth=2, alpha=0.8,
            label=f'Standard DFL (min={lam_min_dfl:.2f})')
    ax.plot(zoom_lam, L_sc_pen[zoom], 'darkorange', linewidth=2, alpha=0.8,
            label=f'Self-consistent (min={lam_min_sc:.2f})')
    ax.axvline(LAMBDA_TRUE, color='red', linestyle='--', linewidth=2, label='$\\lambda_{true}$')
    ax.set_title(f'Both methods overlaid (zoomed)\n$\\rho$={rho}', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Bottom-left: Gradient of Standard DFL + penalty ---
    ax = axes[1, 0]
    dL_dfl = np.gradient(L_dfl_pen, lambda_values)
    for j in range(len(basis_changes)):
        start = basis_changes[j]
        end = basis_changes[j + 1] if j + 1 < len(basis_changes) else len(lambda_values)
        cidx = basis_to_color[basis_list[start]] % 20
        ax.plot(lambda_values[start:end], dL_dfl[start:end],
                color=cmap(cidx), linewidth=1.0, alpha=0.8, marker='.', markersize=3)
    ax.axvline(LAMBDA_TRUE, color='red', linestyle='--', linewidth=2)
    ax.axhline(0, color='black', linewidth=0.5)
    for idx in basis_changes[1:]:
        ax.axvline(lambda_values[idx], color='gray', alpha=0.2, linewidth=0.5)
    ax.set_xlabel('$\\lambda_{pred}$', fontsize=13)
    ax.set_ylabel('$\\partial L / \\partial \\lambda$', fontsize=11)
    ax.set_title('Gradient: Standard DFL + penalty', fontsize=12)
    ax.grid(True, alpha=0.3)

    # --- Bottom-center: Gradient of Self-consistent + penalty ---
    ax = axes[1, 1]
    dL_sc = np.gradient(L_sc_pen, lambda_values)
    for j in range(len(basis_changes)):
        start = basis_changes[j]
        end = basis_changes[j + 1] if j + 1 < len(basis_changes) else len(lambda_values)
        cidx = basis_to_color[basis_list[start]] % 20
        ax.plot(lambda_values[start:end], dL_sc[start:end],
                color=cmap(cidx), linewidth=1.0, alpha=0.8, marker='.', markersize=3)
    ax.axvline(LAMBDA_TRUE, color='red', linestyle='--', linewidth=2)
    ax.axhline(0, color='black', linewidth=0.5)
    for idx in basis_changes[1:]:
        ax.axvline(lambda_values[idx], color='gray', alpha=0.2, linewidth=0.5)
    ax.set_xlabel('$\\lambda_{pred}$', fontsize=13)
    ax.set_ylabel('$\\partial L / \\partial \\lambda$', fontsize=11)
    ax.set_title('Gradient: Self-consistent + penalty', fontsize=12)
    ax.grid(True, alpha=0.3)

    # --- Bottom-right: λ* vs ρ for both methods ---
    ax = axes[1, 2]
    rho_sweep = np.logspace(-2, 4, 100)
    # Use zoomed region for min-finding
    zoom_wide = (lambda_values >= LAMBDA_TRUE - 8) & (lambda_values <= LAMBDA_TRUE + 8)
    zoom_lam_w = lambda_values[zoom_wide]
    zoom_dfl_w = L_rtrue[zoom_wide]
    zoom_sc_w = L_rtheta[zoom_wide]
    zoom_pen_w = penalty_vals[zoom_wide]

    lam_star_dfl = []
    lam_star_sc = []
    for rho_i in rho_sweep:
        aug_dfl = zoom_dfl_w + rho_i * zoom_pen_w
        aug_sc = zoom_sc_w + rho_i * zoom_pen_w
        lam_star_dfl.append(zoom_lam_w[np.argmin(aug_dfl)])
        lam_star_sc.append(zoom_lam_w[np.argmin(aug_sc)])

    ax.semilogx(rho_sweep, lam_star_dfl, 'b-', linewidth=2, label='Standard DFL')
    ax.semilogx(rho_sweep, lam_star_sc, 'darkorange', linewidth=2, label='Self-consistent')
    ax.axhline(LAMBDA_TRUE, color='red', linestyle='--', linewidth=2, label='$\\lambda_{true}$')
    ax.set_xlabel('$\\rho$', fontsize=13)
    ax.set_ylabel('$\\lambda^*$ (minimizer)', fontsize=13)
    ax.set_title('$\\lambda^*$ vs $\\rho$: convergence comparison', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        'Both methods with penalty: $L(\\lambda) = \\text{obj}(\\lambda) '
        '+ \\rho \\|\\gamma_{true} - H_{true} y^*(\\lambda)\\|^2$\n'
        f'Standard DFL uses $r_{{true}}$, Self-consistent uses $r_\\theta$ | $\\rho$={rho}',
        fontsize=14, y=1.02
    )
    plt.tight_layout()
    plt.savefig('mdp_lp/plot_results/both_methods_with_penalty.png', dpi=150, bbox_inches='tight')
    plt.savefig('mdp_lp/plot_results/both_methods_with_penalty.pdf', bbox_inches='tight')
    print("Saved to mdp_lp/plot_results/both_methods_with_penalty.{png,pdf}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"ρ = {rho}")
    print(f"Standard DFL + penalty: min at λ* = {lam_min_dfl:.3f} (shift = {lam_min_dfl - LAMBDA_TRUE:.3f})")
    print(f"Self-consistent + penalty: min at λ* = {lam_min_sc:.3f} (shift = {lam_min_sc - LAMBDA_TRUE:.3f})")
    print(f"\nConvergence as ρ → ∞:")
    for rho_i in [0.1, 1, 10, 100, 1000, 10000]:
        aug_dfl = zoom_dfl_w + rho_i * zoom_pen_w
        aug_sc = zoom_sc_w + rho_i * zoom_pen_w
        ls_dfl = zoom_lam_w[np.argmin(aug_dfl)]
        ls_sc = zoom_lam_w[np.argmin(aug_sc)]
        print(f"  ρ={rho_i:>8.1f}  DFL: λ*={ls_dfl:.3f} (Δ={ls_dfl - LAMBDA_TRUE:+.3f})  "
              f"SC: λ*={ls_sc:.3f} (Δ={ls_sc - LAMBDA_TRUE:+.3f})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
