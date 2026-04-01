"""
Plot the augmented Lagrangian DFL loss landscape:
  L(λ) = -r_true^T y*(λ) + ρ ||γ - H_true y*(λ)||²

Shows:
1. The penalty is 0 at λ_true but DFL gradient ≠ 0 → minimum is shifted
2. For large ρ, minimum approaches λ_true but never reaches it
3. Local quadratic approximation at λ_true

Usage:
    python -m mdp_lp.plot_aug_lag_landscape
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
    mdp = create_inventory_from_lambda(lam, B_MAX, S_MAX, Q_MAX,
                                       FOC, VOC, HC, BC, BETA, p)
    S = mdp.P.shape[1]
    gamma = mdp.build_gamma(1, S, torch.device("cpu"))
    gamma = torch.clamp(gamma, min=1e-8)
    gamma = gamma / gamma.sum(dim=1, keepdim=True)
    H = mdp.build_H(mdp.P)
    r_flat = mdp.r.reshape(1, -1)
    solver = DenseLPSolver()
    y_star, basis, dual = solver.solve(H, r_flat, gamma)
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

    # Sweep λ_pred (dense near λ_true for accuracy)
    lambda_values = np.sort(np.concatenate([
        np.linspace(0.5, 50, 400),
        np.linspace(LAMBDA_TRUE - 5, LAMBDA_TRUE + 5, 200),  # dense near λ_true
    ]))

    dfl_loss_vals = []
    penalty_raw_vals = []  # ||γ - H_true y*||² (without ρ)
    basis_list = []

    print("Computing landscape...")
    for i, lam in enumerate(lambda_values):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(lambda_values)}")
        y_star, _, H_pred, gamma_pred, basis = solve_lp_for_lambda(lam)

        v = (r_true * y_star).sum().item()
        violation = gamma_true - (H_true @ y_star.unsqueeze(-1)).squeeze(-1)
        pen = violation.pow(2).sum().item()

        dfl_loss_vals.append(-v)
        penalty_raw_vals.append(pen)
        basis_list.append(tuple(basis[0].tolist()))

    dfl_loss_vals = np.array(dfl_loss_vals)
    penalty_raw_vals = np.array(penalty_raw_vals)

    # Detect basis changes
    basis_change_idx = [0]
    for i in range(1, len(basis_list)):
        if basis_list[i] != basis_list[i - 1]:
            basis_change_idx.append(i)
    n_basis = len(set(basis_list))

    idx_true = np.argmin(np.abs(lambda_values - LAMBDA_TRUE))

    # ── Figure 1: Three ρ values showing the shift ──
    rho_values = [0.1, 1.0, 10.0, 100.0, 1000.0]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Top row: zoomed landscape near λ_true for different ρ
    zoom = (lambda_values >= LAMBDA_TRUE - 4) & (lambda_values <= LAMBDA_TRUE + 4)
    zoom_lam = lambda_values[zoom]
    zoom_dfl = dfl_loss_vals[zoom]
    zoom_pen = penalty_raw_vals[zoom]

    for ax_idx, rho in enumerate(rho_values[:3]):
        ax = axes[0, ax_idx]
        zoom_aug = zoom_dfl + rho * zoom_pen

        # Find minimum
        min_idx = np.argmin(zoom_aug)
        lam_min = zoom_lam[min_idx]
        L_min = zoom_aug[min_idx]

        # L at λ_true
        idx_t_zoom = np.argmin(np.abs(zoom_lam - LAMBDA_TRUE))
        L_at_true = zoom_aug[idx_t_zoom]

        ax.plot(zoom_lam, zoom_aug, 'b-', linewidth=2, label='$L_{aug}(\\lambda)$')
        ax.axvline(LAMBDA_TRUE, color='red', linestyle='--', linewidth=1.5, label='$\\lambda_{true}$')
        ax.plot(LAMBDA_TRUE, L_at_true, 'r*', markersize=15, zorder=5)
        ax.plot(lam_min, L_min, 'go', markersize=12, zorder=5, label=f'min at λ={lam_min:.2f}')

        # Arrow showing the shift
        if abs(lam_min - LAMBDA_TRUE) > 0.05:
            ax.annotate('', xy=(lam_min, L_min), xytext=(LAMBDA_TRUE, L_min),
                        arrowprops=dict(arrowstyle='<->', color='green', lw=2))
            mid = (lam_min + LAMBDA_TRUE) / 2
            ax.text(mid, L_min + 0.5, f'Δ={lam_min - LAMBDA_TRUE:.2f}',
                    ha='center', fontsize=11, color='green', fontweight='bold')

        ax.set_title(f'ρ = {rho}\nmin at λ*={lam_min:.2f}  (shift = {lam_min - LAMBDA_TRUE:.2f})',
                     fontsize=13)
        ax.set_xlabel('$\\lambda_{pred}$', fontsize=12)
        ax.set_ylabel('$L_{aug}$', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    # Bottom row: more ρ values + convergence of λ*
    for ax_idx, rho in enumerate(rho_values[3:]):
        ax = axes[1, ax_idx]
        zoom_aug = zoom_dfl + rho * zoom_pen
        min_idx = np.argmin(zoom_aug)
        lam_min = zoom_lam[min_idx]
        L_min = zoom_aug[min_idx]
        idx_t_zoom = np.argmin(np.abs(zoom_lam - LAMBDA_TRUE))
        L_at_true = zoom_aug[idx_t_zoom]

        ax.plot(zoom_lam, zoom_aug, 'b-', linewidth=2)
        ax.axvline(LAMBDA_TRUE, color='red', linestyle='--', linewidth=1.5)
        ax.plot(LAMBDA_TRUE, L_at_true, 'r*', markersize=15, zorder=5)
        ax.plot(lam_min, L_min, 'go', markersize=12, zorder=5, label=f'min at λ={lam_min:.2f}')
        ax.set_title(f'ρ = {rho}\nmin at λ*={lam_min:.2f}  (shift = {lam_min - LAMBDA_TRUE:.2f})',
                     fontsize=13)
        ax.set_xlabel('$\\lambda_{pred}$', fontsize=12)
        ax.set_ylabel('$L_{aug}$', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    # Bottom right: λ* vs ρ (convergence plot)
    ax = axes[1, 2]
    rho_sweep = np.logspace(-2, 4, 100)
    lam_star_list = []
    for rho in rho_sweep:
        aug = zoom_dfl + rho * zoom_pen
        lam_star_list.append(zoom_lam[np.argmin(aug)])
    ax.semilogx(rho_sweep, lam_star_list, 'b-', linewidth=2)
    ax.axhline(LAMBDA_TRUE, color='red', linestyle='--', linewidth=2, label='$\\lambda_{true}$')
    ax.set_xlabel('ρ', fontsize=13)
    ax.set_ylabel('$\\lambda^*$ (minimizer)', fontsize=13)
    ax.set_title('$\\lambda^* \\to \\lambda_{true}$ as $\\rho \\to \\infty$\nbut never reaches it', fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Augmented Lagrangian: Minimum is NOT at $\\lambda_{{true}}$ for finite ρ\n'
                 f'$L_{{aug}}(\\lambda) = -r_{{true}}^T y^*(\\lambda) + \\rho \\|\\gamma - H_{{true}} y^*(\\lambda)\\|^2$',
                 fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig('mdp_lp/plot_results/aug_lag_landscape.png', dpi=150, bbox_inches='tight')
    plt.savefig('mdp_lp/plot_results/aug_lag_landscape.pdf', bbox_inches='tight')
    print("Saved to mdp_lp/plot_results/aug_lag_landscape.{png,pdf}")

    # ── Figure 2: Decomposition showing WHY the shift happens ──
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    rho = 10.0

    # Left: DFL loss (linear near λ_true)
    ax = axes2[0]
    ax.plot(zoom_lam, zoom_dfl, 'b-', linewidth=2)
    ax.axvline(LAMBDA_TRUE, color='red', linestyle='--', linewidth=1.5)

    # Fit local slope
    dL_dfl = np.gradient(zoom_dfl, zoom_lam)
    idx_t_zoom = np.argmin(np.abs(zoom_lam - LAMBDA_TRUE))
    a = dL_dfl[idx_t_zoom]
    ax.set_title(f'$-r_{{true}}^T y^*(\\lambda)$\nslope at $\\lambda_{{true}}$ = {a:.1f} ≠ 0', fontsize=13)
    ax.set_xlabel('$\\lambda_{pred}$')
    ax.set_ylabel('DFL loss')
    ax.grid(True, alpha=0.3)

    # Middle: penalty (quadratic, 0 at λ_true)
    ax = axes2[1]
    ax.plot(zoom_lam, rho * zoom_pen, 'orange', linewidth=2)
    ax.axvline(LAMBDA_TRUE, color='red', linestyle='--', linewidth=1.5)
    ax.plot(LAMBDA_TRUE, rho * zoom_pen[idx_t_zoom], 'r*', markersize=15, zorder=5)
    ax.set_title(f'$\\rho \\|\\gamma - H_{{true}} y^*\\|^2$  (ρ={rho})\nquadratic, = 0 at $\\lambda_{{true}}$', fontsize=13)
    ax.set_xlabel('$\\lambda_{pred}$')
    ax.set_ylabel('Penalty')
    ax.grid(True, alpha=0.3)

    # Right: sum = linear + quadratic → minimum shifted
    ax = axes2[2]
    zoom_aug = zoom_dfl + rho * zoom_pen
    min_idx = np.argmin(zoom_aug)
    lam_min = zoom_lam[min_idx]

    ax.plot(zoom_lam, zoom_aug, 'b-', linewidth=2, label='$L_{aug}$ (sum)')
    ax.plot(zoom_lam, zoom_dfl, 'b--', linewidth=1, alpha=0.4, label='DFL only')
    ax.plot(zoom_lam, zoom_dfl[idx_t_zoom] + rho * zoom_pen, '--', color='orange',
            linewidth=1, alpha=0.4, label='penalty only (shifted)')
    ax.axvline(LAMBDA_TRUE, color='red', linestyle='--', linewidth=1.5, label='$\\lambda_{true}$')
    ax.axvline(lam_min, color='green', linestyle='--', linewidth=1.5, label=f'$\\lambda^*$ = {lam_min:.2f}')
    ax.plot(lam_min, zoom_aug[min_idx], 'go', markersize=12, zorder=5)
    ax.set_title(f'Linear (slope={a:.1f}) + Quadratic → min shifted\n'
                 f'$\\lambda^* = \\lambda_{{true}} - a/(2\\rho c^2)$ = {lam_min:.2f}',
                 fontsize=13)
    ax.set_xlabel('$\\lambda_{pred}$')
    ax.set_ylabel('$L_{aug}$')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Why the minimum shifts: DFL has nonzero slope + penalty is 0 at $\\lambda_{true}$',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('mdp_lp/plot_results/aug_lag_decomposition.png', dpi=150, bbox_inches='tight')
    plt.savefig('mdp_lp/plot_results/aug_lag_decomposition.pdf', bbox_inches='tight')
    print("Saved to mdp_lp/plot_results/aug_lag_decomposition.{png,pdf}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"DFL gradient at λ_true:  a = {a:.2f}")
    print(f"Penalty at λ_true:       {penalty_raw_vals[idx_true]:.2e} (≈ 0)")
    print(f"")
    print(f"Minimizer λ* for different ρ:")
    for rho in [0.1, 1, 10, 100, 1000, 10000]:
        aug = zoom_dfl + rho * zoom_pen
        lam_star = zoom_lam[np.argmin(aug)]
        print(f"  ρ = {rho:>8.1f}  →  λ* = {lam_star:.3f}  (shift = {lam_star - LAMBDA_TRUE:.3f})")
    print(f"  ρ → ∞       →  λ* → {LAMBDA_TRUE} (but never reaches)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
