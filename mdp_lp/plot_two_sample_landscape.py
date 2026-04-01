"""
Plot the DFL aug-lag loss landscape for two specific single-sample experiments:
  - seed=42, lambda_true≈5.87, price≈7.55  (converges)
  - seed=1,  lambda_true≈4.42, price≈4.75  (diverges)

Style: like piecewise_linear_landscape.pdf — top: L(λ) with basis coloring,
bottom: gradient. Arrow at the initial lambda_pred shows descent direction.

Usage:
    python -m mdp_lp.plot_two_sample_landscape
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mdp_lp.inventory_elasticity import create_inventory_from_lambda
from mdp_lp.solvers.dense_lp_solver import DenseLPSolver

# ── MDP parameters (matching experiments: exp_b, high_margin) ──
B_MAX, S_MAX, Q_MAX = 5, 50, 50
FOC, VOC, HC, BC, BETA = 0, 0.5, 0.1, 0, 0.95
RHO = 10000.0

# Two samples with their ACTUAL prices from data generation
SAMPLES = [
    {"lambda_true": 6.0847, "lambda_pred_init": 19.82, "price": 6.1193, "label": "converges"},
    {"lambda_true": 4.5933, "lambda_pred_init": 32.15, "price": 7.4354, "label": "diverges"},
]

SOLVER = DenseLPSolver()


def compute_landscape(lambda_true_val, price_val, lambda_sweep):
    """Compute aug-lag loss landscape for a given (lambda_true, price) pair."""
    lam_true_t = torch.tensor([lambda_true_val], dtype=torch.float32)
    p_t = torch.tensor([price_val], dtype=torch.float32)
    mdp_true = create_inventory_from_lambda(lam_true_t, B_MAX, S_MAX, Q_MAX,
                                             FOC, VOC, HC, BC, BETA, p_t)
    r_true = mdp_true.r.reshape(1, -1)
    H_true = mdp_true.build_H(mdp_true.P)
    S = mdp_true.P.shape[1]
    gamma_true = mdp_true.build_gamma(1, S, torch.device("cpu"))
    gamma_true = torch.clamp(gamma_true, min=1e-8)
    gamma_true = gamma_true / gamma_true.sum(dim=1, keepdim=True)

    dfl_vals = []
    pen_vals = []
    basis_list = []

    for lam in lambda_sweep:
        lam_t = torch.tensor([lam], dtype=torch.float32)
        mdp = create_inventory_from_lambda(lam_t, B_MAX, S_MAX, Q_MAX,
                                            FOC, VOC, HC, BC, BETA, p_t)
        H = mdp.build_H(mdp.P)
        r_flat = mdp.r.reshape(1, -1)
        gamma = mdp.build_gamma(1, mdp.P.shape[1], torch.device("cpu"))
        gamma = torch.clamp(gamma, min=1e-8)
        gamma = gamma / gamma.sum(dim=1, keepdim=True)

        y_star, basis, _ = SOLVER.solve(H, r_flat, gamma)

        dfl_vals.append(-(r_true * y_star).sum().item())
        violation = gamma_true - (H_true @ y_star.unsqueeze(-1)).squeeze(-1)
        pen_vals.append(violation.pow(2).sum().item())
        basis_list.append(tuple(basis[0].tolist()))

    return np.array(dfl_vals), np.array(pen_vals), basis_list


def main():
    # Fewer points for speed; dense near the interesting regions
    lambda_sweep = np.sort(np.unique(np.concatenate([
        np.linspace(0.5, 50, 200),
    ])))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10),
                              gridspec_kw={'height_ratios': [3, 1]})

    for col, sample in enumerate(SAMPLES):
        lt = sample["lambda_true"]
        lp0 = sample["lambda_pred_init"]
        price = sample["price"]
        label = sample["label"]

        print(f"Computing landscape: λ_true={lt}, price={price} ...")
        dfl, pen, basis_list = compute_landscape(lt, price, lambda_sweep)
        aug_lag = dfl + RHO * pen

        # Basis coloring
        unique_bases = list(dict.fromkeys(basis_list))
        basis_to_color = {b: i for i, b in enumerate(unique_bases)}
        basis_changes = [0]
        for i in range(1, len(basis_list)):
            if basis_list[i] != basis_list[i - 1]:
                basis_changes.append(i)
        n_basis = len(unique_bases)
        cmap = plt.cm.tab20

        # ── Top panel: L_aug(λ) ──
        ax = axes[0, col]
        for j in range(len(basis_changes)):
            start = basis_changes[j]
            end = basis_changes[j + 1] if j + 1 < len(basis_changes) else len(lambda_sweep)
            cidx = basis_to_color[basis_list[start]] % 20
            ax.plot(lambda_sweep[start:end], aug_lag[start:end],
                    color=cmap(cidx), linewidth=1.5, alpha=0.8)

        for idx in basis_changes[1:]:
            ax.axvline(lambda_sweep[idx], color='gray', alpha=0.15, linewidth=0.5)

        # Mark lambda_true
        idx_true = np.argmin(np.abs(lambda_sweep - lt))
        ax.axvline(lt, color='red', linestyle='--', linewidth=2,
                    label=f'$\\lambda_{{true}}$ = {lt}')
        ax.plot(lt, aug_lag[idx_true], 'r*', markersize=15, zorder=10)

        # Mark lambda_pred_init with arrow showing descent direction
        idx_init = np.argmin(np.abs(lambda_sweep - lp0))
        ax.plot(lp0, aug_lag[idx_init], 'go', markersize=12, zorder=10,
                label=f'$\\lambda_{{pred}}^{{init}}$ = {lp0}')

        grad = np.gradient(aug_lag, lambda_sweep)
        local_grad = grad[idx_init]
        descent_dir = -np.sign(local_grad)
        arrow_len = (lambda_sweep.max() - lambda_sweep.min()) * 0.06
        ax.annotate('', xy=(lp0 + descent_dir * arrow_len, aug_lag[idx_init]),
                     xytext=(lp0, aug_lag[idx_init]),
                     arrowprops=dict(arrowstyle='->', color='green', lw=3))

        toward = "toward" if ((descent_dir < 0 and lp0 > lt) or
                               (descent_dir > 0 and lp0 < lt)) else "AWAY from"
        ax.text(0.03, 0.95,
                f'$\\partial L/\\partial \\lambda$ at init = {local_grad:.1f}\n'
                f'Descent pushes {toward} $\\lambda_{{true}}$\nprice = {price}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

        ax.set_ylabel('$L_{aug}(\\lambda)$', fontsize=12)
        ax.set_title(f'$\\lambda_{{true}}$ = {lt}, price = {price} ({label})\n'
                      f'{n_basis} basis regions', fontsize=13)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)

        # ── Bottom panel: gradient ──
        ax2 = axes[1, col]
        for j in range(len(basis_changes)):
            start = basis_changes[j]
            end = basis_changes[j + 1] if j + 1 < len(basis_changes) else len(lambda_sweep)
            cidx = basis_to_color[basis_list[start]] % 20
            ax2.plot(lambda_sweep[start:end], grad[start:end],
                     color=cmap(cidx), linewidth=1.0, alpha=0.8)

        ax2.axvline(lt, color='red', linestyle='--', linewidth=2)
        ax2.axhline(0, color='black', linewidth=0.5)
        for idx in basis_changes[1:]:
            ax2.axvline(lambda_sweep[idx], color='gray', alpha=0.15, linewidth=0.5)

        # Mark gradient at init
        ax2.plot(lp0, local_grad, 'go', markersize=10, zorder=10)
        ax2.annotate(f'{local_grad:.0f}', (lp0, local_grad),
                      textcoords='offset points', xytext=(8, 8), fontsize=10,
                      color='green', fontweight='bold')

        ax2.set_xlabel('$\\lambda_{pred}$', fontsize=12)
        ax2.set_ylabel('$\\partial L_{aug} / \\partial \\lambda$', fontsize=11)
        ax2.grid(True, alpha=0.3)

    fig.suptitle(
        'Aug-Lag landscape for two single-sample experiments\n'
        '$L_{aug}(\\lambda) = -r_{true}^T y^*(\\lambda) + '
        f'\\rho\\|\\gamma - H_{{true}}\\, y^*(\\lambda)\\|^2$  '
        f'($\\rho$ = {RHO:.0f})',
        fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('mdp_lp/plot_results/two_sample_landscape.png', dpi=150, bbox_inches='tight')
    plt.savefig('mdp_lp/plot_results/two_sample_landscape.pdf', bbox_inches='tight')
    print("\nSaved to mdp_lp/plot_results/two_sample_landscape.{png,pdf}")


if __name__ == "__main__":
    main()
