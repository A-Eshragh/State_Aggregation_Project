"""
Plot the DFL loss landscape L(lambda) for both the original MDP and state-aggregated MDP.

Key insight: state aggregation reduces the number of LP variables (M*A vs S*A),
so the aggregated LP has fewer possible bases => fewer kinks => smoother landscape
=> easier optimization via gradient descent.

Usage:
    python -m mdp_lp.plot_state_agg_landscape
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mdp_lp.inventory_elasticity import create_inventory_from_lambda
from mdp_lp.solvers.dense_lp_solver import DenseLPSolver
from mdp_lp.soft_state_aggregation import SoftAggDense

# ── MDP parameters (same as piecewise_linear plot) ──
B_MAX, S_MAX, Q_MAX, D_MAX = 5, 50, 30, 30
FOC, VOC, HC, BC, BETA, PRICE = 0, 0.5, 0.1, 0, 0.95, 10.0
LAMBDA_TRUE = 10.0
RHO = 10000.0

N_STATES = B_MAX + S_MAX + 1   # 56
N_ACTIONS = Q_MAX + 1           # 31


def build_mdp_and_solve(lam_val, solver, agg=None, tau=0.5):
    """Build MDP from lambda, optionally aggregate, solve LP, return y, basis, mdp, gamma."""
    lam = torch.tensor([lam_val], dtype=torch.float32)
    p = torch.tensor([PRICE], dtype=torch.float32)
    mdp = create_inventory_from_lambda(lam, B_MAX, S_MAX, Q_MAX,
                                       FOC, VOC, HC, BC, BETA, p)
    S = mdp.P.shape[1]
    gamma = mdp.build_gamma(1, S, torch.device("cpu"))
    gamma = torch.clamp(gamma, min=1e-8)
    gamma = gamma / gamma.sum(dim=1, keepdim=True)

    if agg is not None:
        with torch.no_grad():
            P_agg, r_agg, gamma_agg, info = agg(mdp.P, mdp.r, gamma, tau=tau)
        H_agg = mdp.build_H(P_agg)
        r_flat_agg = r_agg.reshape(1, -1)
        y_agg, basis_agg, _ = solver.solve(H_agg, r_flat_agg, gamma_agg)
        # Disaggregate back to original space
        y_orig = SoftAggDense.disaggregate_y(y_agg, info["W"], N_ACTIONS)
        return y_orig, basis_agg, mdp, gamma
    else:
        H = mdp.build_H(mdp.P)
        r_flat = mdp.r.reshape(1, -1)
        y_star, basis, _ = solver.solve(H, r_flat, gamma)
        return y_star, basis, mdp, gamma


def main():
    # Build true MDP reward vector
    lam_true_t = torch.tensor([LAMBDA_TRUE], dtype=torch.float32)
    p_t = torch.tensor([PRICE], dtype=torch.float32)
    mdp_true = create_inventory_from_lambda(lam_true_t, B_MAX, S_MAX, Q_MAX,
                                             FOC, VOC, HC, BC, BETA, p_t)
    r_true = mdp_true.r.reshape(1, -1)
    H_true = mdp_true.build_H(mdp_true.P)
    S_true = mdp_true.P.shape[1]
    gamma_true = mdp_true.build_gamma(1, S_true, torch.device("cpu"))
    gamma_true = torch.clamp(gamma_true, min=1e-8)
    gamma_true = gamma_true / gamma_true.sum(dim=1, keepdim=True)

    solver = DenseLPSolver()

    # Initialize aggregation modules with different cluster counts
    torch.manual_seed(42)
    n_agg_values = [10, 20, 30]
    agg_modules = {}
    for n_agg in n_agg_values:
        agg_modules[n_agg] = SoftAggDense(
            n_states=N_STATES, n_actions=N_ACTIONS,
            n_clusters=n_agg, r_embed=8
        )

    # Sweep lambda_pred
    lambda_values = np.linspace(0.5, 50, 500)

    # Original DFL landscape
    L_orig = []
    basis_orig = []
    print("Computing original DFL landscape...")
    for i, lam in enumerate(lambda_values):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(lambda_values)}")
        y, basis, mdp_pred, gamma_pred = build_mdp_and_solve(lam, solver, agg=None)
        # DFL term: -r_true^T y*(λ)
        dfl = -(r_true * y).sum().item()
        # Penalty: ρ * ||γ_true - H_true y*(λ)||²
        violation = gamma_true - (H_true @ y.unsqueeze(-1)).squeeze(-1)
        pen = violation.pow(2).sum().item()
        L_orig.append(dfl + RHO * pen)
        basis_orig.append(tuple(basis[0].tolist()))
    L_orig = np.array(L_orig)

    # Aggregated landscapes
    L_agg_dict = {}
    basis_agg_dict = {}
    for n_agg in n_agg_values:
        print(f"\nComputing state-aggregated landscape (M={n_agg})...")
        L_agg = []
        basis_agg = []
        for i, lam in enumerate(lambda_values):
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(lambda_values)}")
            y, basis, mdp_pred, gamma_pred = build_mdp_and_solve(lam, solver, agg=agg_modules[n_agg])
            # DFL term: -r_true^T y*(λ)
            dfl = -(r_true * y).sum().item()
            # Penalty: ρ * ||γ_true - H_true y*(λ)||²
            violation = gamma_true - (H_true @ y.unsqueeze(-1)).squeeze(-1)
            pen = violation.pow(2).sum().item()
            L_agg.append(dfl + RHO * pen)
            basis_agg.append(tuple(basis[0].tolist()))
        L_agg_dict[n_agg] = np.array(L_agg)
        basis_agg_dict[n_agg] = basis_agg

    # Count basis changes
    def count_basis_changes(basis_list):
        changes = [0]
        for i in range(1, len(basis_list)):
            if basis_list[i] != basis_list[i - 1]:
                changes.append(i)
        n_distinct = len(set(basis_list))
        return changes, n_distinct

    orig_changes, orig_n_basis = count_basis_changes(basis_orig)
    print(f"\nOriginal LP (S={N_STATES}, SA={N_STATES*N_ACTIONS}): "
          f"{orig_n_basis} distinct bases, {len(orig_changes)-1} basis changes")
    for n_agg in n_agg_values:
        agg_changes, agg_n_basis = count_basis_changes(basis_agg_dict[n_agg])
        print(f"Aggregated LP (M={n_agg}, MA={n_agg*N_ACTIONS}): "
              f"{agg_n_basis} distinct bases, {len(agg_changes)-1} basis changes")

    # ── Figure 1: Landscape comparison ──
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 2]})

    # --- Top panel: loss landscapes ---
    ax = axes[0]

    # Plot original with basis coloring
    unique_bases_orig = list(dict.fromkeys(basis_orig))
    basis_to_color_orig = {b: i for i, b in enumerate(unique_bases_orig)}
    cmap = plt.cm.tab20

    for j in range(len(orig_changes)):
        start = orig_changes[j]
        end = orig_changes[j + 1] if j + 1 < len(orig_changes) else len(lambda_values)
        color_idx = basis_to_color_orig[basis_orig[start]] % 20
        label = f'DFL (S={N_STATES}, {orig_n_basis} bases)' if j == 0 else None
        ax.plot(lambda_values[start:end], L_orig[start:end],
                color=cmap(color_idx), linewidth=1.2, alpha=0.6)

    # Mark basis changes for original
    for idx in orig_changes[1:]:
        ax.axvline(lambda_values[idx], color='gray', alpha=0.15, linewidth=0.5)

    # Plot aggregated landscapes
    agg_colors = {10: '#2196F3', 20: '#4CAF50', 30: '#FF9800'}
    agg_styles = {10: '-', 20: '-', 30: '-'}
    for n_agg in n_agg_values:
        agg_changes_n, agg_n_basis_n = count_basis_changes(basis_agg_dict[n_agg])
        ax.plot(lambda_values, L_agg_dict[n_agg],
                color=agg_colors[n_agg], linewidth=2.5, linestyle=agg_styles[n_agg],
                alpha=0.9, label=f'State Agg (M={n_agg}, {agg_n_basis_n} bases)')

    # Mark lambda_true
    ax.axvline(LAMBDA_TRUE, color='red', linestyle='--', linewidth=2, label=f'$\\lambda_{{true}} = {LAMBDA_TRUE}$')
    idx_true = np.argmin(np.abs(lambda_values - LAMBDA_TRUE))
    ax.plot(LAMBDA_TRUE, L_orig[idx_true], 'r*', markersize=15, zorder=5)

    ax.set_ylabel('$L(\\lambda) = -r_{true}^T y^*(\\lambda) + \\rho\\|\\gamma_{true} - H_{true} y^*\\|^2$', fontsize=12)
    ax.set_title(
        f'Aug-Lag Landscape: Original ({orig_n_basis} bases) vs State Aggregation (fewer bases)\n'
        f'Fewer LP variables $\\Rightarrow$ fewer basis regions $\\Rightarrow$ smoother landscape  ($\\rho$={RHO:.0f})',
        fontsize=14
    )
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)

    # --- Bottom panel: gradient comparison ---
    ax2 = axes[1]

    # Original gradient
    dL_orig = np.gradient(L_orig, lambda_values)
    ax2.plot(lambda_values, dL_orig, color='gray', linewidth=0.8, alpha=0.5,
             label=f'DFL gradient (S={N_STATES})')

    # Aggregated gradients
    for n_agg in n_agg_values:
        dL_agg = np.gradient(L_agg_dict[n_agg], lambda_values)
        ax2.plot(lambda_values, dL_agg,
                 color=agg_colors[n_agg], linewidth=2.0, linestyle=agg_styles[n_agg],
                 alpha=0.9, label=f'State Agg gradient (M={n_agg})')

    ax2.axvline(LAMBDA_TRUE, color='red', linestyle='--', linewidth=2)
    ax2.axhline(0, color='black', linewidth=0.5)
    for idx in orig_changes[1:]:
        ax2.axvline(lambda_values[idx], color='gray', alpha=0.1, linewidth=0.5)

    ax2.set_xlabel('$\\lambda_{pred}$', fontsize=13)
    ax2.set_ylabel('$\\partial L / \\partial \\lambda$\n(finite diff)', fontsize=11)
    ax2.set_title('Gradient: state aggregation produces smoother, more informative gradients', fontsize=12)
    ax2.legend(fontsize=10, loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mdp_lp/plot_results/state_agg_landscape.png', dpi=150, bbox_inches='tight')
    plt.savefig('mdp_lp/plot_results/state_agg_landscape.pdf', bbox_inches='tight')
    print("\nSaved to mdp_lp/plot_results/state_agg_landscape.{png,pdf}")

    # ── Figure 2: Zoomed view near lambda_true ──
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 2]})

    # Zoom range: lambda_true +/- some window
    zoom_lo, zoom_hi = max(0.5, LAMBDA_TRUE - 8), LAMBDA_TRUE + 8
    mask = (lambda_values >= zoom_lo) & (lambda_values <= zoom_hi)

    # Top: zoomed loss
    # Original with basis coloring
    for j in range(len(orig_changes)):
        start = orig_changes[j]
        end = orig_changes[j + 1] if j + 1 < len(orig_changes) else len(lambda_values)
        seg_mask = mask[start:end]
        if seg_mask.any():
            color_idx = basis_to_color_orig[basis_orig[start]] % 20
            ax3.plot(lambda_values[start:end][seg_mask], L_orig[start:end][seg_mask],
                     color=cmap(color_idx), linewidth=1.5, alpha=0.6)

    for n_agg in n_agg_values:
        ax3.plot(lambda_values[mask], L_agg_dict[n_agg][mask],
                 color=agg_colors[n_agg], linewidth=2.5, linestyle=agg_styles[n_agg],
                 alpha=0.9, label=f'State Agg (M={n_agg})')

    for idx in orig_changes[1:]:
        if zoom_lo <= lambda_values[idx] <= zoom_hi:
            ax3.axvline(lambda_values[idx], color='gray', alpha=0.2, linewidth=0.5)

    ax3.axvline(LAMBDA_TRUE, color='red', linestyle='--', linewidth=2, label=f'$\\lambda_{{true}}$')
    ax3.plot(LAMBDA_TRUE, L_orig[idx_true], 'r*', markersize=15, zorder=5)
    ax3.set_xlim(zoom_lo, zoom_hi)
    ax3.set_ylabel('$L_{aug}(\\lambda)$', fontsize=13)
    ax3.set_title(f'Zoomed near $\\lambda_{{true}}$: Aug-Lag ($\\rho$={RHO:.0f}), state aggregation is smoother', fontsize=13)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)

    # Bottom: zoomed gradient
    ax4.plot(lambda_values[mask], dL_orig[mask], color='gray', linewidth=1.0, alpha=0.5,
             label=f'DFL gradient')
    for n_agg in n_agg_values:
        dL_agg = np.gradient(L_agg_dict[n_agg], lambda_values)
        ax4.plot(lambda_values[mask], dL_agg[mask],
                 color=agg_colors[n_agg], linewidth=2.0, linestyle=agg_styles[n_agg],
                 alpha=0.9, label=f'State Agg (M={n_agg})')

    for idx in orig_changes[1:]:
        if zoom_lo <= lambda_values[idx] <= zoom_hi:
            ax4.axvline(lambda_values[idx], color='gray', alpha=0.15, linewidth=0.5)
    ax4.axvline(LAMBDA_TRUE, color='red', linestyle='--', linewidth=2)
    ax4.axhline(0, color='black', linewidth=0.5)
    ax4.set_xlim(zoom_lo, zoom_hi)
    ax4.set_xlabel('$\\lambda_{pred}$', fontsize=13)
    ax4.set_ylabel('$\\partial L / \\partial \\lambda$', fontsize=11)
    ax4.set_title('Zoomed gradient: fewer jumps with state aggregation', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mdp_lp/plot_results/state_agg_landscape_zoomed.png', dpi=150, bbox_inches='tight')
    plt.savefig('mdp_lp/plot_results/state_agg_landscape_zoomed.pdf', bbox_inches='tight')
    print("Saved to mdp_lp/plot_results/state_agg_landscape_zoomed.{png,pdf}")


if __name__ == "__main__":
    main()
