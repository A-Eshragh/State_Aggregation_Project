"""
Demonstrate: original LP is "overly precise" → robust to parameter error (flat regret),
while state aggregation makes the model coarser → parameter deviations cause steeper
performance degradation.

Two panels:
  Top:    Regret = V*(λ_true) - V_true(π(λ))  for original vs aggregated
  Bottom: Policy distance ||y(λ) - y(λ_true)||₁  showing the mechanism

Usage:
    python -m mdp_lp.plot_robustness_comparison
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from mdp_lp.inventory_elasticity import create_inventory_from_lambda
from mdp_lp.solvers.dense_lp_solver import DenseLPSolver
from mdp_lp.soft_state_aggregation import SoftAggDense
from mdp_lp.train_elasticity import eval_y_on_true_mdp

# ── MDP parameters ──
B_MAX, S_MAX, Q_MAX = 5, 50, 30
FOC, VOC, HC, BC, BETA, PRICE = 0, 0.5, 0.1, 0, 0.95, 10.0
LAMBDA_TRUE = 10.0
N_STATES = B_MAX + S_MAX + 1
N_ACTIONS = Q_MAX + 1


def build_and_solve(lam_val, solver, mdp_true, gamma_true, agg=None, tau=0.5):
    """Build MDP from lambda, optionally aggregate, solve, return value and occupancy."""
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
        r_flat = r_agg.reshape(1, -1)
        y_agg, _, _ = solver.solve(H_agg, r_flat, gamma_agg)
        y_orig = SoftAggDense.disaggregate_y(y_agg, info["W"], N_ACTIONS)
    else:
        H = mdp.build_H(mdp.P)
        r_flat = mdp.r.reshape(1, -1)
        y_orig, _, _ = solver.solve(H, r_flat, gamma)

    # Evaluate the implied policy on the TRUE MDP
    v_eval, y_true_occ = eval_y_on_true_mdp(mdp_true, y_orig, gamma_true)
    value = v_eval.sum().item()
    return value, y_true_occ.detach()


def main():
    # Build true MDP
    lam_true_t = torch.tensor([LAMBDA_TRUE], dtype=torch.float32)
    p_t = torch.tensor([PRICE], dtype=torch.float32)
    mdp_true = create_inventory_from_lambda(lam_true_t, B_MAX, S_MAX, Q_MAX,
                                             FOC, VOC, HC, BC, BETA, p_t)
    S_true = mdp_true.P.shape[1]
    gamma_true = mdp_true.build_gamma(1, S_true, torch.device("cpu"))
    gamma_true = torch.clamp(gamma_true, min=1e-8)
    gamma_true = gamma_true / gamma_true.sum(dim=1, keepdim=True)
    solver = DenseLPSolver()

    torch.manual_seed(42)
    n_agg_values = [10, 20, 30]
    agg_modules = {n: SoftAggDense(n_states=N_STATES, n_actions=N_ACTIONS,
                                    n_clusters=n, r_embed=8)
                   for n in n_agg_values}

    # Lambda sweep
    lambda_values = np.linspace(1.0, 40.0, 300)

    # ── Compute values and occupancies ──
    # Original LP
    print("Computing original LP...")
    V_orig = []
    y_orig_list = []
    for i, lam in enumerate(lambda_values):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(lambda_values)}")
        v, y = build_and_solve(lam, solver, mdp_true, gamma_true, agg=None)
        V_orig.append(v)
        y_orig_list.append(y)
    V_orig = np.array(V_orig)

    # Aggregated LPs
    V_agg = {}
    y_agg_list = {}
    for n in n_agg_values:
        print(f"\nComputing aggregated LP (M={n})...")
        vals, ys = [], []
        for i, lam in enumerate(lambda_values):
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(lambda_values)}")
            v, y = build_and_solve(lam, solver, mdp_true, gamma_true,
                                   agg=agg_modules[n])
            vals.append(v)
            ys.append(y)
        V_agg[n] = np.array(vals)
        y_agg_list[n] = ys

    # True optimal value (original LP at λ_true)
    idx_true = np.argmin(np.abs(lambda_values - LAMBDA_TRUE))
    V_star = V_orig[idx_true]
    y_star_orig = y_orig_list[idx_true]

    print(f"\nV*(λ_true) = {V_star:.2f}")

    # ── Regret and policy distance ──
    regret_orig = V_star - V_orig
    regret_agg = {n: V_star - V_agg[n] for n in n_agg_values}

    # Policy distance: L1 norm of occupancy difference
    def policy_dist(y_list, y_ref):
        dists = []
        for y in y_list:
            dists.append(torch.abs(y - y_ref).sum().item())
        return np.array(dists)

    pdist_orig = policy_dist(y_orig_list, y_star_orig)

    # For aggregated, compare to the aggregated method's own occupancy at λ_true
    pdist_agg = {}
    for n in n_agg_values:
        y_ref_agg = y_agg_list[n][idx_true]
        pdist_agg[n] = policy_dist(y_agg_list[n], y_ref_agg)

    # ── Figure ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9),
                                    gridspec_kw={'height_ratios': [3, 2]})

    colors = {'orig': '#616161', 10: '#2196F3', 20: '#4CAF50', 30: '#FF9800'}

    # ── Panel 1: Regret ──
    ax1.plot(lambda_values, regret_orig, lw=2.5, color=colors['orig'],
             label=f'Original LP (S={N_STATES})', alpha=0.7)
    for n in n_agg_values:
        ax1.plot(lambda_values, regret_agg[n], lw=2.5, color=colors[n],
                 label=f'State Agg (M={n})', alpha=0.9)

    ax1.axvline(LAMBDA_TRUE, color='red', ls='--', lw=2,
                label=f'$\\lambda_{{true}} = {LAMBDA_TRUE}$')
    ax1.axhline(0, color='black', lw=0.5)

    # Shade the "near-optimal" band (regret < 5% of V*)
    threshold = 0.02 * abs(V_star)
    ax1.axhline(threshold, color='gray', ls=':', lw=1, alpha=0.5)
    ax1.text(38, threshold + 2, f'2% of $V^*$', fontsize=10, color='gray',
             ha='right', va='bottom')

    # Find and annotate the width of near-optimal region for original
    orig_good = lambda_values[regret_orig < threshold]
    if len(orig_good) > 0:
        lo, hi = orig_good[0], orig_good[-1]
        ax1.annotate('', xy=(hi, -8), xytext=(lo, -8),
                     arrowprops=dict(arrowstyle='<->', color=colors['orig'], lw=2))
        ax1.text((lo + hi) / 2, -15, f'Original: Δλ≈{hi - lo:.0f}',
                 ha='center', fontsize=10, color=colors['orig'], fontweight='bold')

    # For M=10
    agg10_good = lambda_values[regret_agg[10] < threshold]
    if len(agg10_good) > 0:
        lo10, hi10 = agg10_good[0], agg10_good[-1]
        ax1.annotate('', xy=(hi10, -20), xytext=(lo10, -20),
                     arrowprops=dict(arrowstyle='<->', color=colors[10], lw=2))
        ax1.text((lo10 + hi10) / 2, -27, f'M=10: Δλ≈{hi10 - lo10:.0f}',
                 ha='center', fontsize=10, color=colors[10], fontweight='bold')

    ax1.set_ylabel('Regret  $V^*(\\lambda_{true}) - V_{true}(\\pi(\\lambda))$', fontsize=13)
    ax1.set_title(
        'Original LP is "overly precise": near-zero regret even with wrong $\\lambda$\n'
        'Aggregation makes the model coarser $\\Rightarrow$ '
        'regret grows faster with parameter error',
        fontsize=13)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, 40)

    # ── Panel 2: Policy distance ──
    ax2.plot(lambda_values, pdist_orig, lw=2.5, color=colors['orig'],
             label=f'Original LP', alpha=0.7)
    for n in n_agg_values:
        ax2.plot(lambda_values, pdist_agg[n], lw=2.5, color=colors[n],
                 label=f'State Agg (M={n})', alpha=0.9)

    ax2.axvline(LAMBDA_TRUE, color='red', ls='--', lw=2)

    ax2.set_xlabel('$\\lambda_{pred}$', fontsize=13)
    ax2.set_ylabel('Policy distance\n$\\|y(\\lambda) - y(\\lambda_{true})\\|_1$', fontsize=12)
    ax2.set_title(
        'Mechanism: original LP finds similar policies for different $\\lambda$ (robust)\n'
        'Aggregated LP policies diverge faster (sensitive to parameter error)',
        fontsize=12)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, 40)

    plt.tight_layout()
    plt.savefig('mdp_lp/plot_results/robustness_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig('mdp_lp/plot_results/robustness_comparison.pdf', bbox_inches='tight')
    print("\nSaved to mdp_lp/plot_results/robustness_comparison.{png,pdf}")

    # Print summary stats
    print(f"\n{'Method':<25} {'Near-optimal width (2% threshold)':<35} {'Regret at λ=5':<15} {'Regret at λ=20':<15}")
    print("-" * 90)
    for label, regret in [('Original LP', regret_orig)] + [(f'State Agg M={n}', regret_agg[n]) for n in n_agg_values]:
        good = lambda_values[regret < threshold]
        width = f"{good[-1] - good[0]:.1f}" if len(good) > 1 else "0"
        idx5 = np.argmin(np.abs(lambda_values - 5))
        idx20 = np.argmin(np.abs(lambda_values - 20))
        print(f"{label:<25} Δλ = {width:<30} {regret[idx5]:<15.1f} {regret[idx20]:<15.1f}")


if __name__ == "__main__":
    main()
