"""
Robustness comparison: original LP vs state aggregation (random & trained W).

Three panels:
  Top:    Regret = V*(λ_true) - V_true(π(λ))
  Middle: Gradient dV/dλ  (with zero-gradient proportion)
  Bottom: Policy distance ||y(λ) - y(λ_true)||₁

Usage:
    python -m mdp_lp.plot_robustness_comparison                # both random & trained
    python -m mdp_lp.plot_robustness_comparison --mode random   # random W only
    python -m mdp_lp.plot_robustness_comparison --mode trained  # trained W only
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from mdp_lp.elasticity_model import LambdaPredictor
from mdp_lp.inventory_elasticity import create_inventory_from_lambda
from mdp_lp.solvers.dense_lp_solver import DenseLPSolver
from mdp_lp.solvers.differentiable_lp import DifferentiableLPSolver
from mdp_lp.soft_state_aggregation import SoftAggDense
from mdp_lp.train_elasticity import eval_y_on_true_mdp

# ── MDP parameters ──
B_MAX, S_MAX, Q_MAX = 5, 50, 50
FOC, VOC, HC, BC, BETA, PRICE = 0, 0.5, 0.1, 0, 0.95, 10.0
LAMBDA_TRUE = 10.0
N_STATES = B_MAX + S_MAX + 1
N_ACTIONS = Q_MAX + 1


# ═══════════════════════════════════════════════════════════════
# Core computation
# ═══════════════════════════════════════════════════════════════

def build_true_mdp():
    """Build the true MDP and initial distribution at λ_true."""
    lam = torch.tensor([LAMBDA_TRUE], dtype=torch.float32)
    p = torch.tensor([PRICE], dtype=torch.float32)
    mdp = create_inventory_from_lambda(lam, B_MAX, S_MAX, Q_MAX,
                                       FOC, VOC, HC, BC, BETA, p)
    S = mdp.P.shape[1]
    gamma = mdp.build_gamma(1, S, torch.device("cpu"))
    gamma = torch.clamp(gamma, min=1e-8)
    gamma = gamma / gamma.sum(dim=1, keepdim=True)
    return mdp, gamma


def build_and_solve(lam_val, solver, mdp_true, gamma_true, agg=None, tau=0.5):
    """Build MDP at lam_val, optionally aggregate, solve, evaluate on true MDP."""
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

    v_eval, y_true_occ = eval_y_on_true_mdp(mdp_true, y_orig, gamma_true)
    return v_eval.sum().item(), y_true_occ.detach()


def sweep_lambda(lambda_values, solver, mdp_true, gamma_true, agg=None, label=""):
    """Sweep over lambda values, return (V_array, y_list)."""
    V, ys = [], []
    for i, lam in enumerate(lambda_values):
        if (i + 1) % 100 == 0:
            print(f"  [{label}] {i+1}/{len(lambda_values)}")
        v, y = build_and_solve(lam, solver, mdp_true, gamma_true, agg=agg)
        V.append(v)
        ys.append(y)
    return np.array(V), ys


# ═══════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════

def train_agg_dfl(n_clusters, n_epochs=50, pretrain_epochs=100, lr=1e-2,
                  aug_lag_rho=1000000.0, seed=0):
    """Train SoftAggDense via DFL, return trained module."""
    torch.manual_seed(seed)
    device = torch.device("cpu")

    # Use the SAME λ_true and price as the evaluation
    lambda_true = torch.tensor([LAMBDA_TRUE], dtype=torch.float32)
    p = torch.tensor([PRICE], dtype=torch.float32)
    feats = torch.stack([p, torch.zeros(1)], dim=1)

    pred_model = LambdaPredictor(hidden_dim=32, use_p_related=False, use_price_only=True)
    agg = SoftAggDense(n_states=N_STATES, n_actions=N_ACTIONS,
                       n_clusters=n_clusters, r_embed=8)
    params = list(pred_model.parameters()) + list(agg.parameters())
    optimizer = optim.Adam(params, lr=lr)
    pred_loss_fn = nn.MSELoss()

    # Pretrain predictor to output λ_true
    print(f"  [Pretrain M={n_clusters}] {pretrain_epochs} epochs...")
    for epoch in range(pretrain_epochs):
        loss = pred_loss_fn(pred_model(feats), lambda_true)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
        optimizer.step()

    # DFL fine-tuning
    print(f"  [DFL M={n_clusters}] {n_epochs} epochs...")
    for epoch in range(n_epochs):
        lambda_pred = torch.clamp(pred_model(feats), min=1.0)
        mdp = create_inventory_from_lambda(lambda_pred, B_MAX, S_MAX, Q_MAX,
                                           FOC, VOC, HC, BC, BETA, p)
        S = mdp.P.shape[1]
        gamma = mdp.build_gamma(1, S, device)
        gamma = torch.clamp(gamma, min=1e-8)
        gamma = gamma / gamma.sum(dim=1, keepdim=True)

        P_agg, r_agg, gamma_agg, agg_info = agg(mdp.P, mdp.r, gamma, tau=0.5)
        H_agg = mdp.build_H(P_agg)
        y_agg_star = DifferentiableLPSolver.apply(H_agg, r_agg.reshape(1, -1), gamma_agg)
        y_star = SoftAggDense.disaggregate_y(y_agg_star, agg_info["W"], N_ACTIONS)

        with torch.no_grad():
            mdp_t = create_inventory_from_lambda(lambda_true, B_MAX, S_MAX, Q_MAX,
                                                 FOC, VOC, HC, BC, BETA, p)
            r_true = mdp_t.r.reshape(1, -1)
            H_true = mdp_t.build_H(mdp_t.P)

        violation = gamma - (H_true @ y_star.unsqueeze(-1)).squeeze(-1)
        v_star = (r_true * y_star).sum(dim=1)
        total_loss = -v_star.mean() + aug_lag_rho * violation.pow(2).mean()

        optimizer.zero_grad(); total_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}, v*={v_star.item():.1f}, "
                  f"aug_lag={aug_lag_rho * violation.pow(2).mean().item():.2f}")

    agg.eval()
    return agg


# ═══════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════

def plot_results(lambda_values, methods, V_star, output_prefix):
    """
    Plot 3-panel comparison.

    methods: list of dicts with keys:
        label, V, ys, color, ls, alpha
    """
    idx_true = np.argmin(np.abs(lambda_values - LAMBDA_TRUE))

    # Compute derived metrics
    for m in methods:
        m['regret'] = V_star - m['V']
        m['dV'] = np.gradient(m['V'], lambda_values)
        y_ref = m['ys'][idx_true]
        m['pdist'] = np.array([torch.abs(y - y_ref).sum().item() for y in m['ys']])

    # Zero-gradient threshold: 10% of max mean |dV/dλ| across agg methods
    agg_grads = [np.abs(m['dV']).mean() for m in methods if m.get('is_agg')]
    grad_threshold = 0.1 * max(agg_grads) if agg_grads else 1.0

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 13),
                                         gridspec_kw={'height_ratios': [3, 2, 2]})

    # ── Panel 1: Regret ──
    for m in methods:
        ax1.plot(lambda_values, m['regret'], lw=2.5, color=m['color'],
                 ls=m['ls'], label=m['label'], alpha=m['alpha'])

    ax1.axvline(LAMBDA_TRUE, color='red', ls='--', lw=2,
                label=f'$\\lambda_{{true}} = {LAMBDA_TRUE}$')
    ax1.axhline(0, color='black', lw=0.5)
    threshold_2pct = 0.02 * abs(V_star)
    ax1.axhline(threshold_2pct, color='gray', ls=':', lw=1, alpha=0.5)
    ax1.text(lambda_values[-2], threshold_2pct + 2, f'2% of $V^*$',
             fontsize=10, color='gray', ha='right', va='bottom')

    ax1.set_ylabel('Regret  $V^*(\\lambda_{true}) - V_{true}(\\pi(\\lambda))$', fontsize=13)
    ax1.set_title(
        'Original LP is "overly precise": near-zero regret even with wrong $\\lambda$\n'
        'Aggregation makes the model coarser $\\Rightarrow$ '
        'regret grows faster with parameter error', fontsize=13)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(lambda_values[0], lambda_values[-1])

    # ── Panel 2: Gradient dV/dλ ──
    for m in methods:
        zg = 100 * (np.abs(m['dV']) < grad_threshold).mean()
        ax2.plot(lambda_values, m['dV'], lw=2.5, color=m['color'],
                 ls=m['ls'], label=f'{m["label"]}  [near-zero: {zg:.0f}%]',
                 alpha=m['alpha'])

    ax2.axvline(LAMBDA_TRUE, color='red', ls='--', lw=2)
    ax2.axhline(0, color='black', lw=0.5)
    ax2.axhline(grad_threshold, color='gray', ls=':', lw=1, alpha=0.4)
    ax2.axhline(-grad_threshold, color='gray', ls=':', lw=1, alpha=0.4)

    ax2.set_ylabel('$dV_{true}/d\\lambda$', fontsize=13)
    ax2.set_title(
        'DFL gradient signal: original LP has near-zero $dV/d\\lambda$ '
        '(no learning signal)\n'
        'Aggregation amplifies gradient and points toward $\\lambda_{true}$',
        fontsize=13)
    ax2.legend(fontsize=9, loc='lower left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(lambda_values[0], lambda_values[-1])

    # ── Panel 3: Policy distance ──
    for m in methods:
        ax3.plot(lambda_values, m['pdist'], lw=2.5, color=m['color'],
                 ls=m['ls'], label=m['label'], alpha=m['alpha'])

    ax3.axvline(LAMBDA_TRUE, color='red', ls='--', lw=2)
    ax3.set_xlabel('$\\lambda_{pred}$', fontsize=13)
    ax3.set_ylabel('Policy distance\n$\\|y(\\lambda) - y(\\lambda_{true})\\|_1$', fontsize=12)
    ax3.set_title(
        'Mechanism: original LP finds similar policies for different $\\lambda$ (robust)\n'
        'Aggregated LP policies diverge faster (sensitive to parameter error)',
        fontsize=12)
    ax3.legend(fontsize=10, loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(lambda_values[0], lambda_values[-1])

    plt.tight_layout()
    plt.savefig(f'{output_prefix}.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{output_prefix}.pdf', bbox_inches='tight')
    print(f"\nSaved to {output_prefix}.{{png,pdf}}")

    # ── Summary stats ──
    print(f"\n{'Method':<35} {'Near-optimal':<16} {'Zero-grad %':<12} {'Mean |dV/dλ|':<12}")
    print("-" * 75)
    for m in methods:
        good = lambda_values[m['regret'] < threshold_2pct]
        width = f"Δλ={good[-1]-good[0]:.1f}" if len(good) > 1 else "0"
        zg = 100 * (np.abs(m['dV']) < grad_threshold).mean()
        print(f"{m['label']:<35} {width:<16} {zg:<12.0f} {np.abs(m['dV']).mean():<12.2f}")


def plot_gradient_comparison(lambda_values, methods, output_prefix):
    """
    Plot random W and trained W gradients (dV/dλ) together in a single figure.

    Groups by cluster size M so that each M gets its own subplot row,
    with random (solid) and trained (dashed) overlaid.
    """
    # Separate methods by type
    original = [m for m in methods if not m.get('is_agg')]
    random_methods = [m for m in methods if 'Random' in m.get('label', '')]
    trained_methods = [m for m in methods if 'Trained' in m.get('label', '')]

    if not random_methods and not trained_methods:
        print("No aggregation methods to compare.")
        return

    # Compute dV if not already computed
    for m in methods:
        if 'dV' not in m:
            m['dV'] = np.gradient(m['V'], lambda_values)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot original LP gradient as baseline
    for m in original:
        ax.plot(lambda_values, m['dV'], lw=2, color=m['color'],
                ls=m['ls'], label=m['label'], alpha=0.5)

    # Plot random W gradients (solid)
    for m in random_methods:
        ax.plot(lambda_values, m['dV'], lw=2.5, color=m['color'],
                ls='-', label=m['label'], alpha=0.9)

    # Plot trained W gradients (dashed)
    for m in trained_methods:
        ax.plot(lambda_values, m['dV'], lw=2.5, color=m['color'],
                ls='--', label=m['label'], alpha=0.9)

    ax.axvline(LAMBDA_TRUE, color='red', ls='--', lw=2,
               label=f'$\\lambda_{{true}} = {LAMBDA_TRUE}$')
    ax.axhline(0, color='black', lw=0.5)

    ax.set_xlabel('$\\lambda_{pred}$', fontsize=14)
    ax.set_ylabel('$dV_{true}/d\\lambda$', fontsize=14)
    ax.set_title(
        'Gradient comparison: Random W (solid) vs Trained W (dashed)\n'
        'Same color = same cluster size $M$', fontsize=13)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lambda_values[0], lambda_values[-1])

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_gradient_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{output_prefix}_gradient_comparison.pdf', bbox_inches='tight')
    print(f"Saved gradient comparison to {output_prefix}_gradient_comparison.{{png,pdf}}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Robustness comparison plot")
    parser.add_argument('--mode', choices=['random', 'trained', 'both'],
                        default='both', help='Which aggregation to use')
    parser.add_argument('--n-points', type=int, default=300,
                        help='Number of lambda values to sweep')
    parser.add_argument('--clusters', type=int, nargs='+', default=[10, 20, 30],
                        help='Aggregation cluster sizes')
    args = parser.parse_args()

    mdp_true, gamma_true = build_true_mdp()
    solver = DenseLPSolver()
    lambda_values = np.linspace(1.0, 40.0, args.n_points)

    agg_colors = {10: '#2196F3', 20: '#4CAF50', 30: '#FF9800',
                  40: '#9C27B0', 50: '#F44336'}

    # ── Build list of methods ──
    methods = []

    # Original LP (always included)
    print("Computing Original LP...")
    V, ys = sweep_lambda(lambda_values, solver, mdp_true, gamma_true, label="Original LP")
    methods.append(dict(label=f'Original LP (S={N_STATES})', V=V, ys=ys,
                        color='#616161', ls='-', alpha=0.7, is_agg=False))

    # Random W
    if args.mode in ('random', 'both'):
        torch.manual_seed(42)
        for n in args.clusters:
            agg = SoftAggDense(n_states=N_STATES, n_actions=N_ACTIONS,
                               n_clusters=n, r_embed=8)
            label = f'Random W (M={n})' if args.mode == 'both' else f'State Agg (M={n})'
            print(f"Computing {label}...")
            V, ys = sweep_lambda(lambda_values, solver, mdp_true, gamma_true,
                                 agg=agg, label=label)
            methods.append(dict(label=label, V=V, ys=ys,
                                color=agg_colors.get(n, '#607D8B'), ls='-',
                                alpha=0.9, is_agg=True))

    # Trained W
    if args.mode in ('trained', 'both'):
        for n in args.clusters:
            print(f"Training aggregation (M={n})...")
            agg = train_agg_dfl(n)
            label = f'Trained W (M={n})' if args.mode == 'both' else f'State Agg (M={n})'
            print(f"Computing {label}...")
            V, ys = sweep_lambda(lambda_values, solver, mdp_true, gamma_true,
                                 agg=agg, label=label)
            methods.append(dict(label=label, V=V, ys=ys,
                                color=agg_colors.get(n, '#607D8B'), ls='--',
                                alpha=0.9, is_agg=True))

    # ── Reference value ──
    idx_true = np.argmin(np.abs(lambda_values - LAMBDA_TRUE))
    V_star = methods[0]['V'][idx_true]  # Original LP at λ_true
    print(f"\nV*(λ_true) = {V_star:.2f}")

    # ── Plot ──
    plot_results(lambda_values, methods, V_star,
                 'mdp_lp/plot_results/robustness_comparison')

    # ── Gradient comparison: random W vs trained W ──
    if args.mode == 'both':
        plot_gradient_comparison(lambda_values, methods,
                                'mdp_lp/plot_results/robustness_comparison')


if __name__ == "__main__":
    main()
