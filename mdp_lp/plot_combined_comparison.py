import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from mdp_lp.elasticity_model import ElasticityModel, LambdaPredictor
from mdp_lp.inventory_elasticity import create_inventory_from_lambda
from mdp_lp.solvers.dense_lp_solver import DenseLPSolver
from mdp_lp.solvers.differentiable_lp import DifferentiableLPSolver
from mdp_lp.soft_state_aggregation import SoftAggDense
from mdp_lp.train_elasticity import eval_y_on_true_mdp

# ── MDP parameters ──
B_MAX, S_MAX, Q_MAX = 5, 50, 30
FOC, VOC, HC, BC, BETA, PRICE = 0, 0.5, 0.1, 0, 0.95, 10.0
LAMBDA_TRUE = 10.0
RHO = 10000.0
N_STATES = B_MAX + S_MAX + 1   # 56
N_ACTIONS = Q_MAX + 1           # 31


# ═══════════════════════════════════════════════════════════════
# Build true MDP (constants for evaluation)
# ═══════════════════════════════════════════════════════════════

def build_true_mdp():
    """Build the true MDP at λ_true. Returns r_true, H_true, gamma_true (all detached)."""
    lam = torch.tensor([LAMBDA_TRUE], dtype=torch.float32)
    p = torch.tensor([PRICE], dtype=torch.float32)
    mdp = create_inventory_from_lambda(lam, B_MAX, S_MAX, Q_MAX,
                                       FOC, VOC, HC, BC, BETA, p)
    S = mdp.P.shape[1]
    gamma = mdp.build_gamma(1, S, torch.device("cpu"))
    gamma = torch.clamp(gamma, min=1e-8)
    gamma = gamma / gamma.sum(dim=1, keepdim=True)
    r_true = mdp.r.reshape(1, -1).detach()
    H_true = mdp.build_H(mdp.P).detach()
    return r_true, H_true, gamma.detach(), mdp


# ═══════════════════════════════════════════════════════════════
# Compute V_true, V_lag, y*, and their gradients at a single λ
# ═══════════════════════════════════════════════════════════════

def compute_at_lambda(lam_val, mdp_true, r_true, H_true, gamma_true,
                      agg=None, tau=0.5):
    """
    Compute V_true, V_lag, dV_true/dλ, dV_lag/dλ, and y* at a single λ.

    V_true: evaluate the POLICY induced by y*(λ) on the true MDP
            (extract π from y*, compute true occupancy, then r_true^T y_true).
            This guarantees V_true(λ_true) is the maximum.
    V_lag:  augmented Lagrangian loss = -r_true^T y*(λ) + ρ||γ - H_true y*||²

    Uses autograd through DifferentiableLPSolver for exact gradients.
    """
    lam = torch.tensor([lam_val], dtype=torch.float32, requires_grad=True)
    p = torch.tensor([PRICE], dtype=torch.float32)

    mdp = create_inventory_from_lambda(lam, B_MAX, S_MAX, Q_MAX,
                                       FOC, VOC, HC, BC, BETA, p)
    S = mdp.P.shape[1]
    gamma = mdp.build_gamma(1, S, torch.device("cpu"))
    gamma = torch.clamp(gamma, min=1e-8)
    gamma = gamma / gamma.sum(dim=1, keepdim=True)

    if agg is not None:
        P_agg, r_agg, gamma_agg, info = agg(mdp.P, mdp.r, gamma, tau=tau)
        H_agg = mdp.build_H(P_agg)
        r_flat_agg = r_agg.reshape(1, -1)
        y_agg = DifferentiableLPSolver.apply(H_agg, r_flat_agg, gamma_agg)
        y_star = SoftAggDense.disaggregate_y(y_agg, info["W"], N_ACTIONS)
    else:
        H = mdp.build_H(mdp.P)
        r_flat = mdp.r.reshape(1, -1)
        y_star = DifferentiableLPSolver.apply(H, r_flat, gamma)

    # V_true: evaluate the policy π(y*) on the true MDP
    _, y_true_occ = eval_y_on_true_mdp(mdp_true, y_star, gamma_true)
    V_true = (r_true * y_star).sum()

    # V_lag: augmented Lagrangian loss = -r_true^T y* + ρ||γ - H_true y*||²
    violation = gamma_true - (H_true @ y_star.unsqueeze(-1)).squeeze(-1)
    penalty = RHO * violation.pow(2).sum()
    V_lag = -V_true + penalty

    # Compute dV_true/dλ
    dV_true = torch.autograd.grad(V_true, lam, retain_graph=True)[0].item()

    # Compute dV_lag/dλ  (gradient of the loss)
    dV_lag = torch.autograd.grad(V_lag, lam)[0].item()

    return {
        'V_true': V_true.item(),
        'V_lag': V_lag.item(),
        'dV_true': dV_true,
        'dV_lag': dV_lag,
        'y': y_true_occ.detach().squeeze(0),
    }


# ═══════════════════════════════════════════════════════════════
# Sweep over lambda values
# ═══════════════════════════════════════════════════════════════

def sweep_lambda(lambda_values, mdp_true, r_true, H_true, gamma_true,
                 agg=None, label=""):
    """Sweep over λ, return dict of arrays."""
    n = len(lambda_values)
    V_true = np.zeros(n)
    V_lag = np.zeros(n)
    dV_true = np.zeros(n)
    dV_lag = np.zeros(n)
    ys = []

    for i, lam in enumerate(lambda_values):
        if (i + 1) % 50 == 0:
            print(f"  [{label}] {i+1}/{n}")
        res = compute_at_lambda(lam, mdp_true, r_true, H_true, gamma_true,
                                agg=agg)
        V_true[i] = res['V_true']
        V_lag[i] = res['V_lag']
        dV_true[i] = res['dV_true']
        dV_lag[i] = res['dV_lag']
        ys.append(res['y'])

    return {
        'V_true': V_true,
        'V_lag': V_lag,
        'dV_true': dV_true,
        'dV_lag': dV_lag,
        'ys': ys,
    }


# ═══════════════════════════════════════════════════════════════
# DFL Training for soft aggregation
# ═══════════════════════════════════════════════════════════════

def train_soft_agg_dfl(n_clusters, n_samples=50, n_epochs=50, pretrain_epochs=60,
                       lr=1e-2, aug_lag_rho=100000.0, loss_weight=0.1, seed=0,
                       save_dir='mdp_lp/plot_results'):
    """
    Train LambdaPredictor + SoftAggDense (embedding E) via DFL.

    Steps:
      1. Build dataset: prices -> lambda_true via ElasticityModel
      2. Pretrain LambdaPredictor on MSE(lambda_pred, lambda_true)
      3. DFL fine-tune: pred_model + agg jointly
      4. Save learned embedding E

    Returns:
        agg: trained SoftAggDense module (with learned E)
    """
    torch.manual_seed(seed)
    device = torch.device("cpu")

    # ── Step 1: Build dataset ──
    gt = ElasticityModel(beta_0=3.5, beta_1=-1.0, beta_2=0.0, sigma_epsilon=0.5)
    p = torch.exp(torch.randn(n_samples) * 0.6 + np.log(5.0))
    lambda_true = gt(p, None)
    feats = torch.stack([p, torch.zeros(n_samples)], dim=1)  # [price, noise_placeholder]

    print(f"  Dataset: {n_samples} samples, "
          f"lambda_true in [{lambda_true.min():.1f}, {lambda_true.max():.1f}], "
          f"mean={lambda_true.mean():.1f}")

    # ── Step 2: Init models ──
    pred_model = LambdaPredictor(hidden_dim=32, use_p_related=False, use_price_only=True)
    agg = SoftAggDense(n_states=N_STATES, n_actions=N_ACTIONS,
                       n_clusters=n_clusters, r_embed=8)
    params = list(pred_model.parameters()) + list(agg.parameters())
    optimizer = optim.Adam(params, lr=lr)
    pred_loss_fn = nn.MSELoss()

    # ── Step 3: Pretrain predictor ──
    print(f"  [Pretrain M={n_clusters}] {pretrain_epochs} epochs...")
    for epoch in range(pretrain_epochs):
        lambda_pred = pred_model(feats)
        loss = pred_loss_fn(lambda_pred, lambda_true)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{pretrain_epochs}, MSE: {loss.item():.4f}")

    # ── Step 4: DFL fine-tuning ──
    print(f"  [DFL M={n_clusters}] {n_epochs} epochs...")
    for epoch in range(n_epochs):
        lambda_pred = pred_model(feats)
        lambda_pred_pos = torch.clamp(lambda_pred, min=1.0)

        mdp = create_inventory_from_lambda(
            lambda_pred_pos, B_MAX, S_MAX, Q_MAX,
            FOC, VOC, HC, BC, BETA, p)
        S = mdp.P.shape[1]
        batch = lambda_pred.shape[0]
        gamma = mdp.build_gamma(batch, S, device)
        gamma = torch.clamp(gamma, min=1e-8)
        gamma = gamma / gamma.sum(dim=1, keepdim=True)

        P_agg, r_agg, gamma_agg, agg_info = agg(mdp.P, mdp.r, gamma, tau=0.5)
        H_agg = mdp.build_H(P_agg)
        r_flat_agg = r_agg.reshape(batch, -1)
        y_agg_star = DifferentiableLPSolver.apply(H_agg, r_flat_agg, gamma_agg)
        y_star = SoftAggDense.disaggregate_y(y_agg_star, agg_info["W"], N_ACTIONS)

        with torch.no_grad():
            mdp_true_batch = create_inventory_from_lambda(
                lambda_true, B_MAX, S_MAX, Q_MAX,
                FOC, VOC, HC, BC, BETA, p)
            r_true_batch = mdp_true_batch.r.reshape(batch, -1)
            H_true_batch = mdp_true_batch.build_H(mdp_true_batch.P)

        violation = gamma - (H_true_batch @ y_star.unsqueeze(-1)).squeeze(-1)
        v_star = (r_true_batch * y_star).sum(dim=1)
        aug_lag = aug_lag_rho * violation.pow(2).mean()
        pred_loss = pred_loss_fn(lambda_pred, lambda_true)
        total_loss = -v_star.mean() + aug_lag + loss_weight * pred_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}, v*={v_star.mean().item():.1f}, "
                  f"aug_lag={aug_lag.item():.2f}, pred_loss={pred_loss.item():.4f}")

    # ── Step 5: Save learned embedding E ──
    agg.eval()
    save_path = os.path.join(save_dir, f'trained_agg_M{n_clusters}.pt')
    torch.save(agg.state_dict(), save_path)
    print(f"  Saved trained agg (M={n_clusters}) to {save_path}")

    return agg


# ═══════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════

def plot_all(lambda_values, methods, output_dir='mdp_lp/plot_results'):
    """Generate separate figures for each metric."""
    idx_true = np.argmin(np.abs(lambda_values - LAMBDA_TRUE))

    # Compute policy distance for each method
    for m in methods:
        y_ref = m['ys'][idx_true]
        m['pdist'] = np.array([torch.abs(y - y_ref).sum().item() for y in m['ys']])

    n_methods = len(methods)

    # ── Figure 1: V_true  (one subplot per method for own y-scale) ──
    fig, axes = plt.subplots(n_methods, 1, figsize=(12, 4 * n_methods), sharex=True)
    if n_methods == 1:
        axes = [axes]
    for ax, m in zip(axes, methods):
        ax.plot(lambda_values, m['V_true'], lw=0.8, color=m['color'],
                marker='.', markersize=1.5, ls='-', label=m['label'], alpha=0.9)
        ax.axvline(LAMBDA_TRUE, color='red', ls='--', lw=1.5,
                   label=f'$\\lambda_{{true}} = {LAMBDA_TRUE}$')
        ax.set_ylabel('$V_{true}(\\pi(\\lambda))$', fontsize=12)
        ax.set_title(m['label'], fontsize=12)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('$\\lambda_{pred}$', fontsize=13)
    axes[0].set_title(
        f'$V_{{true}}(\\lambda) = r_{{true}}^T y^*(\\lambda)$\n{methods[0]["label"]}',
        fontsize=12)
    axes[-1].set_xlim(lambda_values[0], lambda_values[-1])
    plt.tight_layout()
    plt.savefig(f'{output_dir}/combined_V_true.pdf', bbox_inches='tight')
    plt.close()

    # ── Figure 2: V_lag  (one subplot per method) ──
    fig, axes = plt.subplots(n_methods, 1, figsize=(12, 4 * n_methods), sharex=True)
    if n_methods == 1:
        axes = [axes]
    for ax, m in zip(axes, methods):
        ax.plot(lambda_values, m['V_lag'], lw=0.8, color=m['color'],
                marker='.', markersize=1.5, ls='-', label=m['label'], alpha=0.9)
        ax.axvline(LAMBDA_TRUE, color='red', ls='--', lw=1.5,
                   label=f'$\\lambda_{{true}} = {LAMBDA_TRUE}$')
        ax.set_ylabel('$L_{aug}(\\lambda)$', fontsize=12)
        ax.set_title(m['label'], fontsize=12)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
    axes[0].set_title(
        f'$L_{{aug}} = -r_{{true}}^T y^* + \\rho\\|\\gamma_{{true}} - H_{{true}} y^*\\|^2$'
        f'  ($\\rho={RHO:.0f}$)\n{methods[0]["label"]}', fontsize=12)
    axes[-1].set_xlabel('$\\lambda_{pred}$', fontsize=13)
    axes[-1].set_xlim(lambda_values[0], lambda_values[-1])
    plt.tight_layout()
    plt.savefig(f'{output_dir}/combined_V_lag.pdf', bbox_inches='tight')
    plt.close()

    # ── Figure 3: Policy distance  (one subplot per method) ──
    fig, axes = plt.subplots(n_methods, 1, figsize=(12, 4 * n_methods), sharex=True)
    if n_methods == 1:
        axes = [axes]
    for ax, m in zip(axes, methods):
        ax.plot(lambda_values, m['pdist'], lw=0.8, color=m['color'],
                marker='.', markersize=1.5, ls='-', label=m['label'], alpha=0.9)
        ax.axvline(LAMBDA_TRUE, color='red', ls='--', lw=1.5,
                   label=f'$\\lambda_{{true}} = {LAMBDA_TRUE}$')
        ax.set_ylabel('$\\|y^*(\\lambda) - y^*(\\lambda_{true})\\|_1$', fontsize=12)
        ax.set_title(m['label'], fontsize=12)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
    axes[0].set_title(f'Policy distance from oracle\n{methods[0]["label"]}', fontsize=12)
    axes[-1].set_xlabel('$\\lambda_{pred}$', fontsize=13)
    axes[-1].set_xlim(lambda_values[0], lambda_values[-1])
    plt.tight_layout()
    plt.savefig(f'{output_dir}/combined_policy_distance.pdf', bbox_inches='tight')
    plt.close()

    # ── Figure 4: dV_true/dλ  (one subplot per method) ──
    fig, axes = plt.subplots(n_methods, 1, figsize=(12, 4 * n_methods), sharex=True)
    if n_methods == 1:
        axes = [axes]
    for ax, m in zip(axes, methods):
        ax.plot(lambda_values, m['dV_true'], lw=0.8, color=m['color'],
                marker='.', markersize=1.5, ls='-', label=m['label'], alpha=0.9)
        ax.axvline(LAMBDA_TRUE, color='red', ls='--', lw=1.5,
                   label=f'$\\lambda_{{true}} = {LAMBDA_TRUE}$')
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylabel('$dV_{true}/d\\lambda$', fontsize=12)
        ax.set_title(m['label'], fontsize=12)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
    axes[0].set_title(
        f'$dV_{{true}}/d\\lambda$ (analytical via LP sensitivity)\n{methods[0]["label"]}',
        fontsize=12)
    axes[-1].set_xlabel('$\\lambda_{pred}$', fontsize=13)
    axes[-1].set_xlim(lambda_values[0], lambda_values[-1])
    plt.tight_layout()
    plt.savefig(f'{output_dir}/combined_dV_true.pdf', bbox_inches='tight')
    plt.close()

    # ── Figure 5: dV_lag/dλ  (one subplot per method) ──
    fig, axes = plt.subplots(n_methods, 1, figsize=(12, 4 * n_methods), sharex=True)
    if n_methods == 1:
        axes = [axes]
    for ax, m in zip(axes, methods):
        ax.plot(lambda_values, m['dV_lag'], lw=0.8, color=m['color'],
                marker='.', markersize=1.5, ls='-', label=m['label'], alpha=0.9)
        ax.axvline(LAMBDA_TRUE, color='red', ls='--', lw=1.5,
                   label=f'$\\lambda_{{true}} = {LAMBDA_TRUE}$')
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylabel('$dL_{aug}/d\\lambda$', fontsize=12)
        ax.set_title(m['label'], fontsize=12)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
    axes[0].set_title(
        f'$dL_{{aug}}/d\\lambda$ ($\\rho={RHO:.0f}$, analytical via LP sensitivity)'
        f'\n{methods[0]["label"]}', fontsize=12)
    axes[-1].set_xlabel('$\\lambda_{pred}$', fontsize=13)
    axes[-1].set_xlim(lambda_values[0], lambda_values[-1])
    plt.tight_layout()
    plt.savefig(f'{output_dir}/combined_dV_lag.pdf', bbox_inches='tight')
    plt.close()

    print(f"\nSaved 5 figures to {output_dir}/combined_*.pdf")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Combined comparison: original LP vs soft state aggregation")
    parser.add_argument('--n-points', type=int, default=400,
                        help='Number of lambda values to sweep')
    parser.add_argument('--clusters', type=int, nargs='+', default=[10, 20, 30],
                        help='Aggregation cluster sizes')
    parser.add_argument('--lambda-max', type=float, default=20.0,
                        help='Max lambda value')
    args = parser.parse_args()

    r_true, H_true, gamma_true, mdp_true = build_true_mdp()
    lambda_values = np.linspace(0.5, args.lambda_max, args.n_points)

    agg_colors = {10: '#2196F3', 20: '#4CAF50', 30: '#FF9800',
                  40: '#9C27B0', 50: '#F44336'}
    methods = []

    # Original LP
    print("Computing Original LP...")
    data = sweep_lambda(lambda_values, mdp_true, r_true, H_true, gamma_true,
                        label="Original LP")
    methods.append(dict(
        label=f'Original LP (S={N_STATES})',
        color='#616161', ls='-', alpha=0.7,
        **data,
    ))

    # State aggregation with random W
    torch.manual_seed(42)
    for n_clust in args.clusters:
        agg = SoftAggDense(n_states=N_STATES, n_actions=N_ACTIONS,
                           n_clusters=n_clust, r_embed=8)
        agg.eval()
        label = f'State Agg (M={n_clust})'
        print(f"Computing {label}...")
        data = sweep_lambda(lambda_values, mdp_true, r_true, H_true, gamma_true,
                            agg=agg, label=label)
        methods.append(dict(
            label=label,
            color=agg_colors.get(n_clust, '#607D8B'), ls='-', alpha=0.9,
            **data,
        ))

    plot_all(lambda_values, methods)


if __name__ == "__main__":
    main()
