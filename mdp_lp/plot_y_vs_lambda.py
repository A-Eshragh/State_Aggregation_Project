"""
Train pred_model + soft aggregation using the same DFL pipeline as train_elasticity.py,
then decompose dỹ/dλ = (dW/dλ)·y_agg* + W·(dy_agg*/dλ).

Uses LP sensitivity formula (y_B = H_B^{-1}γ ⟹ dy_B/dλ = H_B^{-1}(dγ/dλ - dH_B/dλ · y_B))
+ small central finite differences for aggregation parameters.
This is exact for the LP part and highly accurate (δ=1e-4) for the aggregation part.

Usage:
    conda run -n bilevel python -m mdp_lp.plot_y_vs_lambda
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

from mdp_lp.elasticity_model import ElasticityModel, LambdaPredictor
from mdp_lp.inventory_elasticity import create_inventory_from_lambda
from mdp_lp.solvers.dense_lp_solver import DenseLPSolver
from mdp_lp.solvers.differentiable_lp import DifferentiableLPSolver
from mdp_lp.soft_state_aggregation import SoftAggDense
from mdp_lp.train_elasticity import eval_y_on_true_mdp

# ── MDP parameters (same as robustness_comparison / exp_b high_margin) ──
B_MAX, S_MAX, Q_MAX = 5, 50, 30
FOC, VOC, HC, BC, BETA, PRICE = 0, 0.5, 0.1, 0, 0.95, 10.0
LAMBDA_TRUE = 10.0
N_STATES = B_MAX + S_MAX + 1   # 56
N_ACTIONS = Q_MAX + 1           # 31
M_SHOW = 10
SA = N_STATES * N_ACTIONS
MA = M_SHOW * N_ACTIONS


# ═══════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════

def train_dfl(n_epochs=10, n_samples=1, lr=1e-2, pretrain_epochs=60,
              loss_weight=0.0, aug_lag_rho=10000.0):
    """Train LambdaPredictor + SoftAggDense using DFL, return trained modules."""
    device = torch.device("cpu")

    torch.manual_seed(0)
    gt = ElasticityModel(beta_0=3.5, beta_1=-1.0, beta_2=0.0, sigma_epsilon=0.5)
    p = torch.exp(torch.randn(n_samples) * 0.6 + np.log(5.0))
    lambda_true = gt(p, None)
    noise = torch.randn(n_samples) * 0.1
    feats = torch.stack([p, noise], dim=1)

    pred_model = LambdaPredictor(hidden_dim=32, use_p_related=False, use_price_only=True)
    agg = SoftAggDense(n_states=N_STATES, n_actions=N_ACTIONS,
                       n_clusters=M_SHOW, r_embed=8)
    params = list(pred_model.parameters()) + list(agg.parameters())
    optimizer = optim.Adam(params, lr=lr)
    pred_loss_fn = nn.MSELoss()

    print(f"[Pretrain] {pretrain_epochs} epochs...")
    for epoch in range(pretrain_epochs):
        lambda_pred = pred_model(feats)
        loss = pred_loss_fn(lambda_pred, lambda_true)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{pretrain_epochs}, MSE: {loss.item():.4f}")

    print(f"\n[DFL] {n_epochs} epochs...")
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
            mdp_true = create_inventory_from_lambda(
                lambda_true, B_MAX, S_MAX, Q_MAX,
                FOC, VOC, HC, BC, BETA, p)
            r_true = mdp_true.r.reshape(batch, -1)
            H_true = mdp_true.build_H(mdp_true.P)

        violation = gamma - (H_true @ y_star.unsqueeze(-1)).squeeze(-1)
        aug_lag = aug_lag_rho * violation.pow(2).mean()
        v_star = (r_true * y_star).sum(dim=1)
        total_loss = -v_star.mean() + aug_lag + loss_weight * pred_loss_fn(lambda_pred, lambda_true)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}, v*={v_star.mean().item():.1f}, "
                  f"aug_lag={aug_lag.item():.2f}")

    pred_model.eval()
    agg.eval()
    return pred_model, agg


# ═══════════════════════════════════════════════════════════════
# LP sensitivity + finite-difference decomposition (FAST)
# ═══════════════════════════════════════════════════════════════

def _build_mdp_gamma(lam_batch):
    """Build MDP and gamma from a (1,) lambda tensor. Preserves autograd."""
    p = torch.tensor([PRICE], dtype=torch.float32)
    mdp = create_inventory_from_lambda(lam_batch, B_MAX, S_MAX, Q_MAX,
                                       FOC, VOC, HC, BC, BETA, p)
    S = mdp.P.shape[1]
    gamma = mdp.build_gamma(1, S, torch.device("cpu"))
    gamma = torch.clamp(gamma, min=1e-8)
    gamma = gamma / gamma.sum(dim=1, keepdim=True)
    return mdp, gamma


def compute_decomposition(lam_val, agg_module, solver):
    """
    dỹ/dλ decomposition: central finite differences for aggregation derivatives
    + LP sensitivity formula. Replaces S×M backward passes with 2 forward passes.
    """
    M, A, S = M_SHOW, N_ACTIONS, N_STATES
    delta = 1e-4

    # ── Step 1: Forward pass at lam (for W, y_agg*, basis) ──
    with torch.no_grad():
        lam_t = torch.tensor([lam_val], dtype=torch.float32)
        mdp, gamma = _build_mdp_gamma(lam_t)
        P_agg, r_agg, gamma_agg, info = agg_module(mdp.P, mdp.r, gamma, tau=0.5)
        H_agg = mdp.build_H(P_agg)
        W = info["W"].squeeze(0)              # (S, M)
        H_agg_sq = H_agg.squeeze(0)           # (M, MA)
        gamma_agg_sq = gamma_agg.squeeze(0)   # (M,)

    # ── Step 2: LP solve ──
    with torch.no_grad():
        r_flat = r_agg.reshape(1, -1)
        y_agg, basis, _ = solver.solve(H_agg, r_flat, gamma_agg)
    y_agg_flat = y_agg.squeeze(0)   # (MA,)
    B = basis.squeeze(0).long()     # (M,)
    y_B = y_agg_flat[B]             # (M,)
    H_B = H_agg_sq[:, B]           # (M, M)

    # ── Step 3: Finite differences at lam±delta (2 forward passes) ──
    with torch.no_grad():
        lam_p = torch.tensor([lam_val + delta], dtype=torch.float32)
        mdp_p, gamma_p = _build_mdp_gamma(lam_p)
        P_agg_p, _, gamma_agg_p, info_p = agg_module(mdp_p.P, mdp_p.r, gamma_p, tau=0.5)
        H_agg_p = mdp_p.build_H(P_agg_p).squeeze(0)
        W_p = info_p["W"].squeeze(0)
        gamma_agg_p_sq = gamma_agg_p.squeeze(0)

        lam_m = torch.tensor([lam_val - delta], dtype=torch.float32)
        mdp_m, gamma_m = _build_mdp_gamma(lam_m)
        P_agg_m, _, gamma_agg_m, info_m = agg_module(mdp_m.P, mdp_m.r, gamma_m, tau=0.5)
        H_agg_m = mdp_m.build_H(P_agg_m).squeeze(0)
        W_m = info_m["W"].squeeze(0)
        gamma_agg_m_sq = gamma_agg_m.squeeze(0)

    dW_dlam = (W_p - W_m) / (2 * delta)
    dgamma_dlam = (gamma_agg_p_sq - gamma_agg_m_sq) / (2 * delta)
    dH_B_dlam = (H_agg_p[:, B] - H_agg_m[:, B]) / (2 * delta)

    # ── Step 4: LP sensitivity ──
    rhs_grad = dgamma_dlam - dH_B_dlam @ y_B
    dy_B_dlam = torch.linalg.solve(H_B, rhs_grad)
    dy_agg_dlam = torch.zeros(M * A)
    dy_agg_dlam[B] = dy_B_dlam

    # ── Decompose dỹ/dλ ──
    y_agg_3d = y_agg_flat.reshape(M, A)
    dy_agg_3d = dy_agg_dlam.reshape(M, A)

    steering = dW_dlam @ y_agg_3d             # (S, A)
    lp_response = W @ dy_agg_3d               # (S, A)
    total = steering + lp_response

    y_tilde = (W @ y_agg_3d).reshape(-1).numpy()

    steering_flat = steering.reshape(-1)
    lp_resp_flat = lp_response.reshape(-1)
    total_flat = total.reshape(-1)

    return {
        'norm_steering': float(steering_flat.abs().sum()),
        'norm_lp_resp': float(lp_resp_flat.abs().sum()),
        'norm_total': float(total_flat.abs().sum()),
        'y_tilde': y_tilde,
        'dy_tilde': total_flat.numpy(),
        'steering_vec': steering_flat.numpy(),
        'lp_resp_vec': lp_resp_flat.numpy(),
    }


def compute_original_gradient(lam_val, solver):
    """dy*/dλ for original LP via finite differences + LP sensitivity."""
    S = N_STATES
    delta = 1e-4

    # ── Forward at lam ──
    with torch.no_grad():
        lam_t = torch.tensor([lam_val], dtype=torch.float32)
        mdp, gamma = _build_mdp_gamma(lam_t)
        H = mdp.build_H(mdp.P).squeeze(0)       # (S, SA)
        gamma_sq = gamma.squeeze(0)              # (S,)

    # ── LP solve ──
    with torch.no_grad():
        r_flat = mdp.r.reshape(1, -1)
        y, basis, _ = solver.solve(H.unsqueeze(0), r_flat, gamma)
    y_flat = y.squeeze(0)         # (SA,)
    B = basis.squeeze(0).long()   # (S,)
    y_B = y_flat[B]               # (S,)
    H_B = H[:, B]                 # (S, S)

    # ── Finite differences for dH[:,B]/dλ (2 forward passes) ──
    with torch.no_grad():
        lam_p = torch.tensor([lam_val + delta], dtype=torch.float32)
        mdp_p, _ = _build_mdp_gamma(lam_p)
        H_p = mdp_p.build_H(mdp_p.P).squeeze(0)

        lam_m = torch.tensor([lam_val - delta], dtype=torch.float32)
        mdp_m, _ = _build_mdp_gamma(lam_m)
        H_m = mdp_m.build_H(mdp_m.P).squeeze(0)

    dH_B_dlam = (H_p[:, B] - H_m[:, B]) / (2 * delta)
    # γ is uniform (independent of λ), so dγ/dλ = 0
    rhs_grad = -dH_B_dlam @ y_B

    dy_B_dlam = torch.linalg.solve(H_B, rhs_grad)
    dy_dlam = torch.zeros(SA)
    dy_dlam[B] = dy_B_dlam

    return float(dy_dlam.abs().sum()), y_flat.numpy(), dy_dlam.numpy()


# ═══════════════════════════════════════════════════════════════
# Analysis and plotting
# ═══════════════════════════════════════════════════════════════

def analyze(agg_module, label, lambda_range=(5.0, 15.0), n_points=100):
    """Run decomposition over a lambda sweep."""
    solver = DenseLPSolver()
    lambda_values = np.linspace(*lambda_range, n_points)

    norm_steering = np.zeros(n_points)
    norm_lp_resp = np.zeros(n_points)
    norm_dy_tilde = np.zeros(n_points)
    norm_dy_orig = np.zeros(n_points)
    y_tilde_all = np.zeros((n_points, SA))
    y_orig_all = np.zeros((n_points, SA))
    dy_tilde_all = np.zeros((n_points, SA))
    dy_orig_all = np.zeros((n_points, SA))
    steering_all = np.zeros((n_points, SA))
    lp_resp_all = np.zeros((n_points, SA))

    print(f"\n[{label}] Computing LP-sensitivity decomposition for {n_points} points...")
    t0 = time.time()

    for i, lam in enumerate(lambda_values):
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{n_points}  ({elapsed:.0f}s)")

        # Aggregated decomposition
        res = compute_decomposition(lam, agg_module, solver)
        norm_steering[i] = res['norm_steering']
        norm_lp_resp[i] = res['norm_lp_resp']
        norm_dy_tilde[i] = res['norm_total']
        y_tilde_all[i] = res['y_tilde']
        dy_tilde_all[i] = res['dy_tilde']
        steering_all[i] = res['steering_vec']
        lp_resp_all[i] = res['lp_resp_vec']

        # Original
        ndo, yo, dyo = compute_original_gradient(lam, solver)
        norm_dy_orig[i] = ndo
        y_orig_all[i] = yo
        dy_orig_all[i] = dyo

    print(f"  Done in {time.time()-t0:.0f}s")

    # Build r_true for V_true computation
    lam_true_t = torch.tensor([LAMBDA_TRUE], dtype=torch.float32)
    p_t = torch.tensor([PRICE], dtype=torch.float32)
    mdp_true = create_inventory_from_lambda(lam_true_t, B_MAX, S_MAX, Q_MAX,
                                             FOC, VOC, HC, BC, BETA, p_t)
    r_true = mdp_true.r.reshape(-1).detach().numpy()  # (SA,)

    # V_true = r_true^T y(λ)
    V_orig = y_orig_all @ r_true
    V_tilde = y_tilde_all @ r_true

    # dV/dλ = r_true^T dy/dλ (exact, using LP sensitivity derivatives)
    dV_orig = dy_orig_all @ r_true
    dV_tilde = dy_tilde_all @ r_true
    dV_steering = steering_all @ r_true
    dV_lp_resp = lp_resp_all @ r_true

    idx_true = np.argmin(np.abs(lambda_values - LAMBDA_TRUE))
    return {
        'lambda_values': lambda_values,
        'norm_steering': norm_steering,
        'norm_lp_resp': norm_lp_resp,
        'norm_dy_tilde': norm_dy_tilde,
        'norm_dy_orig': norm_dy_orig,
        'dist_orig': np.abs(y_orig_all - y_orig_all[idx_true]).sum(axis=1),
        'dist_tilde': np.abs(y_tilde_all - y_tilde_all[idx_true]).sum(axis=1),
        'threshold': 0.1 * norm_dy_orig.mean(),
        'V_orig': V_orig,
        'V_tilde': V_tilde,
        'dV_orig': dV_orig,
        'dV_tilde': dV_tilde,
        'dV_steering': dV_steering,
        'dV_lp_resp': dV_lp_resp,
    }


def make_plot(data_random, data_trained):
    fig, axes = plt.subplots(3, 2, figsize=(20, 16))
    colors = {'orig': '#616161', 'steering': '#E91E63', 'lp_resp': '#2196F3',
              'total': '#9C27B0'}

    for col, (data, title) in enumerate([
        (data_random, 'Random W (untrained)'),
        (data_trained, 'Trained W (after DFL)'),
    ]):
        lv = data['lambda_values']

        ax = axes[0, col]
        ax.plot(lv, data['norm_steering'], color=colors['steering'], lw=2.5,
                alpha=0.9, label='Smooth steering $\\|dW/d\\lambda \\cdot y_{agg}^*\\|_1$')
        ax.plot(lv, data['norm_lp_resp'], color=colors['lp_resp'], lw=2.5,
                alpha=0.9, label='LP response $\\|W \\cdot dy_{agg}^*/d\\lambda\\|_1$')
        ax.plot(lv, data['norm_dy_tilde'], color=colors['total'], lw=2, ls='--',
                alpha=0.8, label='Total $\\|d\\tilde{y}/d\\lambda\\|_1$')
        ax.axvline(LAMBDA_TRUE, color='red', ls='--', lw=2)
        ax.set_ylabel('$\\ell_1$ norm', fontsize=12)
        ax.set_title(f'{title}:\ndecomposition of $d\\tilde{{y}}/d\\lambda$', fontsize=13)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(lv[0], lv[-1])

        ax = axes[1, col]
        ax.plot(lv, data['norm_dy_orig'], color=colors['orig'], lw=1.5,
                alpha=0.6, label='Original LP $\\|dy^*/d\\lambda\\|_1$')
        ax.plot(lv, data['norm_dy_tilde'], color=colors['total'], lw=2.5,
                alpha=0.9, label=f'State Agg (M={M_SHOW}) $\\|d\\tilde{{y}}/d\\lambda\\|_1$')
        ax.plot(lv, data['norm_steering'], color=colors['steering'], lw=1.5,
                ls=':', alpha=0.7, label='  of which: smooth steering')
        ax.axvline(LAMBDA_TRUE, color='red', ls='--', lw=2)
        ax.set_ylabel('$\\|dy/d\\lambda\\|_1$', fontsize=12)
        ax.set_title(f'{title}: gradient comparison', fontsize=13)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(lv[0], lv[-1])

        ax = axes[2, col]
        ax.plot(lv, data['dist_orig'], color=colors['orig'], lw=2.5,
                alpha=0.7, label='Original LP')
        ax.plot(lv, data['dist_tilde'], color=colors['total'], lw=2.5,
                alpha=0.9, label=f'State Agg (M={M_SHOW})')
        ax.axvline(LAMBDA_TRUE, color='red', ls='--', lw=2, label='$\\lambda_{true}$')
        ax.set_xlabel('$\\lambda_{pred}$', fontsize=13)
        ax.set_ylabel('$\\|y(\\lambda) - y(\\lambda_{true})\\|_1$', fontsize=12)
        ax.set_title(f'{title}: policy distance from oracle', fontsize=13)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(lv[0], lv[-1])

    plt.tight_layout()
    plt.savefig('mdp_lp/plot_results/y_vs_lambda_decomposition.png', dpi=150,
                bbox_inches='tight')
    plt.savefig('mdp_lp/plot_results/y_vs_lambda_decomposition.pdf',
                bbox_inches='tight')
    print("\nSaved to mdp_lp/plot_results/y_vs_lambda_decomposition.{png,pdf}")


def print_stats(label, data):
    threshold = data['threshold']
    ns, nlr, ndt, ndo = (data['norm_steering'], data['norm_lp_resp'],
                          data['norm_dy_tilde'], data['norm_dy_orig'])
    print(f"\n{'Metric':<50} {label}")
    print("-" * 65)
    print(f"{'Original LP: mean ||dy*/dλ||_1':<50} {ndo.mean():.2f}")
    print(f"{'Original LP: % near-zero gradient':<50} {100*(ndo < threshold).mean():.1f}%")
    print(f"{'State Agg: mean ||dỹ/dλ||_1':<50} {ndt.mean():.2f}")
    print(f"{'  smooth steering contribution':<50} {ns.mean():.2f}")
    print(f"{'  LP response contribution':<50} {nlr.mean():.2f}")
    print(f"{'  steering / total ratio':<50} {100*ns.mean()/(ndt.mean()+1e-10):.1f}%")
    print(f"{'State Agg: % near-zero gradient':<50} {100*(ndt < threshold).mean():.1f}%")


def make_components_plot(data_trained):
    """y_vs_lambda_components: V_true and dV/dλ with decomposition."""
    colors = {'orig': '#616161', 'steering': '#E91E63', 'lp_resp': '#2196F3',
              'total': '#9C27B0'}
    lv = data_trained['lambda_values']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                    gridspec_kw={'height_ratios': [3, 2]})

    # ── Top: V_true(y*(λ)) ──
    ax1.plot(lv, data_trained['V_orig'], color=colors['orig'], lw=2.5, alpha=0.7,
             label=f'Original LP (S={N_STATES})')
    ax1.plot(lv, data_trained['V_tilde'], color=colors['total'], lw=2.5, alpha=0.9,
             label=f'State Agg (M={M_SHOW})')
    ax1.axvline(LAMBDA_TRUE, color='red', ls='--', lw=2,
                label=f'$\\lambda_{{true}} = {LAMBDA_TRUE}$')
    idx_true = np.argmin(np.abs(lv - LAMBDA_TRUE))
    ax1.plot(LAMBDA_TRUE, data_trained['V_orig'][idx_true], 'r*', markersize=15, zorder=5)
    ax1.set_ylabel('$V_{true}(y^*(\\lambda)) = r_{true}^T y^*(\\lambda)$', fontsize=13)
    ax1.set_title('DFL objective vs $\\lambda_{pred}$', fontsize=13)
    ax1.legend(fontsize=10, loc='lower left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(lv[0], lv[-1])

    # ── Bottom: dV/dλ with decomposition ──
    ax2.plot(lv, data_trained['dV_orig'], color=colors['orig'], lw=1.2, alpha=0.6,
             label='Original LP: $dV/d\\lambda$')
    ax2.plot(lv, data_trained['dV_tilde'], color=colors['total'], lw=2.0, alpha=0.9,
             label=f'State Agg total: $dV/d\\lambda$')
    ax2.plot(lv, data_trained['dV_steering'], color=colors['steering'], lw=1.5,
             ls=':', alpha=0.8, label='Smooth steering component')
    ax2.plot(lv, data_trained['dV_lp_resp'], color=colors['lp_resp'], lw=1.5,
             ls=':', alpha=0.8, label='LP response component')
    ax2.axvline(LAMBDA_TRUE, color='red', ls='--', lw=2)
    ax2.axhline(0, color='black', lw=0.5)
    ax2.set_xlabel('$\\lambda_{pred}$', fontsize=13)
    ax2.set_ylabel('$dV_{true}/d\\lambda$  (exact via LP sensitivity)', fontsize=12)
    ax2.set_title(
        'Gradient of DFL objective: decomposed into smooth steering + LP response',
        fontsize=12)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(lv[0], lv[-1])

    plt.tight_layout()
    plt.savefig('mdp_lp/plot_results/y_vs_lambda_components.png', dpi=150,
                bbox_inches='tight')
    plt.savefig('mdp_lp/plot_results/y_vs_lambda_components.pdf',
                bbox_inches='tight')
    print("Saved to mdp_lp/plot_results/y_vs_lambda_components.{png,pdf}")


def make_policy_plot(data_trained):
    """y_vs_lambda_policy: policy distance and its derivative with decomposition."""
    colors = {'orig': '#616161', 'steering': '#E91E63', 'lp_resp': '#2196F3',
              'total': '#9C27B0'}
    lv = data_trained['lambda_values']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                    gridspec_kw={'height_ratios': [3, 2]})

    # ── Top: policy distance ──
    ax1.plot(lv, data_trained['dist_orig'], color=colors['orig'], lw=2.5, alpha=0.7,
             label=f'Original LP (S={N_STATES})')
    ax1.plot(lv, data_trained['dist_tilde'], color=colors['total'], lw=2.5, alpha=0.9,
             label=f'State Agg (M={M_SHOW})')
    ax1.axvline(LAMBDA_TRUE, color='red', ls='--', lw=2,
                label=f'$\\lambda_{{true}}$')
    ax1.set_ylabel('$\\|y^*(\\lambda) - y^*(\\lambda_{true})\\|_1$', fontsize=13)
    ax1.set_title('Policy distance from oracle: aggregation is more sensitive and smoother',
                  fontsize=13)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(lv[0], lv[-1])

    # ── Bottom: ||dy/dλ||_1 with decomposition ──
    ax2.plot(lv, data_trained['norm_dy_orig'], color=colors['orig'], lw=1.5, alpha=0.6,
             label='Original LP: $\\|dy^*/d\\lambda\\|_1$')
    ax2.plot(lv, data_trained['norm_dy_tilde'], color=colors['total'], lw=2.0, alpha=0.9,
             label=f'State Agg total: $\\|d\\tilde{{y}}/d\\lambda\\|_1$')
    ax2.plot(lv, data_trained['norm_steering'], color=colors['steering'], lw=1.5,
             ls=':', alpha=0.8, label='Smooth steering: $\\|dW/d\\lambda \\cdot y_{agg}^*\\|_1$')
    ax2.plot(lv, data_trained['norm_lp_resp'], color=colors['lp_resp'], lw=1.5,
             ls=':', alpha=0.8, label='LP response: $\\|W \\cdot dy_{agg}^*/d\\lambda\\|_1$')
    ax2.axvline(LAMBDA_TRUE, color='red', ls='--', lw=2)
    ax2.set_xlabel('$\\lambda_{pred}$', fontsize=13)
    ax2.set_ylabel('$\\|dy/d\\lambda\\|_1$  (exact)', fontsize=12)
    ax2.set_title(
        'Gradient magnitude: smooth steering provides non-zero baseline; '
        'LP response adds spikes at basis changes',
        fontsize=12)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(lv[0], lv[-1])

    plt.tight_layout()
    plt.savefig('mdp_lp/plot_results/y_vs_lambda_policy.png', dpi=150,
                bbox_inches='tight')
    plt.savefig('mdp_lp/plot_results/y_vs_lambda_policy.pdf',
                bbox_inches='tight')
    print("Saved to mdp_lp/plot_results/y_vs_lambda_policy.{png,pdf}")


def main():
    # ── 1. Random W ──
    print("=" * 60)
    print("PHASE 1: Random W (untrained)")
    print("=" * 60)
    torch.manual_seed(42)
    agg_random = SoftAggDense(n_states=N_STATES, n_actions=N_ACTIONS,
                               n_clusters=M_SHOW, r_embed=8)
    data_random = analyze(agg_random, "Random W")
    print_stats("Random W", data_random)

    # ── 2. Train ──
    print("\n" + "=" * 60)
    print("PHASE 2: Training DFL")
    print("=" * 60)
    _, agg_trained = train_dfl()

    # ── 3. Analyze trained W ──
    print("\n" + "=" * 60)
    print("PHASE 3: Trained W analysis")
    print("=" * 60)
    data_trained = analyze(agg_trained, "Trained W")
    print_stats("Trained W", data_trained)

    # ── 4. Plot all figures ──
    make_plot(data_random, data_trained)
    make_components_plot(data_trained)
    make_policy_plot(data_trained)


if __name__ == "__main__":
    main()
