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
B_MAX, S_MAX, Q_MAX = 5, 30, 30
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
    gt = ElasticityModel(beta_0=3.5, beta_1=-1.0, beta_2=0.09, sigma_epsilon=0.3)
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


def compute_loss_grad(lam_val, r_true, H_true, gamma_true,
                      agg=None, rho=0.0, tau=0.5):
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

    V = (r_true * y_star).sum()
    loss = -V
    if rho > 0.0:
        violation = gamma_true - (H_true @ y_star.unsqueeze(-1)).squeeze(-1)
        loss = loss + rho * violation.pow(2).sum()

    grad = torch.autograd.grad(loss, lam)[0].item()
    return loss.item(), grad


def sweep_loss_grad(lambda_values, r_true, H_true, gamma_true,
                    agg=None, rho=0.0, label=""):
    """Sweep λ and return (losses, grads) arrays for a given formulation."""
    n = len(lambda_values)
    losses = np.zeros(n)
    grads = np.zeros(n)
    for i, lam in enumerate(lambda_values):
        if (i + 1) % 50 == 0:
            print(f"  [{label}] {i+1}/{n}")
        ell, g = compute_loss_grad(lam, r_true, H_true, gamma_true,
                                   agg=agg, rho=rho)
        losses[i] = ell
        grads[i] = g
    return losses, grads


def _longest_flat_run(lam_vals, g, rel_thresh=0.05, min_frac=0.25):
    """Return (x0, x1) of the longest contiguous run where |g| < rel_thresh·max|g|.

    Returns None if the longest such run spans less than `min_frac` of the grid.
    Used to highlight "vanishing gradient" plateaus on the gradient row.
    """
    g_abs = np.abs(np.asarray(g))
    g_max = g_abs.max()
    if g_max == 0:
        return None
    mask = g_abs < rel_thresh * g_max
    n = len(mask)
    best_start, best_end, best_len = 0, 0, 0
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i > best_len:
                best_len = j - i
                best_start, best_end = i, j - 1
            i = j
        else:
            i += 1
    if best_len < min_frac * n:
        return None
    return float(lam_vals[best_start]), float(lam_vals[best_end])


def load_trained_agg(ckpt_path, n_clusters_fallback=10, r_embed_fallback=8,
                     seed=1):
    """Load a SoftAggDense module saved by train_elasticity.py.

    Supports two checkpoint formats:
      1. Payload dict: {'state_dict', 'n_states', 'n_actions',
         'n_clusters', 'r_embed', ...}  (format written by
         train_elasticity.py to `agg_state_final.pt`).
      2. Raw state_dict (legacy format used by
         `trained_agg_M{n_clusters}.pt`).
    """
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        agg = SoftAggDense(
            n_states=int(ckpt.get('n_states', N_STATES)),
            n_actions=int(ckpt.get('n_actions', N_ACTIONS)),
            n_clusters=int(ckpt.get('n_clusters', n_clusters_fallback)),
            r_embed=int(ckpt.get('r_embed', r_embed_fallback)),
        )
        agg.load_state_dict(ckpt['state_dict'])
    else:
        torch.manual_seed(seed)
        agg = SoftAggDense(n_states=N_STATES, n_actions=N_ACTIONS,
                           n_clusters=n_clusters_fallback,
                           r_embed=r_embed_fallback)
        agg.load_state_dict(ckpt)
    agg.eval()
    return agg


def plot_landscape_comparison(lambda_values, r_true, H_true, gamma_true,
                              n_clusters=10, rho_small=10.0, seed=1,
                              output_dir='mdp_lp/plot_results',
                              agg_ckpt=None):
    L_orig, g_orig = sweep_loss_grad(
        lambda_values, r_true, H_true, gamma_true,
        rho=0.0, label="Original")

    print(f"[3-col] Aug. Lagrangian (rho={rho_small}) ...")
    L_lag, g_lag = sweep_loss_grad(
        lambda_values, r_true, H_true, gamma_true,
        rho=rho_small, label=f"Lag rho={rho_small:g}")

    torch.manual_seed(seed)
    agg = SoftAggDense(n_states=N_STATES, n_actions=N_ACTIONS,
                       n_clusters=n_clusters, r_embed=8)
    agg_src = f'random init (seed={seed})'

    candidate_paths = []
    if agg_ckpt is not None:
        candidate_paths.append(agg_ckpt)
    candidate_paths.append(os.path.join(output_dir, f'trained_agg_M{n_clusters}.pt'))

    for path in candidate_paths:
        if not os.path.exists(path):
            continue
        try:
            print(f"Loading trained aggregation model from {path}")
            agg = load_trained_agg(path, n_clusters_fallback=n_clusters, seed=seed)
            agg_src = f'trained ({path})'
            break
        except RuntimeError as e:
            print(f"  [3-col] Skipping {path} (schema mismatch: {e.__class__.__name__})")
    agg.eval()
    print(f"[3-col] Soft aggregation + penalty (M={n_clusters}, rho={rho_small}, {agg_src}) ...")
    L_agg, g_agg = sweep_loss_grad(
        lambda_values, r_true, H_true, gamma_true,
        agg=agg, rho=rho_small, label=f"SoftAgg+pen M={n_clusters}")

    losses = [L_orig, L_lag, L_agg]
    def _moving_avg(y, window):
        y = np.asarray(y, dtype=float)
        kernel = np.ones(window) / window
        padded = np.pad(y, (window // 2, window // 2), mode='edge')
        return np.convolve(padded, kernel, mode='valid')[:len(y)]

    smooth_win = max(5, len(lambda_values) // 40)   # ~10 points for 400-sample sweep
    grads = [np.gradient(_moving_avg(L, smooth_win), lambda_values) for L in losses]

    lambda_stars = [float(lambda_values[int(np.argmin(L))]) for L in losses]
    shifts = [ls - LAMBDA_TRUE for ls in lambda_stars]

    prev_rc = plt.rcParams.copy()
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 13,
        'axes.linewidth': 0.9,
        'legend.fontsize': 9.5,
        'legend.frameon': True,
        'legend.framealpha': 0.92,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    fig, axes = plt.subplots(
        2, 3, figsize=(14.8, 7.4),
        gridspec_kw={'hspace': 0.34, 'wspace': 0.30})

    col_titles = [
        r'(a) Original loss:  $\mathcal{L}=-r_{\mathrm{true}}^{\!\top}\,y^{*}(\lambda)$',
        rf'(b) Aug. Lagrangian  ($\rho={rho_small:g}$)',
        rf'(c) Aug. Lagrangian + Soft agg.  ($\rho={rho_small:g},\,M={n_clusters}$)',
    ]
    col_colors = ['#455A64', '#E65100', '#1565C0']   # charcoal / orange / blue
    lam_true_color = '#D32F2F'          # red dashed (matches both_methods_with_penalty.pdf)
    minimiser_color = '#2E7D32'         # green minimiser dot (matches reference)
    bias_band_color = '#D32F2F'         # red bias band |λ* − λ_true|
    plateau_color = '#9E9E9E'           # gray vanishing-gradient plateau

    # ───────── Row 1 — loss landscapes ─────────
    for ax, L, ttl, c, lam_s in zip(axes[0], losses, col_titles, col_colors,
                                    lambda_stars):
        ax.plot(lambda_values, L, lw=1.8, color=c, solid_capstyle='round')
        ax.axvline(LAMBDA_TRUE, color=lam_true_color, ls='--', lw=1.3,
                   alpha=0.9, label=rf'$\lambda_{{\mathrm{{true}}}}={LAMBDA_TRUE:g}$')
        ax.plot(lam_s, float(np.min(L)), marker='o', mfc=minimiser_color,
                mec='white', ms=9, mew=1.3, ls='', zorder=5,
                label=rf'$\lambda^{{*}}={lam_s:.2f}$')
        ax.set_title(ttl, pad=8)
        ax.grid(True, alpha=0.35, ls=':', lw=0.7)
        ax.legend(loc='best')
    axes[0, 0].set_ylabel(r'Loss  $\mathcal{L}(\lambda)$')

    # ───────── Row 2 — gradients with diagnostic overlays ─────────
    dlam = float(lambda_values[1] - lambda_values[0])
    for ax, g, c, lam_s, shift in zip(axes[1], grads, col_colors,
                                       lambda_stars, shifts):
        ax.plot(lambda_values, g, lw=1.8, color=c, solid_capstyle='round')

        # (a)-type pathology: shade vanishing-gradient plateau if one exists.
        flat = _longest_flat_run(lambda_values, g, rel_thresh=0.10, min_frac=0.35)
        if flat is not None:
            ax.axvspan(flat[0], flat[1], color=plateau_color, alpha=0.22, lw=0,
                       label='vanishing gradient')

        # (a)/(b)-type pathology: shade the bias band between λ* and λ_true.
        if abs(shift) >= dlam:
            x0, x1 = sorted([lam_s, LAMBDA_TRUE])
            ax.axvspan(x0, x1, color=bias_band_color, alpha=0.18, lw=0,
                       label=rf'$|\Delta\lambda|={abs(shift):.2f}$')

        ax.axvline(LAMBDA_TRUE, color=lam_true_color, ls='--', lw=1.3,
                   alpha=0.9, label=rf'$\lambda_{{\mathrm{{true}}}}={LAMBDA_TRUE:g}$')
        ax.axhline(0.0, color='black', lw=1.0, alpha=0.80)

        # Where gradient descent actually stops.
        g_at_star = float(np.interp(lam_s, lambda_values, g))
        ax.plot(lam_s, g_at_star, marker='o', mfc=minimiser_color, mec='white',
                ms=9, mew=1.3, ls='', zorder=5,
                label=rf'$\lambda^{{*}}={lam_s:.2f}$')

        ax.set_xlabel(r'$\lambda_{\mathrm{pred}}$')
        ax.grid(True, alpha=0.35, ls=':', lw=0.7)
        ax.legend(loc='best')
    axes[1, 0].set_ylabel(r'Gradient  $d\mathcal{L}/d\lambda$')

    for ax in axes.flat:
        ax.set_xlim(lambda_values[0], lambda_values[-1])

    fig.suptitle(
        r'Gradient descent stops where $d\mathcal{L}/d\lambda\!=\!0$:  '
        r'(a) vanishing signal $\Rightarrow$ no descent;  '
        r'(b) biased stationary point $\Rightarrow$ wrong $\lambda^{*}$;  '
        r'(c) soft aggregation $\Rightarrow$ $\lambda^{*}\!\approx\!\lambda_{\mathrm{true}}$.',
        fontsize=11.5, y=1.02)

    out_path = os.path.join(output_dir, 'landscape_comparison_3col.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    plt.rcParams.update(prev_rc)

    print(f"\nSaved 2×3 landscape comparison to {out_path}")
    print(f"  λ*  (a) original          = {lambda_stars[0]:.3f}   shift {shifts[0]:+.3f}")
    print(f"  λ*  (b) aug-lag ρ={rho_small:<6g}= {lambda_stars[1]:.3f}   shift {shifts[1]:+.3f}")
    print(f"  λ*  (c) soft agg M={n_clusters:<4}   = {lambda_stars[2]:.3f}   shift {shifts[2]:+.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Combined comparison: original LP vs soft state aggregation")
    parser.add_argument('--n-points', type=int, default=400,
                        help='Number of lambda values to sweep')
    parser.add_argument('--clusters', type=int, nargs='+', default=[10, 20, 30],
                        help='Aggregation cluster sizes')
    parser.add_argument('--lambda-max', type=float, default=20.0,
                        help='Max lambda value')
    parser.add_argument('--plot-3col-only', action='store_true',
                        help='Only generate the 2×3 landscape-vs-gradient figure '
                             '(skips the original per-metric combined_*.pdf sweeps)')
    parser.add_argument('--cluster-3col', type=int, default=10,
                        help='Cluster count M used for the soft-agg panel of the 2×3 figure')
    parser.add_argument('--rho-3col', type=float, default=10.0,
                        help='Small ρ used for the augmented-Lagrangian panel of the 2×3 figure')
    parser.add_argument('--seed-3col', type=int, default=1,
                        help='Seed for the random SoftAggDense init in the 2×3 figure')
    parser.add_argument('--agg-ckpt', type=str, default=None,
                        help='Path to a SoftAggDense checkpoint saved by '
                             'train_elasticity.py (agg_state_final.pt). If '
                             'provided, this overrides the default '
                             'trained_agg_M{M}.pt lookup.')
    args = parser.parse_args()

    r_true, H_true, gamma_true, mdp_true = build_true_mdp()
    lambda_values = np.linspace(0.5, args.lambda_max, args.n_points)

    if args.plot_3col_only:
        plot_landscape_comparison(
            lambda_values, r_true, H_true, gamma_true,
            n_clusters=args.cluster_3col,
            rho_small=args.rho_3col,
            seed=args.seed_3col,
            agg_ckpt=args.agg_ckpt)
        return

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
    torch.manual_seed(1)
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

    plot_landscape_comparison(
        lambda_values, r_true, H_true, gamma_true,
        n_clusters=args.cluster_3col,
        rho_small=args.rho_3col,
        seed=args.seed_3col,
        agg_ckpt=args.agg_ckpt)


if __name__ == "__main__":
    main()
