import argparse
import os
import json
import hashlib
from pathlib import Path
from typing import Optional
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
from mdp_lp.utils import set_seed

# ── MDP parameters ──
B_MAX, S_MAX, Q_MAX = 5, 30, 30
FOC, VOC, HC, BC, BETA, PRICE = 0, 0.5, 0.1, 0, 0.95, 10.0
LAMBDA_TRUE = 10.0
RHO = 10000.0
N_STATES = B_MAX + S_MAX + 1   # 56
N_ACTIONS = Q_MAX + 1           # 31


def _sample_prices_and_lambdas_from_cfg(args_cfg: dict):
    """Recreate the experiment's sampled prices and lambda_true values."""
    set_seed(int(args_cfg["seed"]))
    device = torch.device("cpu")
    n_samples = int(args_cfg.get("generated_n_samples", args_cfg["n_samples"]))
    experiment_type = str(args_cfg["experiment_type"])

    if experiment_type == "a":
        p = torch.ones(n_samples, device=device) * float(args_cfg["p_constant"])
        p_related = None
    elif experiment_type == "b":
        if bool(args_cfg.get("fixed_p", False)):
            p = torch.arange(1, n_samples + 1, dtype=torch.float32, device=device)
        else:
            p = torch.exp(
                torch.randn(n_samples, device=device) * float(args_cfg["p_lognormal_std"])
                + np.log(float(args_cfg["p_lognormal_mean"]))
            )
        p_related = None
    elif experiment_type == "c":
        p = torch.exp(
            torch.randn(n_samples, device=device) * float(args_cfg["p_lognormal_std"])
            + np.log(float(args_cfg["p_lognormal_mean"]))
        )
        p_related = torch.exp(
            torch.randn(n_samples, device=device) * float(args_cfg["p_related_lognormal_std"])
            + np.log(float(args_cfg["p_related_lognormal_mean"]))
        )
    else:
        raise ValueError(f"Unsupported experiment_type={experiment_type!r}")

    gt_elasticity = ElasticityModel(
        beta_0=float(args_cfg["beta_0"]),
        beta_1=float(args_cfg["beta_1"]),
        beta_2=float(args_cfg["beta_2"]),
        sigma_epsilon=float(args_cfg["sigma_epsilon"]),
    )
    lambda_true = gt_elasticity(p, p_related)
    return p.cpu(), None if p_related is None else p_related.cpu(), lambda_true.cpu()


def _split_dataset_indices(n_samples: int, seed: int):
    if n_samples <= 1:
        return [0], [0]
    train_size = n_samples // 2
    perm = torch.randperm(n_samples, generator=torch.Generator().manual_seed(int(seed))).tolist()
    return perm[:train_size], perm[train_size:]


def _load_torch_payload(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _resolve_saved_seed(
    explicit_seed: Optional[int],
    args_cfg: Optional[dict],
    agg_ckpt: Optional[str],
):
    if explicit_seed is not None:
        return int(explicit_seed), "explicit"

    if agg_ckpt is not None:
        ckpt_path = Path(agg_ckpt).expanduser()
        if ckpt_path.exists():
            try:
                payload = _load_torch_payload(str(ckpt_path))
            except Exception:
                payload = None
            if isinstance(payload, dict):
                if "fixed_W_seed" in payload:
                    return int(payload["fixed_W_seed"]), f"{ckpt_path.name}:fixed_W_seed"
                if "seed" in payload:
                    return int(payload["seed"]), f"{ckpt_path.name}:seed"

    if args_cfg is not None:
        if bool(args_cfg.get("agg_fixed_W", False)) and args_cfg.get("agg_fixed_W_seed") is not None:
            return int(args_cfg["agg_fixed_W_seed"]), "results.json:agg_fixed_W_seed"
        if args_cfg.get("seed") is not None:
            return int(args_cfg["seed"]), "results.json:seed"

    return 1, "default"


def _resolve_reference_values(
    args_cfg: dict,
    ref_split: str,
    ref_index: int,
    lambda_ref_override: Optional[float],
    price_ref_override: Optional[float],
):
    p_saved = args_cfg.get("p_values_used")
    p_related_saved = args_cfg.get("p_related_values_used")
    p, p_related, lambda_true = _sample_prices_and_lambdas_from_cfg(args_cfg)
    if p_saved is not None:
        p = torch.as_tensor(p_saved, dtype=torch.float32)
    if p_related_saved is not None:
        p_related = torch.as_tensor(p_related_saved, dtype=torch.float32)

    n_samples = int(p.shape[0])
    train_idx, test_idx = _split_dataset_indices(n_samples, int(args_cfg["seed"]))
    if ref_split == "train":
        split_indices = train_idx
    elif ref_split == "test":
        split_indices = test_idx
    elif ref_split == "all":
        split_indices = list(range(n_samples))
    else:
        raise ValueError(f"Unsupported reference split {ref_split!r}")

    if not split_indices:
        raise ValueError(f"No samples available in split {ref_split!r}")
    if ref_index < 0 or ref_index >= len(split_indices):
        raise IndexError(
            f"reference_index={ref_index} is out of range for split {ref_split!r} "
            f"(size={len(split_indices)})"
        )

    sample_index = split_indices[ref_index]
    inferred_lambda = float(lambda_true[sample_index].item())
    inferred_price = float(p[sample_index].item())
    lambda_ref = inferred_lambda if lambda_ref_override is None else float(lambda_ref_override)
    price_ref = inferred_price if price_ref_override is None else float(price_ref_override)
    p_related_ref = None if p_related is None else float(p_related[sample_index].item())
    source = "explicit" if (lambda_ref_override is not None or price_ref_override is not None) else (
        "reconstructed" if p_saved is None else "saved_prices+reconstructed_lambda"
    )
    return lambda_ref, price_ref, {
        "source": source,
        "seed": int(args_cfg["seed"]),
        "split": ref_split,
        "ref_index": ref_index,
        "sample_index": sample_index,
        "p_related_ref": p_related_ref,
    }


def _apply_experiment_context(args_cfg: dict, lambda_ref: float, price_ref: float):
    """Override module-level problem constants from an experiment config."""
    global B_MAX, S_MAX, Q_MAX
    global FOC, VOC, HC, BC, BETA, PRICE
    global LAMBDA_TRUE, RHO, N_STATES, N_ACTIONS

    B_MAX = int(args_cfg["b_max"])
    S_MAX = int(args_cfg["s_max"])
    Q_MAX = int(args_cfg["q_max"])
    FOC = float(args_cfg["foc"])
    VOC = float(args_cfg["voc"])
    HC = float(args_cfg["hc"])
    BC = float(args_cfg["bc"])
    BETA = float(args_cfg["beta"])
    PRICE = float(price_ref)
    LAMBDA_TRUE = float(lambda_ref)
    RHO = float(args_cfg.get("aug_lag_rho", RHO))
    N_STATES = B_MAX + S_MAX + 1
    N_ACTIONS = Q_MAX + 1


def _current_problem_meta() -> dict:
    return {
        "B_MAX": int(B_MAX),
        "S_MAX": int(S_MAX),
        "Q_MAX": int(Q_MAX),
        "FOC": float(FOC),
        "VOC": float(VOC),
        "HC": float(HC),
        "BC": float(BC),
        "BETA": float(BETA),
        "PRICE": float(PRICE),
        "LAMBDA_TRUE": float(LAMBDA_TRUE),
        "RHO": float(RHO),
        "N_STATES": int(N_STATES),
        "N_ACTIONS": int(N_ACTIONS),
    }


# ═══════════════════════════════════════════════════════════════
# Lightweight on-disk cache (for expensive sweeps)
# ═══════════════════════════════════════════════════════════════

CACHE_VERSION = 3  # bumped: agg now defaults to fixed_W to match diagnose-solver-grad


def _default_plot_results_dir() -> str:
    # Resolve relative to this file so `python -m ...` works from any CWD.
    return os.path.join(os.path.dirname(__file__), "plot_results")


def _sha256_json(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _cache_file(cache_dir: str, tag: str, meta: dict) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    h = _sha256_json(meta)
    return os.path.join(cache_dir, f"{tag}-{h}.npz")


def _stable_cache_file(cache_dir: str, tag: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{tag}.npz")


def _maybe_load_npz(path: str):
    if not os.path.exists(path):
        return None
    try:
        return dict(np.load(path, allow_pickle=True))
    except Exception:
        return None


def _meta_matches(loaded_meta, expected_meta: dict) -> bool:
    if loaded_meta is None:
        return False
    try:
        if hasattr(loaded_meta, "item"):
            loaded_meta = loaded_meta.item()
        return dict(loaded_meta) == expected_meta
    except Exception:
        return False


def cached_sweep_loss_grad(
    *,
    tag: str,
    cache_dir: str,
    recompute_tags: set,
    lambda_values,
    r_true,
    H_true,
    gamma_true,
    agg=None,
    rho: float = 0.0,
    label: str = "",
    extra_meta: dict | None = None,
    sketch_cfg: dict | None = None,
    mdp_true=None,
):
    meta = {
        "cache_version": CACHE_VERSION,
        "tag": tag,
        "n_points": int(len(lambda_values)),
        "lambda_min": float(lambda_values[0]),
        "lambda_max": float(lambda_values[-1]),
        "rho": float(rho),
        "extra": extra_meta or {},
        "sketch": sketch_cfg or {},
    }
    cache_path = _stable_cache_file(cache_dir, tag)

    if tag not in recompute_tags:
        loaded = _maybe_load_npz(cache_path)
        if (
            loaded is not None
            and "losses" in loaded
            and "grads" in loaded
            and _meta_matches(loaded.get("meta"), meta)
        ):
            return np.asarray(loaded["losses"], dtype=float), np.asarray(loaded["grads"], dtype=float)

        # Back-compat: if the old hashed filename exists for this meta, load it and
        # promote to the stable filename for next run.
        legacy_path = _cache_file(cache_dir, tag, meta)
        loaded_legacy = _maybe_load_npz(legacy_path)
        if loaded_legacy is not None and "losses" in loaded_legacy and "grads" in loaded_legacy:
            np.savez_compressed(
                cache_path,
                losses=np.asarray(loaded_legacy["losses"], dtype=float),
                grads=np.asarray(loaded_legacy["grads"], dtype=float),
                meta=meta,
            )
            return np.asarray(loaded_legacy["losses"], dtype=float), np.asarray(loaded_legacy["grads"], dtype=float)

    losses, grads = sweep_loss_grad(
        lambda_values, r_true, H_true, gamma_true,
        agg=agg, rho=rho, label=label, sketch_cfg=sketch_cfg,
        mdp_true=mdp_true)
    np.savez_compressed(cache_path, losses=np.asarray(losses, dtype=float), grads=np.asarray(grads, dtype=float), meta=meta)
    return losses, grads


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
    if output_dir is None:
        output_dir = _default_plot_results_dir()
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


def _sketched_lp_with_cost(H, r, gamma, *,
                            n_sketches=8, sketch_ratio=0.7,
                            sketch_mode="gaussian", sketch_seed=0,
                            beta=0.95, cost_sigma=0.0):
    """Local extension of ``sketched_lp.sketched_lp`` that supports
    cost-side perturbation in addition to constraint-side row sketching.

    For each ensemble member k = 1..K, we solve
        y_k = argmax_{y >= 0}  (r + sigma * eps_k)^T y
              s.t.   (S_k H_aug) y = (S_k gamma_aug)
    and return the ensemble mean. When cost_sigma == 0 this matches
    ``sketched_lp`` exactly (row sketch only). When sketch_ratio == 1.0
    and use_anchor=False, the row sketch reduces to the identity and we
    get cost-side smoothing only (matching ``sketch_smoothing_demo``'s
    cost-sketch curve). The combined setting decorrelates basis hops via
    BOTH a different polytope and a different cost direction per
    ensemble member, which is the regime that produces the demo's smooth
    landscape on this MDP-scale problem.
    """
    import math
    if H.dim() != 3:
        raise ValueError(f"H must be (B, M, M*A); got {tuple(H.shape)}")
    B, M, MA = H.shape
    if r.shape != (B, MA):
        raise ValueError(f"r must be (B, M*A); got {tuple(r.shape)}")
    if gamma.shape != (B, M):
        raise ValueError(f"gamma must be (B, M); got {tuple(gamma.shape)}")

    K = int(n_sketches)
    if K < 1:
        raise ValueError("n_sketches must be >= 1")
    k = max(1, int(round(float(sketch_ratio) * M)))
    k = min(k, M)
    dtype, device = H.dtype, H.device

    use_anchor = k < M
    if use_anchor:
        total_mass = 1.0 / max(1e-6, 1.0 - float(beta))
        anchor_row = torch.ones(B, 1, MA, dtype=dtype, device=device)
        anchor_rhs = torch.full((B, 1), total_mass, dtype=dtype, device=device)
        H_aug = torch.cat([H, anchor_row], dim=1)
        gamma_aug = torch.cat([gamma, anchor_rhs], dim=1)
    else:
        H_aug = H
        gamma_aug = gamma

    gen = torch.Generator(device=device).manual_seed(int(sketch_seed))

    S_list = []
    for _ in range(K):
        if sketch_mode == "gaussian":
            S_rand = (torch.randn(k, M, generator=gen, dtype=dtype,
                                  device=device) / math.sqrt(float(k)))
        elif sketch_mode == "rademacher":
            signs = (torch.randint(0, 2, (k, M), generator=gen,
                                    dtype=torch.int64, device=device).to(dtype)
                     * 2.0 - 1.0)
            S_rand = signs / math.sqrt(float(k))
        elif sketch_mode == "subsample":
            perm = torch.randperm(M, generator=gen, device=device)[:k]
            S_rand = torch.zeros(k, M, dtype=dtype, device=device)
            S_rand[torch.arange(k, device=device), perm] = math.sqrt(
                float(M) / float(k))
        else:
            raise ValueError(f"Unknown sketch mode {sketch_mode!r}")
        if use_anchor:
            S_top = torch.cat(
                [S_rand, torch.zeros(k, 1, dtype=dtype, device=device)], dim=1)
            anchor_pick = torch.zeros(1, M + 1, dtype=dtype, device=device)
            anchor_pick[0, M] = 1.0
            S_m = torch.cat([S_top, anchor_pick], dim=0)
        else:
            S_m = S_rand
        S_list.append(S_m)
    S_stack = torch.stack(S_list, dim=0)
    k_eff = S_stack.shape[1]

    SH = torch.einsum("kij,bjp->kbip", S_stack, H_aug)
    Sg = torch.einsum("kij,bj->kbi", S_stack, gamma_aug)

    KB = K * B
    SH_flat = SH.reshape(KB, k_eff, MA)
    Sg_flat = Sg.reshape(KB, k_eff)

    if float(cost_sigma) > 0.0:
        cost_eps = torch.randn(K, B, MA, generator=gen, dtype=dtype,
                                device=device)
        r_per = r.unsqueeze(0) + float(cost_sigma) * cost_eps  # (K, B, MA)
        r_flat = r_per.reshape(KB, MA).contiguous()
    else:
        r_flat = r.unsqueeze(0).expand(K, B, MA).reshape(KB, MA).contiguous()

    y_flat = DifferentiableLPSolver.apply(SH_flat, r_flat, Sg_flat)
    y_per = y_flat.reshape(K, B, MA)
    return y_per.mean(dim=0)


def compute_loss_grad(lam_val, r_true, H_true, gamma_true,
                      agg=None, rho=0.0, tau=0.5,
                      sketch_cfg=None,
                      mdp_true=None):
    """Loss/grad at a single λ.

    When ``agg`` is None: original-space loss
        L = -r_true^T y* + rho * sum((gamma_true - H_true y*)^2)
    When ``agg`` is not None: aggregated true-projected loss
        L = -r_true_agg^T y_agg + rho * mean((gamma_true_agg - H_true_agg y_agg)^2)
    where r_true_agg, P_true_agg, gamma_true_agg are built from W (the
    membership returned by ``agg``) and ``mdp_true``. This matches the
    formulation used by ``diagnose_solver_gradients`` and the actual DFL
    training loss in ``train_elasticity.py``.
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
        if mdp_true is None:
            raise ValueError(
                "compute_loss_grad: mdp_true must be provided when agg is not None "
                "(needed for the true-projected aggregated loss)"
            )
        P_agg, r_agg, gamma_agg, info = agg(mdp.P, mdp.r, gamma, tau=tau)
        H_agg = mdp.build_H(P_agg)
        r_flat_agg = r_agg.reshape(1, -1)
        if sketch_cfg is not None:
            from mdp_lp.solvers.sketched_lp import sketched_lp
            y_agg = sketched_lp(
                H_agg, r_flat_agg, gamma_agg,
                n_sketches=int(sketch_cfg.get("n_sketches", 4)),
                sketch_ratio=float(sketch_cfg.get("sketch_ratio", 0.7)),
                sketch_mode=str(sketch_cfg.get("sketch_mode", "gaussian")),
                # Hold the sketch noise fixed across the lambda sweep so the
                # landscape is a deterministic curve (not Monte-Carlo noise).
                sketch_seed=int(sketch_cfg.get("sketch_seed", 0)),
                beta=float(BETA),
            )
        else:
            y_agg = DifferentiableLPSolver.apply(H_agg, r_flat_agg, gamma_agg)

        # Project the TRUE MDP into the aggregated space using the SAME W
        # that came from the predicted-MDP aggregator call. Mirrors
        # diagnose_solver_gradients and train_elasticity.py.
        W = info["W"]                                              # (1, S, M)
        mass = W.sum(dim=1).clamp_min(1e-8)                        # (1, M)
        r_true_3d = mdp_true.r                                     # (1, S, A)
        r_true_agg = torch.einsum("bsm,bsa->bma", W, r_true_3d) / mass.unsqueeze(-1)
        right = torch.einsum("bsaj,bjm->bsam", mdp_true.P, W)
        p_hat = torch.einsum("bsm,bsan->bman", W, right)
        P_true_agg = (p_hat / mass[:, :, None, None]).clamp_min(0.0)
        P_true_agg = P_true_agg / P_true_agg.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        gamma_true_agg = torch.einsum("bs,bsm->bm", gamma, W).clamp_min(1e-8)
        gamma_true_agg = gamma_true_agg / gamma_true_agg.sum(dim=1, keepdim=True).clamp_min(1e-8)
        H_true_agg = mdp_true.build_H(P_true_agg)                  # (1, M, M*A)

        v = torch.sum(r_true_agg.reshape(1, -1) * y_agg, dim=1)    # (1,)
        loss = -v.mean()
        if rho > 0.0:
            viol = gamma_true_agg - (H_true_agg @ y_agg.unsqueeze(-1)).squeeze(-1)
            loss = loss + rho * viol.pow(2).mean()
    else:
        H = mdp.build_H(mdp.P)
        r_flat = mdp.r.reshape(1, -1)
        if sketch_cfg is not None:
            cost_sigma = float(sketch_cfg.get("cost_sigma", 0.0))
            if cost_sigma > 0.0:
                # Combined row + cost-side sketching (local extension; does
                # NOT modify sketched_lp.py).
                y_star = _sketched_lp_with_cost(
                    H, r_flat, gamma,
                    n_sketches=int(sketch_cfg.get("n_sketches", 4)),
                    sketch_ratio=float(sketch_cfg.get("sketch_ratio", 0.7)),
                    sketch_mode=str(sketch_cfg.get("sketch_mode", "gaussian")),
                    sketch_seed=int(sketch_cfg.get("sketch_seed", 0)),
                    beta=float(BETA),
                    cost_sigma=cost_sigma,
                )
            else:
                from mdp_lp.solvers.sketched_lp import sketched_lp
                y_star = sketched_lp(
                    H, r_flat, gamma,
                    n_sketches=int(sketch_cfg.get("n_sketches", 4)),
                    sketch_ratio=float(sketch_cfg.get("sketch_ratio", 0.7)),
                    sketch_mode=str(sketch_cfg.get("sketch_mode", "gaussian")),
                    sketch_seed=int(sketch_cfg.get("sketch_seed", 0)),
                    beta=float(BETA),
                )
        else:
            y_star = DifferentiableLPSolver.apply(H, r_flat, gamma)

        V = (r_true * y_star).sum()
        loss = -V
        if rho > 0.0:
            violation = gamma_true - (H_true @ y_star.unsqueeze(-1)).squeeze(-1)
            loss = loss + rho * violation.pow(2).sum()

    grad = torch.autograd.grad(loss, lam)[0].item()
    return loss.item(), grad


def sweep_loss_grad(lambda_values, r_true, H_true, gamma_true,
                    agg=None, rho=0.0, label="",
                    sketch_cfg=None,
                    mdp_true=None):
    """Sweep λ and return (losses, grads) arrays for a given formulation."""
    n = len(lambda_values)
    losses = np.zeros(n)
    grads = np.zeros(n)
    for i, lam in enumerate(lambda_values):
        if (i + 1) % 50 == 0:
            print(f"  [{label}] {i+1}/{n}")
        ell, g = compute_loss_grad(lam, r_true, H_true, gamma_true,
                                   agg=agg, rho=rho, sketch_cfg=sketch_cfg,
                                   mdp_true=mdp_true)
        losses[i] = ell
        grads[i] = g
    return losses, grads


def _longest_flat_run(lam_vals, g, abs_thresh=0.01, min_frac=0.25):
    """Return (x0, x1) of the longest contiguous run where |g| <= abs_thresh.

    Returns None if the longest such run spans less than `min_frac` of the grid.
    Used to highlight "vanishing gradient" plateaus on the gradient row.
    """
    g_abs = np.abs(np.asarray(g))
    mask = g_abs <= abs_thresh
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
                     seed=1, return_meta: bool = False):
    """Load a SoftAggDense module saved by train_elasticity.py.

    Supports two checkpoint formats:
      1. Payload dict: {'state_dict', 'n_states', 'n_actions',
         'n_clusters', 'r_embed', ...}  (format written by
         train_elasticity.py to `agg_state_final.pt`).
      2. Raw state_dict (legacy format used by
         `trained_agg_M{n_clusters}.pt`).
    """
    ckpt = _load_torch_payload(ckpt_path)
    loaded_seed = int(seed)
    seed_source = "fallback"
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        loaded_seed = int(ckpt.get('seed', seed))
        seed_source = "checkpoint.seed" if 'seed' in ckpt else "fallback"
        agg = SoftAggDense(
            n_states=int(ckpt.get('n_states', N_STATES)),
            n_actions=int(ckpt.get('n_actions', N_ACTIONS)),
            n_clusters=int(ckpt.get('n_clusters', n_clusters_fallback)),
            r_embed=int(ckpt.get('r_embed', r_embed_fallback)),
        )
        agg.load_state_dict(ckpt['state_dict'])
    elif isinstance(ckpt, dict) and 'W_fixed' in ckpt:
        w_fixed = ckpt['W_fixed']
        if not isinstance(w_fixed, torch.Tensor) or w_fixed.dim() != 2:
            raise RuntimeError(f"Expected W_fixed tensor with shape (S,M), got {type(w_fixed)} {getattr(w_fixed, 'shape', None)}")
        loaded_seed = int(ckpt.get('fixed_W_seed', seed))
        seed_source = "checkpoint.fixed_W_seed" if 'fixed_W_seed' in ckpt else "fallback"
        agg = SoftAggDense(
            n_states=int(ckpt.get('n_states', w_fixed.shape[0])),
            n_actions=int(ckpt.get('n_actions', N_ACTIONS)),
            n_clusters=int(ckpt.get('n_clusters', w_fixed.shape[1])),
            r_embed=int(ckpt.get('r_embed', r_embed_fallback)),
            fixed_W=True,
            fixed_W_seed=loaded_seed,
        )
        if tuple(agg.W_fixed.shape) != tuple(w_fixed.shape):
            raise RuntimeError(
                f"W_fixed shape mismatch: module has {tuple(agg.W_fixed.shape)} but checkpoint has {tuple(w_fixed.shape)}"
            )
        with torch.no_grad():
            agg.W_fixed.copy_(w_fixed.to(dtype=agg.W_fixed.dtype))
    else:
        torch.manual_seed(seed)
        agg = SoftAggDense(n_states=N_STATES, n_actions=N_ACTIONS,
                           n_clusters=n_clusters_fallback,
                           r_embed=r_embed_fallback)
        agg.load_state_dict(ckpt)
    agg.eval()
    if return_meta:
        return agg, {"seed": loaded_seed, "seed_source": seed_source}
    return agg


def plot_landscape_comparison(lambda_values, r_true, H_true, gamma_true,
                              n_clusters=10, rho_small=10.0, seed=1,
                              output_dir='mdp_lp/plot_results',
                              agg_ckpt=None,
                              cache_dir=None,
                              recompute=None,
                              sketch_cfg=None,
                              mdp_true=None,
                              fixed_W=True):
    if output_dir is None:
        output_dir = _default_plot_results_dir()
    if cache_dir is None:
        cache_dir = os.path.join(output_dir, "cache")
    recompute_tags = set(recompute or [])

    problem_meta = _current_problem_meta()

    # Cache tags requested by user: ori, dfl, dfl_soft_agg
    L_orig, g_orig = cached_sweep_loss_grad(
        tag="ori",
        cache_dir=cache_dir,
        recompute_tags=recompute_tags,
        lambda_values=lambda_values,
        r_true=r_true,
        H_true=H_true,
        gamma_true=gamma_true,
        rho=0.0,
        label="Original",
        extra_meta={**problem_meta, "seed": int(seed), "n_clusters": int(n_clusters), "agg_ckpt": agg_ckpt},
    )

    RHO = 0.01

    print(f"[3-col] Aug. Lagrangian (rho={RHO:g}) ...")
    L_lag, g_lag = cached_sweep_loss_grad(
        tag="dfl",
        cache_dir=cache_dir,
        recompute_tags=recompute_tags,
        lambda_values=lambda_values,
        r_true=r_true,
        H_true=H_true,
        gamma_true=gamma_true,
        rho=RHO,
        label=f"Lag rho={RHO:g}",
        extra_meta={**problem_meta, "seed": int(seed), "n_clusters": int(n_clusters), "agg_ckpt": agg_ckpt},
    )

    agg_seed = int(seed)
    torch.manual_seed(agg_seed)
    agg = SoftAggDense(n_states=N_STATES, n_actions=N_ACTIONS,
                       n_clusters=n_clusters, r_embed=8,
                       fixed_W=bool(fixed_W), fixed_W_seed=agg_seed)
    agg_src = (f'fixed-W init (seed={agg_seed})' if fixed_W
               else f'random init (seed={agg_seed})')

    candidate_paths = []
    if agg_ckpt is not None:
        candidate_paths.append(agg_ckpt)
    candidate_paths.append(os.path.join(output_dir, f'trained_agg_M{n_clusters}.pt'))

    for path in candidate_paths:
        if not os.path.exists(path):
            continue
        try:
            print(f"Loading trained aggregation model from {path}")
            agg, agg_meta = load_trained_agg(
                path,
                n_clusters_fallback=n_clusters,
                seed=agg_seed,
                return_meta=True,
            )
            n_clusters = int(getattr(agg, "n_clusters", n_clusters))
            agg_seed = int(agg_meta["seed"])
            agg_src = f'trained ({path}, seed={agg_seed})'
            break
        except RuntimeError as e:
            print(f"  [3-col] Skipping {path} (schema mismatch: {e.__class__.__name__})")
    agg.eval()
    print(f"[3-col] Soft aggregation + penalty (M={n_clusters}, rho={RHO:g}, {agg_src}) ...")
    L_agg, g_agg = cached_sweep_loss_grad(
        tag="dfl_soft_agg",
        cache_dir=cache_dir,
        recompute_tags=recompute_tags,
        lambda_values=lambda_values,
        r_true=r_true,
        H_true=H_true,
        gamma_true=gamma_true,
        agg=agg,
        rho=RHO,
        label=f"SoftAgg+pen M={n_clusters}",
        extra_meta={**problem_meta, "seed": int(agg_seed), "n_clusters": int(n_clusters), "agg_ckpt": agg_ckpt, "agg_src": agg_src},
        mdp_true=mdp_true,
    )

    L_sketch, g_sketch = (None, None)
    if sketch_cfg is not None:
        sk_label = (f"SoftAgg+pen+Sketch K={int(sketch_cfg.get('n_sketches', 4))}, "
                    f"k/M={float(sketch_cfg.get('sketch_ratio', 0.7)):.2f}")
        print(f"[3-col] Soft aggregation + penalty + sketch ({sk_label}) ...")
        L_sketch, g_sketch = cached_sweep_loss_grad(
            tag="dfl_soft_agg_sketch",
            cache_dir=cache_dir,
            recompute_tags=recompute_tags,
            lambda_values=lambda_values,
            r_true=r_true,
            H_true=H_true,
            gamma_true=gamma_true,
            agg=agg,
            rho=RHO,
            label=sk_label,
            extra_meta={**problem_meta, "seed": int(agg_seed),
                        "n_clusters": int(n_clusters), "agg_ckpt": agg_ckpt,
                        "agg_src": agg_src},
            sketch_cfg=sketch_cfg,
            mdp_true=mdp_true,
        )

    losses = [L_orig, L_lag, L_agg]
    # Use the autograd dL/dλ returned by sweep_loss_grad (i.e. backprop through
    # DifferentiableLPSolver), so the gradient row matches the autograd curve
    # plotted by diagnose_solver_gradients (solver_gradient_comparison.pdf).
    grads = [np.asarray(g_orig, dtype=float),
             np.asarray(g_lag, dtype=float),
             np.asarray(g_agg, dtype=float)]
    if L_sketch is not None:
        losses.append(L_sketch)
        grads.append(np.asarray(g_sketch, dtype=float))

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

    n_cols = len(losses)
    fig_width = 4.93 * n_cols
    fig, axes = plt.subplots(
        2, n_cols, figsize=(fig_width, 7.4),
        gridspec_kw={'hspace': 0.34, 'wspace': 0.30})

    col_titles = [
        r'(a) Original loss:  $\mathcal{L}=-r_{\mathrm{true}}^{\!\top}\,y^{*}(\lambda)$',
        rf'(b) Aug. Lagrangian  ($\rho={RHO:g}$)',
        rf'(c) Aug. Lagrangian + Soft agg.  ($\rho={RHO:g},\,M={n_clusters}$)',
    ]
    col_colors = ['#455A64', '#E65100', '#1565C0', '#6A1B9A']   # charcoal / orange / blue / purple
    if L_sketch is not None:
        K_show = int(sketch_cfg.get('n_sketches', 4))
        ratio_show = float(sketch_cfg.get('sketch_ratio', 0.7))
        col_titles.append(
            rf'(d) Soft agg. + Sketch  ($K={K_show},\,k/M={ratio_show:.2f}$)'
        )
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
        flat = _longest_flat_run(lambda_values, g, abs_thresh=0.01, min_frac=0.35)
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

    out_name = 'landscape_comparison_3col.pdf' if n_cols == 3 else 'landscape_comparison_4col.pdf'
    out_path = os.path.join(output_dir, out_name)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    plt.rcParams.update(prev_rc)

    print(f"\nSaved 2×{n_cols} landscape comparison to {out_path}")
    print(f"  λ*  (a) original          = {lambda_stars[0]:.3f}   shift {shifts[0]:+.3f}")
    print(f"  λ*  (b) aug-lag ρ={rho_small:<6g}= {lambda_stars[1]:.3f}   shift {shifts[1]:+.3f}")
    print(f"  λ*  (c) soft agg M={n_clusters:<4}   = {lambda_stars[2]:.3f}   shift {shifts[2]:+.3f}")
    if n_cols == 4:
        print(f"  λ*  (d) sketched          = {lambda_stars[3]:.3f}   shift {shifts[3]:+.3f}")


def plot_loss_landscape_sketch_rho(
    lambda_values, r_true, H_true, gamma_true, agg, mdp_true,
    rhos=(0.01, 1.0, 100.0),
    sketch_ratios=(0.2, 0.7),
    sketch_K=8,
    sketch_seed=0,
    sketch_mode="gaussian",
    output_dir=None,
    cache_dir=None,
    recompute=None,
):
    """2x3 loss-landscape comparison across (sketch on/off) × ρ in soft-agg.

    Each column corresponds to one ρ; each subplot overlays the no-sketch
    curve and one curve per sketch ratio. Top row is L(λ); bottom row is
    autograd dL/dλ via DifferentiableLPSolver / sketched_lp.
    """
    if output_dir is None:
        output_dir = _default_plot_results_dir()
    if cache_dir is None:
        cache_dir = os.path.join(output_dir, "cache")
    recompute_tags = set(recompute or [])

    sketch_cfgs = [None] + [
        {"n_sketches": int(sketch_K),
         "sketch_ratio": float(r),
         "sketch_mode": str(sketch_mode),
         "sketch_seed": int(sketch_seed)}
        for r in sketch_ratios
    ]
    sketch_labels = ["no sketch"] + [
        rf"sketch $k/M={float(r):.2f}$" for r in sketch_ratios
    ]
    # charcoal / blue / orange — matches plot_landscape_comparison palette
    base_colors = ['#455A64', '#1565C0', '#E65100']
    sketch_colors = base_colors[:len(sketch_cfgs)]

    problem_meta = _current_problem_meta()
    n_clusters = int(getattr(agg, 'n_clusters', 0)) if agg is not None else 0
    agg_tag = f"softaggM{n_clusters}" if agg is not None else "noagg"

    L_grid, g_grid = {}, {}
    for rho in rhos:
        for cfg, lbl in zip(sketch_cfgs, sketch_labels):
            cfg_tag = "none" if cfg is None else f"r{cfg['sketch_ratio']:.2f}_K{cfg['n_sketches']}"
            tag = f"sketchrho_{agg_tag}__rho{rho:g}__sk_{cfg_tag}"
            print(f"[sketch-rho] sweeping rho={rho:g}  cfg={cfg_tag}")
            L, g = cached_sweep_loss_grad(
                tag=tag,
                cache_dir=cache_dir,
                recompute_tags=recompute_tags,
                lambda_values=lambda_values,
                r_true=r_true,
                H_true=H_true,
                gamma_true=gamma_true,
                agg=agg,
                rho=float(rho),
                label=lbl,
                sketch_cfg=cfg,
                mdp_true=mdp_true,
                extra_meta={**problem_meta, "n_clusters": n_clusters},
            )
            L_grid[(rho, lbl)] = np.asarray(L, dtype=float)
            g_grid[(rho, lbl)] = np.asarray(g, dtype=float)

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

    lam_true_color = '#D32F2F'
    minimiser_color = '#2E7D32'
    plateau_color = '#9E9E9E'

    n_cols = len(rhos)
    fig_width = 4.93 * n_cols
    try:
        fig, axes = plt.subplots(
            2, n_cols, figsize=(fig_width, 7.4),
            gridspec_kw={'hspace': 0.34, 'wspace': 0.30})

        for col, rho in enumerate(rhos):
            ax_top, ax_bot = axes[0, col], axes[1, col]
            for cfg, lbl, c in zip(sketch_cfgs, sketch_labels, sketch_colors):
                L = L_grid[(rho, lbl)]
                g = g_grid[(rho, lbl)]

                ax_top.plot(lambda_values, L, color=c, lw=1.6,
                            solid_capstyle='round', label=lbl)
                lam_star = float(lambda_values[int(np.argmin(L))])
                ax_top.plot(lam_star, float(L.min()), marker='o',
                            mfc=minimiser_color, mec='white', ms=6,
                            mew=1.0, ls='', zorder=5)

                ax_bot.plot(lambda_values, g, color=c, lw=1.6,
                            solid_capstyle='round')
                if cfg is None:
                    flat = _longest_flat_run(lambda_values, g,
                                             abs_thresh=0.01, min_frac=0.35)
                    if flat is not None:
                        ax_bot.axvspan(*flat, color=plateau_color,
                                       alpha=0.22, lw=0)

            for ax in (ax_top, ax_bot):
                ax.axvline(LAMBDA_TRUE, color=lam_true_color, ls='--',
                           lw=1.3, alpha=0.9, zorder=3)
                ax.grid(True, alpha=0.35, ls=':', lw=0.7)
                ax.set_xlim(lambda_values[0], lambda_values[-1])

            ax_bot.axhline(0.0, color='black', lw=1.0, alpha=0.80)
            ax_top.set_title(rf'$\rho={rho:g}$', pad=8)
            ax_bot.set_xlabel(r'$\lambda_{\mathrm{pred}}$')

        axes[0, 0].set_ylabel(r'Loss  $\mathcal{L}(\lambda)$')
        axes[1, 0].set_ylabel(r'Gradient  $d\mathcal{L}/d\lambda$')
        axes[0, 0].legend(loc='best', fontsize=9)

        suffix = f'_softaggM{n_clusters}' if agg is not None else '_noagg'
        out_path = os.path.join(output_dir, f'loss_landscape_sketch_rho{suffix}.pdf')
        plt.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
    finally:
        plt.rcParams.update(prev_rc)

    print(f"\nSaved sketch×ρ landscape (M={n_clusters}) to {out_path}")
    for rho in rhos:
        for lbl in sketch_labels:
            L = L_grid[(rho, lbl)]
            lam_star = float(lambda_values[int(np.argmin(L))])
            print(f"  λ* ({lbl:<22s}, ρ={rho:>6g}) = {lam_star:6.3f}   "
                  f"shift {lam_star - LAMBDA_TRUE:+.3f}")


def plot_three_methods_row(
    lambda_values, r_true, H_true, gamma_true,
    rho=10.0,
    sketch_K=24,
    sketch_ratio=0.5,
    sketch_mode="gaussian",
    sketch_seed=0,
    cost_sigma=0.3,
    output_dir=None,
    cache_dir=None,
    recompute=None,
    out_name='landscape_three_methods_row.pdf',
):
    """2x3 figure on the original (un-aggregated) LP. Top row: loss
    landscape; bottom row: autograd dL/dλ. Columns:
        (a) -r_true^T y*(λ)                                          — original DFL loss
        (b) L_aug    = -r_true^T y*       + ρ ||γ_true - H_true y*||²  — aug. Lagrangian
        (c) L_sketch = -r_true^T ȳ_sketch + ρ ||γ_true - H_true ȳ||²   — aug. Lagrangian + sketch

    Panel (a) exhibits the bare staircase. Panel (b) keeps y* and adds the
    feasibility penalty (curvature where (a) is flat, but the kinks remain
    — penalty changes the value but not the underlying y*, so the kinks
    persist). Panel (c) replaces y* with the K-sketch ensemble mean ȳ
    using BOTH constraint-side row sketching (k/M ratio) and cost-side
    Gaussian perturbation (cost_sigma). The two sources of stochasticity
    decorrelate basis hops across the K ensemble members, which is the
    regime that produces the smooth landscape seen in
    sketch_smoothing_demo.pdf.

    Caching: each sweep is keyed by (tag, problem constants, rho, sketch
    config) via cached_sweep_loss_grad's npz cache. Results are reused
    whenever the meta matches and recomputed otherwise; every fresh sweep
    is written back to {cache_dir}/three_row_*.npz.
    """
    if output_dir is None:
        output_dir = _default_plot_results_dir()
    if cache_dir is None:
        cache_dir = os.path.join(output_dir, "cache")
    recompute_tags = set(recompute or [])
    problem_meta = _current_problem_meta()

    # ── (a) plain LP, rho=0 → loss = -r_true^T y* ──
    print("[3-row] (a) -r_true^T y*, rho=0 ...")
    L_a, g_a = cached_sweep_loss_grad(
        tag="three_row_orig",
        cache_dir=cache_dir, recompute_tags=recompute_tags,
        lambda_values=lambda_values,
        r_true=r_true, H_true=H_true, gamma_true=gamma_true,
        rho=0.0, label="orig",
        extra_meta={**problem_meta, "panel": "a"},
    )
    L_orig = np.asarray(L_a, dtype=float)
    g_orig = np.asarray(g_a, dtype=float)

    # ── (b) plain LP + augmented Lagrangian, rho>0 ──
    print(f"[3-row] (b) augmented Lagrangian, rho={rho:g} ...")
    L_b, g_b = cached_sweep_loss_grad(
        tag="three_row_aug",
        cache_dir=cache_dir, recompute_tags=recompute_tags,
        lambda_values=lambda_values,
        r_true=r_true, H_true=H_true, gamma_true=gamma_true,
        rho=float(rho), label=f"aug-lag rho={rho:g}",
        extra_meta={**problem_meta, "panel": "b"},
    )
    L_aug = np.asarray(L_b, dtype=float)
    g_aug = np.asarray(g_b, dtype=float)

    # ── (c) sketched LP + augmented Lagrangian, same rho ──
    sketch_cfg = {
        "n_sketches": int(sketch_K),
        "sketch_ratio": float(sketch_ratio),
        "sketch_mode": str(sketch_mode),
        "sketch_seed": int(sketch_seed),
        "cost_sigma": float(cost_sigma),
    }
    print(f"[3-row] (c) aug. Lagrangian + sketch, rho={rho:g}, "
          f"K={sketch_K}, k/M={sketch_ratio:g}, σ={cost_sigma:g}, "
          f"mode={sketch_mode} ...")
    L_c, g_c = cached_sweep_loss_grad(
        tag="three_row_aug_sketch",
        cache_dir=cache_dir, recompute_tags=recompute_tags,
        lambda_values=lambda_values,
        r_true=r_true, H_true=H_true, gamma_true=gamma_true,
        rho=float(rho),
        label=f"aug-lag+sketch rho={rho:g} K={sketch_K} σ={cost_sigma:g}",
        sketch_cfg=sketch_cfg,
        extra_meta={**problem_meta, "panel": "c"},
    )
    L_sketch = np.asarray(L_c, dtype=float)
    g_sketch = np.asarray(g_c, dtype=float)

    # ── Diagnostics: penalty contribution vs reward range ──
    # Both (a) and (b) share the same y*, so L_aug - L_orig isolates the
    # penalty term ρ ||γ_true - H_true y*||² across the λ sweep. If the
    # ratio below is small, panels (a) and (b) will look visually
    # identical even though their gradients differ — bump rho.
    pen = L_aug - L_orig
    rew_range = float(L_orig.max() - L_orig.min())
    pen_max = float(pen.max())
    pen_min = float(pen.min())
    pen_mean = float(pen.mean())
    print(f"[3-row] diagnostics @ ρ={rho:g}:")
    print(f"  reward range  -r^T y*  : {rew_range:.4e}")
    print(f"  penalty range          : [{pen_min:.4e}, {pen_max:.4e}]   "
          f"mean {pen_mean:.4e}")
    if rew_range > 0:
        ratio = pen_max / rew_range
        print(f"  max(penalty) / range(-r^T y*) = {ratio:.3%}")
        if ratio < 0.05:
            print("  WARNING: penalty < 5% of reward range — panels (a) and "
                  "(b) will look visually identical even though their "
                  "gradients differ. Bump rho (try ρ × 100).")

    # All three panels are losses → use argmin everywhere. (Tried
    # zero-crossing of g, but on LP-staircased losses g can hit zero in a
    # locally-flat segment while L keeps dropping past it via basis-change
    # discontinuities — argmin(L) is the honest visual minimum.)
    idx_a = int(np.argmin(L_orig))
    idx_b = int(np.argmin(L_aug))
    idx_c = int(np.argmin(L_sketch))
    lam_stars = [float(lambda_values[idx_a]),
                 float(lambda_values[idx_b]),
                 float(lambda_values[idx_c])]
    star_vals = [float(L_orig[idx_a]),
                 float(L_aug[idx_b]),
                 float(L_sketch[idx_c])]

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

    # Finite-difference gradients (centred, raw — basis-change spikes are
    # part of the honest picture). The dot's y-value is computed below
    # via a wide-stride FD that skips the local spike at argmin.
    fd_curves = [np.gradient(L_orig, lambda_values),
                 np.gradient(L_aug,  lambda_values),
                 np.gradient(L_sketch, lambda_values)]
    fd_dot_stride = max(1, len(lambda_values) // 80)
    def _wide_fd_at(L_arr, idx):
        i_lo = max(0, idx - fd_dot_stride)
        i_hi = min(len(L_arr) - 1, idx + fd_dot_stride)
        if i_hi == i_lo:
            return 0.0
        return float((L_arr[i_hi] - L_arr[i_lo]) /
                     (lambda_values[i_hi] - lambda_values[i_lo]))
    fd_dot_vals = [
        _wide_fd_at(L_orig,   int(np.argmin(L_orig))),
        _wide_fd_at(L_aug,    int(np.argmin(L_aug))),
        _wide_fd_at(L_sketch, int(np.argmin(L_sketch))),
    ]

    try:
        fig, axes = plt.subplots(3, 3, figsize=(14.8, 10.8),
                                 gridspec_kw={'hspace': 0.34, 'wspace': 0.30})

        titles = [
            r'(a) $-r_{\mathrm{true}}^{\!\top}\,y^{*}(\lambda)$',
            r'(b) Augmented Lagrangian',
            r'(c) Aug. Lagrangian + Sketch',
        ]
        ylabels = [
            r'$-r_{\mathrm{true}}^{\!\top}\,y^{*}(\lambda)$',
            r'$\mathcal{L}_{\mathrm{aug}}(\lambda)$',
            r'$\mathcal{L}_{\mathrm{aug}}^{\mathrm{sketch}}(\lambda)$',
        ]
        loss_curves = [L_orig, L_aug, L_sketch]
        grad_curves = [g_orig, g_aug, g_sketch]
        colors = ['#455A64', '#E65100', '#1565C0']
        lam_true_color = '#D32F2F'
        minimiser_color = '#2E7D32'

        # Row 1: loss landscapes.
        for ax, y, ttl, lbl, c, lam_s, sv in zip(
                axes[0], loss_curves, titles, ylabels, colors, lam_stars,
                star_vals):
            ax.plot(lambda_values, y, lw=1.8, color=c, solid_capstyle='round')
            ax.axvline(LAMBDA_TRUE, color=lam_true_color, ls='--', lw=1.3,
                       alpha=0.9,
                       label=rf'$\lambda_{{\mathrm{{true}}}}={LAMBDA_TRUE:g}$')
            ax.plot(lam_s, sv, marker='o', mfc=minimiser_color, mec='white',
                    ms=9, mew=1.3, ls='', zorder=5,
                    label=rf'$\lambda^{{*}}={lam_s:.2f}$')
            ax.set_title(ttl, pad=8)
            ax.set_ylabel(lbl)
            ax.set_xlim(lambda_values[0], lambda_values[-1])
            ax.grid(True, alpha=0.35, ls=':', lw=0.7)
            ax.legend(loc='best')

        # Row 2: autograd dL/dλ (within-basis derivative; misses LP kinks).
        autograd_ylabels = [
            r'$d(-r_{\mathrm{true}}^{\!\top} y^{*})/d\lambda$  (autograd)',
            r'$d\mathcal{L}_{\mathrm{aug}}/d\lambda$  (autograd)',
            r'$d\mathcal{L}_{\mathrm{aug}}^{\mathrm{sketch}}/d\lambda$  (autograd)',
        ]
        for ax, g, c, lam_s, lbl in zip(
                axes[1], grad_curves, colors, lam_stars, autograd_ylabels):
            ax.plot(lambda_values, g, lw=1.8, color=c, solid_capstyle='round')
            ax.axvline(LAMBDA_TRUE, color=lam_true_color, ls='--', lw=1.3,
                       alpha=0.9,
                       label=rf'$\lambda_{{\mathrm{{true}}}}={LAMBDA_TRUE:g}$')
            ax.axhline(0.0, color='black', lw=1.0, alpha=0.80)

            g_at_star = float(np.interp(lam_s, lambda_values, g))
            ax.plot(lam_s, g_at_star, marker='o', mfc=minimiser_color,
                    mec='white', ms=9, mew=1.3, ls='', zorder=5,
                    label=rf'$\lambda^{{*}}={lam_s:.2f}$')

            ax.set_ylabel(lbl)
            ax.set_xlim(lambda_values[0], lambda_values[-1])
            ax.grid(True, alpha=0.35, ls=':', lw=0.7)
            ax.legend(loc='best')

        # Row 3: finite-difference dL/dλ. FD captures basis-change drops, so
        # its zero coincides with argmin(L) on smooth panels.
        fd_ylabels = [
            r'$d(-r_{\mathrm{true}}^{\!\top} y^{*})/d\lambda$  (finite diff.)',
            r'$d\mathcal{L}_{\mathrm{aug}}/d\lambda$  (finite diff.)',
            r'$d\mathcal{L}_{\mathrm{aug}}^{\mathrm{sketch}}/d\lambda$  (finite diff.)',
        ]
        for ax, gfd, c, lam_s, lbl, gfd_dot in zip(
                axes[2], fd_curves, colors, lam_stars, fd_ylabels,
                fd_dot_vals):
            ax.plot(lambda_values, gfd, lw=1.8, color=c, solid_capstyle='round')
            ax.axvline(LAMBDA_TRUE, color=lam_true_color, ls='--', lw=1.3,
                       alpha=0.9,
                       label=rf'$\lambda_{{\mathrm{{true}}}}={LAMBDA_TRUE:g}$')
            ax.axhline(0.0, color='black', lw=1.0, alpha=0.80)

            ax.plot(lam_s, gfd_dot, marker='o', mfc=minimiser_color,
                    mec='white', ms=9, mew=1.3, ls='', zorder=5,
                    label=rf'$\lambda^{{*}}={lam_s:.2f}$')

            ax.set_xlabel(r'$\lambda_{\mathrm{pred}}$')
            ax.set_ylabel(lbl)
            ax.set_xlim(lambda_values[0], lambda_values[-1])
            ax.grid(True, alpha=0.35, ls=':', lw=0.7)
            ax.legend(loc='best')

        out_path = os.path.join(output_dir, out_name)
        plt.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
    finally:
        plt.rcParams.update(prev_rc)

    print(f"\nSaved 3x3 three-methods landscape+gradient to {out_path}")
    print(f"  λ*  (a) -r^T y*                                                        = {lam_stars[0]:.3f}   loss {star_vals[0]:.4f}")
    print(f"  λ*  (b) aug-lag (ρ={rho:g})                                             = {lam_stars[1]:.3f}   loss {star_vals[1]:.4f}")
    print(f"  λ*  (c) aug-lag + sketch (ρ={rho:g}, K={sketch_K}, k/M={sketch_ratio:g}, σ={cost_sigma:g}) = {lam_stars[2]:.3f}   loss {star_vals[2]:.4f}")
    print(f"  λ_true                                                                  = {LAMBDA_TRUE:.3f}")
    print(f"  TV(g_orig)={float(np.abs(np.diff(g_orig)).sum()):.3e}   "
          f"TV(g_aug)={float(np.abs(np.diff(g_aug)).sum()):.3e}   "
          f"TV(g_sketch)={float(np.abs(np.diff(g_sketch)).sum()):.3e}")
    print("  ↳ TV(g_sketch) should be << TV(g_orig); if not, increase K or "
          "cost_sigma.")
    return {
        'lambda_values': np.asarray(lambda_values, dtype=float),
        'L_orig': L_orig,
        'L_aug': L_aug,
        'L_sketch': L_sketch,
        'g_orig': g_orig,
        'g_aug': g_aug,
        'g_sketch': g_sketch,
        'lam_stars': lam_stars,
        'star_vals': star_vals,
    }


def diagnose_solver_gradients(lambda_values, agg, output_dir,
                              rho=0.01, qp_reg=1.0, recompute=False):
    """Evidence that DifferentiableLPSolver's autograd gradient is
    sign-inconsistent with the landscape gradient (np.gradient of L),
    and the QP-regularized solver fixes it.

    Loss: the projected-true aggregated loss used by the fixed
    train_elasticity.py (r_true, P_true, γ_true projected into the
    agg space via the SAME W as the λ_pred aggregation).

    Reuses cached arrays from solver_gradient_comparison{_panel_a}.npz
    when the lambda grid and meta (rho, qp_reg, λ_true, price, panel_a)
    match. Pass ``recompute=True`` to force a fresh sweep.
    """
    from mdp_lp.solvers.qpth_qp_solver import solve_occupancy_qp_qpth
    from qpth.qp import QPFunction, QPSolvers

    suffix = '_panel_a' if agg is None else ''
    npz_path = os.path.join(output_dir, f'solver_gradient_comparison{suffix}.npz')
    expected_meta = dict(
        rho=float(rho), qp_reg=float(qp_reg),
        lambda_true=float(LAMBDA_TRUE), price=float(PRICE),
        panel_a=bool(agg is None),
    )

    cached = None
    if not recompute and os.path.exists(npz_path):
        try:
            loaded = np.load(npz_path, allow_pickle=True)
            lam_loaded = np.asarray(loaded["lambda_values"], dtype=float)
            meta_obj = loaded["meta"]
            meta_loaded = meta_obj.item() if hasattr(meta_obj, "item") else dict(meta_obj)
            grid_match = (
                lam_loaded.shape == np.asarray(lambda_values).shape
                and np.allclose(lam_loaded, lambda_values)
            )
            meta_match = all(
                np.isclose(float(meta_loaded.get(k, np.nan)), float(expected_meta[k]))
                if isinstance(expected_meta[k], float)
                else meta_loaded.get(k) == expected_meta[k]
                for k in expected_meta
            )
            if grid_match and meta_match:
                cached = loaded
                print(f"[solver-grad] loaded cached results from {npz_path}")
            else:
                print(f"[solver-grad] cache exists at {npz_path} but "
                      f"grid_match={grid_match} meta_match={meta_match}; recomputing")
        except Exception as e:
            print(f"[solver-grad] could not reuse {npz_path} ({e}); recomputing")

    qp_solver = QPFunction(eps=1e-10, maxIter=20, solver=QPSolvers.PDIPM_BATCHED)

    lam_true_t = torch.tensor([LAMBDA_TRUE], dtype=torch.float32)
    price_t = torch.tensor([PRICE], dtype=torch.float32)
    mdp_true = create_inventory_from_lambda(lam_true_t, B_MAX, S_MAX, Q_MAX,
                                            FOC, VOC, HC, BC, BETA, price_t)
    S = mdp_true.P.shape[1]
    gamma = mdp_true.build_gamma(1, S, torch.device("cpu"))
    gamma = torch.clamp(gamma, min=1e-8); gamma = gamma / gamma.sum(dim=1, keepdim=True)

    r_true_flat = mdp_true.r.reshape(1, -1)

    def projected_loss(lam_val, which):
        lam = torch.tensor([lam_val], dtype=torch.float32, requires_grad=True)
        mdp = create_inventory_from_lambda(lam, B_MAX, S_MAX, Q_MAX,
                                           FOC, VOC, HC, BC, BETA, price_t)
        if agg is None:
            # Panel-(a) bare LP loss: L = -r_true^T y*(λ).  No aggregation, no penalty.
            H = mdp.build_H(mdp.P)
            r_flat = mdp.r.reshape(1, -1)
            if which == "lp":
                y = DifferentiableLPSolver.apply(H, r_flat, gamma)
            else:
                y = solve_occupancy_qp_qpth(H, r_flat, gamma,
                                            reg=qp_reg, qp_solver=qp_solver)
            loss = -(r_true_flat * y).sum()
            return loss, lam

        P_use, r_use, gamma_agg, info = agg(mdp.P, mdp.r, gamma, tau=0.5)
        H_use = mdp.build_H(P_use)
        if which == "lp":
            y = DifferentiableLPSolver.apply(H_use, r_use.reshape(1, -1), gamma_agg)
        else:
            y = solve_occupancy_qp_qpth(H_use, r_use.reshape(1, -1), gamma_agg,
                                        reg=qp_reg, qp_solver=qp_solver)
        W = info["W"]; mass = W.sum(dim=1).clamp_min(1e-8)
        r_true_agg = torch.einsum("bsm,bsa->bma", W, mdp_true.r) / mass.unsqueeze(-1)
        right = torch.einsum("bsaj,bjm->bsam", mdp_true.P, W)
        p_hat = torch.einsum("bsm,bsan->bman", W, right)
        P_true_agg = (p_hat / mass[:, :, None, None]).clamp_min(0.0)
        P_true_agg = P_true_agg / P_true_agg.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        gamma_true_agg = torch.einsum("bs,bsm->bm", gamma, W).clamp_min(1e-8)
        gamma_true_agg = gamma_true_agg / gamma_true_agg.sum(dim=1, keepdim=True).clamp_min(1e-8)
        H_true_agg = mdp_true.build_H(P_true_agg)
        viol = gamma_true_agg - (H_true_agg @ y.unsqueeze(-1)).squeeze(-1)
        v = torch.sum(r_true_agg.reshape(1, -1) * y, dim=1)
        loss = -v.mean() + rho * viol.pow(2).mean()
        return loss, lam

    if cached is not None:
        L_lp = np.asarray(cached["L_lp"], dtype=float)
        L_qp = np.asarray(cached["L_qp"], dtype=float)
        g_auto_lp = np.asarray(cached["g_auto_lp"], dtype=float)
        g_auto_qp = np.asarray(cached["g_auto_qp"], dtype=float)
        g_land_lp = np.asarray(cached["g_land_lp"], dtype=float)
        g_land_qp = np.asarray(cached["g_land_qp"], dtype=float)
    else:
        n = len(lambda_values)
        L_lp = np.zeros(n); L_qp = np.zeros(n)
        g_auto_lp = np.zeros(n); g_auto_qp = np.zeros(n)
        for i, lv in enumerate(lambda_values):
            if (i + 1) % 25 == 0:
                print(f"  [solver-grad] {i+1}/{n}")
            Llp, lam = projected_loss(lv, "lp")
            L_lp[i] = Llp.item(); g_auto_lp[i] = torch.autograd.grad(Llp, lam)[0].item()
            Lqp, lam = projected_loss(lv, "qp")
            L_qp[i] = Lqp.item(); g_auto_qp[i] = torch.autograd.grad(Lqp, lam)[0].item()
        g_land_lp = np.gradient(L_lp, lambda_values)
        g_land_qp = np.gradient(L_qp, lambda_values)

    sign_agree_lp = float(np.mean(np.sign(g_auto_lp) == np.sign(g_land_lp)))
    sign_agree_qp = float(np.mean(np.sign(g_auto_qp) == np.sign(g_land_qp)))

    print(f"\n=== Solver-gradient evidence (ρ={rho}, qp_reg={qp_reg}) ===")
    print(f"LP solver: sign(autograd) == sign(np.gradient(L)) in "
          f"{sign_agree_lp:.1%} of grid points")
    print(f"QP solver: sign(autograd) == sign(np.gradient(L)) in "
          f"{sign_agree_qp:.1%} of grid points")
    print(f"argmin(L_lp) = λ={lambda_values[int(np.argmin(L_lp))]:.3f}   "
          f"argmin(L_qp) = λ={lambda_values[int(np.argmin(L_qp))]:.3f}   "
          f"(λ_true={LAMBDA_TRUE:.3f})")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes[0, 0].plot(lambda_values, L_lp, color='#E65100', lw=1.6)
    axes[0, 0].axvline(LAMBDA_TRUE, color='r', ls='--', lw=1, alpha=0.8)
    axes[0, 0].set_title('LP solver: loss landscape')
    axes[0, 0].set_ylabel(r'$\mathcal{L}(\lambda)$'); axes[0, 0].grid(alpha=0.3)

    axes[1, 0].plot(lambda_values, L_qp, color='#1565C0', lw=1.6)
    axes[1, 0].axvline(LAMBDA_TRUE, color='r', ls='--', lw=1, alpha=0.8)
    axes[1, 0].set_title(f'QP solver (reg={qp_reg}): loss landscape')
    axes[1, 0].set_ylabel(r'$\mathcal{L}(\lambda)$'); axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_xlabel(r'$\lambda$')

    axes[0, 1].plot(lambda_values, g_land_lp, color='k', lw=1.4,
                    label=r'np.gradient($\mathcal{L}$) — landscape')
    axes[0, 1].plot(lambda_values, g_auto_lp, color='#E65100', lw=1.4,
                    label='autograd (LP solver)')
    axes[0, 1].axvline(LAMBDA_TRUE, color='r', ls='--', lw=1, alpha=0.8)
    axes[0, 1].axhline(0, color='k', lw=0.5, alpha=0.5)
    axes[0, 1].set_title(f'LP autograd vs landscape   (sign-match: {sign_agree_lp:.0%})')
    axes[0, 1].set_ylabel(r'$d\mathcal{L}/d\lambda$'); axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend(loc='best')

    axes[1, 1].plot(lambda_values, g_land_qp, color='k', lw=1.4,
                    label=r'np.gradient($\mathcal{L}$) — landscape')
    axes[1, 1].plot(lambda_values, g_auto_qp, color='#1565C0', lw=1.4,
                    label='autograd (QP solver)')
    axes[1, 1].axvline(LAMBDA_TRUE, color='r', ls='--', lw=1, alpha=0.8)
    axes[1, 1].axhline(0, color='k', lw=0.5, alpha=0.5)
    axes[1, 1].set_title(f'QP autograd vs landscape   (sign-match: {sign_agree_qp:.0%})')
    axes[1, 1].set_ylabel(r'$d\mathcal{L}/d\lambda$'); axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend(loc='best')
    axes[1, 1].set_xlabel(r'$\lambda$')

    for ax in axes.flat:
        ax.set_xlim(lambda_values[0], lambda_values[-1])
    out = os.path.join(output_dir, f'solver_gradient_comparison{suffix}.pdf')
    plt.tight_layout(); plt.savefig(out, bbox_inches='tight'); plt.close(fig)
    print(f"Saved: {out}")

    if cached is None:
        np.savez_compressed(
            npz_path,
            lambda_values=np.asarray(lambda_values, dtype=float),
            L_lp=L_lp, L_qp=L_qp,
            g_auto_lp=g_auto_lp, g_auto_qp=g_auto_qp,
            g_land_lp=g_land_lp, g_land_qp=g_land_qp,
            g_auto_lp_abs=np.abs(g_auto_lp), g_auto_qp_abs=np.abs(g_auto_qp),
            g_land_lp_abs=np.abs(g_land_lp), g_land_qp_abs=np.abs(g_land_qp),
            meta=dict(
                **expected_meta,
                sign_agree_lp=sign_agree_lp, sign_agree_qp=sign_agree_qp,
            ),
        )
        print(f"Saved gradient arrays: {npz_path}")


def replot_solver_gradient_from_npz(npz_path, out_pdf=None,
                                    grad_ylim_quantile=0.99,
                                    grad_ylim_pad=1.10):
    """Re-render solver_gradient_comparison{_panel_a}.pdf from a saved .npz.

    Loss panels are plotted in full. Gradient panels are clipped to a robust
    y-range derived from the (1 - q, q) quantile of |g_*|, so isolated spikes
    don't compress the visible scale.
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"No such file: {npz_path}")
    d = np.load(npz_path, allow_pickle=True)
    lam = np.asarray(d["lambda_values"], dtype=float)
    L_lp = np.asarray(d["L_lp"], dtype=float)
    L_qp = np.asarray(d["L_qp"], dtype=float)
    g_auto_lp = np.asarray(d["g_auto_lp"], dtype=float)
    g_auto_qp = np.asarray(d["g_auto_qp"], dtype=float)
    g_land_lp = np.asarray(d["g_land_lp"], dtype=float)
    g_land_qp = np.asarray(d["g_land_qp"], dtype=float)
    meta_obj = d["meta"]
    meta = meta_obj.item() if hasattr(meta_obj, "item") else dict(meta_obj)
    lam_true = float(meta.get("lambda_true", LAMBDA_TRUE))
    qp_reg = float(meta.get("qp_reg", 1.0))

    sign_agree_lp = float(np.mean(np.sign(g_auto_lp) == np.sign(g_land_lp)))
    sign_agree_qp = float(np.mean(np.sign(g_auto_qp) == np.sign(g_land_qp)))

    def landscape_ylim(g_land, g_auto):
        """Symmetric y-limit set by the landscape gradient's full range, with
        a small extra margin if every autograd value is also smaller. Spikes
        in autograd that exceed this bound saturate at the axes, which is the
        intended visual: same-scale comparison favoring the smooth trend.
        """
        ga = np.abs(np.asarray(g_land))
        ga = ga[np.isfinite(ga)]
        if ga.size == 0:
            return None
        ymax = float(ga.max()) * grad_ylim_pad
        au = np.abs(np.asarray(g_auto))
        au = au[np.isfinite(au)]
        if au.size and au.max() <= ymax:
            ymax = float(au.max()) * grad_ylim_pad
        if ymax <= 0:
            return None
        return -ymax, ymax

    def plot_grad_shared(ax, g_auto, g_land, auto_color, auto_label, land_label):
        ax.axvline(lam_true, color='r', ls='--', lw=1, alpha=0.8)
        ax.axhline(0, color='k', lw=0.5, alpha=0.5)
        ax.plot(lam, g_land, color='k', lw=1.4, label=land_label)
        ax.plot(lam, g_auto, color=auto_color, lw=1.2, alpha=0.85,
                label=auto_label)
        ax.set_ylabel(r'$d\mathcal{L}/d\lambda$')
        yl = landscape_ylim(g_land, g_auto)
        if yl is not None:
            ax.set_ylim(*yl)
        ax.grid(alpha=0.3)
        ax.legend(loc='best', fontsize=9)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes[0, 0].plot(lam, L_lp, color='#E65100', lw=1.6)
    axes[0, 0].axvline(lam_true, color='r', ls='--', lw=1, alpha=0.8)
    axes[0, 0].set_title('LP solver: loss landscape')
    axes[0, 0].set_ylabel(r'$\mathcal{L}(\lambda)$'); axes[0, 0].grid(alpha=0.3)

    axes[1, 0].plot(lam, L_qp, color='#1565C0', lw=1.6)
    axes[1, 0].axvline(lam_true, color='r', ls='--', lw=1, alpha=0.8)
    axes[1, 0].set_title(f'QP solver (reg={qp_reg}): loss landscape')
    axes[1, 0].set_ylabel(r'$\mathcal{L}(\lambda)$'); axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_xlabel(r'$\lambda$')

    plot_grad_shared(axes[0, 1], g_auto_lp, g_land_lp,
                     auto_color='#E65100',
                     auto_label='autograd (LP solver)',
                     land_label=r'np.gradient($\mathcal{L}$) — landscape')
    axes[0, 1].set_title(f'LP autograd vs landscape   (sign-match: {sign_agree_lp:.0%})')

    plot_grad_shared(axes[1, 1], g_auto_qp, g_land_qp,
                     auto_color='#1565C0',
                     auto_label='autograd (QP solver)',
                     land_label=r'np.gradient($\mathcal{L}$) — landscape')
    axes[1, 1].set_title(f'QP autograd vs landscape   (sign-match: {sign_agree_qp:.0%})')
    axes[1, 1].set_xlabel(r'$\lambda$')

    for ax in axes.flat:
        ax.set_xlim(lam[0], lam[-1])

    if out_pdf is None:
        out_pdf = npz_path[:-4] + '.pdf' if npz_path.endswith('.npz') else npz_path + '.pdf'
    plt.tight_layout(); plt.savefig(out_pdf, bbox_inches='tight'); plt.close(fig)
    print(f"Replotted from {npz_path}  ->  {out_pdf}  "
          f"(grad ylim q={grad_ylim_quantile})")


def _diag_sweep(lambda_values, r_true, H_true, gamma_true, agg=None, label=""):
    """Sweep λ and return (L_reward, L_pen) components separately.

    L_reward(λ) = -r_true^T y*(λ)
    L_pen(λ)    = ||γ_true - H_true y*(λ)||²
    """
    n = len(lambda_values)
    L_reward = np.zeros(n)
    L_pen = np.zeros(n)
    p_ref = torch.tensor([PRICE], dtype=torch.float32)
    for i, lam_val in enumerate(lambda_values):
        if (i + 1) % 50 == 0:
            print(f"  [{label}] {i+1}/{n}")
        lam = torch.tensor([lam_val], dtype=torch.float32)
        mdp = create_inventory_from_lambda(lam, B_MAX, S_MAX, Q_MAX,
                                           FOC, VOC, HC, BC, BETA, p_ref)
        S = mdp.P.shape[1]
        gamma = mdp.build_gamma(1, S, torch.device("cpu"))
        gamma = torch.clamp(gamma, min=1e-8)
        gamma = gamma / gamma.sum(dim=1, keepdim=True)
        if agg is not None:
            P_agg, r_agg, gamma_agg, info = agg(mdp.P, mdp.r, gamma, tau=0.5)
            H_agg = mdp.build_H(P_agg)
            y_agg = DifferentiableLPSolver.apply(H_agg, r_agg.reshape(1, -1), gamma_agg)
            y_star = SoftAggDense.disaggregate_y(y_agg, info["W"], N_ACTIONS)
        else:
            H = mdp.build_H(mdp.P)
            y_star = DifferentiableLPSolver.apply(H, mdp.r.reshape(1, -1), gamma)
        L_reward[i] = -(r_true * y_star).sum().item()
        viol = gamma_true - (H_true @ y_star.unsqueeze(-1)).squeeze(-1)
        L_pen[i] = viol.pow(2).sum().item()
    return L_reward, L_pen


def diagnose_rho_decomposition(lambda_values, r_true, H_true, gamma_true,
                               agg, n_clusters, output_dir,
                               rho_list=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e4)):
    """Produce evidence for: (1) penalty-bias floor in soft agg, and
    (2) smoothing analogous to perturbed optimizers."""
    print("[diagnose] sweeping exact LP ...")
    R_lp, P_lp = _diag_sweep(lambda_values, r_true, H_true, gamma_true,
                             agg=None, label="LP")
    print(f"[diagnose] sweeping soft agg (M={n_clusters}) ...")
    R_ag, P_ag = _diag_sweep(lambda_values, r_true, H_true, gamma_true,
                             agg=agg, label="Agg")

    idx = int(np.argmin(np.abs(lambda_values - LAMBDA_TRUE)))
    dr_lp = np.gradient(R_lp, lambda_values)
    dr_ag = np.gradient(R_ag, lambda_values)
    scale = max(1e-6, 1e-3 * float(np.max(np.abs(dr_lp) + np.abs(dr_ag))))
    flat_lp = float(np.mean(np.abs(dr_lp) < scale))
    flat_ag = float(np.mean(np.abs(dr_ag) < scale))

    print("\n" + "=" * 78)
    print(f"Evidence (λ_true = {LAMBDA_TRUE},  grid λ[{idx}] = {lambda_values[idx]:.3f})")
    print("=" * 78)
    print("(1) Feasibility floor at λ_true  —  L_pen = ||γ_true - H_true y*||²")
    print(f"     Exact LP :  L_pen(λ_true) = {P_lp[idx]:.3e}   "
          f"argmin_λ L_pen = {lambda_values[int(np.argmin(P_lp))]:.3f}")
    print(f"     Soft Agg :  L_pen(λ_true) = {P_ag[idx]:.3e}   "
          f"argmin_λ L_pen = {lambda_values[int(np.argmin(P_ag))]:.3f}")
    print("     → Exact LP feasibility is exact at λ_true (zero floor).")
    print("     → Soft Agg has an irreducible aggregation residual >0 at λ_true,")
    print("       and its argmin can sit away from λ_true.")
    print()
    print("(2) Smoothing of L_reward(λ) = -r_true^T y*(λ)  [perturbed-opt analogy]")
    print(f"     std of dL_reward/dλ :   LP = {np.std(dr_lp):.3e}   Agg = {np.std(dr_ag):.3e}")
    print(f"     fraction of grid with |dL_reward/dλ| < {scale:.1e} :"
          f"   LP = {flat_lp:.1%}   Agg = {flat_ag:.1%}")
    print("     → LP reward is piecewise-flat (vertex structure of the exact LP).")
    print("     → Agg reward is globally smoother — disaggregation via soft W")
    print("       mixes vertex solutions of the reduced LP, just as a perturbed")
    print("       optimizer's expectation mixes argmaxes over noise.")
    print()
    print("(3) argmin of L_total(λ;ρ) = L_reward + ρ·L_pen  as ρ varies")
    print(f"     {'rho':>10}   {'argmin LP':>10}   {'argmin Agg':>10}   {'|Δλ_agg|':>10}")
    for rho in rho_list:
        tot_lp = R_lp + rho * P_lp
        tot_ag = R_ag + rho * P_ag
        lam_lp = lambda_values[int(np.argmin(tot_lp))]
        lam_ag = lambda_values[int(np.argmin(tot_ag))]
        print(f"     {rho:>10.3g}   {lam_lp:>10.3f}   {lam_ag:>10.3f}   "
              f"{abs(lam_ag - LAMBDA_TRUE):>10.3f}")
    print(f"     (λ_true = {LAMBDA_TRUE})")
    print("     → ρ ↓  :  LP drifts AWAY from λ_true (penalty is its only identifier);")
    print("              Agg drifts TOWARD λ_true (unloads the biased penalty term).")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panels = [
        (r'$L_{\rm reward}=-r_{\rm true}^\top y^*(\lambda)$', R_lp, R_ag),
        (r'$L_{\rm pen}=\|\gamma_{\rm true}-H_{\rm true}\,y^*(\lambda)\|^2$', P_lp, P_ag),
        (r'$dL_{\rm reward}/d\lambda$', dr_lp, dr_ag),
    ]
    for col, (ttl, y_lp, y_ag) in enumerate(panels):
        for ax, y, name, c in [(axes[0, col], y_lp, 'Exact LP', '#455A64'),
                               (axes[1, col], y_ag, f'Soft Agg (M={n_clusters})', '#1565C0')]:
            ax.plot(lambda_values, y, lw=1.4, color=c)
            ax.axvline(LAMBDA_TRUE, color='r', ls='--', lw=1, alpha=0.8)
            ax.axhline(0, color='k', lw=0.5, alpha=0.4)
            ax.set_title(f'{name}: {ttl}', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(lambda_values[0], lambda_values[-1])
    for ax in axes[1]:
        ax.set_xlabel(r'$\lambda_{\rm pred}$')
    out = os.path.join(output_dir, 'rho_decomposition.pdf')
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved decomposition figure to {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Combined comparison: original LP vs soft state aggregation")
    parser.add_argument('--n-points', '--n-grid', dest='n_grid', type=int, default=400,
                        help='Number of lambda values to sweep')
    parser.add_argument('--clusters', type=int, nargs='+', default=[10, 20, 30],
                        help='Aggregation cluster sizes')
    parser.add_argument('--lam-min', type=float, default=0.5,
                        help='Min lambda value')
    parser.add_argument('--lambda-max', '--lam-max', dest='lam_max', type=float, default=20.0,
                        help='Max lambda value')
    parser.add_argument('--plot-3col-only', action='store_true',
                        help='Only generate the 2×3 landscape-vs-gradient figure '
                             '(skips the original per-metric combined_*.pdf sweeps)')
    parser.add_argument('--cluster-3col', type=int, default=30,
                        help='Cluster count M used for the soft-agg panel of the 2×3 figure')
    parser.add_argument('--rho-3col', type=float, default=10.0,
                        help='Small ρ used for the augmented-Lagrangian panel of the 2×3 figure')
    parser.add_argument('--seed-3col', type=int, default=None,
                        help='Seed for the random SoftAggDense init in the 2×3 figure. '
                             'If omitted, infer it from the checkpoint/results.json when possible.')
    parser.add_argument('--agg-ckpt', type=str, default=None,
                        help='Path to a SoftAggDense checkpoint saved by '
                             'train_elasticity.py (agg_state_final.pt) or a fixed-W '
                             'checkpoint (agg_W_fixed.pt). If '
                             'provided, this overrides the default '
                             'trained_agg_M{M}.pt lookup.')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Optional experiment directory containing results.json. '
                             'If omitted, we auto-detect it from --agg-ckpt when possible.')
    parser.add_argument('--lambda-ref', type=float, default=None,
                        help='Optional explicit reference lambda. If omitted and results.json '
                             'is available, infer from the saved experiment sample.')
    parser.add_argument('--price-ref', type=float, default=None,
                        help='Optional explicit reference price. If omitted and results.json '
                             'is available, infer from the saved experiment sample.')
    parser.add_argument('--reference-split', type=str, choices=['train', 'test', 'all'],
                        default='train',
                        help='Which split to use when inferring lambda/price from results.json.')
    parser.add_argument('--reference-index', type=int, default=0,
                        help='Index within the chosen split when inferring lambda/price.')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Cache directory for expensive sweeps (default: mdp_lp/plot_results/cache)')
    parser.add_argument('--recompute', type=str, nargs='*', default=[],
                        help='Which cached results to recompute: ori dfl dfl_soft_agg (or all). '
                             'Example: --recompute dfl_soft_agg')
    parser.add_argument('--diagnose-rho', action='store_true',
                        help='Decompose the loss into L_reward + ρ·L_pen and sweep ρ '
                             'to show: (i) soft-agg has non-zero feasibility floor at λ_true; '
                             '(ii) soft-agg reward landscape is smoother than exact LP '
                             '(perturbed-optimizer analogy).')
    parser.add_argument('--diagnose-solver-grad', action='store_true',
                        help='Compare autograd dL/dλ from DifferentiableLPSolver vs '
                             'QP-regularized solver against the landscape gradient '
                             'np.gradient(L). Uses the projected-true aggregated loss.')
    parser.add_argument('--diagnose-rho-value', type=float, default=0.01,
                        help='ρ used in --diagnose-solver-grad.')
    parser.add_argument('--diagnose-qp-reg', type=float, default=1.0,
                        help='QP regularizer used in --diagnose-solver-grad.')
    parser.add_argument('--diagnose-trainable-w', action='store_true',
                        help='Use trainable soft W (fixed_W=False) in '
                             '--diagnose-solver-grad so that W depends smoothly on λ.')
    parser.add_argument('--recompute-solver-grad', action='store_true',
                        help='Force a fresh sweep in --diagnose-solver-grad even if '
                             'solver_gradient_comparison{_panel_a}.npz already exists.')
    parser.add_argument('--replot-solver-grad', action='store_true',
                        help='Skip computation; load saved '
                             'solver_gradient_comparison{_panel_a}.npz and replot the '
                             'PDF with robust y-limits on the gradient panels.')
    parser.add_argument('--replot-npz', type=str, nargs='*', default=None,
                        help='Explicit list of .npz files to replot. Defaults to '
                             '{both panel-a and non-panel-a} solver_gradient_comparison*.npz '
                             'in plot_results/ when --replot-solver-grad is set.')
    parser.add_argument('--replot-grad-quantile', type=float, default=0.99,
                        help='Quantile of |grad| used to set y-limits on the gradient '
                             'panels when replotting (default: 0.99).')
    parser.add_argument('--diagnose-panel-a', action='store_true',
                        help='Run --diagnose-solver-grad on panel-(a) bare LP loss '
                             'L=-r_true^T y*(λ) (no aggregation, no penalty).')
    # Sketch-and-solve column for the landscape comparison plot.
    parser.add_argument('--with-sketch', action='store_true',
                        help='Add a 4th column "(d) Soft agg + Sketch" to the landscape '
                             'figure, sweeping λ with the sketched-LP solver and saving as '
                             'landscape_comparison_4col.pdf.')
    parser.add_argument('--sketch-K', type=int, default=4,
                        help='Number of sketches K averaged per λ in (d).')
    parser.add_argument('--sketch-ratio', type=float, default=0.7,
                        help='Sketched-row count k as fraction of M for (d).')
    parser.add_argument('--sketch-mode', type=str, default='gaussian',
                        choices=['gaussian', 'rademacher', 'subsample'],
                        help='Sketch matrix distribution for (d).')
    parser.add_argument('--sketch-seed', type=int, default=0,
                        help='Sketch RNG seed (held fixed across λ so the curve is '
                             'deterministic and reproducible).')
    # 1x3 three-methods row: r^T y* | aug. Lagrangian | sketch-and-solve.
    parser.add_argument('--three-methods-row', action='store_true',
                        help='Produce a 2x3 figure: row 1 = loss landscape, '
                             'row 2 = autograd dL/dλ. Columns: '
                             '(a) -r_true^T y*, (b) augmented Lagrangian, '
                             '(c) augmented Lagrangian + sketch. Saved as '
                             'landscape_three_methods_row.pdf.')
    parser.add_argument('--three-methods-rho', type=float, default=10.0,
                        help='ρ used in the augmented-Lagrangian columns (b) '
                             'and (c) of --three-methods-row.')
    parser.add_argument('--three-methods-K', type=int, default=128,
                        help='Number of sketches K averaged in panel (c) of '
                             '--three-methods-row. Override of --sketch-K so '
                             'this figure can use a more aggressive default '
                             'without affecting other figures.')
    parser.add_argument('--three-methods-ratio', type=float, default=0.7,
                        help='Row-sketch keep ratio k/M in panel (c) of '
                             '--three-methods-row. Smaller = more aggressive '
                             'constraint relaxation (more decorrelation '
                             'between sketches, but also more bias in the '
                             'ensemble-mean LP solution — which shifts the '
                             'augmented-Lagrangian minimum away from '
                             'lambda_true).')
    parser.add_argument('--three-methods-cost-sigma', type=float, default=0.0,
                        help='Cost-side perturbation noise scale σ in panel '
                             '(c) of --three-methods-row. Each of the K '
                             'sketches solves max (r + σ ε_k)^T y under a '
                             'different row-sketched constraint set. Larger '
                             'σ adds bias to the LP optimum (each perturbed '
                             'argmax sits at a different vertex), which '
                             'shifts the augmented-Lagrangian minimum away '
                             'from lambda_true. Default 0 keeps cost '
                             'unbiased and relies on row sketching alone for '
                             'smoothing.')
    # 2x3 sketch × ρ landscape figure.
    parser.add_argument('--sketch-rho-fig', action='store_true',
                        help='Produce the 2x3 sketch × ρ landscape comparison figure '
                             '(loss_landscape_sketch_rho.pdf).')
    parser.add_argument('--sketch-rho-fig-rhos', type=float, nargs='+',
                        default=[0.01, 1.0, 100.0],
                        help='ρ values for --sketch-rho-fig columns.')
    parser.add_argument('--sketch-rho-fig-ratios', type=float, nargs='+',
                        default=[0.2, 0.7],
                        help='Sketch k/M ratios overlaid in each subplot of '
                             '--sketch-rho-fig (in addition to the no-sketch curve).')
    parser.add_argument('--sketch-rho-fig-no-agg', action='store_true',
                        help='Run --sketch-rho-fig on the original (S=N_STATES) LP '
                             'instead of the soft-aggregated LP. Useful for showing '
                             'sketching\'s smoothing effect without the smoothing that '
                             'soft-agg already provides.')
    parser.add_argument('--landscape-trainable-w', action='store_true',
                        help='Build SoftAggDense in plot_landscape_comparison with the '
                             'trainable embedding-network W. Default is fixed random W '
                             '(matches diagnose_solver_gradients).')
    args = parser.parse_args()
    cache_dir = args.cache_dir or os.path.join(_default_plot_results_dir(), 'cache')
    recompute = args.recompute

    # d = np.load("mdp_lp/plot_results/solver_gradient_comparison_panel_a.npz", allow_pickle=True)
    d = np.load("mdp_lp/plot_results/solver_gradient_comparison.npz", allow_pickle=True)
    lam = d["lambda_values"]
    g_auto_lp = d["g_auto_lp"]; g_auto_qp = d["g_auto_qp"]
    g_land_lp = d["g_land_lp"]; g_land_qp = d["g_land_qp"]
    meta = d["meta"].item()

    if any(x.lower() == "all" for x in recompute):
        recompute = ["ori", "dfl", "dfl_soft_agg"]

    results_dir = None
    cfg = None
    if args.results_dir:
        results_dir = Path(args.results_dir)
    elif args.agg_ckpt:
        ckpt_parent = Path(args.agg_ckpt).expanduser().resolve().parent
        if (ckpt_parent / "results.json").exists():
            results_dir = ckpt_parent

    if results_dir is not None:
        cfg_path = results_dir / "results.json"
        if not cfg_path.exists():
            raise ValueError(f"Could not find results.json in {results_dir}")
        cfg = json.loads(cfg_path.read_text())
        lambda_ref, price_ref, ref_info = _resolve_reference_values(
            cfg,
            ref_split=args.reference_split,
            ref_index=int(args.reference_index),
            lambda_ref_override=args.lambda_ref,
            price_ref_override=args.price_ref,
        )
        _apply_experiment_context(cfg, lambda_ref, price_ref)
        ref_msg = (
            f"[Config] loaded {cfg_path} | seed={ref_info['seed']} split={ref_info['split']} "
            f"ref_index={ref_info['ref_index']} sample_index={ref_info['sample_index']} "
            f"lambda_true={LAMBDA_TRUE:.6f} price={PRICE:.6f}"
        )
        if ref_info["p_related_ref"] is not None:
            ref_msg += f" p_related_ref={ref_info['p_related_ref']:.6f}"
        print(ref_msg)
    else:
        if args.lambda_ref is not None:
            globals()["LAMBDA_TRUE"] = float(args.lambda_ref)
        if args.price_ref is not None:
            globals()["PRICE"] = float(args.price_ref)

    seed_3col, seed_3col_source = _resolve_saved_seed(
        explicit_seed=args.seed_3col,
        args_cfg=cfg,
        agg_ckpt=args.agg_ckpt,
    )
    print(f"[Config] 3-col seed={seed_3col} (source={seed_3col_source})")

    r_true, H_true, gamma_true, mdp_true = build_true_mdp()
    lambda_values = np.linspace(float(args.lam_min), float(args.lam_max), int(args.n_grid))
    print(f"[Scan] lambda range=[{float(args.lam_min):.4f}, {float(args.lam_max):.4f}]  n_grid={int(args.n_grid)}")

    if args.replot_solver_grad:
        out_dir = _default_plot_results_dir()
        if args.replot_npz:
            paths = list(args.replot_npz)
        else:
            paths = [
                os.path.join(out_dir, 'solver_gradient_comparison.npz'),
                os.path.join(out_dir, 'solver_gradient_comparison_panel_a.npz'),
            ]
            paths = [p for p in paths if os.path.exists(p)]
        if not paths:
            print("[replot] No solver_gradient_comparison*.npz found in "
                  f"{out_dir}; run --diagnose-solver-grad first.")
            return
        for p in paths:
            replot_solver_gradient_from_npz(
                p, grad_ylim_quantile=float(args.replot_grad_quantile))
        return

    sketch_cfg = None
    if bool(args.with_sketch):
        sketch_cfg = {
            "n_sketches": int(args.sketch_K),
            "sketch_ratio": float(args.sketch_ratio),
            "sketch_mode": str(args.sketch_mode),
            "sketch_seed": int(args.sketch_seed),
        }

    if args.three_methods_row:
        out_dir = _default_plot_results_dir()
        plot_three_methods_row(
            lambda_values, r_true, H_true, gamma_true,
            rho=float(args.three_methods_rho),
            sketch_K=int(args.three_methods_K),
            sketch_ratio=float(args.three_methods_ratio),
            sketch_mode=str(args.sketch_mode),
            sketch_seed=int(args.sketch_seed),
            cost_sigma=float(args.three_methods_cost_sigma),
            output_dir=out_dir,
            cache_dir=cache_dir,
            recompute=recompute,
        )
        return

    if args.sketch_rho_fig:
        out_dir = _default_plot_results_dir()
        if args.sketch_rho_fig_no_agg:
            agg = None
            print("[sketch-rho] using ORIGINAL LP (no soft aggregation)")
        else:
            torch.manual_seed(seed_3col)
            agg = SoftAggDense(n_states=N_STATES, n_actions=N_ACTIONS,
                               n_clusters=args.cluster_3col, r_embed=8,
                               fixed_W=True, fixed_W_seed=seed_3col)
            agg.eval()
            print(f"[sketch-rho] using soft-agg LP (M={args.cluster_3col})")
        plot_loss_landscape_sketch_rho(
            lambda_values, r_true, H_true, gamma_true,
            agg=agg, mdp_true=mdp_true,
            rhos=tuple(float(r) for r in args.sketch_rho_fig_rhos),
            sketch_ratios=tuple(float(r) for r in args.sketch_rho_fig_ratios),
            sketch_K=int(args.sketch_K),
            sketch_seed=int(args.sketch_seed),
            sketch_mode=str(args.sketch_mode),
            output_dir=out_dir,
            cache_dir=cache_dir,
            recompute=recompute,
        )
        return

    if args.plot_3col_only:
        plot_landscape_comparison(
            lambda_values, r_true, H_true, gamma_true,
            n_clusters=args.cluster_3col,
            rho_small=args.rho_3col,
            seed=seed_3col,
            agg_ckpt=args.agg_ckpt,
            cache_dir=cache_dir,
            recompute=recompute,
            sketch_cfg=sketch_cfg,
            mdp_true=mdp_true,
            fixed_W=(not args.landscape_trainable_w))
        return

    if args.diagnose_solver_grad:
        out_dir = _default_plot_results_dir()
        torch.manual_seed(seed_3col)
        if bool(args.diagnose_panel_a):
            agg = None
            print("[solver-grad] panel (a): bare LP, no aggregation, no penalty")
        else:
            use_fixed = not bool(args.diagnose_trainable_w)
            agg = SoftAggDense(n_states=N_STATES, n_actions=N_ACTIONS,
                               n_clusters=args.cluster_3col, r_embed=8,
                               fixed_W=use_fixed, fixed_W_seed=seed_3col)
            print(f"[solver-grad] agg fixed_W={use_fixed}")
        if agg is not None and args.agg_ckpt is not None and os.path.exists(args.agg_ckpt):
            ck = _load_torch_payload(args.agg_ckpt)
            sd = ck.get('state_dict', ck) if isinstance(ck, dict) else ck
            has_W_fixed = isinstance(sd, dict) and 'W_fixed' in sd
            agg = SoftAggDense(
                n_states=int(ck.get('n_states', N_STATES)),
                n_actions=int(ck.get('n_actions', N_ACTIONS)),
                n_clusters=int(ck.get('n_clusters', args.cluster_3col)),
                r_embed=int(ck.get('r_embed', 8)),
                fixed_W=has_W_fixed,
                fixed_W_seed=int(ck.get('seed', seed_3col)),
            )
            agg.load_state_dict(sd)
            print(f"[solver-grad] loaded agg from {args.agg_ckpt} (fixed_W={has_W_fixed})")
        if agg is not None:
            agg.eval()
        diagnose_solver_gradients(lambda_values, agg, out_dir,
                                  rho=float(args.diagnose_rho_value),
                                  qp_reg=float(args.diagnose_qp_reg),
                                  recompute=bool(args.recompute_solver_grad))
        return

    if args.diagnose_rho:
        out_dir = _default_plot_results_dir()
        torch.manual_seed(seed_3col)
        agg = SoftAggDense(n_states=N_STATES, n_actions=N_ACTIONS,
                           n_clusters=args.cluster_3col, r_embed=8)
        if args.agg_ckpt is not None and os.path.exists(args.agg_ckpt):
            ck = _load_torch_payload(args.agg_ckpt)
            sd = ck.get('state_dict', ck) if isinstance(ck, dict) else ck
            has_W_fixed = isinstance(sd, dict) and 'W_fixed' in sd
            agg = SoftAggDense(
                n_states=int(ck.get('n_states', N_STATES)),
                n_actions=int(ck.get('n_actions', N_ACTIONS)),
                n_clusters=int(ck.get('n_clusters', args.cluster_3col)),
                r_embed=int(ck.get('r_embed', 8)),
                fixed_W=has_W_fixed,
                fixed_W_seed=int(ck.get('seed', seed_3col)),
            )
            agg.load_state_dict(sd)
            print(f"[diagnose] loaded agg from {args.agg_ckpt} (fixed_W={has_W_fixed})")
        agg.eval()
        M = int(getattr(agg, "n_clusters", args.cluster_3col))
        diagnose_rho_decomposition(lambda_values, r_true, H_true, gamma_true,
                                   agg, M, out_dir)
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
        seed=seed_3col,
        agg_ckpt=args.agg_ckpt,
        cache_dir=cache_dir,
        recompute=recompute,
        sketch_cfg=sketch_cfg,
        mdp_true=mdp_true)


if __name__ == "__main__":
    main()
