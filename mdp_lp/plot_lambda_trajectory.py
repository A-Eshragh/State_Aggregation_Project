"""
Sandbox: reproduce DFL training from run_elasticity_experiments.py
and plot per-sample lambda_pred trajectories.

Uses the same data generation (prepare_elasticity_data) and training logic
(train_eval_elasticity) but tracks lambda_pred for specific samples.

Usage:
    python -m mdp_lp.plot_lambda_trajectory
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

from mdp_lp.elasticity_model import LambdaPredictor
from mdp_lp.inventory_elasticity import create_inventory_from_lambda
from mdp_lp.solvers.differentiable_lp import DifferentiableLPSolver
from mdp_lp.soft_state_aggregation import SoftAggDense
from mdp_lp.train_elasticity import prepare_elasticity_data
from mdp_lp.utils import set_seed

# ── Experiment parameters (matching run_elasticity_experiments: exp_b, high_margin) ──
ARGS = SimpleNamespace(
    experiment_type='b',
    beta_0=3.5, beta_1=-1.0, beta_2=0.0, sigma_epsilon=0.5,
    p_lognormal_mean=5.0, p_lognormal_std=0.6,
    p_constant=5.0, fixed_p=False,
    p_related_lognormal_mean=8.0, p_related_lognormal_std=1.0,
    b_max=5, s_max=50, q_max=50,
    foc=0, voc=0.5, hc=0.1, bc=0, beta=0.95,
    n_samples=20,  # will be doubled inside prepare_elasticity_data
    n_epochs=15,
    lr=1e-2, hidden_dim=32,
    algo='dfl',
    loss_weight=0.0,
    aug_lag_rho=10000.0,
    grad_clip=10.0,
    n_agg=-1,
    sa_tau=0.5, sa_r_embed=8,
)

N_STATES = ARGS.b_max + ARGS.s_max + 1
N_ACTIONS = ARGS.q_max + 1


def train_dfl_track(train_loader, pretrain=False, agg=None, n_epochs=15):
    """
    Run DFL training (mimicking train_eval_elasticity) and record
    per-epoch (lambda_pred, lambda_true) for ALL samples seen in that epoch.

    Returns:
        all_epochs: list of dicts, each with 'lam_pred' and 'lam_true' arrays
    """
    device = torch.device("cpu")
    pred_model = LambdaPredictor(ARGS.hidden_dim, use_p_related=False, use_price_only=True).to(device)

    params = list(pred_model.parameters())
    if agg is not None:
        params += list(agg.parameters())

    # Pretrain (same as train_eval_elasticity)
    if pretrain:
        pt_optimizer = optim.Adam(pred_model.parameters(), lr=1e-2)
        pt_loss_fn = nn.MSELoss()
        pred_model.train()
        for ep in range(60):
            for batch in train_loader:
                feats, obj_true, lambda_true, prices, y_true, dual_true = batch
                lp = pred_model(feats)
                loss = pt_loss_fn(lp, lambda_true)
                pt_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pred_model.parameters(), max_norm=10.0)
                pt_optimizer.step()
        print(f"    Pretrain done.")

    optimizer = optim.Adam(params, lr=ARGS.lr)
    pred_loss_fn = nn.MSELoss()

    all_epochs = []

    pred_model.train()
    for epoch in range(n_epochs):
        epoch_lam_pred = []
        epoch_lam_true = []

        for batch_data in train_loader:
            feats, obj_true, lambda_true, prices, y_true, dual_true = batch_data
            batch_size = feats.shape[0]
            p = prices if prices.dim() == 1 else prices

            lambda_pred = pred_model(feats)
            lambda_pred_pos = torch.clamp(lambda_pred, min=1.0)

            # Record
            epoch_lam_pred.append(lambda_pred.detach().clone())
            epoch_lam_true.append(lambda_true.detach().clone())

            # Build predicted MDP
            mdp = create_inventory_from_lambda(
                lambda_pred_pos, ARGS.b_max, ARGS.s_max, ARGS.q_max,
                ARGS.foc, ARGS.voc, ARGS.hc, ARGS.bc, ARGS.beta, p
            )
            S = mdp.P.shape[1]
            gamma = mdp.build_gamma(batch_size, S, device)
            gamma = torch.clamp(gamma, min=1e-8)
            gamma = gamma / gamma.sum(dim=1, keepdim=True)

            P_use, r_use = mdp.P, mdp.r
            gamma_lp = gamma
            _agg_info = None
            if agg is not None:
                P_use, r_use, gamma_agg, _agg_info = agg(P_use, r_use, gamma, tau=ARGS.sa_tau)
                gamma_lp = gamma_agg

            H_use = mdp.build_H(P_use)
            r_flat_use = r_use.reshape(batch_size, -1)
            y_star = DifferentiableLPSolver.apply(H_use, r_flat_use, gamma_lp)

            if agg is not None and _agg_info is not None:
                y_star = SoftAggDense.disaggregate_y(y_star, _agg_info["W"], N_ACTIONS)

            # True MDP for loss
            with torch.no_grad():
                mdp_true = create_inventory_from_lambda(
                    lambda_true, ARGS.b_max, ARGS.s_max, ARGS.q_max,
                    ARGS.foc, ARGS.voc, ARGS.hc, ARGS.bc, ARGS.beta, p
                )
                r_true = mdp_true.r.reshape(batch_size, -1)
                H_true = mdp_true.build_H(mdp_true.P)

            violation = gamma - (H_true @ y_star.unsqueeze(-1)).squeeze(-1)
            aug_lag = ARGS.aug_lag_rho * violation.pow(2).mean()
            v_star = torch.sum(r_true * y_star, dim=1)
            dfl_loss = -v_star.mean() + aug_lag

            pred_loss = pred_loss_fn(lambda_pred, lambda_true)
            total_loss = dfl_loss + ARGS.loss_weight * pred_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=ARGS.grad_clip)
            optimizer.step()

        lp_cat = torch.cat(epoch_lam_pred).numpy()
        lt_cat = torch.cat(epoch_lam_true).numpy()
        all_epochs.append({'lam_pred': lp_cat, 'lam_true': lt_cat})

        print(f"    Epoch {epoch+1}/{n_epochs}: "
              f"mean_pred={lp_cat.mean():.2f}, mean_true={lt_cat.mean():.2f}, "
              f"MAE={np.abs(lp_cat - lt_cat).mean():.2f}")

    return all_epochs


def build_per_sample_trajectories(all_epochs):
    """
    Match samples across epochs by their lambda_true value.
    Returns dict: lambda_true_val -> list of lambda_pred per epoch.
    """
    # Use epoch 0 lambda_true values as reference
    ref_true = all_epochs[0]['lam_true']
    n_samples = len(ref_true)

    trajectories = {round(float(ref_true[i]), 4): [] for i in range(n_samples)}

    for ep_data in all_epochs:
        lt = ep_data['lam_true']
        lp = ep_data['lam_pred']
        # Match each reference sample to the closest lambda_true in this epoch
        for i in range(n_samples):
            key = round(float(ref_true[i]), 4)
            dists = np.abs(lt - ref_true[i])
            j = np.argmin(dists)
            trajectories[key].append(float(lp[j]))

    return trajectories


def main():
    set_seed(1)
    args = SimpleNamespace(**vars(ARGS))
    train_loader, test_loader, gt_elasticity, p, p_related = prepare_elasticity_data(args)

    # ── DFL, no pretraining ──
    print("\n[1/1] DFL, no pretraining...")
    set_seed(1)
    epochs_data = train_dfl_track(train_loader, pretrain=False, agg=None,
                                   n_epochs=ARGS.n_epochs)

    trajectories = build_per_sample_trajectories(epochs_data)

    # Show all trajectories with final error
    all_true = sorted(trajectories.keys())
    print(f"\nAll lambda_true values and final lambda_pred:")
    for lt in all_true:
        traj = trajectories[lt]
        err = abs(traj[-1] - lt)
        direction = "CONVERGE" if err < abs(traj[0] - lt) else "DIVERGE"
        print(f"  λ_true={lt:6.2f}  λ_pred: {traj[0]:6.1f} → {traj[-1]:6.1f}  "
              f"|err|={err:.1f}  {direction}")

    # Pick one that converges (closest to 6) and one that diverges (closest to 20)
    target_converge = min(all_true, key=lambda x: abs(x - 6.0))
    target_diverge = min(all_true, key=lambda x: abs(x - 20.0))
    print(f"\nSelected: converge={target_converge:.2f}, diverge={target_diverge:.2f}")

    # ── Plot: two panels ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, target, label in zip(axes,
                                  [target_converge, target_diverge],
                                  ['converges', 'diverges']):
        traj = trajectories[target]
        epochs = np.arange(len(traj))

        ax.plot(epochs, traj, 'b-', linewidth=2, marker='o', markersize=4,
                label='$\\lambda_{pred}$')
        ax.axhline(target, color='red', linestyle='--', linewidth=2,
                    label=f'$\\lambda_{{true}}$ = {target:.2f}')

        ax.plot(0, traj[0], 'go', markersize=10, zorder=5)
        ax.plot(len(traj) - 1, traj[-1], 'rs', markersize=10, zorder=5)

        ax.set_xlabel('Epoch', fontsize=13)
        ax.set_ylabel('$\\lambda_{pred}$', fontsize=13)
        ax.set_title(f'DFL (no pretrain), $\\lambda_{{true}}$ = {target:.2f} ({label})\n'
                      f'$\\lambda_{{pred}}$: {traj[0]:.1f} → {traj[-1]:.1f}',
                      fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle('DFL without pretraining: $\\lambda_{pred}$ trajectory',
                 fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig('mdp_lp/plot_results/lambda_trajectory.png', dpi=150, bbox_inches='tight')
    plt.savefig('mdp_lp/plot_results/lambda_trajectory.pdf', bbox_inches='tight')
    print("\nSaved to mdp_lp/plot_results/lambda_trajectory.{png,pdf}")


if __name__ == "__main__":
    main()
