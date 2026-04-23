"""
Evidence probe: with a FIXED random W, does DFL training trace the loss landscape?

Setup (single-sample, scalar λ_pred):
  - SoftAggDense with fixed_W=True — W is a random row-stochastic matrix, frozen,
    independent of P, r, and λ.
  - True problem: one λ_true, one price p. We hold them fixed and sweep λ_pred.
  - Loss (same form used in training):
        L(λ) = -r_true^T y_disagg(λ) + ρ · ||γ_true - H_true y_disagg(λ)||²
    where y_disagg(λ) = disaggregate( LP_agg(H(λ), r(λ), γ) ).

Three pieces of evidence:
  (A) Scan λ over a grid, compute L(λ) via the forward-only LP solver (Gurobi).
  (B) Treat λ_pred as a torch.Parameter and run Adam via the DIFFERENTIABLE LP.
      Track the trajectory (λ_t, L_t) and overlay on the landscape.
  (C) At each λ in the grid, compute the DFL autograd gradient and compare
      against the finite-difference gradient of L. Agreement within basis cells
      is the cleanest evidence that DFL is descending the landscape.

Usage:
    python -m mdp_lp.test_fixed_W_landscape
"""

import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from mdp_lp.inventory_elasticity import create_inventory_from_lambda
from mdp_lp.solvers.dense_lp_solver import DenseLPSolver
from mdp_lp.solvers.differentiable_lp import DifferentiableLPSolver
from mdp_lp.soft_state_aggregation import SoftAggDense


# ─── Problem parameters (match plot_state_agg_landscape.py for comparability) ───
B_MAX, S_MAX, Q_MAX = 5, 50, 30
FOC, VOC, HC, BC, BETA, PRICE = 0, 0.5, 0.1, 0, 0.95, 10.0
LAMBDA_TRUE = 10.0
RHO = 10000.0

N_STATES = B_MAX + S_MAX + 1   # 56
N_ACTIONS = Q_MAX + 1           # 31
M_CLUSTERS = 10
TAU = 0.5
NORMALIZE_DISAGG = True
EXPECTED_SUM = 1.0 / (1.0 - BETA)


def build_mdp(lam_val):
    lam = torch.as_tensor([float(lam_val)], dtype=torch.float32)
    p = torch.tensor([PRICE], dtype=torch.float32)
    return create_inventory_from_lambda(lam, B_MAX, S_MAX, Q_MAX,
                                        FOC, VOC, HC, BC, BETA, p)


def uniform_gamma(mdp):
    S = mdp.P.shape[1]
    g = mdp.build_gamma(1, S, torch.device("cpu"))
    g = torch.clamp(g, min=1e-8)
    return g / g.sum(dim=1, keepdim=True)


def dfl_loss_forward_only(lam_val, agg, r_true_flat, H_true, gamma_true, solver):
    """L(λ) using the non-differentiable Gurobi solver (for landscape scan)."""
    with torch.no_grad():
        mdp = build_mdp(lam_val)
        gamma = uniform_gamma(mdp)
        P_agg, r_agg, gamma_agg, info = agg(mdp.P, mdp.r, gamma, tau=TAU)
        H_agg = mdp.build_H(P_agg)
        r_flat_agg = r_agg.reshape(1, -1)
        y_agg, _, _ = solver.solve(H_agg, r_flat_agg, gamma_agg)
        y_orig = SoftAggDense.disaggregate_y(y_agg, info["W"], N_ACTIONS)
        if NORMALIZE_DISAGG:
            y_sum = y_orig.sum(dim=1, keepdim=True).clamp_min(1e-8)
            y_orig = y_orig * (EXPECTED_SUM / y_sum)
        dfl = -(r_true_flat * y_orig).sum().item()
        violation = gamma_true - (H_true @ y_orig.unsqueeze(-1)).squeeze(-1)
        pen = violation.pow(2).sum().item()
    return dfl + RHO * pen


def dfl_loss_differentiable(lam_tensor, agg, r_true_flat, H_true, gamma_true):
    """L(λ) built through the DIFFERENTIABLE LP (autograd flows back to lam_tensor)."""
    lam = lam_tensor.view(1)
    p = torch.tensor([PRICE], dtype=torch.float32)
    mdp = create_inventory_from_lambda(lam, B_MAX, S_MAX, Q_MAX,
                                       FOC, VOC, HC, BC, BETA, p)
    gamma = uniform_gamma(mdp)
    P_agg, r_agg, gamma_agg, info = agg(mdp.P, mdp.r, gamma, tau=TAU)
    H_agg = mdp.build_H(P_agg)
    r_flat_agg = r_agg.reshape(1, -1)
    y_agg = DifferentiableLPSolver.apply(H_agg, r_flat_agg, gamma_agg)
    y_orig = SoftAggDense.disaggregate_y(y_agg, info["W"], N_ACTIONS)
    if NORMALIZE_DISAGG:
        y_sum = y_orig.sum(dim=1, keepdim=True).clamp_min(1e-8)
        y_orig = y_orig * (EXPECTED_SUM / y_sum)
    dfl = -(r_true_flat * y_orig).sum()
    violation = gamma_true - (H_true @ y_orig.unsqueeze(-1)).squeeze(-1)
    pen = violation.pow(2).sum()
    return dfl + RHO * pen


def main():
    out_dir = "mdp_lp/plot_results"

    # ── Fixed-W aggregator (non-trainable; independent of P, λ) ──
    agg = SoftAggDense(
        n_states=N_STATES, n_actions=N_ACTIONS, n_clusters=M_CLUSTERS,
        r_embed=8, fixed_W=True, fixed_W_seed=42,
    )
    agg.eval()
    for p in agg.parameters():  # no parameters, but buffers already frozen.
        p.requires_grad_(False)

    # Sanity: W is a valid row-stochastic matrix.
    W = agg.W_fixed
    row_sums = W.sum(dim=1)
    print(f"[W] shape={tuple(W.shape)}  row_sum min/max=({row_sums.min():.6f}, {row_sums.max():.6f})")

    # ── True problem ──
    mdp_true = build_mdp(LAMBDA_TRUE)
    r_true_flat = mdp_true.r.reshape(1, -1).detach()
    H_true = mdp_true.build_H(mdp_true.P).detach()
    gamma_true = uniform_gamma(mdp_true).detach()

    solver = DenseLPSolver()

    # (A) Landscape scan
    lambda_grid = np.linspace(0.5, 50.0, 150)
    t0 = time.time()
    L_grid = np.array([
        dfl_loss_forward_only(lam, agg, r_true_flat, H_true, gamma_true, solver)
        for lam in lambda_grid
    ])
    print(f"[Landscape] scanned {len(lambda_grid)} λ values in {time.time()-t0:.1f}s")

    # (B) Training trajectories from several inits
    init_lams = [2.0, 6.0, 18.0, 35.0]
    n_steps = 80
    lr = 0.3
    trajectories = {}
    for init in init_lams:
        lam_param = torch.tensor(float(init), requires_grad=True)
        opt = torch.optim.Adam([lam_param], lr=lr)
        traj_lams, traj_losses = [], []
        for _ in range(n_steps):
            opt.zero_grad()
            loss = dfl_loss_differentiable(lam_param, agg, r_true_flat, H_true, gamma_true)
            loss.backward()
            traj_lams.append(float(lam_param.detach().clone()))
            traj_losses.append(float(loss.detach().clone()))
            opt.step()
            # soft box: keep λ strictly positive and within scan range.
            with torch.no_grad():
                lam_param.clamp_(min=0.5, max=50.0)
        traj_lams.append(float(lam_param.detach().clone()))
        trajectories[init] = (np.array(traj_lams), np.array(traj_losses))
        print(f"[SGD] init λ₀={init:5.1f} → final λ={traj_lams[-1]:.3f}   final L={traj_losses[-1]:.2f}")

    # (C) Autograd vs finite-difference gradient on a coarse grid.
    # Use a SMALL h for the FD so it stays within-basis (matches what autograd
    # computes). Basis changes will still show up as ratio ≠ 1 at a few points.
    grad_grid = np.linspace(1.0, 45.0, 90)
    g_autograd = np.zeros_like(grad_grid)
    g_fd_small_h = np.zeros_like(grad_grid)
    h_fd = 0.05
    for i, lam in enumerate(grad_grid):
        lam_t = torch.tensor(float(lam), requires_grad=True)
        L = dfl_loss_differentiable(lam_t, agg, r_true_flat, H_true, gamma_true)
        (g,) = torch.autograd.grad(L, lam_t)
        g_autograd[i] = float(g)
        L_hi = dfl_loss_forward_only(lam + h_fd, agg, r_true_flat, H_true, gamma_true, solver)
        L_lo = dfl_loss_forward_only(lam - h_fd, agg, r_true_flat, H_true, gamma_true, solver)
        g_fd_small_h[i] = (L_hi - L_lo) / (2 * h_fd)
    # Also keep the coarse FD from the landscape scan — useful to see basis jumps
    g_fd_scan = np.gradient(L_grid, lambda_grid)

    # ────────────────────────────── Plot ──────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(13, 12),
                             gridspec_kw={"height_ratios": [3, 2, 2]})
    ax = axes[0]
    ax.plot(lambda_grid, L_grid, "k-", lw=1.6, alpha=0.55,
            label=f"Landscape L(λ) — forward LP scan")
    cmap = plt.cm.viridis(np.linspace(0.05, 0.9, len(init_lams)))
    for (init, (traj_lams, traj_losses)), color in zip(trajectories.items(), cmap):
        ax.plot(traj_lams[:-1], traj_losses, "o-", color=color, ms=3, lw=1.2, alpha=0.9,
                label=f"Adam from λ₀={init}")
        ax.plot(traj_lams[0], traj_losses[0], "s", color=color, ms=10, mec="k", mew=0.6)
        # evaluate L at final λ (closed-form via forward LP)
        L_final = dfl_loss_forward_only(traj_lams[-1], agg, r_true_flat, H_true, gamma_true, solver)
        ax.plot(traj_lams[-1], L_final, "*", color=color, ms=16, mec="k", mew=0.6)
    ax.axvline(LAMBDA_TRUE, color="red", ls="--", lw=1.4, label=f"λ_true={LAMBDA_TRUE}")
    ax.set_xlabel("λ_pred")
    ax.set_ylabel(f"L(λ) = -r_true·y*(λ) + {RHO:g}·‖γ-Hy*‖²")
    ax.set_title(f"Fixed random W (seed=42, M={M_CLUSTERS}):  DFL trajectories vs loss landscape\n"
                 f"■ = init, ★ = end.   Adam lr={lr}, {n_steps} steps")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Middle panel: autograd vs small-h FD on the same scale (symlog).
    ax2 = axes[1]
    ax2.plot(grad_grid, g_autograd, "o-", color="#1f77b4", ms=4,
             label="DFL autograd  ∂L/∂λ")
    ax2.plot(grad_grid, g_fd_small_h, "x--", color="#d62728", ms=5, lw=1.0,
             label=f"Finite diff (h={h_fd})")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.axvline(LAMBDA_TRUE, color="red", ls="--", lw=1.2)
    ax2.set_yscale("symlog", linthresh=10)
    ax2.set_xlabel("λ")
    ax2.set_ylabel("∂L/∂λ  (symlog)")
    ax2.set_title("Within-basis gradient agreement: DFL autograd ≈ FD with small h.\n"
                  "Isolated spikes = basis changes occurring inside [λ-h, λ+h].")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Bottom panel: ratio autograd / FD (should be ≈1 within basis).
    ax3 = axes[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(g_fd_small_h) > 1e-6,
                         g_autograd / g_fd_small_h, np.nan)
    ax3.plot(grad_grid, ratio, "o-", color="#2ca02c", ms=4,
             label="autograd / FD")
    ax3.axhline(1.0, color="k", lw=0.7, ls="--", label="ideal = 1")
    ax3.axvline(LAMBDA_TRUE, color="red", ls="--", lw=1.2)
    ax3.set_ylim(-0.5, 2.0)
    ax3.set_xlabel("λ")
    ax3.set_ylabel("ratio")
    ax3.set_title("Autograd/FD ratio — clustering near 1.0 confirms the DFL gradient = true landscape slope.")
    ax3.legend(loc="best", fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    png = f"{out_dir}/fixed_W_landscape_trajectory.png"
    pdf = f"{out_dir}/fixed_W_landscape_trajectory.pdf"
    plt.savefig(png, dpi=150, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    print(f"[Saved] {png}")
    print(f"[Saved] {pdf}")

    # ── Quantitative summary ──
    lam_min = float(lambda_grid[np.argmin(L_grid)])
    print(f"\n[Summary] λ with min L on scan:  {lam_min:.3f}   (λ_true={LAMBDA_TRUE})")
    print(f"[Summary] min L on scan:         {L_grid.min():.3f}")
    # Fraction of grad grid where autograd and FD have the same sign
    signs_match = np.sign(g_autograd) == np.sign(g_fd_small_h)
    print(f"[Summary] sign(autograd)==sign(FD[h={h_fd}]) on grid: "
          f"{signs_match.mean()*100:.1f}% ({signs_match.sum()}/{len(signs_match)})")
    # Ratio close to 1 quantifies true within-basis agreement
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(g_fd_small_h) > 1e-6, g_autograd / g_fd_small_h, np.nan)
    in_band = np.isfinite(ratio) & (np.abs(ratio - 1.0) < 0.1)
    print(f"[Summary] |ratio-1|<0.1 (within-basis agreement): "
          f"{in_band.mean()*100:.1f}% ({in_band.sum()}/{np.isfinite(ratio).sum()})")


if __name__ == "__main__":
    main()
