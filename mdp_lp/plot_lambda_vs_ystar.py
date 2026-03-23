"""
Visualize the relationship between lambda_pred and y_star / v_star / DFL gradient.

Demonstrates:
1. v*(λ_pred) = r_true^T · y*(λ_pred) — piecewise smooth, non-convex
2. DFL loss landscape and gradient direction
3. Why DFL collapses λ to 0 from wrong initialization

Usage:
    python -m mdp_lp.plot_lambda_vs_ystar
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mdp_lp.inventory_elasticity import create_inventory_from_lambda
from mdp_lp.solvers.dense_lp_solver import DenseLPSolver
from mdp_lp.solvers.differentiable_lp import DifferentiableLPSolver
from mdp_lp.inventory_poisson_grad import poisson_pmf

# ── MDP parameters (match your experiments) ──────────────────────────
B_MAX = 5
S_MAX = 100
Q_MAX = 80
D_MAX = 90
FOC = 0
VOC = 0.5
HC = 0.1          # high_margin
BC = 0
BETA = 0.95
PRICE = 10.0      # selling price (constant for this analysis)

# Ground truth lambda for evaluation
LAMBDA_TRUE = 30.0


def solve_for_lambda(lam_val, price=PRICE):
    """Solve the inventory LP for a single lambda value. Returns y_star, v_pred, r_flat."""
    lam = torch.tensor([lam_val], dtype=torch.float32)
    p = torch.tensor([price], dtype=torch.float32)
    mdp = create_inventory_from_lambda(lam, B_MAX, S_MAX, Q_MAX, D_MAX,
                                       FOC, VOC, HC, BC, BETA, p)
    S = mdp.P.shape[1]
    gamma = mdp.build_gamma(1, S, torch.device("cpu"))
    gamma = torch.clamp(gamma, min=1e-8)
    gamma = gamma / gamma.sum(dim=1, keepdim=True)

    H = mdp.build_H(mdp.P)
    r_flat = mdp.r.reshape(1, -1)

    solver = DenseLPSolver()
    y_star, basis, dual = solver.solve(H, r_flat, gamma)
    v_pred = (r_flat * y_star).sum().item()

    return y_star, v_pred, r_flat, H, gamma, basis


def compute_dfl_gradient(lam_val, lambda_true=LAMBDA_TRUE, price=PRICE):
    """
    Compute the DFL gradient dL/dλ at a given λ_pred value.
    L = -r_true^T · y*(λ_pred)
    """
    lam_pred = torch.tensor([lam_val], dtype=torch.float32, requires_grad=True)
    p = torch.tensor([price], dtype=torch.float32)

    # Build predicted MDP (differentiable path)
    mdp_pred = create_inventory_from_lambda(lam_pred, B_MAX, S_MAX, Q_MAX, D_MAX,
                                             FOC, VOC, HC, BC, BETA, p)
    S = mdp_pred.P.shape[1]
    gamma = mdp_pred.build_gamma(1, S, torch.device("cpu"))
    gamma = torch.clamp(gamma, min=1e-8)
    gamma = gamma / gamma.sum(dim=1, keepdim=True)

    H_pred = mdp_pred.build_H(mdp_pred.P)
    r_pred_flat = mdp_pred.r.reshape(1, -1)

    # Solve LP (differentiable)
    y_star = DifferentiableLPSolver.apply(H_pred, r_pred_flat, gamma)

    # Build true MDP (detached)
    with torch.no_grad():
        lam_true_t = torch.tensor([lambda_true], dtype=torch.float32)
        mdp_true = create_inventory_from_lambda(lam_true_t, B_MAX, S_MAX, Q_MAX, D_MAX,
                                                 FOC, VOC, HC, BC, BETA, p)
        r_true_flat = mdp_true.r.reshape(1, -1)

    # DFL loss
    v_star = (r_true_flat * y_star).sum()
    dfl_loss = -v_star

    # Compute gradient
    dfl_loss.backward()
    grad_val = lam_pred.grad.item() if lam_pred.grad is not None else 0.0

    return grad_val, v_star.item()


def main():
    print("Computing LP solutions across lambda range...")

    # ── Sweep lambda values ───────────────────────────────────────────
    lambda_values = np.concatenate([
        np.linspace(0.5, 5, 20),
        np.linspace(5, 30, 26),
        np.linspace(30, 80, 50),
        np.linspace(80, 200, 80),
    ])

    # Build true MDP once
    lam_true_t = torch.tensor([LAMBDA_TRUE], dtype=torch.float32)
    p_t = torch.tensor([PRICE], dtype=torch.float32)
    _, v_true_opt, r_true_flat, _, _, _ = solve_for_lambda(LAMBDA_TRUE)

    # Arrays to store results
    v_true_at_pred = []   # r_true^T · y*(λ_pred)  — DFL objective
    v_pred_at_pred = []   # r_pred^T · y*(λ_pred)  — LP objective
    dfl_grads = []        # dL/dλ_pred
    basis_ids = []        # track basis changes

    for i, lam in enumerate(lambda_values):
        if (i + 1) % 20 == 0:
            print(f"  λ = {lam:.1f}  ({i+1}/{len(lambda_values)})")

        # Solve LP at this lambda
        y_star, v_pred, r_pred_flat, H, gamma, basis = solve_for_lambda(lam)

        # Evaluate with true reward
        mdp_true = create_inventory_from_lambda(lam_true_t, B_MAX, S_MAX, Q_MAX, D_MAX,
                                                 FOC, VOC, HC, BC, BETA, p_t)
        r_true = mdp_true.r.reshape(1, -1)
        v_true = (r_true * y_star).sum().item()

        v_true_at_pred.append(v_true)
        v_pred_at_pred.append(v_pred)
        basis_ids.append(tuple(basis[0].tolist()))

        # Compute DFL gradient
        try:
            grad, _ = compute_dfl_gradient(lam, LAMBDA_TRUE, PRICE)
        except Exception:
            grad = 0.0
        dfl_grads.append(grad)

    v_true_at_pred = np.array(v_true_at_pred)
    v_pred_at_pred = np.array(v_pred_at_pred)
    dfl_grads = np.array(dfl_grads)

    # Detect basis changes
    unique_bases = []
    basis_change_lambdas = []
    prev = None
    for i, b in enumerate(basis_ids):
        if b != prev:
            unique_bases.append(b)
            if prev is not None:
                basis_change_lambdas.append(lambda_values[i])
            prev = b
    print(f"\nFound {len(unique_bases)} distinct LP bases across lambda range")

    # ── Plot ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    # ── Panel 1: DFL objective v*(λ_pred) = r_true^T · y*(λ_pred) ──
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(lambda_values, v_true_at_pred, 'b-', linewidth=2, label=r'$v^*(\lambda_{pred}) = r_{true}^T \cdot y^*(\lambda_{pred})$')
    ax1.axvline(x=LAMBDA_TRUE, color='r', linestyle='--', linewidth=1.5, label=f'λ_true = {LAMBDA_TRUE}')

    # Mark basis changes
    for bc_lam in basis_change_lambdas:
        ax1.axvline(x=bc_lam, color='gray', linestyle=':', alpha=0.5)

    # Mark initial conditions from user's experiments
    for init_lam, label, color in [(1, 'init λ≈1 (Result 1)', 'orange'),
                                    (42, 'init λ≈42 (Current)', 'green'),
                                    (200, 'init λ≈1600 (Result 2)', 'purple')]:
        if init_lam <= lambda_values.max():
            idx = np.argmin(np.abs(lambda_values - init_lam))
            ax1.plot(init_lam, v_true_at_pred[idx], 'o', color=color, markersize=10, label=label, zorder=5)

    ax1.set_xlabel('λ_pred', fontsize=12)
    ax1.set_ylabel('r_true^T · y*(λ_pred)', fontsize=12)
    ax1.set_title('DFL Objective: True Reward Under Predicted Policy\n'
                   '(Gray lines = LP basis changes; peak should be near λ_true)',
                   fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: DFL gradient ──
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(lambda_values, dfl_grads, 'r-', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.axvline(x=LAMBDA_TRUE, color='r', linestyle='--', linewidth=1.5, alpha=0.5)
    for bc_lam in basis_change_lambdas:
        ax2.axvline(x=bc_lam, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('λ_pred', fontsize=12)
    ax2.set_ylabel('∂L/∂λ_pred (DFL gradient)', fontsize=12)
    ax2.set_title('DFL Gradient: dL/dλ\n'
                   '(Negative → SGD increases λ; Positive → SGD decreases λ)',
                   fontsize=13)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Predicted LP objective ──
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(lambda_values, v_pred_at_pred, 'g-', linewidth=2, label=r'$r_{pred}^T \cdot y^*(\lambda_{pred})$')
    ax3.axvline(x=LAMBDA_TRUE, color='r', linestyle='--', linewidth=1.5, alpha=0.5)
    ax3.set_xlabel('λ_pred', fontsize=12)
    ax3.set_ylabel('r_pred^T · y*(λ_pred)', fontsize=12)
    ax3.set_title('Predicted LP Objective\n(what the LP maximizes internally)', fontsize=13)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: Gradient direction arrows ──
    ax4 = fig.add_subplot(gs[2, :])

    # Subsample for arrow plot
    step = max(1, len(lambda_values) // 30)
    lam_sub = lambda_values[::step]
    grad_sub = dfl_grads[::step]

    # Color by gradient direction
    for i in range(len(lam_sub)):
        color = 'red' if grad_sub[i] > 0 else 'blue'
        arrow_len = -grad_sub[i]  # negative gradient = SGD direction
        ax4.annotate('', xy=(lam_sub[i] + np.sign(arrow_len) * 2, 0),
                     xytext=(lam_sub[i], 0),
                     arrowprops=dict(arrowstyle='->', color=color, lw=2))
        ax4.plot(lam_sub[i], 0, 'o', color=color, markersize=5)

    ax4.axvline(x=LAMBDA_TRUE, color='r', linestyle='--', linewidth=2, label=f'λ_true = {LAMBDA_TRUE}')
    ax4.set_xlabel('λ_pred', fontsize=12)
    ax4.set_title('SGD Update Direction at Each λ_pred\n'
                   '(Blue → = SGD pushes right; Red ← = SGD pushes left)',
                   fontsize=13)
    ax4.set_yticks([])
    ax4.legend(fontsize=11)
    ax4.grid(True, axis='x', alpha=0.3)

    plt.savefig('mdp_lp/plot_results/lambda_vs_ystar.png', dpi=150, bbox_inches='tight')
    plt.savefig('mdp_lp/plot_results/lambda_vs_ystar.pdf', bbox_inches='tight')
    print("\nSaved to mdp_lp/plot_results/lambda_vs_ystar.{png,pdf}")
    plt.show()

    # save basis to a file
    import json
    with open('mdp_lp/plot_results/basis_ids.json', 'w') as f:
        json.dump(basis_ids, f)
    print(f"Saved basis_ids to mdp_lp/plot_results/basis_ids.json")
    print(f"basis_ids: {basis_ids}")
    f.close()
    with open('mdp_lp/plot_results/lambda_values.json', 'w') as f:
        json.dump(lambda_values, f)
    print(f"Saved lambda_values to mdp_lp/plot_results/lambda_values.json")
    print(f"lambda_values: {lambda_values}")
    f.close()
    with open('mdp_lp/plot_results/v_true_at_pred.json', 'w') as f:
        json.dump(v_true_at_pred, f)
    print(f"Saved v_true_at_pred to mdp_lp/plot_results/v_true_at_pred.json")


if __name__ == "__main__":
    main()
