"""
Verify that DFL training can achieve the oracle mean surrogate.

Setup: 3 samples with p_true = [5, 10, 15]. Lambda from elasticity model:
  ln(λ) = β₀ + β₁·ln(p) => λ = exp(β₀ + β₁·ln(p)).
Uses LambdaPredictor to learn feats [p, noise] -> lambda_pred.
Oracle: mean surrogate when lambda_pred_i = lambda_true_i for each sample.

Diagnostic plots show basis changes, gradient flow, and convergence anatomy.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.optim as optim

from mdp_lp.inventory_elasticity import create_inventory_from_lambda
from mdp_lp.solvers.differentiable_lp import DifferentiableLPSolver
from mdp_lp.solvers.dense_lp_solver import solve_lp_gurobi


class DifferentiableLPWithBasis(torch.autograd.Function):
    """Wraps DifferentiableLPSolver but also stores the basis externally."""
    _last_basis = None

    @staticmethod
    def forward(ctx, H, r, gamma):
        with torch.no_grad():
            y_star, basis, _ = solve_lp_gurobi(H, r, gamma)
        ctx.save_for_backward(H, r, gamma, y_star, basis)
        DifferentiableLPWithBasis._last_basis = basis.clone()
        return y_star

    @staticmethod
    def backward(ctx, grad_y):
        return DifferentiableLPSolver.backward(ctx, grad_y)


from mdp_lp.elasticity_model import LambdaPredictor

config = {
    "s_max": 50, "b_max": 5, "q_max": 70, "d_max": 100,
    "foc": 0, "voc": 0.5, "hc": 0.1, "bc": 0, "beta": 0.95,
    "p_constant": 10.0,
}
ELASTICITY_BETA_0 = 6.5
ELASTICITY_BETA_1 = -1.5


def _lambda_from_p(p, beta_0=ELASTICITY_BETA_0, beta_1=ELASTICITY_BETA_1):
    return float(np.exp(beta_0 + beta_1 * np.log(p)))


def get_true_optimum(p_true_list, rho=10.0):
    device = torch.device("cpu")
    torch.manual_seed(0)
    surrogates = []
    for p_val in p_true_list:
        lam_true = _lambda_from_p(p_val)
        lam_t = torch.tensor([lam_true], device=device)
        p = torch.tensor([p_val], dtype=torch.float32, device=device)
        mdp = create_inventory_from_lambda(
            lam_t, config["b_max"], config["s_max"], config["q_max"],
            config["foc"], config["voc"], config["hc"], config["bc"], config["beta"], p)
        S = config["s_max"] + config["b_max"] + 1
        gamma = torch.ones(1, S, device=device) / S
        gamma = torch.clamp(gamma, min=1e-8)
        gamma = gamma / gamma.sum(dim=1, keepdim=True)
        H = mdp.build_H(mdp.P)
        r = mdp.r.reshape(1, -1)
        y_star = DifferentiableLPSolver.apply(H, r, gamma)
        with torch.no_grad():
            reward = (r * y_star).sum().item()
            violation = (H @ y_star.unsqueeze(-1)).squeeze(-1) - gamma
            penalty = (violation ** 2).mean().item()
        surrogates.append(reward - rho * penalty)
    return float(np.mean(surrogates))


def train_dfl_mean_surrogate(
    p_true_list=(5.0, 10.0, 15.0),
    rho=10.0,
    n_epochs=200,
    lr=0.01,
    hidden_dim=32,
):
    """
    Train DFL with full diagnostics: basis tracking, per-sample decomposition,
    gradient norms through H vs r.
    """
    device = torch.device("cpu")
    torch.manual_seed(0)
    np.random.seed(0)

    batch_size = len(p_true_list)
    lambda_true_list = [_lambda_from_p(pv) for pv in p_true_list]
    p_vals = np.array(p_true_list, dtype=np.float32)
    feats = torch.tensor(
        np.column_stack([p_vals, np.random.randn(batch_size) * 0.1]),
        dtype=torch.float32, device=device,
    )
    p = torch.tensor(p_vals, dtype=torch.float32, device=device)
    lambda_true = torch.tensor(lambda_true_list, dtype=torch.float32, device=device)

    predictor = LambdaPredictor(
        hidden_dim=hidden_dim, use_p_related=False, use_price_only=True,
    ).to(device)
    optimizer = optim.Adam(predictor.parameters(), lr=lr)

    S = config["s_max"] + config["b_max"] + 1
    gamma = torch.ones(batch_size, S, device=device) / S
    gamma = torch.clamp(gamma, min=1e-8)
    gamma = gamma / gamma.sum(dim=1, keepdim=True)

    # --- Diagnostic storage ---
    history = {
        "lambda_pred": [],           # (n_epochs, batch) individual predictions
        "mean_surrogate": [],
        "reward_per_sample": [],     # (n_epochs, batch)
        "penalty_per_sample": [],    # (n_epochs, batch)
        "surrogate_per_sample": [],  # (n_epochs, batch)
        "basis_per_sample": [],      # (n_epochs, batch, S) basis column indices
        "basis_changes": [],         # (n_epochs, batch) number of columns changed
        "grad_norm_H": [],           # (n_epochs,) gradient norm through H
        "grad_norm_r": [],           # (n_epochs,) gradient norm through r (always 0)
        "grad_norm_total": [],       # (n_epochs,) total param gradient norm
    }
    prev_basis = None

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        lambda_pred = predictor(feats)

        mdp_pred = create_inventory_from_lambda(
            lambda_pred,
            config["b_max"], config["s_max"], config["q_max"],
            config["foc"], config["voc"], config["hc"], config["bc"], config["beta"], p,
        )
        mdp_true = create_inventory_from_lambda(
            lambda_true,
            config["b_max"], config["s_max"], config["q_max"],
            config["foc"], config["voc"], config["hc"], config["bc"], config["beta"], p,
        )

        H_pred = mdp_pred.build_H(mdp_pred.P)
        r_pred = mdp_pred.r.reshape(batch_size, -1)

        # Retain grad on non-leaf tensors for diagnostic gradient norms
        H_pred.retain_grad()
        r_pred.retain_grad()

        y_star = DifferentiableLPWithBasis.apply(H_pred, r_pred, gamma)
        basis_np = DifferentiableLPWithBasis._last_basis.cpu().numpy()  # (batch, S)

        r_true = mdp_true.r.reshape(batch_size, -1)
        H_true = mdp_true.build_H(mdp_true.P)

        reward = (r_true * y_star).sum(dim=1)
        violation = (H_true @ y_star.unsqueeze(-1)).squeeze(-1) - gamma
        penalty = (violation ** 2).mean(dim=1)
        surrogate = reward - rho * penalty

        loss = -surrogate.mean()
        loss.backward()

        # --- Record gradient norms before step ---
        with torch.no_grad():
            gn_H = H_pred.grad.norm().item() if H_pred.grad is not None else 0.0
            gn_r = r_pred.grad.norm().item() if r_pred.grad is not None else 0.0
            total_gn = sum(
                par.grad.norm().item() ** 2
                for par in predictor.parameters() if par.grad is not None
            ) ** 0.5

        optimizer.step()

        # Count basis column changes vs previous epoch
        if prev_basis is not None:
            changes = []
            for b in range(batch_size):
                prev_set = set(prev_basis[b].tolist())
                curr_set = set(basis_np[b].tolist())
                changes.append(len(prev_set.symmetric_difference(curr_set)))
            history["basis_changes"].append(changes)
        else:
            history["basis_changes"].append([0] * batch_size)
        prev_basis = basis_np.copy()

        # --- Store diagnostics ---
        with torch.no_grad():
            lam_vals = lambda_pred.tolist()
            rew_vals = reward.tolist()
            pen_vals = penalty.tolist()
            surr_vals = surrogate.tolist()

        history["lambda_pred"].append(lam_vals)
        history["mean_surrogate"].append(float(np.mean(surr_vals)))
        history["reward_per_sample"].append(rew_vals)
        history["penalty_per_sample"].append(pen_vals)
        history["surrogate_per_sample"].append(surr_vals)
        history["basis_per_sample"].append(basis_np.tolist())
        history["grad_norm_H"].append(gn_H)
        history["grad_norm_r"].append(gn_r)
        history["grad_norm_total"].append(total_gn)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            lam_str = ", ".join(f"{x:.1f}" for x in lam_vals)
            bc_str = ", ".join(str(c) for c in history["basis_changes"][-1])
            print(f"  Epoch {epoch+1:4d}: λ=[{lam_str}]  surr={np.mean(surr_vals):.2f}"
                  f"  basis_changes=[{bc_str}]")

    # Convert lists to arrays for easy plotting
    for key in ["lambda_pred", "reward_per_sample", "penalty_per_sample",
                "surrogate_per_sample", "basis_changes"]:
        history[key] = np.array(history[key])

    with torch.no_grad():
        lambda_final = predictor(feats).cpu().numpy()
    return lambda_final, history["mean_surrogate"][-1], history


def plot_diagnostics(history, lambda_true_list, p_true_list, surr_opt, rho, save_path=None):
    """
    Four-panel diagnostic figure:
      (a) Lambda trajectories with basis-change markers
      (b) Basis changes per epoch (bar chart)
      (c) Surrogate decomposition: reward vs penalty
      (d) Mean surrogate with basis-change shading
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n_epochs = len(history["mean_surrogate"])
    batch_size = history["lambda_pred"].shape[1]
    epochs = np.arange(1, n_epochs + 1)
    colors = ["#e74c3c", "#2ecc71", "#3498db"]  # red, green, blue
    labels = [f"p={p:.0f}" for p in p_true_list]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"DFL Convergence Diagnostics  (lr={history.get('lr', '?')}, "
        r"$\rho$" + f"={rho})",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # ── (a) Lambda trajectories ──
    ax = axes[0, 0]
    # Shade epochs where ANY sample had a basis change
    any_change = history["basis_changes"].sum(axis=1) > 0
    for e in range(n_epochs):
        if any_change[e]:
            ax.axvspan(e + 0.5, e + 1.5, color="#fee2e2", zorder=0)

    for i in range(batch_size):
        ax.plot(epochs, history["lambda_pred"][:, i], color=colors[i],
                linewidth=1.8, label=labels[i])
        ax.axhline(y=lambda_true_list[i], color=colors[i], linestyle="--",
                   alpha=0.5, linewidth=1)
        # Mark basis-change epochs with dots
        bc_epochs = np.where(history["basis_changes"][:, i] > 0)[0]
        if len(bc_epochs) > 0:
            ax.scatter(bc_epochs + 1, history["lambda_pred"][bc_epochs, i],
                       color=colors[i], s=30, zorder=5, edgecolors="k", linewidths=0.5)

    ax.set_ylabel(r"$\lambda_{\mathrm{pred}}$", fontsize=12)
    ax.set_title("(a) Lambda trajectories", fontsize=11)
    legend1 = ax.legend(loc="upper right", fontsize=9)
    # Add explanation for markers
    marker_legend = Line2D([0], [0], marker="o", color="gray", markeredgecolor="k",
                           markersize=6, linestyle="None", label="basis change")
    shade_legend = Line2D([0], [0], color="#fee2e2", linewidth=8, label="basis-change epoch")
    ax.legend(handles=[*legend1.legend_handles, marker_legend, shade_legend],
              loc="upper right", fontsize=8)

    # ── (b) Basis changes per epoch ──
    ax = axes[0, 1]
    bottom = np.zeros(n_epochs)
    for i in range(batch_size):
        ax.bar(epochs, history["basis_changes"][:, i], bottom=bottom,
               color=colors[i], alpha=0.8, width=1.0, label=labels[i])
        bottom += history["basis_changes"][:, i]
    ax.set_ylabel("# basis columns changed", fontsize=12)
    ax.set_title("(b) Basis changes per epoch", fontsize=11)
    ax.legend(fontsize=9)
    # Annotate total changes
    total_changes = int(history["basis_changes"].sum())
    ax.text(0.98, 0.95, f"total = {total_changes} changes",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))

    # ── (c) Surrogate decomposition: reward and penalty per sample ──
    ax = axes[1, 0]
    ax_pen = ax.twinx()
    for i in range(batch_size):
        ax.plot(epochs, history["reward_per_sample"][:, i], color=colors[i],
                linewidth=1.2, linestyle="-")
        ax_pen.plot(epochs, history["penalty_per_sample"][:, i], color=colors[i],
                    linewidth=1.2, linestyle=":")
    ax.set_ylabel(r"reward  $r_{\mathrm{true}}^\top y^*$  (solid)", fontsize=11)
    ax_pen.set_ylabel(r"penalty  $\|H_{\mathrm{true}} y^* - \gamma\|^2$  (dotted)",
                      fontsize=11)
    ax.set_title("(c) Per-sample reward (solid) & penalty (dotted)", fontsize=11)
    # Custom legend
    solid = Line2D([0], [0], color="gray", linewidth=1.5, linestyle="-", label="reward")
    dotted = Line2D([0], [0], color="gray", linewidth=1.5, linestyle=":", label="penalty")
    sample_handles = [Line2D([0], [0], color=c, linewidth=1.5, label=l)
                      for c, l in zip(colors, labels)]
    ax.legend(handles=[solid, dotted] + sample_handles, fontsize=8, loc="center right")

    # ── (d) Mean surrogate with basis-change shading ──
    ax = axes[1, 1]
    for e in range(n_epochs):
        if any_change[e]:
            ax.axvspan(e + 0.5, e + 1.5, color="#fee2e2", zorder=0)
    ax.plot(epochs, history["mean_surrogate"], "k-", linewidth=2, label="mean surrogate")
    ax.axhline(y=surr_opt, color="green", linestyle="--", linewidth=1.5,
               label=f"oracle = {surr_opt:.1f}")
    # Mark surrogate drops at basis changes
    surr = np.array(history["mean_surrogate"])
    drops = np.where((np.diff(surr) < 0) & any_change[1:])[0] + 1
    if len(drops) > 0:
        ax.scatter(drops + 1, surr[drops], color="red", s=40, zorder=5,
                   marker="v", label="drop at basis change")
    ax.set_ylabel("mean surrogate", fontsize=12)
    ax.set_xlabel("epoch", fontsize=12)
    ax.set_title("(d) Mean surrogate convergence", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")

    for a in axes.flat:
        a.set_xlabel("epoch", fontsize=11)
        a.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


def plot_gradient_anatomy(history, save_path=None):
    """
    Show why grad_r = 0 matters:
      (a) ||∂L/∂H|| vs ||∂L/∂r|| over training (r is always 0)
      (b) Total parameter gradient norm
    """
    import matplotlib.pyplot as plt

    n_epochs = len(history["grad_norm_H"])
    epochs = np.arange(1, n_epochs + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Gradient Anatomy: LP Backward Pass", fontsize=13, fontweight="bold")

    # (a) grad norms through H vs r
    ax1.semilogy(epochs, history["grad_norm_H"], "b-", linewidth=1.5,
                 label=r"$\|\partial L / \partial H_{\mathrm{pred}}\|$  (non-zero)")
    ax1.semilogy(epochs, [max(g, 1e-15) for g in history["grad_norm_r"]],
                 "r--", linewidth=1.5,
                 label=r"$\|\partial L / \partial r_{\mathrm{pred}}\|$  (≡ 0, hardcoded)")
    ax1.set_ylabel("gradient norm (log scale)", fontsize=11)
    ax1.set_xlabel("epoch", fontsize=11)
    ax1.set_title(r"(a)  $\nabla_H$ active  vs  $\nabla_r \equiv 0$  (missing signal)",
                  fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Annotate the problem
    ax1.annotate(
        "LP backward hardcodes\n" r"$\partial y^*/\partial r = 0$" "\n(fixed-basis assumption)",
        xy=(n_epochs * 0.5, 1e-15), fontsize=9, color="red",
        ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fee2e2", alpha=0.8),
    )

    # (b) Total parameter gradient norm
    ax2.plot(epochs, history["grad_norm_total"], "k-", linewidth=1.5)
    ax2.set_ylabel("total param gradient norm", fontsize=11)
    ax2.set_xlabel("epoch", fontsize=11)
    ax2.set_title("(b)  Parameter gradient norm (predictor weights)", fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Shade basis-change epochs
    if "basis_changes" in history:
        any_change = history["basis_changes"].sum(axis=1) > 0
        for e in range(n_epochs):
            if any_change[e]:
                ax2.axvspan(e + 0.5, e + 1.5, color="#fee2e2", zorder=0)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


def main():
    p_true_list = [5.0, 10.0, 15.0]
    lambda_true_list = [_lambda_from_p(pv) for pv in p_true_list]
    rho = 10.0

    print("=" * 60)
    print("DFL Mean Surrogate — Basis Change Diagnostics")
    print("=" * 60)
    print(f"  p_true     = {p_true_list}")
    print(f"  λ_true     = {[f'{x:.2f}' for x in lambda_true_list]}")
    print(f"  ρ          = {rho}")
    print()

    surr_opt = get_true_optimum(p_true_list, rho=rho)
    print(f"  Oracle mean_surrogate = {surr_opt:.2f}")
    print()

    # --- Run two learning rates to contrast behavior ---
    results = {}
    for lr in [1e-3, 1e-4]:
        print(f"[lr={lr}] Training...")
        torch.manual_seed(0)
        np.random.seed(0)
        lam_final, surr_final, hist = train_dfl_mean_surrogate(
            p_true_list=p_true_list, rho=rho, n_epochs=300, lr=lr,
        )
        hist["lr"] = lr
        results[lr] = (lam_final, surr_final, hist)
        mae = float(np.mean(np.abs(lam_final - np.array(lambda_true_list))))
        print(f"  Final: λ_pred={[f'{x:.1f}' for x in lam_final]}, "
              f"MAE={mae:.2f}, surrogate={surr_final:.2f}")
        total_bc = int(hist["basis_changes"].sum())
        print(f"  Total basis changes: {total_bc}")
        print()

    # --- Plot for each lr ---
    out_dir = os.path.dirname(__file__)
    for lr, (lam_final, surr_final, hist) in results.items():
        lr_tag = f"{lr:.0e}".replace("-", "m")
        plot_diagnostics(
            hist, lambda_true_list, p_true_list, surr_opt, rho,
            save_path=os.path.join(out_dir, f"basis_diagnostics_lr{lr_tag}.pdf"),
        )
        plot_gradient_anatomy(
            hist,
            save_path=os.path.join(out_dir, f"gradient_anatomy_lr{lr_tag}.pdf"),
        )

    # --- Side-by-side lr comparison ---
    plot_lr_comparison(results, lambda_true_list, surr_opt, rho,
                       save_path=os.path.join(out_dir, "lr_comparison.pdf"))


def plot_lr_comparison(results, lambda_true_list, surr_opt, rho, save_path=None):
    """Side-by-side comparison of two learning rates."""
    import matplotlib.pyplot as plt

    lrs = sorted(results.keys(), reverse=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        "Effect of Learning Rate on Basis Stability & Convergence",
        fontsize=14, fontweight="bold", y=0.98,
    )

    colors = ["#e74c3c", "#2ecc71", "#3498db"]

    for col, lr in enumerate(lrs):
        _, _, hist = results[lr]
        n_epochs = len(hist["mean_surrogate"])
        epochs = np.arange(1, n_epochs + 1)
        batch_size = hist["lambda_pred"].shape[1]
        any_change = hist["basis_changes"].sum(axis=1) > 0

        # Top row: lambda trajectories
        ax = axes[0, col]
        for e in range(n_epochs):
            if any_change[e]:
                ax.axvspan(e + 0.5, e + 1.5, color="#fee2e2", zorder=0)
        for i in range(batch_size):
            ax.plot(epochs, hist["lambda_pred"][:, i], color=colors[i], linewidth=1.5)
            ax.axhline(y=lambda_true_list[i], color=colors[i], linestyle="--", alpha=0.4)
        total_bc = int(hist["basis_changes"].sum())
        ax.set_title(f"lr = {lr}  ({total_bc} basis changes)", fontsize=11)
        ax.set_ylabel(r"$\lambda_{\mathrm{pred}}$", fontsize=11)

        # Bottom row: mean surrogate
        ax = axes[1, col]
        for e in range(n_epochs):
            if any_change[e]:
                ax.axvspan(e + 0.5, e + 1.5, color="#fee2e2", zorder=0)
        ax.plot(epochs, hist["mean_surrogate"], "k-", linewidth=1.8)
        ax.axhline(y=surr_opt, color="green", linestyle="--", linewidth=1.2,
                   label=f"oracle = {surr_opt:.1f}")
        # Mark drops
        surr = np.array(hist["mean_surrogate"])
        drops = np.where((np.diff(surr) < -5) & any_change[1:])[0] + 1
        if len(drops) > 0:
            ax.scatter(drops + 1, surr[drops], color="red", s=40, zorder=5,
                       marker="v", label=f"{len(drops)} drops")
        ax.set_ylabel("mean surrogate", fontsize=11)
        ax.set_xlabel("epoch", fontsize=11)
        ax.legend(fontsize=9)

    for a in axes.flat:
        a.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
