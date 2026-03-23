import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt

from mdp_lp.inventory_elasticity import create_inventory_from_lambda
from mdp_lp.solvers.differentiable_lp import DifferentiableLPSolver


config = {
    "s_max": 50,
    "b_max": 5,
    "q_max": 70,
    "d_max": 100,
    "foc": 0,
    "voc": 0.5,
    "hc": 0.1,
    "bc": 0,
    "beta": 0.95,
    "p_constant": 10.0,
    "p_lognormal_mean": 10.0,
    "p_lognormal_std": 0.5,
}


def scan_lambda_surrogate(
    lambda_true_value=30.0,
    lambda_min=0.0,
    lambda_max=100.0,
    n_grid=201,
    rho=10.0,
    normalize_terms=False,
):
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cpu")

    lambda_pred_grid = torch.linspace(lambda_min, lambda_max, n_grid, device=device)

    S = config["s_max"] + config["b_max"] + 1
    A = config["q_max"] + 1
    batch_size = lambda_pred_grid.shape[0]

    lambda_true = torch.full((batch_size,), float(lambda_true_value), device=device)
    p = torch.ones(batch_size, device=device) * config["p_constant"]

    # predicted MDP for each lambda_pred
    mdp_pred = create_inventory_from_lambda(
        lambda_pred_grid,
        config["b_max"], config["s_max"], config["q_max"], config["d_max"],
        config["foc"], config["voc"], config["hc"], config["bc"], config["beta"],
        p
    )

    H_pred = mdp_pred.build_H(mdp_pred.P)                 # (B, S, S*A)
    r_pred = mdp_pred.r.reshape(batch_size, -1)           # (B, S*A)
    gamma = mdp_pred.build_gamma(batch_size, S, device)   # (B, S)

    # Solve occupancy of pred MDP
    y_star = DifferentiableLPSolver.apply(H_pred, r_pred, gamma)   # (B, S*A)

    # true MDP (same true lambda for all grid points)
    mdp_true = create_inventory_from_lambda(
        lambda_true,
        config["b_max"], config["s_max"], config["q_max"], config["d_max"],
        config["foc"], config["voc"], config["hc"], config["bc"], config["beta"],
        p
    )

    H_true = mdp_true.build_H(mdp_true.P)                  # (B, S, S*A)
    r_true = mdp_true.r.reshape(batch_size, -1)            # (B, S*A)

    with torch.no_grad():
        reward_term = (r_true * y_star).sum(dim=1)         # (B,)

        violation = (H_true @ y_star.unsqueeze(-1)).squeeze(-1) - gamma   # (B, S)
        penalty_term = (violation ** 2).mean(dim=1)        # (B,)

        # 3) surrogate if viewed as maximization objective
        surrogate_raw = reward_term - rho * penalty_term

        # 4) surrogate if viewed as training loss to minimize
        surrogate_loss = -reward_term + rho * penalty_term

        if normalize_terms:
            def _norm(x):
                return (x - x.mean()) / (x.std() + 1e-8)
            reward_plot = _norm(reward_term)
            penalty_plot = _norm(penalty_term)
            surrogate_raw_plot = _norm(surrogate_raw)
            surrogate_loss_plot = _norm(surrogate_loss)
        else:
            reward_plot = reward_term
            penalty_plot = penalty_term
            surrogate_raw_plot = surrogate_raw
            surrogate_loss_plot = surrogate_loss

    return {
        "lambda_pred_grid": lambda_pred_grid.cpu().numpy(),
        "lambda_true_value": float(lambda_true_value),
        "reward_term": reward_term.cpu().numpy(),
        "penalty_term": penalty_term.cpu().numpy(),
        "surrogate_raw": surrogate_raw.cpu().numpy(),
        "surrogate_loss": surrogate_loss.cpu().numpy(),
        "reward_plot": reward_plot.cpu().numpy(),
        "penalty_plot": penalty_plot.cpu().numpy(),
        "surrogate_raw_plot": surrogate_raw_plot.cpu().numpy(),
        "surrogate_loss_plot": surrogate_loss_plot.cpu().numpy(),
    }


def plot_lambda_surrogate_curves(
    lambda_true_value=30.0,
    lambda_min=0.0,
    lambda_max=100.0,
    n_grid=201,
    rho=10.0,
    normalize_terms=False,
    save_prefix="lambda_surrogate_scan",
):
    out = scan_lambda_surrogate(
        lambda_true_value=lambda_true_value,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        n_grid=n_grid,
        rho=rho,
        normalize_terms=normalize_terms,
    )

    x = out["lambda_pred_grid"]
    lam_true = out["lambda_true_value"]
    idx_best_surrogate = int(np.argmax(out["surrogate_raw"]))
    lam_best_surrogate = float(x[idx_best_surrogate])

    # Fig 1：plot three curves
    plt.figure(figsize=(9, 5))
    plt.plot(x, out["reward_plot"], label=r"$r_{\mathrm{true}}^\top y_\star(\lambda_{\mathrm{pred}})$")
    plt.plot(x, out["penalty_plot"], label=r"$\|H_{\mathrm{true}} y_\star - \gamma\|^2$")
    plt.plot(x, out["surrogate_raw_plot"], linewidth=2.5,
             label=rf"surrogate = reward - {rho} * penalty")
    plt.axvline(x=lam_true, linestyle="--", label=rf"$\lambda_{{true}}={lam_true:.1f}$")
    plt.axvline(x=lam_best_surrogate, linestyle=":", color="green",
                label=rf"$\lambda_{{\mathrm{{pred}}}}^*={lam_best_surrogate:.1f}$ (surrogate max)")
    plt.xlabel(r"$\lambda_{\mathrm{pred}}$")
    plt.ylabel("value" + (" (normalized)" if normalize_terms else ""))
    plt.title("Surrogate components vs lambda_pred")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_raw.pdf", dpi=300)
    plt.show()

    # Fig 2：training loss
    plt.figure(figsize=(9, 5))
    plt.plot(x, out["surrogate_loss_plot"], linewidth=2.5,
             label=rf"loss = - reward + {rho} * penalty")
    plt.axvline(x=lam_true, linestyle="--", label=rf"$\lambda_{{true}}={lam_true:.1f}$")
    plt.axvline(x=lam_best_surrogate, linestyle=":", color="green",
                label=rf"$\lambda_{{\mathrm{{pred}}}}^*={lam_best_surrogate:.1f}$ (loss min)")
    plt.xlabel(r"$\lambda_{\mathrm{pred}}$")
    plt.ylabel("loss" + (" (normalized)" if normalize_terms else ""))
    plt.title("Training loss vs lambda_pred")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_loss.pdf", dpi=300)
    plt.show()

    # Fig 3: plot the violation of the true MDP
    plt.figure(figsize=(9, 5))
    plt.plot(x, out["penalty_plot"], label=r"$\|H_{\mathrm{true}} y_\star - \gamma\|$")
    plt.axvline(x=lam_true, linestyle="--", label=rf"$\lambda_{{true}}={lam_true:.1f}$")
    plt.xlabel(r"$\lambda_{\mathrm{pred}}$")
    plt.ylabel("violation")
    plt.title("Violation vs lambda_pred")
    plt.savefig(f"{save_prefix}_violation.pdf", dpi=300)
    plt.legend()
    plt.tight_layout()

    # Print best points, help to judge whether it is biased towards small λ or very flat
    idx_best_raw = int(np.argmax(out["surrogate_raw"]))
    idx_best_loss = int(np.argmin(out["surrogate_loss"]))
    print(f"[scan] lambda_true = {lam_true:.4f}")
    print(f"[scan] best lambda_pred for surrogate maximize = {x[idx_best_raw]:.4f}")
    print(f"[scan] best lambda_pred for loss minimize    = {x[idx_best_loss]:.4f}")

    return out


if __name__ == "__main__":
    # out = plot_lambda_surrogate_curves(
    #     lambda_true_value=60.0,
    #     lambda_min=0.0,
    #     lambda_max=100.0,
    #     n_grid=201,
    #     rho=10.0,
    #     normalize_terms=False,
    #     save_prefix="lambda_surrogate_scan"
    # )

    # out_norm = plot_lambda_surrogate_curves(
    #     lambda_true_value=60.0,
    #     lambda_min=0.0,
    #     lambda_max=100.0,
    #     n_grid=201,
    #     rho=10.0,
    #     normalize_terms=True,
    #     save_prefix="lambda_surrogate_scan_norm"
    # )

    lambda_true_list = [10.0, 30.0, 60.0]
    all_surrogates = []

    for lam_true in lambda_true_list:
        out = scan_lambda_surrogate(lambda_true_value=lam_true, rho=10.0)
        all_surrogates.append(out["surrogate_raw"])

    mean_surrogate = np.mean(np.stack(all_surrogates, axis=0), axis=0)
    idx_best_mean = int(np.argmax(mean_surrogate))
    lam_best_mean = float(out["lambda_pred_grid"][idx_best_mean])

    plt.figure(figsize=(9, 5))
    for lam_true, surr in zip(lambda_true_list, all_surrogates):
        plt.plot(out["lambda_pred_grid"], surr, label=f"lambda_true={lam_true}")
    plt.plot(out["lambda_pred_grid"], mean_surrogate, linewidth=3, label="mean surrogate")
    plt.axvline(x=lam_best_mean, linestyle=":", color="green",
                label=rf"$\lambda_{{\mathrm{{pred}}}}^*={lam_best_mean:.1f}$ (mean surrogate max)")
    plt.legend()
    plt.xlabel("lambda_pred")
    plt.ylabel("surrogate")
    plt.title("Individual vs averaged surrogate")
    plt.tight_layout()
    plt.savefig("individual_vs_averaged_surrogate.pdf", dpi=300)
    plt.show()