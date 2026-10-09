"""
Figure-1-style landscape (Loss top, gradient dL/dlambda bottom), 3 columns:

  (1) Original DFL             L = -r_true^T y*(l)                              (biased)
  (2) Full aug-Lagrangian      L = -r_true^T y* + v*^T(H_true y*-g) + rho||.||^2 (min at l_true)
  (3) DFL + State Aggregation  same full aug-Lag loss, but y from the M-state
                               aggregated LP (occupancy disaggregated back)

Columns 1-2 reuse cached arrays (plot_aug_lag_linear_compare.npz). Column 3 is
computed fresh: aggregate the predicted MDP with a fixed contiguous M=10
membership, solve the reduced LP with scipy/HiGHS (no Gurobi -> no contention
with a running training sweep), disaggregate, and score under the TRUE MDP with
the same v* used in column 2.

Usage:  python -m mdp_lp.plot_fig1_sa
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linprog

from mdp_lp.inventory_elasticity import create_inventory_from_lambda
from mdp_lp.soft_state_aggregation import aggregate_with_W, SoftAggDense

# config identical to plot_aug_lag_linear_compare (matches the cached npz)
B_MAX, S_MAX, Q_MAX, D_MAX = 5, 50, 30, 30
FOC, VOC, HC, BC, BETA, PRICE = 0, 0.5, 0.1, 0, 0.95, 10.0
LAMBDA_TRUE = 10.0
NPZ = "mdp_lp/plot_results/aug_lag_linear_compare.npz"
RHO = 30.0
XLO, XHI = 8.0, 12.0
M_AGG = 10          # aggregated states (paper DFL-SA uses M=10)


def build_mdp(lam_val):
    lam = torch.tensor([lam_val], dtype=torch.float32)
    p = torch.tensor([PRICE], dtype=torch.float32)
    mdp = create_inventory_from_lambda(lam, B_MAX, S_MAX, Q_MAX, FOC, VOC, HC, BC, BETA, p)
    S = mdp.P.shape[1]
    gamma = mdp.build_gamma(1, S, torch.device("cpu"))
    gamma = torch.clamp(gamma, min=1e-8)
    gamma = gamma / gamma.sum(dim=1, keepdim=True)
    return mdp, gamma


def solve_np(H, r, gamma):
    """max r^T y s.t. H y = gamma, y>=0 via scipy; returns y (np)."""
    res = linprog(c=-r, A_eq=H, b_eq=gamma, bounds=(0, None), method="highs")
    if not res.success:
        return None
    return res.x


def contiguous_W(S, M):
    """Fixed hard membership: contiguous bins of states -> (1,S,M)."""
    W = torch.zeros(1, S, M)
    binf = (torch.arange(S) * M // S).clamp(max=M - 1)
    W[0, torch.arange(S), binf] = 1.0
    return W


def main():
    d = np.load(NPZ)
    lam_all = d["lambda_values"]
    m = (lam_all >= XLO) & (lam_all <= XHI)
    idx = np.argsort(lam_all[m])
    lam = lam_all[m][idx]
    dfl = d["dfl_term"][m][idx]           # -r^T y*
    lin = d["lin_term"][m][idx]           # v*^T (H y* - g)
    pen = d["pen_term"][m][idx]           # ||H y* - g||^2
    v_star = torch.tensor(d["v_star"], dtype=torch.float32)   # (S,)

    L_orig = dfl
    L_full = dfl + lin + RHO * pen

    # true MDP tensors for the SA column
    mdp_t, gamma_t = build_mdp(LAMBDA_TRUE)
    H_true = mdp_t.build_H(mdp_t.P)                 # (1,S,SA)
    r_true = mdp_t.r.reshape(1, -1)                 # (1,SA)
    S = mdp_t.P.shape[1]
    A = mdp_t.r.shape[-1]
    W = contiguous_W(S, M_AGG)

    print(f"S={S}, A={A}, M={M_AGG}; computing SA landscape over {len(lam)} points...")
    L_full_sa = np.full_like(lam, np.nan)
    for i, lv in enumerate(lam):
        mdp_p, gamma_p = build_mdp(lv)
        with torch.no_grad():
            P_agg, r_agg, gamma_agg = aggregate_with_W(mdp_p.P, mdp_p.r, gamma_p, W)
            H_agg = mdp_p.build_H(P_agg)                       # (1,M,MA)
        y_agg = solve_np(H_agg[0].detach().numpy().astype(np.float64),
                         r_agg.reshape(1, -1)[0].detach().numpy().astype(np.float64),
                         gamma_agg[0].detach().numpy().astype(np.float64))
        if y_agg is None:
            continue
        y_sa = SoftAggDense.disaggregate_y(torch.tensor(y_agg, dtype=torch.float32).unsqueeze(0),
                                           W, A)               # (1,SA)
        c = (H_true @ y_sa.unsqueeze(-1)).squeeze(-1) - gamma_t   # (1,S)
        reward = (r_true * y_sa).sum().item()
        linear = (v_star.unsqueeze(0) * c).sum().item()
        penalty = c.pow(2).sum().item()
        L_full_sa[i] = -reward + linear + RHO * penalty

    cols = [
        ("(1) Original DFL\n$-r_{true}^{\\top} y^*(\\lambda)$", L_orig, "tab:red"),
        ("(2) Full aug-Lagrangian (ours)\n$+\\,(v^*)^{\\top}(H_{true}y^*-\\gamma)+\\rho\\|\\cdot\\|^2$",
         L_full, "tab:green"),
        (f"(3) DFL + State Aggregation (M={M_AGG})\nsame loss, aggregated LP",
         L_full_sa, "tab:blue"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8), sharex=True)
    print("\nminimizers:")
    for j, (title, L, c) in enumerate(cols):
        good = np.isfinite(L)
        lam_g, L_g = lam[good], L[good]
        imin = int(np.argmin(L_g))
        lam_star = lam_g[imin]
        grad = np.gradient(L_g, lam_g)
        print(f"  col{j+1}: lambda* = {lam_star:.3f}  (shift {lam_star-LAMBDA_TRUE:+.3f})")

        ax = axes[0, j]
        ax.plot(lam_g, L_g, color=c, lw=2.2)
        ax.axvline(LAMBDA_TRUE, color="red", ls="--", lw=1.4)
        ax.plot(lam_star, L_g[imin], "o", color="green", ms=11, zorder=5)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Loss" if j == 0 else "")
        ax.grid(alpha=0.25)
        ax.text(0.04, 0.06,
                f"$\\lambda_{{true}}={LAMBDA_TRUE:g}$\n$\\lambda^*={lam_star:.2f}$  (shift {lam_star-LAMBDA_TRUE:+.2f})",
                transform=ax.transAxes, fontsize=11, va="bottom",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

        ax = axes[1, j]
        ax.plot(lam_g, grad, color=c, lw=1.6)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.axvline(LAMBDA_TRUE, color="red", ls="--", lw=1.4)
        ax.plot(lam_star, 0.0, "o", color="green", ms=9, zorder=5)
        ax.set_xlabel("$\\lambda_{pred}$")
        ax.set_ylabel("$dL/d\\lambda$" if j == 0 else "")
        ax.grid(alpha=0.25)

    axes[0, 2].plot([], [], "r--", label="$\\lambda_{true}$")
    axes[0, 2].plot([], [], "go", label="minimizer $\\lambda^*$")
    axes[0, 2].legend(loc="upper right", fontsize=10, framealpha=0.9)
    plt.suptitle("Loss (top) and gradient (bottom) vs $\\lambda_{pred}$:  "
                 "corrected DFL loss (2) is unbiased; state aggregation (3) stays close and smoother",
                 fontsize=13, y=1.0)
    plt.tight_layout()
    out = "mdp_lp/plot_results/fig1_sa"
    plt.savefig(out + ".png", dpi=160, bbox_inches="tight")
    plt.savefig(out + ".pdf", bbox_inches="tight")
    print(f"\nsaved {out}.png / .pdf")


if __name__ == "__main__":
    main()
