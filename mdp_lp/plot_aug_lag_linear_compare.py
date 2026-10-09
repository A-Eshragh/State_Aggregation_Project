"""
Compare two DFL loss landscapes as a function of the predicted parameter lambda:

  penalty-only (what has been run so far):
      L_pen(l)  = -r_true^T y*(l)                       + rho ||H_true y*(l) - gamma||^2

  full augmented Lagrangian (adds the linear dual term, v* = true optimal dual):
      L_full(l) = -r_true^T y*(l) + v*^T(H_true y*(l)-gamma) + rho ||H_true y*(l) - gamma||^2

y*(l) is the occupancy that solves the PREDICTED MDP (parameter l); r_true, H_true,
gamma are the TRUE MDP; v* is the dual of the true LP (H_true y = gamma).

Key question: does the linear term put the minimizer at lambda_true (removing the
shift the penalty-only loss has for every finite rho)?

Uses scipy/HiGHS for the LP (primal y* + equality dual v*); Gurobi license expired.

Usage:
    python -m mdp_lp.plot_aug_lag_linear_compare
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linprog

from mdp_lp.inventory_elasticity import create_inventory_from_lambda

# ── MDP parameters (match plot_aug_lag_landscape.py for comparability) ──
B_MAX, S_MAX, Q_MAX, D_MAX = 5, 50, 30, 30
FOC, VOC, HC, BC, BETA, PRICE = 0, 0.5, 0.1, 0, 0.95, 10.0
LAMBDA_TRUE = 10.0
RHO_VALUES = [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0]


def build(lam_val):
    lam = torch.tensor([lam_val], dtype=torch.float32)
    p = torch.tensor([PRICE], dtype=torch.float32)
    mdp = create_inventory_from_lambda(lam, B_MAX, S_MAX, Q_MAX, FOC, VOC, HC, BC, BETA, p)
    S = mdp.P.shape[1]
    gamma = mdp.build_gamma(1, S, torch.device("cpu"))
    gamma = torch.clamp(gamma, min=1e-8)
    gamma = gamma / gamma.sum(dim=1, keepdim=True)
    H = mdp.build_H(mdp.P)
    r = mdp.r.reshape(1, -1)
    return (H[0].detach().numpy().astype(np.float64),
            r[0].detach().numpy().astype(np.float64),
            gamma[0].detach().numpy().astype(np.float64))


def solve_lp(H, r, gamma):
    """max r^T y s.t. H y = gamma, y>=0.  Returns y*, v* (value-fn dual of H y=gamma)."""
    res = linprog(c=-r, A_eq=H, b_eq=gamma, bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return res.x, -np.asarray(res.eqlin.marginals)


def main():
    # True MDP + true dual v*
    H_true, r_true, gamma_true = build(LAMBDA_TRUE)
    y_true, v_star = solve_lp(H_true, r_true, gamma_true)
    print(f"Strong duality check: r^T y* = {r_true @ y_true:.4f}, "
          f"gamma^T v* = {gamma_true @ v_star:.4f}")

    # Sweep lambda_pred: coarse over the full range + dense near lambda_true
    # (dense region is what determines the local-basin minimizer we report).
    lambda_values = np.sort(np.unique(np.concatenate([
        np.linspace(0.5, 50, 70),
        np.linspace(LAMBDA_TRUE - 4, LAMBDA_TRUE + 4, 161),  # 0.05 spacing
    ])))

    dfl_term = np.zeros_like(lambda_values)   # -r_true^T y*(l)
    lin_term = np.zeros_like(lambda_values)   #  v*^T (H_true y*(l) - gamma)
    pen_term = np.zeros_like(lambda_values)   #  ||H_true y*(l) - gamma||^2

    print(f"Computing landscape over {len(lambda_values)} points...")
    for i, lam in enumerate(lambda_values):
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(lambda_values)}", flush=True)
        y, _ = solve_lp(*build(lam))
        c = H_true @ y - gamma_true
        dfl_term[i] = -(r_true @ y)
        lin_term[i] = v_star @ c
        pen_term[i] = c @ c

    idx_true = int(np.argmin(np.abs(lambda_values - LAMBDA_TRUE)))

    # ── Summary table: minimizer of each loss vs rho ──
    # Restrict to a window around lambda_true so we report the LOCAL basin
    # (the penalty-only shift is a local effect; far regions have other minima).
    win = (lambda_values >= LAMBDA_TRUE - 4) & (lambda_values <= LAMBDA_TRUE + 4)
    lam_w = lambda_values[win]
    dfl_w, lin_w, pen_w = dfl_term[win], lin_term[win], pen_term[win]

    print("\n" + "=" * 72)
    print(f"{'rho':>8} | {'penalty-only lam*':>18} {'shift':>8} | "
          f"{'full (linear) lam*':>18} {'shift':>8}")
    print("-" * 72)
    rows = []
    for rho in RHO_VALUES:
        L_pen = dfl_w + rho * pen_w
        L_full = dfl_w + lin_w + rho * pen_w
        lam_pen = lam_w[np.argmin(L_pen)]
        lam_full = lam_w[np.argmin(L_full)]
        rows.append((rho, lam_pen, lam_full))
        print(f"{rho:>8.1f} | {lam_pen:>18.3f} {lam_pen-LAMBDA_TRUE:>+8.3f} | "
              f"{lam_full:>18.3f} {lam_full-LAMBDA_TRUE:>+8.3f}")
    print("=" * 72)

    # ── Figure ──
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))

    # Panel A: penalty-only landscape for several rho
    ax = axes[0]
    for rho in [0.1, 1.0, 10.0, 100.0]:
        L = dfl_w + rho * pen_w
        L = L - L.min()
        lam_min = lam_w[np.argmin(L)]
        ax.plot(lam_w, L, lw=2, label=f"rho={rho:g}  (min {lam_min:.2f})")
    ax.axvline(LAMBDA_TRUE, color="red", ls="--", lw=1.5, label="lambda_true")
    ax.set_title("Penalty-only  $-r^T y^* + \\rho\\,\\|H y^*-\\gamma\\|^2$\n"
                 "minimizer shifts with $\\rho$", fontsize=12)
    ax.set_xlabel("$\\lambda_{pred}$"); ax.set_ylabel("loss (shifted to 0 min)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Panel B: full (with linear dual term) landscape for several rho
    ax = axes[1]
    for rho in [0.0, 0.1, 1.0, 10.0, 100.0]:
        L = dfl_w + lin_w + rho * pen_w
        L = L - L.min()
        lam_min = lam_w[np.argmin(L)]
        ax.plot(lam_w, L, lw=2, label=f"rho={rho:g}  (min {lam_min:.2f})")
    ax.axvline(LAMBDA_TRUE, color="red", ls="--", lw=1.5, label="lambda_true")
    ax.set_title("Full aug-Lag  $+\\,(v^*)^T(H y^*-\\gamma)$\n"
                 "minimizer pinned at $\\lambda_{true}$ for all $\\rho$", fontsize=12)
    ax.set_xlabel("$\\lambda_{pred}$"); ax.set_ylabel("loss (shifted to 0 min)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Panel C: minimizer vs rho for both losses
    ax = axes[2]
    rho_sweep = np.concatenate([[0.0], np.logspace(-2, 4, 80)])
    lam_pen_sweep, lam_full_sweep = [], []
    for rho in rho_sweep:
        lam_pen_sweep.append(lam_w[np.argmin(dfl_w + rho * pen_w)])
        lam_full_sweep.append(lam_w[np.argmin(dfl_w + lin_w + rho * pen_w)])
    ax.semilogx(np.clip(rho_sweep, 1e-3, None), lam_pen_sweep, "b-o", ms=3,
                lw=1.8, label="penalty-only")
    ax.semilogx(np.clip(rho_sweep, 1e-3, None), lam_full_sweep, "g-s", ms=3,
                lw=1.8, label="full (linear dual term)")
    ax.axhline(LAMBDA_TRUE, color="red", ls="--", lw=2, label="lambda_true")
    ax.set_xlabel("$\\rho$"); ax.set_ylabel("minimizer $\\lambda^*$")
    ax.set_title("Minimizer vs $\\rho$\npenalty-only only reaches "
                 "$\\lambda_{true}$ as $\\rho\\to\\infty$", fontsize=12)
    ax.legend(fontsize=10); ax.grid(alpha=0.3, which="both")

    plt.suptitle("Adding the linear dual term $(v^*)^T(H_{true}y^*-\\gamma)$ "
                 "removes the penalty-only minimizer shift", fontsize=14, y=1.03)
    plt.tight_layout()
    out = "mdp_lp/plot_results/aug_lag_linear_compare"
    plt.savefig(out + ".png", dpi=150, bbox_inches="tight")
    plt.savefig(out + ".pdf", bbox_inches="tight")
    print(f"\nSaved figure to {out}.png / .pdf")

    # Save arrays for numeric inspection (per the 'verify plots with numbers' habit)
    np.savez(out + ".npz", lambda_values=lambda_values, dfl_term=dfl_term,
             lin_term=lin_term, pen_term=pen_term, v_star=v_star,
             lambda_true=LAMBDA_TRUE, rho_values=np.array(RHO_VALUES),
             rows=np.array(rows))
    print(f"Saved arrays to {out}.npz")


if __name__ == "__main__":
    main()
