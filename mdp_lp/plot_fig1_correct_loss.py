"""
Figure-1-style landscape (Loss top row, gradient dL/dlambda bottom row) as
lambda_pred is swept around lambda_true, for three DFL objectives:

  (a) Original DFL regret        L = -r_true^T y*(l)                          (biased)
  (b) Augmented Lagrangian       L = -r_true^T y* + rho||H_true y*-gamma||^2   (biased, current paper Eq.5)
  (c) Full aug-Lagrangian (ours) L = (b) + (v*)^T(H_true y*-gamma)            (minimizer AT lambda_true)

Reuses the cached landscape arrays from plot_aug_lag_linear_compare
(dfl_term = -r^T y*, lin_term = v*^T(Hy*-gamma), pen_term = ||Hy*-gamma||^2),
so no LP re-solve is needed. Mirrors Figure 1 of the paper but replaces the
sketch column with the dual-corrected loss.

Usage:  python -m mdp_lp.plot_fig1_correct_loss
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NPZ = "mdp_lp/plot_results/aug_lag_linear_compare.npz"
RHO = 30.0                 # penalty weight (any rho>0; full loss is unbiased for all)
XLO, XHI = 8.0, 12.0       # match Figure 1's x-range


def main():
    d = np.load(NPZ)
    lam = d["lambda_values"]
    lam_true = float(d["lambda_true"])
    m = (lam >= XLO) & (lam <= XHI)
    lam = lam[m]
    order = np.argsort(lam)
    lam = lam[order]
    dfl = d["dfl_term"][ (d["lambda_values"] >= XLO) & (d["lambda_values"] <= XHI) ][order]  # -r^T y*
    lin = d["lin_term"][ (d["lambda_values"] >= XLO) & (d["lambda_values"] <= XHI) ][order]   # v*^T(Hy-g)
    pen = d["pen_term"][ (d["lambda_values"] >= XLO) & (d["lambda_values"] <= XHI) ][order]    # ||Hy-g||^2

    # the three objectives
    L_orig = dfl
    L_pen = dfl + RHO * pen
    L_full = dfl + lin + RHO * pen
    cols = [
        ("(a) Original DFL\n$-r_{true}^{\\top} y^*(\\lambda)$", L_orig, "tab:red"),
        (f"(b) Augmented Lagrangian ($\\rho$={RHO:g})\n$-r_{{true}}^{{\\top}} y^* + \\rho\\|H_{{true}}y^*-\\gamma\\|^2$",
         L_pen, "tab:orange"),
        ("(c) Full aug-Lagrangian (ours)\n$+\\,(v^*)^{\\top}(H_{true}y^*-\\gamma)$", L_full, "tab:green"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8), sharex=True)
    for j, (title, L, c) in enumerate(cols):
        imin = int(np.argmin(L))
        lam_star = lam[imin]
        grad = np.gradient(L, lam)

        # top: loss
        ax = axes[0, j]
        ax.plot(lam, L, color=c, lw=2.2)
        ax.axvline(lam_true, color="red", ls="--", lw=1.4)
        ax.plot(lam_star, L[imin], "o", color="green", ms=11, zorder=5)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Loss" if j == 0 else "")
        ax.grid(alpha=0.25)
        shift = lam_star - lam_true
        ax.text(0.04, 0.06,
                f"$\\lambda_{{true}}={lam_true:g}$\n$\\lambda^*={lam_star:.2f}$  (shift {shift:+.2f})",
                transform=ax.transAxes, fontsize=11, va="bottom",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

        # bottom: gradient
        ax = axes[1, j]
        ax.plot(lam, grad, color=c, lw=1.6)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.axvline(lam_true, color="red", ls="--", lw=1.4)
        ax.plot(lam_star, 0.0, "o", color="green", ms=9, zorder=5)
        ax.set_xlabel("$\\lambda_{pred}$")
        ax.set_ylabel("$dL/d\\lambda$" if j == 0 else "")
        ax.grid(alpha=0.25)

    # shared legend cue
    axes[0, 2].plot([], [], "r--", label="$\\lambda_{true}$")
    axes[0, 2].plot([], [], "go", label="minimizer $\\lambda^*$")
    axes[0, 2].legend(loc="upper right", fontsize=10, framealpha=0.9)

    plt.suptitle("Loss (top) and gradient (bottom) vs $\\lambda_{pred}$:  the linear dual term (c) "
                 "moves the minimizer back to $\\lambda_{true}$",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    out = "mdp_lp/plot_results/fig1_correct_loss"
    plt.savefig(out + ".png", dpi=160, bbox_inches="tight")
    plt.savefig(out + ".pdf", bbox_inches="tight")
    print(f"saved {out}.png / .pdf")
    for name, L in [("original", L_orig), ("penalty", L_pen), ("full(ours)", L_full)]:
        ls = lam[int(np.argmin(L))]
        print(f"  {name:>12}: lambda* = {ls:.3f}  (shift {ls-lam_true:+.3f})")


if __name__ == "__main__":
    main()
