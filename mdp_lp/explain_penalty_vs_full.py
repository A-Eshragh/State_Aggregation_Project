"""
Intuition for penalty-only vs full augmented Lagrangian, on a 1-D toy that has
the SAME structure as the MDP-DFL loss:

    min_x  -x        s.t.   c(x) = x - 1 = 0        (true answer x*=1, multiplier v*=1)

    -x                        <->  -r_true^T y        (linear objective)
    c(x) = x-1                <->  H_true y - gamma    (constraint violation)
    v* = 1                    <->  dual / value fn v*  (exchange rate obj<->constraint)

    L_pen(x)  = -x            + rho (x-1)^2   -> min at x = 1 + 1/(2 rho)   (biased)
    L_full(x) = -x + v*(x-1)  + rho (x-1)^2   = -1 + rho(x-1)^2 -> min at x = 1 (any rho)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

X_TRUE, V_STAR = 1.0, 1.0
x = np.linspace(-0.6, 3.0, 400)

obj = -x                         # objective, wants x up
lin = V_STAR * (x - X_TRUE)      # linear dual term
c2 = (x - X_TRUE) ** 2           # constraint penalty (without rho)

RHO = 1.0                        # a single, deliberately SMALL rho to expose the bias
pen = RHO * c2
L_pen = obj + pen
L_full = obj + lin + pen

xmin_pen = X_TRUE + 1.0 / (2 * RHO)   # = 1.5 for rho=1

fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))

# ---- Panel A: penalty-only -> minimizer drifts off the constraint ----
a = ax[0]
a.plot(x, obj, "--", color="tab:blue", lw=1.6, label="objective  $-x$  (slope $-1$)")
a.plot(x, pen, "--", color="tab:orange", lw=1.6, label=r"penalty  $\rho(x-1)^2$")
a.plot(x, L_pen, "-", color="black", lw=2.6, label=r"$L_{pen}=-x+\rho(x-1)^2$")
a.axvline(X_TRUE, color="red", ls=":", lw=1.8)
a.plot(X_TRUE, (obj + pen)[np.argmin(np.abs(x - X_TRUE))], "r*", ms=16, zorder=5)
a.plot(xmin_pen, L_pen.min(), "go", ms=12, zorder=5)
a.annotate("", xy=(xmin_pen, L_pen.min()), xytext=(X_TRUE, L_pen.min()),
           arrowprops=dict(arrowstyle="<->", color="green", lw=2))
a.text((X_TRUE + xmin_pen) / 2, L_pen.min() + 0.28,
       r"bias $=\frac{1}{2\rho}$", ha="center", color="green", fontsize=12, fontweight="bold")
a.text(X_TRUE + 0.05, -1.9, "constraint\n$x^*=1$", color="red", fontsize=10)
a.set_title(r"Penalty-only  ($\rho=1$)"
            "\nweak restoring force $2\\rho c$ loses to the constant pull $\\Rightarrow$ min at $1+\\frac{1}{2\\rho}$",
            fontsize=11)
a.set_xlabel("$x$"); a.set_ylabel("loss"); a.set_ylim(-2.3, 3.2)
a.legend(fontsize=9, loc="upper center"); a.grid(alpha=0.3)

# ---- Panel B: full -> linear term flattens the objective, bowl stays centered ----
b = ax[1]
b.plot(x, obj + lin, "--", color="tab:blue", lw=1.6,
       label=r"$-x + v^*(x-1) = -1$  (flat!)")
b.plot(x, pen, "--", color="tab:orange", lw=1.6, label=r"penalty  $\rho(x-1)^2$")
b.plot(x, L_full, "-", color="black", lw=2.6,
       label=r"$L_{full}=-1+\rho(x-1)^2$")
b.axvline(X_TRUE, color="red", ls=":", lw=1.8)
b.plot(X_TRUE, L_full.min(), "go", ms=12, zorder=5)
b.text(X_TRUE + 0.05, -1.9, "min $=$ constraint\n$x^*=1$", color="green", fontsize=10)
b.set_title(r"Full aug-Lag: add $v^*(x-1)$"
            "\nlinear term cancels the objective's slope $\\Rightarrow$ min at $1$ for every $\\rho$",
            fontsize=11)
b.set_xlabel("$x$"); b.set_ylabel("loss"); b.set_ylim(-2.3, 3.2)
b.legend(fontsize=9, loc="upper center"); b.grid(alpha=0.3)

# ---- Panel C: minimizer vs rho (analytic) ----
c = ax[2]
rho = np.logspace(-1.3, 3, 200)
c.semilogx(rho, X_TRUE + 1.0 / (2 * rho), "b-", lw=2.4, label=r"penalty-only  $1+\frac{1}{2\rho}$")
c.semilogx(rho, np.full_like(rho, X_TRUE), "g-", lw=2.4, label=r"full  $=1$ (any $\rho$)")
c.axhline(X_TRUE, color="red", ls=":", lw=1.6)
c.fill_between(rho, X_TRUE, X_TRUE + 1.0 / (2 * rho), color="tab:blue", alpha=0.12)
c.text(3, 1.45, "bias only vanishes\nas $\\rho\\to\\infty$", color="tab:blue", fontsize=10)
c.set_xlabel(r"$\rho$"); c.set_ylabel("minimizer $x^\\star$")
c.set_title("Minimizer vs $\\rho$\npenalty-only is biased for every finite $\\rho$", fontsize=11)
c.set_ylim(0.9, 1.9); c.legend(fontsize=10); c.grid(alpha=0.3, which="both")

plt.suptitle(r"Why the linear dual term $v^*(x-1)$ fixes the bias: it is the exchange rate "
             r"that flattens the objective along the constraint",
             fontsize=13, y=1.02)
plt.tight_layout()
out = "mdp_lp/plot_results/explain_penalty_vs_full"
plt.savefig(out + ".png", dpi=150, bbox_inches="tight")
plt.savefig(out + ".pdf", bbox_inches="tight")
print("saved", out + ".png / .pdf")

# numeric check
print(f"\nrho=1:  penalty-only min = {xmin_pen:.3f} (bias +0.500),  full min = 1.000 (bias 0)")
for r in [0.1, 1, 10, 100]:
    print(f"  rho={r:>5}:  penalty-only x* = {1 + 1/(2*r):.4f}   full x* = 1.0000")
