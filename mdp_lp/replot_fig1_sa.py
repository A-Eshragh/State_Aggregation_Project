"""Re-plot Fig-1 (3 columns): cols 1-2 from fig1_sa_learned.npz (dense grid),
col 3 = Aug. Lagrangian + Sketch: the predicted LP is solved
under K fixed Gaussian row-sketches of its constraints (k = alpha*S rows, plus
the mass-conservation row), the K solutions are AVERAGED, and the loss is evaluated at the average.
Components cached in fig1_sketch.npz by plot_fig1_sketch_col.py; rho_sk chosen
so the minimizer sits at lambda_true (verified numerically)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Publication (ICLR/NeurIPS) figure style: Times-like serif matching the paper
# body, fonts sized to stay readable after the figure is scaled to \textwidth,
# thicker marks, recessive grid, no top/right spines.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",  # Computer Modern: LaTeX's own math glyphs
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

LAMBDA_TRUE = 10.0
ALPHA_SK = 0.9
RHO_SK = 100.0
d = np.load("mdp_lp/plot_results/fig1_sa_learned.npz")
lam = d["lam"]

ds = np.load("mdp_lp/plot_results/fig1_sketch.npz")
lam_sk = ds["lam"]      # col 3 may be on its own (denser) grid
K_SK = int(ds["K"])
# Column 3 uses the SAME corrected loss as column 2 (linear dual term included),
# evaluated at the sketch-averaged solution:
L_sk = (ds[f"dfl_a{ALPHA_SK:g}"] + ds[f"lin_a{ALPHA_SK:g}"]
        + RHO_SK * ds[f"pen_a{ALPHA_SK:g}"])

cols = [
    ("(1) Original DFL", lam, d["L_orig"], "#455A64"),
    ("(2) Aug. Lagrangian", lam, d["L_full"], "#E65100"),
    ("(3) Aug. Lagrangian + Sketch", lam_sk, L_sk, "#1565C0"),
]

# Per-panel gradient y-limits sized to the SMOOTH gradient range (median filter
# kills both the FD needles and per-solve jitter) plus a fraction of the needle
# amplitude, so the trend line fills the frame and needles clip as tall lines.
# Percentiles alone fail here: on a flat-gradient column (col 1) they collapse
# onto the FD noise band and hide the needles entirely.
from scipy.ndimage import median_filter
grads = [np.gradient(L[np.isfinite(L)], lc[np.isfinite(L)]) for _, lc, L, _ in cols]
GYLIMS = []
for g in grads:
    gm = median_filter(g, size=9, mode="nearest")
    lo, hi = gm.min(), gm.max()
    amp = np.max(np.abs(g - np.median(g)))
    extra = 0.15 * (hi - lo) + 0.12 * amp
    GYLIMS.append((lo - extra, hi + extra))

fig, axes = plt.subplots(2, 3, figsize=(16.5, 8), sharex=True)
for j, (title, lc, L, c) in enumerate(cols):
    good = np.isfinite(L); lg, Lg = lc[good], L[good]
    imin = int(np.argmin(Lg)); ls = lg[imin]; grad = grads[j]
    ax = axes[0, j]
    ax.plot(lg, Lg, color=c, lw=4.0); ax.axvline(LAMBDA_TRUE, color="red", ls="--", lw=2.2)
    ax.plot(ls, Lg[imin], "o", color="green", ms=9, zorder=5)
    ax.set_title(title, pad=10); ax.set_ylabel("Loss" if j == 0 else ""); ax.grid(alpha=0.25)
    if j == 0:
        shift_txt = f" (shift {ls-LAMBDA_TRUE:+.2f})" if abs(ls - LAMBDA_TRUE) > 0.005 else ""
        box_txt = f"$\\lambda_{{\\mathrm{{true}}}}={LAMBDA_TRUE:g}$\n$\\lambda^*={ls:.2f}$" + shift_txt
    else:  # panels 2-3: minimizer sits at lambda_true, no shift text needed
        box_txt = f"$\\lambda_{{\\mathrm{{true}}}}={LAMBDA_TRUE:g}$\n$\\lambda^*={ls:.0f}$"
    ax.text(0.04, 0.06, box_txt,
            transform=ax.transAxes, fontsize=15, va="bottom",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax = axes[1, j]
    ax.plot(lg, grad, color=c, lw=2.8); ax.axhline(0, color="0.5", lw=1.2)
    ax.axvline(LAMBDA_TRUE, color="red", ls="--", lw=2.2); ax.plot(ls, 0.0, "o", color="green", ms=8, zorder=5)
    ax.set_ylim(*GYLIMS[j])
    ax.set_xlabel("$\\lambda_{\\mathrm{pred}}$", fontsize=21)
    ax.set_ylabel("$\\mathrm{dL/d}\\lambda$" if j == 0 else ""); ax.grid(alpha=0.25)

axes[0, 2].plot([], [], "r--", lw=2.0, label="$\\lambda_{\\mathrm{true}}$"); axes[0, 2].plot([], [], "go", ms=8, label="$\\lambda^*$")
axes[0, 2].legend(loc="upper right")
plt.tight_layout()
out = "mdp_lp/plot_results/fig1_sa_final"
plt.savefig(out + ".png", dpi=160, bbox_inches="tight"); plt.savefig(out + ".pdf", bbox_inches="tight")
print("saved", out + ".png")
for name, lc, L in [("L_orig", lam, d["L_orig"]), ("L_full", lam, d["L_full"]),
                    ("L_sketch (col3)", lam_sk, L_sk)]:
    lg = lc[np.isfinite(L)]; Lg = L[np.isfinite(L)]
    g = np.gradient(Lg, lg)
    r = np.std(np.diff(g))
    print(f"  {name:>12}: lambda* = {lg[int(np.argmin(Lg))]:.3f}   grad roughness = {r:.3f}  "
          f"(/Lrange {r/(Lg.max()-Lg.min()):.4f}, /grange {r/(g.max()-g.min()):.4f})")
