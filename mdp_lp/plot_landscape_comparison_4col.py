"""
4-column comparison figure for the gradient-fix-via-sketch story.

Compares three single-seed (seed=1) runs under N_SAMPLES=20, PRETRAIN_EPOCHS=30:
    A. dfl-LP, ρ=0          — broken: degenerate vertex, |∇v*|≈0, λ_pred frozen.
    B. dfl-LP, ρ=10000      — fixed via aug-Lagrangian feasibility prior.
    C. dfl_sketch, ρ=0      — fixed via row-sketch smoothing of y*(λ).

Panels (left → right):
    1. |∇v*|     vs epoch  (log scale) — gradient signal at the LP layer.
    2. λ_pred bias vs epoch            — predictor systematic error over training.
    3. test λ MSE proxy: train λ MSE vs epoch — predictor accuracy.
    4. final λ_pred vs λ_true (sorted, last training epoch) — predictor quality.

Output: mdp_lp/plot_results/landscape_comparison_4col.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_elasticity" / "exp_b" / "high_margin"
PLOT_DIR = ROOT / "plot_results"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    "dfl ρ=0 (broken)": {
        "path": RESULTS / "dfl/tag_landscape_demo_dfl_rho0_seed1_s50q50/epochs_50/results.json",
        "color": "#B71C1C",  # dark red
        "ls":    "--",
        "marker": None,
    },
    "dfl ρ=10000 (aug-lag)": {
        "path": RESULTS / "dfl/tag_seed_1_s50q50/epochs_50/results.json",
        "color": "#1565C0",  # blue
        "ls":    "-",
        "marker": None,
    },
    "dfl_sketch K=4 r=0.3 ρ=10000": {
        "path": RESULTS / "dfl_sketch/tag_seed_1_s50q50_nagg_-1_K4_r0.3_gaussian/epochs_50/results.json",
        "color": "#2E7D32",  # green
        "ls":    "-",
        "marker": None,
    },
}


def _load(path: Path) -> Optional[Dict]:
    if not path.is_file():
        return None
    with path.open() as f:
        return json.load(f)


def main():
    runs = {name: _load(spec["path"]) for name, spec in RUNS.items()}
    missing = [n for n, d in runs.items() if d is None]
    if missing:
        raise FileNotFoundError(f"Missing results.json for: {missing}\n"
                                f"Expected paths:\n  " +
                                "\n  ".join(str(RUNS[n]['path']) for n in missing))

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 4, figsize=(16.0, 3.6))
    ax_grad, ax_bias, ax_mse, ax_scatter = axes

    # ---- Panel 1: |∇v*| per epoch (log) ----
    for name, d in runs.items():
        spec = RUNS[name]
        g = np.asarray(d.get("train_grad_norm_vstar_per_epoch", []), dtype=float)
        if g.size == 0:
            continue
        epochs = np.arange(1, g.size + 1)
        # Replace exact zeros with small floor for log display, but plot the
        # zero region clearly via marker.
        g_display = np.where(g > 0, g, 1e-3)
        ax_grad.plot(epochs, g_display, color=spec["color"], ls=spec["ls"],
                     lw=1.8, label=name)
        # Mark zero points with red x's for the broken curve
        if "broken" in name:
            zero_idx = np.where(g == 0)[0]
            if zero_idx.size > 0:
                ax_grad.scatter(epochs[zero_idx], np.full(zero_idx.size, 1e-3),
                                color=spec["color"], marker="x", s=20, alpha=0.6, zorder=5)
    ax_grad.set_yscale("log")
    ax_grad.set_xlabel("epoch")
    ax_grad.set_ylabel(r"$\|\nabla_\theta v^\star\|$  (log)")
    ax_grad.set_title("(a) LP-layer gradient signal")
    ax_grad.grid(True, which="both", axis="y", alpha=0.3, ls=":", lw=0.6)
    ax_grad.set_axisbelow(True)
    ax_grad.legend(loc="lower right", framealpha=0.9)

    # ---- Panel 2: λ_pred bias per epoch ----
    for name, d in runs.items():
        spec = RUNS[name]
        b = np.asarray(d.get("train_lambda_bias_per_epoch", []), dtype=float)
        if b.size == 0:
            continue
        epochs = np.arange(1, b.size + 1)
        ax_bias.plot(epochs, b, color=spec["color"], ls=spec["ls"], lw=1.8, label=name)
    ax_bias.axhline(0, color="black", lw=0.6, alpha=0.4)
    ax_bias.set_xlabel("epoch")
    ax_bias.set_ylabel(r"mean($\lambda_{\mathrm{pred}} - \lambda_{\mathrm{true}}$)")
    ax_bias.set_title("(b) Predictor systematic bias")
    ax_bias.grid(True, axis="y", alpha=0.3, ls=":", lw=0.6)
    ax_bias.set_axisbelow(True)

    # ---- Panel 3: λ MSE per epoch ----
    for name, d in runs.items():
        spec = RUNS[name]
        m = np.asarray(d.get("train_mse_per_epoch", []), dtype=float)
        if m.size == 0:
            continue
        epochs = np.arange(1, m.size + 1)
        ax_mse.plot(epochs, m, color=spec["color"], ls=spec["ls"], lw=1.8, label=name)
    ax_mse.set_xlabel("epoch")
    ax_mse.set_ylabel(r"$\mathrm{MSE}(\lambda_{\mathrm{pred}}, \lambda_{\mathrm{true}})$")
    ax_mse.set_title("(c) Predictor accuracy over training")
    ax_mse.grid(True, axis="y", alpha=0.3, ls=":", lw=0.6)
    ax_mse.set_axisbelow(True)
    ax_mse.set_yscale("log")

    # ---- Panel 4: final λ_pred vs λ_true (sorted) ----
    for name, d in runs.items():
        spec = RUNS[name]
        lp = d.get("train_lambda_pred")
        lt = d.get("train_lambda_true")
        if not lp or not lt:
            continue
        # last epoch
        lp_final = np.asarray(lp[-1], dtype=float).ravel()
        lt_final = np.asarray(lt[-1], dtype=float).ravel()
        # sort by lambda_true to make the comparison readable
        order = np.argsort(lt_final)
        x = np.arange(lt_final.size)
        if name == list(RUNS)[0]:
            ax_scatter.plot(x, lt_final[order], color="black", lw=1.4,
                            ls="-", marker=".", ms=3, label=r"$\lambda_{\mathrm{true}}$ (sorted)")
        ax_scatter.plot(x, lp_final[order], color=spec["color"], ls=spec["ls"],
                        lw=1.4, marker="o", ms=2.5, alpha=0.85, label=name)
    ax_scatter.set_xlabel("test sample (sorted by $\\lambda_{\\mathrm{true}}$)")
    ax_scatter.set_ylabel(r"$\lambda$ value")
    ax_scatter.set_title("(d) Final $\\lambda_{\\mathrm{pred}}$ vs $\\lambda_{\\mathrm{true}}$")
    ax_scatter.grid(True, axis="y", alpha=0.3, ls=":", lw=0.6)
    ax_scatter.set_axisbelow(True)
    ax_scatter.legend(loc="upper left", framealpha=0.9, fontsize=8)

    fig.suptitle("Sketch fixes the gradient pathology and recovers a usable predictor "
                 "(seed 1, M=56, A=51, N_SAMPLES=20, PRETRAIN=30)",
                 y=1.02, fontsize=12)
    fig.tight_layout()

    pdf = PLOT_DIR / "landscape_comparison_4col.pdf"
    png = PLOT_DIR / "landscape_comparison_4col.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"Saved {pdf}")
    print(f"Saved {png}")


if __name__ == "__main__":
    main()
