#!/usr/bin/env python3
"""
Publication-style PNGs from ``results_n120`` (regime v4, k=7, TS vs DFL).

Unlike ``plot_publication_cw`` (dose over k with ``results_n40``), this
suite uses the **training-sample sweep** n (default ``20,40,80,120,160``) at fixed k=7.
**qpth** / **ssa** use v4 runs when present (``run_v4_hard_experiments --algos``);
otherwise they fall back to **v3 dose** at k=7 via ``--dose_root``.

Usage::

    python -m mdp_cw.plot_v4_hard_publication \\
        --v4_root mdp_cw/results_n120 \\
        --dose_root mdp_cw/results_n40 \\
        --out_dir mdp_cw/best_results_final/v4_hard_k7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats as sp_stats

# Re-use NeurIPS rc + save helper from the dose publication module.
import mdp_cw.plot_publication_cw as _pub  # noqa: F401 — applies rcParams
from mdp_cw.plot_publication_cw import (
    _epoch_backward,
    _iqr_band,
    _matrix,
    _save,
    load_seeds as load_seeds_dose,
)

DEFAULT_NS = (20, 40, 80, 120, 160)
K_FIXED = 7

TS_STY = dict(color="#E41A1C", label="Two-Stage", ls="-", marker="o", zorder=5)
DFL_STY = dict(color="#377EB8", label="DFL", ls="-", marker="s", zorder=6)
QPTH_STY = dict(color="#4DAF4A", label="DFL-qpth", ls="--", marker="^", zorder=4)
SSA_STY = dict(color="#984EA3", label="DFL-SSA", ls=":", marker="D", zorder=3)
REF_NOTE = "† v3 dose, k=7"


def _load_run(v4_root: Path, n: int, k: int, prefix: str, s: int) -> Optional[Dict]:
    p = v4_root / f"n{n}" / f"k{k}" / f"{prefix}_s{s}" / "results.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def load_seeds_v4(
    v4_root: Path, n: int, k: int, prefix: str, n_seeds: int,
) -> List[Dict]:
    return [d for s in range(n_seeds) if (d := _load_run(v4_root, n, k, prefix, s)) is not None]


def _arr(runs: List[Dict], key: str) -> np.ndarray:
    vals = [float(r[key]) for r in runs if key in r]
    return np.array(vals, dtype=np.float64)


def fig1_sample_scaling(
    v4_root: Path,
    out: Path,
    n_seeds: int,
    ns: Tuple[int, ...],
    dose_root: Optional[Path],
) -> None:
    """Median test regret vs training sample size n (k=7) — all four methods on one axes."""
    xs = np.array(ns, dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    all_sty = ((TS_STY, "ts"), (DFL_STY, "dfl"), (QPTH_STY, "qpth"), (SSA_STY, "ssa"))
    for sty, key in all_sty:
        med, q1, q3 = [], [], []
        for n in ns:
            runs = load_seeds_v4(v4_root, n, K_FIXED, key, n_seeds)
            vals = _arr(runs, "test_mean_regret")
            if vals.size == 0:
                med.append(np.nan)
                q1.append(np.nan)
                q3.append(np.nan)
            else:
                med.append(float(np.median(vals)))
                q1.append(float(np.percentile(vals, 25)))
                q3.append(float(np.percentile(vals, 75)))
        m, lo, hi = np.array(med), np.array(q1), np.array(q3)
        if np.any(np.isfinite(m)):
            lab = sty["label"] + (" (v4)" if key in ("qpth", "ssa") else "")
            ax.plot(
                xs, m, f"{sty['marker']}{sty['ls']}", color=sty["color"],
                lw=2.0, ms=6, label=lab, zorder=sty["zorder"],
            )
            ax.fill_between(xs, lo, hi, color=sty["color"], alpha=0.11, zorder=1)
        elif dose_root is not None and dose_root.is_dir() and key in ("qpth", "ssa"):
            runs = load_seeds_dose(dose_root, K_FIXED, key, n_seeds)
            vals = _arr(runs, "test_mean_regret")
            if vals.size == 0:
                continue
            med = float(np.median(vals))
            x0, x1 = float(xs.min()), float(xs.max())
            ax.hlines(
                med, x0, x1, colors=sty["color"], linestyles=sty["ls"],
                linewidth=2.0,
                label=f"{sty['label']} ({REF_NOTE})",
                zorder=sty["zorder"],
            )
    ax.set_xticks(list(ns))
    ax.set_xlabel("Training samples  n  (k = 7, regime v4 hard)")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylabel("Test regret (median ± IQR; symlog y)")
    ax.set_title("Sample scaling — TS, DFL, DFL-qpth, DFL-SSA")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.grid(True, alpha=0.25)
    fig.suptitle("v4 hard — Test regret vs sample size  n", fontsize=11, y=1.02)
    fig.tight_layout()
    _save(fig, out)


def fig2_learning_curves(
    v4_root: Path,
    out: Path,
    n_seeds: int,
    n_lo: int,
    n_hi: int,
    dose_root: Optional[Path],
) -> None:
    """Two panels: train reward vs epoch at k=7 for n=n_lo and n=n_hi."""
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), sharey=False)
    titles = [
        f"n = {n_lo}  (k = 7)",
        f"n = {n_hi}  (k = 7)",
    ]
    all_four = ((TS_STY, "ts"), (DFL_STY, "dfl"), (QPTH_STY, "qpth"), (SSA_STY, "ssa"))
    for ax, n, title in zip(axes, (n_lo, n_hi), titles):
        oracle_vals = []
        for sty, key in all_four:
            runs = load_seeds_v4(v4_root, n, K_FIXED, key, n_seeds)
            lab_note = "(v4)"
            if not runs and dose_root is not None and dose_root.is_dir():
                runs = load_seeds_dose(dose_root, K_FIXED, key, n_seeds)
                lab_note = f"({REF_NOTE})"
            mat = _matrix(runs, "train_reward_per_epoch")
            if mat is None:
                continue
            epochs = np.arange(mat.shape[1])
            med, q1, q3 = _iqr_band(mat)
            ax.plot(epochs, med, color=sty["color"], ls=sty["ls"],
                    lw=1.5, label=f"{sty['label']} {lab_note}", zorder=sty["zorder"])
            ax.fill_between(epochs, q1, q3, color=sty["color"], alpha=0.12, zorder=1)
            for r in runs:
                ov = r.get("train_optimal_reward")
                if ov is not None:
                    oracle_vals.append(float(ov))
        if oracle_vals:
            om = float(np.mean(oracle_vals))
            ax.axhline(om, color="#555555", lw=1.1, ls="--",
                       label=f"Oracle ({om:.2f})", zorder=1)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        if ax is axes[0]:
            ax.set_ylabel("Train reward (policy on predicted MDP)")
        ax.legend(loc="lower right", fontsize=7.5)
    fig.suptitle("Learning curves — v4 hard, k = 7  (all four methods)", fontsize=11, y=1.01)
    fig.tight_layout()
    _save(fig, out)


def fig3_compute_profile(
    v4_root: Path,
    out: Path,
    n_seeds: int,
    n_lo: int,
    n_hi: int,
    dose_root: Optional[Path],
) -> None:
    """Stacked solver vs backward — all four methods (v4 per n; v3 k7 fallback if missing)."""
    method_rows = (
        ("ts", TS_STY["label"], TS_STY["color"]),
        ("dfl", DFL_STY["label"], DFL_STY["color"]),
        ("qpth", "DFL-qpth", QPTH_STY["color"]),
        ("ssa", "DFL-SSA", SSA_STY["color"]),
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), sharey=True)
    for ax, n, title in zip(axes, (n_lo, n_hi), (f"n = {n_lo}", f"n = {n_hi}")):
        solver_means, bwd_means, labels, colors = [], [], [], []
        for key, lab0, col in method_rows:
            pool = load_seeds_v4(v4_root, n, K_FIXED, key, n_seeds)
            note = "v4"
            if not pool and dose_root is not None and dose_root.is_dir():
                pool = load_seeds_dose(dose_root, K_FIXED, key, n_seeds)
                note = "v3 k7"
            if not pool:
                continue
            solver_pool, bwd_pool = [], []
            for run in pool:
                st = run.get("solver_time_sec_per_epoch")
                if st and len(st) > 0:
                    solver_pool.append(float(np.mean(st)))
                bwd_pool.append(_epoch_backward(run))
            solver_means.append(float(np.mean(solver_pool)) if solver_pool else np.nan)
            bwd_means.append(float(np.mean(bwd_pool)) if bwd_pool else 0.0)
            labels.append(f"{lab0}\n({note})")
            colors.append(col)
        x = np.arange(len(labels))
        bar_w = 0.55
        solver_arr = np.array(solver_means)
        bwd_arr = np.array(bwd_means)
        ax.bar(x, solver_arr, bar_w, color=colors, alpha=0.85,
               edgecolor="white", linewidth=0.5, label="LP/QP solve", zorder=3)
        ax.bar(x, bwd_arr, bar_w, bottom=np.nan_to_num(solver_arr),
               color=colors, alpha=0.40, edgecolor="white", linewidth=0.5,
               hatch="//", label="Backward pass", zorder=3)
        for xi, (sv, bw) in enumerate(zip(solver_arr, bwd_arr)):
            total = sv + bw
            if np.isfinite(total):
                ax.text(xi, total + total * 0.05, f"{total*1000:.1f} ms",
                        ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title)
        if ax is axes[0]:
            ax.set_ylabel("Avg wall-clock per epoch  (s)")
        if ax is axes[0]:
            ax.legend(loc="upper left", fontsize=7.5)
    fig.suptitle("Compute profile — v4 hard, k = 7", fontsize=11, y=1.01)
    fig.tight_layout()
    _save(fig, out)


def fig4_paired_k7_n120(
    v4_root: Path,
    out: Path,
    n_seeds: int,
    n: int,
    dose_root: Optional[Path],
) -> None:
    """Per-seed regret lines + boxplots — all four methods (v4 n; v3 k7 fallback)."""
    rows = (
        ("ts", "Two-Stage", "#E41A1C"),
        ("dfl", "DFL", "#377EB8"),
        ("qpth", "DFL-qpth", "#4DAF4A"),
        ("ssa", "DFL-SSA", "#984EA3"),
    )
    regret_rows: List[np.ndarray] = []
    labels: List[str] = []
    colors: List[str] = []
    for key, lab0, col in rows:
        runs = load_seeds_v4(v4_root, n, K_FIXED, key, n_seeds)
        lab = lab0
        if not runs and dose_root is not None and dose_root.is_dir():
            runs = load_seeds_dose(dose_root, K_FIXED, key, n_seeds)
            lab = f"{lab0}†"
        if not runs:
            continue
        regret_rows.append(
            np.array([float(r["test_mean_regret"]) for r in runs if "test_mean_regret" in r])
        )
        labels.append(lab)
        colors.append(col)

    if len(regret_rows) < 2:
        fig, ax = plt.subplots(figsize=(6.0, 2.5))
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        _save(fig, out)
        return

    min_n = min(len(r) for r in regret_rows)
    trim = [r[:min_n] for r in regret_rows]

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    ax = axes[0]
    x = np.arange(min_n)
    for arr, lab, c in zip(trim, labels, colors):
        ax.plot(x, arr, lw=1.35, label=lab, color=c, alpha=0.92)
    ax.set_xlabel("Seed index")
    ax.set_ylabel("Test mean regret")
    ax.set_title(f"(a)  Per-seed  n = {n}, k = 7")
    ax.legend(loc="upper right", fontsize=6.8)
    ax.grid(True, alpha=0.25)

    ax2 = axes[1]
    bp = ax2.boxplot(
        trim,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker=".", markersize=3, alpha=0.5),
        medianprops=dict(color="#111111", linewidth=2.0),
        widths=0.52,
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.28)
        patch.set_edgecolor(c)
        patch.set_linewidth(1.5)
    for xi, (arr, c) in enumerate(zip(trim, colors), start=1):
        ax2.scatter([xi], [np.mean(arr)], color=c, s=28, marker="D",
                    zorder=10, edgecolors="white", linewidths=0.5)
    ax2.set_xticks(range(1, len(labels) + 1))
    ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylabel("Test regret")
    ax2.set_yscale("symlog", linthresh=1.0)
    med_line = ", ".join(f"{labels[i].split('†')[0]}={float(np.median(trim[i])):.4f}" for i in range(len(trim)))
    ax2.set_title(f"(b)  Distribution  k = 7\n{med_line}")
    mean_patch = Line2D([0], [0], marker="D", color="gray", markersize=6,
                        linestyle="none", label="mean")
    ax2.legend(handles=[mean_patch], fontsize=7.5, loc="upper right")

    note = ""
    if "Two-Stage" in labels and "DFL-qpth" in labels:
        ts_arr = trim[labels.index("Two-Stage")]
        qp_arr = trim[labels.index("DFL-qpth")]
        if len(ts_arr) == len(qp_arr):
            try:
                p_q = sp_stats.wilcoxon(ts_arr, qp_arr, alternative="greater").pvalue
                med_d = float(np.median(ts_arr - qp_arr))
                note = f"  median(TS−qpth)={med_d:.4f}  Wilcoxon p={p_q:.3g}"
            except Exception as e:
                note = f"  ({e})"

    fig.suptitle(f"v4 hard — all four methods  n = {n}, k = 7{note}", fontsize=10.5, y=1.05)
    fig.tight_layout()
    _save(fig, out)


def fig5_regret_boxplot_n_compare(
    v4_root: Path,
    out: Path,
    n_seeds: int,
    n_lo: int,
    n_hi: int,
    dose_root: Optional[Path],
) -> None:
    """All four methods per n (v4; v3 k7 fallback for missing qpth/ssa)."""
    method_rows = (
        ("ts", "Two-Stage", "#E41A1C"),
        ("dfl", "DFL", "#377EB8"),
        ("qpth", "DFL-qpth", "#4DAF4A"),
        ("ssa", "DFL-SSA", "#984EA3"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    for ax, n, subtitle in zip(axes, (n_lo, n_hi), (f"n = {n_lo}", f"n = {n_hi}")):
        data, tick_labels, colors = [], [], []
        for key, lab0, col in method_rows:
            runs = load_seeds_v4(v4_root, n, K_FIXED, key, n_seeds)
            lab = lab0
            if not runs and dose_root is not None and dose_root.is_dir():
                runs = load_seeds_dose(dose_root, K_FIXED, key, n_seeds)
                lab = f"{lab0}†"
            vals = np.array([float(r["test_mean_regret"]) for r in runs
                             if "test_mean_regret" in r])
            if vals.size == 0:
                continue
            data.append(vals)
            tick_labels.append(lab)
            colors.append(col)
        if not data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        bp = ax.boxplot(data, patch_artist=True, showfliers=True,
                        flierprops=dict(marker=".", markersize=3.5, alpha=0.5),
                        medianprops=dict(color="#111111", linewidth=2.0),
                        widths=0.5)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.28)
            patch.set_edgecolor(c)
            patch.set_linewidth(1.5)
        for xi, (arr, c) in enumerate(zip(data, colors), start=1):
            ax.scatter([xi], [np.mean(arr)], color=c, s=28, marker="D",
                       zorder=10, edgecolors="white", linewidths=0.5)
        ax.set_xticks(range(1, len(tick_labels) + 1))
        ax.set_xticklabels(tick_labels, fontsize=8.5)
        ax.set_title(f"k = 7  ({subtitle})")
        ax.set_yscale("symlog", linthresh=1.0)
        if ax is axes[0]:
            ax.set_ylabel("Test regret  (oracle − predicted)")
    mean_patch = Line2D([0], [0], marker="D", color="gray", markersize=6,
                        linestyle="none", label="mean")
    axes[-1].legend(handles=[mean_patch], fontsize=7.5, loc="upper right")
    fig.suptitle("Test regret — all four methods (k = 7; † v3 ref if missing)", fontsize=11, y=1.01)
    fig.tight_layout()
    _save(fig, out)


def write_summary_v4(
    v4_root: Path,
    out: Path,
    n_seeds: int,
    ns: Tuple[int, ...],
    dose_root: Optional[Path],
) -> None:
    summary: Dict[str, object] = {"v4_hard_k7": {}}
    v4 = summary["v4_hard_k7"]
    assert isinstance(v4, dict)
    for n in ns:
        v4[f"n{n}"] = {}
        for key in ("ts", "dfl", "qpth", "ssa"):
            runs = load_seeds_v4(v4_root, n, K_FIXED, key, n_seeds)
            vals = _arr(runs, "test_mean_regret")
            if vals.size == 0:
                continue
            v4[f"n{n}"][key] = {
                "n_seeds": int(vals.size),
                "median_regret": float(np.median(vals)),
                "mean_regret": float(np.mean(vals)),
                "q1_regret": float(np.percentile(vals, 25)),
                "q3_regret": float(np.percentile(vals, 75)),
            }
    if dose_root is not None and dose_root.is_dir():
        summary["dose_v3_reference_k7"] = {}
        ref = summary["dose_v3_reference_k7"]
        assert isinstance(ref, dict)
        for key in ("qpth", "ssa"):
            runs = load_seeds_dose(dose_root, K_FIXED, key, n_seeds)
            vals = _arr(runs, "test_mean_regret")
            if vals.size == 0:
                continue
            ref[key] = {
                "n_seeds": int(vals.size),
                "median_regret": float(np.median(vals)),
                "mean_regret": float(np.mean(vals)),
                "q1_regret": float(np.percentile(vals, 25)),
                "q3_regret": float(np.percentile(vals, 75)),
            }
    p = out / "summary_v4_n_scaling.json"
    p.write_text(json.dumps(summary, indent=2))
    print(f"  saved: {p.name}")


def render_v4_bundle(
    out: Path,
    v4_root: Path,
    *,
    dose_root: Optional[Path] = None,
    n_seeds: int = 50,
    ns: Tuple[int, ...] = DEFAULT_NS,
    n_highlight: int = 160,
    n_compare_lo: int = 40,
    n_compare_hi: int = 160,
) -> None:
    """Write fig1–fig5 PNGs + summary under ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    dr = dose_root if (dose_root is not None and dose_root.is_dir()) else None
    print(f"[v4_hard] v4_root={v4_root}  dose_ref={dr}  ->  {out}")
    fig1_sample_scaling(v4_root, out / "fig1_dose_response.png", n_seeds, ns, dr)
    fig2_learning_curves(
        v4_root, out / "fig2_learning_curves.png", n_seeds, n_compare_lo, n_compare_hi, dr,
    )
    fig3_compute_profile(
        v4_root, out / "fig3_compute_profile.png", n_seeds, n_compare_lo, n_compare_hi, dr,
    )
    fig4_paired_k7_n120(v4_root, out / "fig4_paired_k7.png", n_seeds, n_highlight, dr)
    fig5_regret_boxplot_n_compare(
        v4_root, out / "fig5_regret_boxplot.png", n_seeds, n_compare_lo, n_compare_hi, dr,
    )
    write_summary_v4(v4_root, out, n_seeds, ns, dr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v4_root", type=Path, default=Path("mdp_cw/results_n120"))
    ap.add_argument("--dose_root", type=Path, default=Path("mdp_cw/results_n40"),
                      help="v3 dose runs for qpth/ssa baselines at k=7.")
    ap.add_argument("--out_dir", type=Path, default=Path("mdp_cw/best_results_final/v4_hard_k7"))
    ap.add_argument("--n_seeds", type=int, default=50)
    ap.add_argument(
        "--ns",
        type=str,
        default="20,40,80,120,160",
        help="Comma-separated training sample sizes for fig1 / summary (must match run dirs nN/).",
    )
    args = ap.parse_args()
    ns_tuple = tuple(int(x.strip()) for x in args.ns.split(",") if x.strip())
    if not ns_tuple:
        raise SystemExit("empty --ns")
    n_hi = ns_tuple[-1]
    n_lo = ns_tuple[1] if len(ns_tuple) >= 2 else ns_tuple[0]
    render_v4_bundle(
        args.out_dir,
        args.v4_root,
        dose_root=args.dose_root,
        n_seeds=args.n_seeds,
        ns=ns_tuple,
        n_highlight=n_hi,
        n_compare_lo=n_lo,
        n_compare_hi=n_hi,
    )


if __name__ == "__main__":
    main()
