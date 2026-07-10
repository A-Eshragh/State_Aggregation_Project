"""
Reviewer-oriented figures from saved dose-response runs (no retraining required
except for the skew-scaled loss ablation).

Produces:
  * Regime-stratified median regret (TS vs DFL) at a chosen k.
  * Ambient slip “risk mass” on ICE test samples (TS vs DFL).
  * Mechanism: greedy policy grids on one ICE test instance (TS vs DFL).

Example::

    python -m mdp_cw.generate_reviewer_figures \\
        --dose_root mdp_cw/results_n40 --k 7 --seed_demo 0 \\
        --out_dir mdp_cw/report_results/reviewer_figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from mdp_cw.cliff_mdp import CLIFF_STATES, GOAL_STATE, N_COLS, N_ROWS, START_STATE
from mdp_cw.cliff_run_utils import (
    read_run_config,
    regime_names_for_set,
    test_regime_index_order,
)
from mdp_cw.load_run_for_eval import greedy_policy_from_occupancy, load_stack_for_eval
from mdp_cw.train_cliff import _build_gamma, _solve_with_algo

# Publication-style defaults (aligned with plot_publication_cw)
_COL_TS = "#E41A1C"
_COL_DFL = "#377EB8"


def _apply_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.55,
            "axes.facecolor": "#fafafa",
            "figure.facecolor": "white",
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
        }
    )


def _pretty_regime_label(name: str) -> str:
    return name.replace("_", " ").title()


def _results_dict(run_dir: Path) -> dict:
    return json.loads((run_dir / "results.json").read_text())


def regime_regret_vector(run_dir: Path, regime_names: list[str]) -> dict[str, float]:
    """Mean test regret conditional on latent regime (one scalar per regime)."""
    run_dir = Path(run_dir)
    rj = _results_dict(run_dir)
    if "test_mean_regret_by_regime" in rj:
        raw = rj["test_mean_regret_by_regime"]
        return {
            name: float(raw[name]) if name in raw else float("nan")
            for name in regime_names
        }
    regret = np.asarray(rj["test_regret"], dtype=float)
    if "test_regime_ids" in rj and len(rj["test_regime_ids"]) == len(regret):
        z = np.asarray(rj["test_regime_ids"], dtype=int)
    else:
        z = test_regime_index_order(run_dir)
    names_idx = regime_names_for_set(str(read_run_config(run_dir).get("regime_set", "v3")))
    out: dict[str, float] = {}
    for i, name in enumerate(names_idx):
        m = z == i
        out[name] = float(regret[m].mean()) if np.any(m) else float("nan")
    # Fill missing keys from caller's regime_names with nan
    for name in regime_names:
        out.setdefault(name, float("nan"))
    return {name: out.get(name, float("nan")) for name in regime_names}


def plot_regime_stratified_bar(
    root: Path,
    k: int,
    n_seeds: int,
    out_path: Path,
):
    _apply_style()
    root = Path(root)
    ts_mat = []
    dfl_mat = []
    names: list[str] | None = None
    for s in range(n_seeds):
        ts_p = root / f"k{k}" / f"ts_s{s}" / "results.json"
        dfl_p = root / f"k{k}" / f"dfl_s{s}" / "results.json"
        if not ts_p.exists() or not dfl_p.exists():
            continue
        if names is None:
            rc = read_run_config(ts_p.parent)
            names = regime_names_for_set(str(rc.get("regime_set", "v3")))
        ts_mat.append([regime_regret_vector(ts_p.parent, names)[nm] for nm in names])
        dfl_mat.append([regime_regret_vector(dfl_p.parent, names)[nm] for nm in names])
    if not ts_mat:
        raise RuntimeError(f"No paired ts/dfl runs under {root}/k{k}/")
    assert names is not None
    labels = [_pretty_regime_label(n) for n in names]

    ts_arr = np.asarray(ts_mat, dtype=float)
    dfl_arr = np.asarray(dfl_mat, dtype=float)
    ts_med = np.nanmedian(ts_arr, axis=0)
    dfl_med = np.nanmedian(dfl_arr, axis=0)

    x = np.arange(len(names))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 4.35))
    ax.bar(
        x - w / 2, ts_med, width=w, label="Two-stage (TS)", color=_COL_TS,
        edgecolor="#8b0000", linewidth=0.85, zorder=3,
    )
    ax.bar(
        x + w / 2, dfl_med, width=w, label="DFL", color=_COL_DFL,
        edgecolor="#1a4480", linewidth=0.85, zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Test latent regime (ground truth)")
    ax.set_ylabel("Median regret across seeds")
    ax.set_title(
        f"Regret by regime at skew $k$ = {k}\n"
        r"(each bar: median over seeds of mean test regret in that regime)",
        pad=10,
    )
    ymax = float(np.nanmax(np.r_[ts_med, dfl_med])) * 1.12
    ax.set_ylim(0.0, max(ymax, 1e-6))
    ax.legend(frameon=True, fancybox=False, edgecolor="#cccccc", loc="upper right")
    ax.grid(axis="y", which="major")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _ice_ambient_risk_mass(slip_pred: np.ndarray) -> float:
    """Scalar summary: mass on non-intended slip outcomes, averaged over actions."""
    # slip_pred: (A, K) ambient, K=5 categories; index 0 = intended
    return float((1.0 - slip_pred[:, 0]).mean())


def plot_slip_bias_ice_only(
    root: Path,
    k: int,
    n_seeds: int,
    out_path: Path,
):
    _apply_style()
    root = Path(root)
    ts_risks, dfl_risks = [], []
    for s in range(n_seeds):
        for label, bucket in (("ts", ts_risks), ("dfl", dfl_risks)):
            p = root / f"k{k}" / f"{label}_s{s}" / "results.json"
            if not p.exists():
                continue
            rj = _results_dict(p.parent)
            if "test_ambient_slip_pred" not in rj or "test_regime_ids" not in rj:
                continue
            z = np.asarray(rj["test_regime_ids"], dtype=int)
            amb = np.asarray(rj["test_ambient_slip_pred"], dtype=float)
            ice_mask = z >= 1
            if not np.any(ice_mask):
                continue
            for row in amb[ice_mask]:
                bucket.append(_ice_ambient_risk_mass(row))

    fig, ax = plt.subplots(figsize=(5.4, 4.1))
    if ts_risks or dfl_risks:
        bp = ax.boxplot(
            [ts_risks if ts_risks else [np.nan], dfl_risks if dfl_risks else [np.nan]],
            tick_labels=["Two-stage", "DFL"],
            patch_artist=True,
            widths=0.55,
            medianprops=dict(color="#222222", linewidth=1.6),
            whiskerprops=dict(color="#555555", linewidth=1.0),
            capprops=dict(color="#555555", linewidth=1.0),
            flierprops=dict(marker="o", markersize=4, markerfacecolor="#666666", alpha=0.65),
        )
        colors = [_COL_TS, _COL_DFL]
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.35)
            patch.set_edgecolor("#333333")
            patch.set_linewidth(1.0)
    else:
        ax.text(
            0.5,
            0.5,
            "Re-train with current train_cliff.py to log\n"
            "test_ambient_slip_pred on test points.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="#444444",
        )
    ax.set_ylabel('Ambient slip mass not on "intended" outcome\n(mean over actions; ICE test draws only)')
    ax.set_title(f"Predicted slip caution vs method ($k$ = {k})", pad=10)
    ax.grid(axis="y", which="major")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# Greedy action 0=UP,1=RIGHT,2=DOWN,3=LEFT — distinct arrow colors for readability
_ACTION_QUIVER_COLORS = ("#0d47a1", "#c62828", "#2e7d32", "#6a1b9a")


def _draw_policy_panel(ax, actions: np.ndarray, title: str, *, show_action_legend: bool = False):
    """4×12 grid: greedy LP policy as **quiver** (centered arrows, color by action)."""
    cliff = set(CLIFF_STATES)
    vecs = {0: (0.0, -1.0), 1: (1.0, 0.0), 2: (0.0, 1.0), 3: (-1.0, 0.0)}

    for s in range(N_ROWS * N_COLS):
        row, col = divmod(s, N_COLS)
        if s in cliff:
            fc, ec = "#e8c4c4", "#a07070"
        elif s == START_STATE:
            fc, ec = "#c8e6c9", "#558b2f"
        elif s == GOAL_STATE:
            fc, ec = "#bbdefb", "#1565c0"
        else:
            fc, ec = "#ffffff", "#cfcfcf"
        ax.add_patch(
            Rectangle(
                (col - 0.5, row - 0.5), 1, 1,
                facecolor=fc, edgecolor=ec, linewidth=1.05, zorder=1,
            )
        )

    cols = np.arange(N_COLS, dtype=float)
    rows = np.arange(N_ROWS, dtype=float)
    X, Y = np.meshgrid(cols, rows)
    U = np.zeros_like(X, dtype=float)
    V = np.zeros_like(Y, dtype=float)
    C_list: list[str] = []
    for s in range(N_ROWS * N_COLS):
        row, col = divmod(s, N_COLS)
        a = int(actions[s])
        ux, uy = vecs[a]
        U[row, col] = ux
        V[row, col] = uy
        C_list.append(_ACTION_QUIVER_COLORS[a])

    # Arrow length ~0.36 of a cell for unit direction; thick shaft + visible heads
    ax.quiver(
        X,
        Y,
        U,
        V,
        color=C_list,
        angles="xy",
        scale_units="xy",
        scale=1.0 / 0.36,
        width=0.022,
        headwidth=4.2,
        headlength=4.8,
        headaxislength=4.0,
        minshaft=2.0,
        pivot="mid",
        zorder=4,
        clip_on=True,
    )

    ax.set_xlim(-0.52, N_COLS - 0.48)
    ax.set_ylim(N_ROWS - 0.48, -0.52)
    ax.set_xticks(np.arange(N_COLS))
    ax.set_xticklabels([str(c) for c in range(N_COLS)], fontsize=7, color="#666666")
    ax.set_yticks(np.arange(N_ROWS))
    ax.set_yticklabels([str(r) for r in range(N_ROWS)], fontsize=7, color="#666666")
    ax.tick_params(length=0)
    ax.set_xlabel("Column", fontsize=8, color="#555555")
    ax.set_ylabel("Row", fontsize=8, color="#555555")
    ax.set_title(title, pad=8, fontweight="medium")
    ax.set_aspect("equal")

    if show_action_legend:
        leg = [
            Line2D([0], [0], color=_ACTION_QUIVER_COLORS[0], marker="^", linestyle="None", markersize=9, label="Up"),
            Line2D([0], [0], color=_ACTION_QUIVER_COLORS[1], marker=">", linestyle="None", markersize=9, label="Right"),
            Line2D([0], [0], color=_ACTION_QUIVER_COLORS[2], marker="v", linestyle="None", markersize=9, label="Down"),
            Line2D([0], [0], color=_ACTION_QUIVER_COLORS[3], marker="<", linestyle="None", markersize=9, label="Left"),
        ]
        ax.legend(
            handles=leg, loc="upper left", bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0, frameon=True, fancybox=False, edgecolor="#cccccc",
            title="Action", title_fontsize=8, fontsize=8,
        )


def _test_z_array(run_dir: Path) -> np.ndarray:
    rj = _results_dict(run_dir)
    regret = np.asarray(rj["test_regret"], dtype=float)
    if "test_regime_ids" in rj and len(rj["test_regime_ids"]) == len(regret):
        return np.asarray(rj["test_regime_ids"], dtype=np.int64)
    return test_regime_index_order(run_dir)


def scan_best_ice_cases(
    root: Path,
    k: int,
    n_seeds: int,
    *,
    top: int = 15,
) -> list[tuple[float, int, int, int, float, float]]:
    """Return sorted rows: (TS−DFL regret, seed, flat_test_idx, z_id, ts_regret, dfl_regret).

    Only indices with latent ICE (z ≥ 1). Larger delta = stronger DFL win on that draw.
    """
    root = Path(root)
    rows: list[tuple[float, int, int, int, float, float]] = []
    for s in range(n_seeds):
        ts_dir = root / f"k{k}" / f"ts_s{s}"
        dfl_dir = root / f"k{k}" / f"dfl_s{s}"
        if not (ts_dir / "results.json").exists() or not (dfl_dir / "results.json").exists():
            continue
        ts_r = np.asarray(_results_dict(ts_dir)["test_regret"], dtype=float)
        dfl_r = np.asarray(_results_dict(dfl_dir)["test_regret"], dtype=float)
        if ts_r.shape != dfl_r.shape:
            continue
        z = _test_z_array(ts_dir)
        if len(z) != len(ts_r):
            continue
        ice = z >= 1
        if not np.any(ice):
            continue
        delta = ts_r - dfl_r
        ice_idx = np.where(ice)[0]
        j = int(ice_idx[int(np.argmax(delta[ice_idx]))])
        rows.append((float(delta[j]), s, j, int(z[j]), float(ts_r[j]), float(dfl_r[j])))
    rows.sort(key=lambda t: t[0], reverse=True)
    return rows[:top]


def pick_flat_test_example(test_loader, flat_idx: int):
    """Return (w[1], z, P_true, r_true, z_int) for the flat test index (DataLoader order)."""
    device = torch.device("cpu")
    n = 0
    for batch in test_loader:
        w, z, slip_compact, P_true, r_true, y_true, v_true = batch
        for j in range(z.shape[0]):
            if n == flat_idx:
                j = int(j)
                return (
                    w[j : j + 1].to(device),
                    z[j : j + 1].to(device),
                    P_true[j : j + 1].to(device),
                    r_true[j : j + 1].to(device),
                    int(z[j].item()),
                )
            n += 1
    return None


def pick_first_ice_example(test_loader):
    device = torch.device("cpu")
    for batch in test_loader:
        w, z, slip_compact, P_true, r_true, y_true, v_true = batch
        for j in range(z.shape[0]):
            if int(z[j].item()) >= 1:
                idx = int(j)
                return (
                    w[idx : idx + 1].to(device),
                    z[idx : idx + 1].to(device),
                    P_true[idx : idx + 1].to(device),
                    r_true[idx : idx + 1].to(device),
                    int(z[idx].item()),
                )
    return None


def plot_mechanism_grids(
    run_dir_ts: Path,
    run_dir_dfl: Path,
    out_path: Path,
    *,
    flat_test_index: int | None = None,
):
    _apply_style()
    run_dir_ts, run_dir_dfl = Path(run_dir_ts), Path(run_dir_dfl)
    bundle_ts = load_stack_for_eval(run_dir_ts)
    bundle_dfl = load_stack_for_eval(run_dir_dfl)
    args_ts, model_ts, _, tl_ts, _, meta_ts, lp, qp, tau = bundle_ts
    args_dfl, model_dfl, _, _, _, _, _, _, _ = bundle_dfl

    if flat_test_index is not None:
        ex = pick_flat_test_example(tl_ts, flat_test_index)
        if ex is None:
            raise RuntimeError(f"flat_test_index={flat_test_index} out of range for test_loader")
    else:
        ex = pick_first_ice_example(tl_ts)
    if ex is None:
        raise RuntimeError("No ICE regime in test_loader for mechanism plot.")
    w1, _, _, _, zid = ex

    def infer_actions(model, args):
        model.eval()
        with torch.no_grad():
            P_pred, r_pred, _ = model(w1)
            gamma = _build_gamma(
                1, P_pred.shape[1], w1.device, P_pred.dtype,
                start_state=START_STATE,
            )
            y_star, _ = _solve_with_algo(
                args, P_pred, r_pred, lp, qp, agg=None, sa_tau=tau,
            )
        return greedy_policy_from_occupancy(y_star[0].detach()).cpu().numpy()

    a_ts = infer_actions(model_ts, args_ts)
    a_dfl = infer_actions(model_dfl, args_dfl)
    names = regime_names_for_set(str(meta_ts.get("regime_set", "v3")))
    zname = _pretty_regime_label(names[zid] if zid < len(names) else str(zid))

    fig, axes = plt.subplots(
        1, 2, figsize=(12.4, 4.75), sharey=True,
        gridspec_kw={"wspace": 0.38},
    )
    _draw_policy_panel(axes[0], a_ts, f"Two-stage — latent: {zname}", show_action_legend=False)
    _draw_policy_panel(axes[1], a_dfl, "DFL — identical weather draw", show_action_legend=True)

    leg = [
        Patch(facecolor="#c8e6c9", edgecolor="#558b2f", linewidth=1, label="Start"),
        Patch(facecolor="#bbdefb", edgecolor="#1565c0", linewidth=1, label="Goal"),
        Patch(facecolor="#e8c4c4", edgecolor="#a07070", linewidth=1, label="Cliff"),
    ]
    fig.legend(
        handles=leg, loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, -0.02), fontsize=9,
    )
    sub = f" (test index {flat_test_index})" if flat_test_index is not None else ""
    fig.suptitle(
        f"Greedy policy implied by predicted MDP — same test draw{sub}",
        fontsize=11, fontweight="medium", y=1.02,
    )
    fig.subplots_adjust(bottom=0.2, top=0.86)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dose_root", type=Path, default=Path("mdp_cw/results_n40"))
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--n_seeds", type=int, default=50)
    ap.add_argument("--seed_demo", type=int, default=0, help="Paired seed for mechanism figure.")
    ap.add_argument(
        "--print_best_ice",
        action="store_true",
        help="Print best ICE test draws (max TS−DFL regret) and exit.",
    )
    ap.add_argument(
        "--mechanism_auto_best",
        action="store_true",
        help="Mechanism figure: use seed + flat index with largest TS−DFL regret among ICE tests.",
    )
    ap.add_argument(
        "--mechanism_flat_index",
        type=int,
        default=None,
        help="0-based flat test index (DataLoader order). Overrides first-ICE default.",
    )
    ap.add_argument("--out_dir", type=Path, default=Path("mdp_cw/report_results/reviewer_figures"))
    args = ap.parse_args()

    if args.print_best_ice:
        rows = scan_best_ice_cases(args.dose_root, args.k, args.n_seeds, top=20)
        names = regime_names_for_set("v3")
        print(
            f"Top ICE test draws by (TS regret − DFL regret) at k={args.k} "
            f"(paired seeds, n_seeds≤{args.n_seeds}):\n"
            "  rank  Δ_regret  seed  test_idx  z  regime              TS_reg   DFL_reg"
        )
        for rank, row in enumerate(rows, start=1):
            d, s, idx, zid, ts_r, dfl_r = row
            rname = names[zid] if 0 <= zid < len(names) else "?"
            print(f"  {rank:2d}   {d:+.5f}   {s:3d}   {idx:2d}      {zid}  {rname:16s}  {ts_r:.5f}  {dfl_r:.5f}")
        if not rows:
            print("(no paired runs found)")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_regime_stratified_bar(
        args.dose_root, args.k, args.n_seeds,
        args.out_dir / f"fig_regime_stratified_k{args.k}.png",
    )
    plot_slip_bias_ice_only(
        args.dose_root, args.k, args.n_seeds,
        args.out_dir / f"fig_slip_bias_ice_k{args.k}.png",
    )

    mech_seed = args.seed_demo
    mech_flat: int | None = args.mechanism_flat_index
    if args.mechanism_auto_best and args.mechanism_flat_index is None:
        rows = scan_best_ice_cases(args.dose_root, args.k, args.n_seeds, top=1)
        if rows:
            _, mech_seed, mech_flat, _, _, _ = rows[0]

    ts_demo = args.dose_root / f"k{args.k}" / f"ts_s{mech_seed}"
    dfl_demo = args.dose_root / f"k{args.k}" / f"dfl_s{mech_seed}"
    if (ts_demo / "results.json").exists() and (dfl_demo / "results.json").exists():
        tag = f"k{args.k}_s{mech_seed}_i{mech_flat}" if mech_flat is not None else f"k{args.k}_s{mech_seed}"
        plot_mechanism_grids(
            ts_demo, dfl_demo,
            args.out_dir / f"fig_mechanism_policies_{tag}.png",
            flat_test_index=mech_flat,
        )
    print(f"wrote figures under {args.out_dir}")


if __name__ == "__main__":
    main()