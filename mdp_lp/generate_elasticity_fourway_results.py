#!/usr/bin/env python3
"""
Generate a "fourway results" directory that consolidates plots/results for:

1) DFL (LP) no-SSA              : dfl/epochs_{E}
2) DFL (LP) + SSA               : dfl/n_agg_{M}/epochs_{E}
3) DFL (qpth QP) + SSA          : dfl_qpth/n_agg_{M}/qp_reg_{R}/epochs_{E}
4) Two-stage (LP eval)          : twostage/epochs_{E}

The goal is to be *efficient*: copy existing per-run PNGs where possible and only
generate lightweight overlays/combined plots (reward/loss overlays + 4-way test boxplot).

Usage (from repo root):
  python -m mdp_lp.generate_elasticity_fourway_results --epochs 100 --n_agg 20 --qp_reg 0.5

Output:
  mdp_lp/fourway_results/exp_{a,b,c}/{high_margin,low_margin}/[beta2_{...}/]...
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

Key = Tuple[str, str, Optional[float]]  # (experiment_type, cost_case, beta_2)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _safe_get_series(d: Dict[str, Any], k: str) -> Optional[np.ndarray]:
    v = d.get(k)
    if v is None:
        return None
    if not isinstance(v, list) or len(v) == 0:
        return None
    return np.asarray(v, dtype=np.float64)


def _expected_avg_prices(d: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], str]:
    def _f(x: Any) -> Optional[float]:
        try:
            if x is None:
                return None
            return float(x)
        except Exception:
            return None

    exp = d.get("experiment_type")
    if exp not in ("a", "b", "c"):
        return None, None, ""

    realized = _f(d.get("test_avg_price")) or _f(d.get("train_avg_price"))
    realized_related = _f(d.get("test_avg_related_price")) or _f(d.get("train_avg_related_price"))
    if realized is not None:
        return realized, realized_related, "avg"

    if exp == "a":
        p = _f(d.get("p_constant"))
        return p, None, "E"

    p_mean = _f(d.get("p_lognormal_mean"))
    p_std = _f(d.get("p_lognormal_std"))
    if p_mean is None or p_std is None:
        return None, None, ""
    p = float(p_mean * np.exp(0.5 * (p_std**2)))

    pr = None
    if exp == "c":
        pr_mean = _f(d.get("p_related_lognormal_mean"))
        pr_std = _f(d.get("p_related_lognormal_std"))
        if pr_mean is not None and pr_std is not None:
            pr = float(pr_mean * np.exp(0.5 * (pr_std**2)))
    return p, pr, "E"


def _param_text(d: Dict[str, Any]) -> str:
    def fnum(x: Any, nd: int = 3) -> str:
        try:
            if x is None:
                return "None"
            if isinstance(x, bool):
                return str(x)
            if isinstance(x, (int, np.integer)):
                return str(int(x))
            if isinstance(x, (float, np.floating)):
                return f"{float(x):.{nd}f}"
            if isinstance(x, str) and x.replace(".", "", 1).replace("-", "", 1).isdigit():
                return f"{float(x):.{nd}f}"
            return str(x)
        except Exception:
            return str(x)

    avg_p, avg_pr, avg_kind = _expected_avg_prices(d)
    if avg_p is not None:
        avg_p_txt = f"{avg_kind}[p]={fnum(avg_p,2)}"
        if avg_pr is not None:
            avg_p_txt += f"  {avg_kind}[p_rel]={fnum(avg_pr,2)}"
    else:
        avg_p_txt = "avg price: N/A"

    lines = [
        f"beta0={fnum(d.get('beta_0'),2)}  beta1={fnum(d.get('beta_1'),2)}  beta2={fnum(d.get('beta_2'),2)}  sigma={fnum(d.get('sigma_epsilon'),2)}",
        f"foc={fnum(d.get('foc'),2)}  voc={fnum(d.get('voc'),2)}  hc={fnum(d.get('hc'),2)}  bc={fnum(d.get('bc'),2)}",
        f"b_max={fnum(d.get('b_max'),0)}  s_max={fnum(d.get('s_max'),0)}  q_max={fnum(d.get('q_max'),0)}  d_max={fnum(d.get('d_max'),0)}  beta={fnum(d.get('beta'),3)}",
        avg_p_txt,
    ]
    return "\n".join(lines)


def _stamp_params_fig(fig: plt.Figure, text: str) -> None:
    fig.text(
        0.01,
        0.01,
        text,
        ha="left",
        va="bottom",
        fontsize=8.4,
        family="monospace",
        color="#222222",
        bbox=dict(
            boxstyle="round,pad=0.30",
            facecolor="white",
            edgecolor="#bdbdbd",
            linewidth=0.8,
            alpha=0.92,
        ),
    )


def _copy_pngs(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    if not src_dir.exists():
        return
    for p in src_dir.glob("*.png"):
        try:
            shutil.copy(p, dst_dir / p.name)
        except Exception:
            # best-effort; keep going
            pass


def _is_all_zero(x: Optional[np.ndarray], *, atol: float = 1e-12) -> bool:
    if x is None:
        return True
    if x.size == 0:
        return True
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return True
    return bool(np.all(np.abs(x) <= atol))


def _ensure_dfl_loss_in_results(rd: Dict[str, Any]) -> None:
    """
    Some older runs produced stale/placeholder DFL loss series (all zeros) even when
    reward/MSE were logged correctly (or were backfilled later). For DFL variants we can
    reconstruct the intended training loss per epoch:

      dfl_loss = -reward + loss_weight * MSE

    This mutates `rd` in-memory only.
    """
    rw = _safe_get_series(rd, "train_reward_per_epoch")
    ms = _safe_get_series(rd, "train_mse_per_epoch")
    dl = _safe_get_series(rd, "train_dfl_loss_per_epoch")
    if rw is None or ms is None:
        return

    if (dl is None) or _is_all_zero(dl):
        w = rd.get("loss_weight", 1.0)
        try:
            w = float(w)
        except Exception:
            w = 1.0
        rd["train_dfl_loss_per_epoch"] = (-rw + w * ms).tolist()


def _write_variant_curve_plots(dst_dir: Path, vname: str, rd: Dict[str, Any]) -> None:
    """
    Overwrite a few key per-run plots in the *output* directory using `results.json`.
    This avoids copying stale PNGs from older runs (e.g., curves saved as all zeros).
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Train reward
    rw = _safe_get_series(rd, "train_reward_per_epoch")
    if rw is not None:
        fig, ax = plt.subplots(figsize=(7.0, 2.6))
        x = np.arange(1, len(rw) + 1)
        ax.plot(x, rw, linewidth=2.2, color="#2E59A8")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Train objective value")
        ax.set_title("Train objective per epoch")
        fig.savefig(dst_dir / "train_reward_per_epoch.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # MSE (DFL prediction loss)
    ms = _safe_get_series(rd, "train_mse_per_epoch")
    if ms is not None:
        fig, ax = plt.subplots(figsize=(7.0, 2.6))
        x = np.arange(1, len(ms) + 1)
        ax.plot(x, ms, linewidth=2.2, color="#1B9E77")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("MSE (Prediction Loss)")
        ax.set_title("DFL - Prediction Loss (MSE) per Epoch")
        fig.savefig(dst_dir / "train_mse_per_epoch.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # DFL full loss
    dl = _safe_get_series(rd, "train_dfl_loss_per_epoch")
    if dl is not None:
        fig, ax = plt.subplots(figsize=(7.0, 2.6))
        x = np.arange(1, len(dl) + 1)
        ax.plot(x, dl, linewidth=2.2, color="#7B52AB")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("DFL Loss  (-v* + w·MSE)")
        ax.set_title("DFL - Full Loss per Epoch")
        fig.savefig(dst_dir / "train_dfl_loss_per_epoch.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def _overlay_rewards(series: Dict[str, np.ndarray], out_path: Path, params_text: str) -> None:
    if not series:
        return
    fig, ax = plt.subplots(figsize=(8.6, 3.0))
    colors = {
        "dfl_no_ssa": "#2E59A8",
        "dfl_ssa": "#1B9E77",
        "dfl_qpth_ssa": "#7B52AB",
        "twostage": "#C84B31",
    }
    labels = {
        "dfl_no_ssa": "DFL (LP) no-SSA",
        "dfl_ssa": "DFL (LP) + SSA",
        "dfl_qpth_ssa": "DFL (qpth) + SSA",
        "twostage": "Two-stage",
    }

    for key, y in series.items():
        x = np.arange(1, len(y) + 1)
        ax.plot(x, y, label=labels.get(key, key), linewidth=2.1, color=colors.get(key, None))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train objective value")
    ax.set_title("Train reward per epoch (overlay)")
    ax.legend(frameon=False, loc="best", ncol=2)
    fig.subplots_adjust(bottom=0.30)
    _stamp_params_fig(fig, params_text)
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def _overlay_losses(
    dfl_losses: Dict[str, np.ndarray],
    mse_losses: Dict[str, np.ndarray],
    ts_loss: Optional[np.ndarray],
    out_path: Path,
    params_text: str,
) -> None:
    # Only plot if there is at least one curve
    if not dfl_losses and not mse_losses and ts_loss is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2))
    ax0, ax1 = axes

    # Left: DFL objective (full loss) overlay
    colors_dfl = {
        "dfl_no_ssa": "#2E59A8",
        "dfl_ssa": "#1B9E77",
        "dfl_qpth_ssa": "#7B52AB",
    }
    labels_dfl = {
        "dfl_no_ssa": "DFL (LP) no-SSA",
        "dfl_ssa": "DFL (LP) + SSA",
        "dfl_qpth_ssa": "DFL (qpth) + SSA",
    }
    for key, y in dfl_losses.items():
        x = np.arange(1, len(y) + 1)
        ax0.plot(x, y, label=labels_dfl.get(key, key), linewidth=2.0, color=colors_dfl.get(key, None))
    ax0.set_title("DFL loss per epoch")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("DFL loss  (-v* + w·MSE)")
    ax0.legend(frameon=False, loc="best")

    # Right: prediction losses (MSE + two-stage)
    for key, y in mse_losses.items():
        x = np.arange(1, len(y) + 1)
        ax1.plot(x, y, label=f"{labels_dfl.get(key, key)} MSE", linewidth=1.9, color=colors_dfl.get(key, None), alpha=0.95)
    if ts_loss is not None:
        x = np.arange(1, len(ts_loss) + 1)
        ax1.plot(x, ts_loss, label="Two-stage pred loss", linewidth=2.0, color="#C84B31")
    ax1.set_title("Prediction loss per epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(frameon=False, loc="best")

    fig.subplots_adjust(bottom=0.32, wspace=0.28)
    _stamp_params_fig(fig, params_text)
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def _fourway_test_boxplot(values: Dict[str, np.ndarray], out_path: Path, params_text: str) -> None:
    if not values:
        return
    order = ["dfl_no_ssa", "dfl_ssa", "dfl_qpth_ssa", "twostage"]
    labels = ["DFL (LP)\nno-SSA", "DFL (LP)\n+SSA", "DFL (qpth)\n+SSA", "Two-stage"]
    data = []
    used_labels = []
    for k, lab in zip(order, labels):
        if k in values and values[k].size > 0:
            data.append(values[k])
            used_labels.append(lab)

    if len(data) == 0:
        return

    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)
    palette = ["#2E59A8", "#1B9E77", "#7B52AB", "#C84B31"]
    for patch, c in zip(bp["boxes"], palette[: len(data)]):
        patch.set_facecolor(c)
        patch.set_alpha(0.25)
        patch.set_edgecolor(c)
        patch.set_linewidth(1.5)
    for med in bp["medians"]:
        med.set_color("#111111")
        med.set_linewidth(1.8)

    ax.set_xticklabels(used_labels)
    ax.set_ylabel("Test objective value")
    ax.set_title("Test objective distribution (4-way)")
    fig.subplots_adjust(bottom=0.30)
    _stamp_params_fig(fig, params_text)
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def _variant_paths(root: Path, *, exp: str, cost: str, beta2: Optional[float], epochs: int, n_agg: int, qp_reg: float) -> Dict[str, Path]:
    base = root / f"exp_{exp}" / cost
    if exp == "c":
        assert beta2 is not None
        base_dfl = base / "dfl" / f"beta2_{beta2}"
        base_qp = base / "dfl_qpth" / f"beta2_{beta2}"
        base_ts = base / "twostage" / f"beta2_{beta2}"
    else:
        base_dfl = base / "dfl"
        base_qp = base / "dfl_qpth"
        base_ts = base / "twostage"

    return {
        "dfl_no_ssa": base_dfl / f"epochs_{epochs}",
        "dfl_ssa": base_dfl / f"n_agg_{n_agg}" / f"epochs_{epochs}",
        "dfl_qpth_ssa": base_qp / f"n_agg_{n_agg}" / f"qp_reg_{qp_reg}" / f"epochs_{epochs}",
        "twostage": base_ts / f"epochs_{epochs}",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n_agg", type=int, default=20)
    ap.add_argument("--qp_reg", type=float, default=0.5)
    ap.add_argument("--results_root", type=str, default="mdp_lp/results_elasticity")
    ap.add_argument("--out_dir", type=str, default="mdp_lp/fourway_results")
    args = ap.parse_args()

    root = Path(args.results_root)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    settings: list[Key] = [
        ("a", "high_margin", None),
        ("a", "low_margin", None),
        ("b", "high_margin", None),
        ("b", "low_margin", None),
        ("c", "high_margin", 0.09),
        ("c", "high_margin", -0.09),
        ("c", "low_margin", 0.09),
        ("c", "low_margin", -0.09),
    ]

    for exp, cost, beta2 in settings:
        sub = out_root / f"exp_{exp}" / cost
        if beta2 is not None:
            sub = sub / f"beta2_{beta2:+.2f}"
        sub.mkdir(parents=True, exist_ok=True)

        paths = _variant_paths(
            root,
            exp=exp,
            cost=cost,
            beta2=beta2,
            epochs=args.epochs,
            n_agg=args.n_agg,
            qp_reg=args.qp_reg,
        )

        # Copy all per-run PNGs into variant subfolders (fast).
        for vname, src_dir in paths.items():
            _copy_pngs(src_dir, sub / vname)

        # Load results.json for combined plots (best-effort).
        results: Dict[str, Dict[str, Any]] = {}
        for vname, src_dir in paths.items():
            rfile = src_dir / "results.json"
            if rfile.exists():
                try:
                    results[vname] = _read_json(rfile)
                except Exception:
                    pass
            else:
                # allow backups
                for alt in ("results_backup2.json", "results_backup.json"):
                    af = src_dir / alt
                    if af.exists():
                        try:
                            results[vname] = _read_json(af)
                            break
                        except Exception:
                            pass

        # Ensure we have a usable DFL loss series (and overwrite stale per-run plots in output).
        for vname, rd in results.items():
            if vname.startswith("dfl"):
                _ensure_dfl_loss_in_results(rd)
            _write_variant_curve_plots(sub / vname, vname, rd)

        # Choose params stamp from any available results (prefer DFL no-SSA).
        params_src = results.get("dfl_no_ssa") or results.get("dfl_ssa") or results.get("dfl_qpth_ssa") or results.get("twostage")
        params_text = _param_text(params_src) if params_src else ""

        # Combined reward overlay
        reward_series: Dict[str, np.ndarray] = {}
        for vname, rd in results.items():
            s = _safe_get_series(rd, "train_reward_per_epoch")
            if s is not None:
                reward_series[vname] = s
        _overlay_rewards(reward_series, sub / "fourway_train_reward_overlay.png", params_text)

        # Combined losses overlay
        dfl_losses: Dict[str, np.ndarray] = {}
        mse_losses: Dict[str, np.ndarray] = {}
        for vname, rd in results.items():
            if vname.startswith("dfl"):
                dl = _safe_get_series(rd, "train_dfl_loss_per_epoch")
                ms = _safe_get_series(rd, "train_mse_per_epoch")
                if dl is not None:
                    dfl_losses[vname] = dl
                if ms is not None:
                    mse_losses[vname] = ms
        ts_loss = None
        if "twostage" in results:
            ts_loss = _safe_get_series(results["twostage"], "train_twostage_loss_per_epoch")
        _overlay_losses(dfl_losses, mse_losses, ts_loss, sub / "fourway_losses_overlay.png", params_text)

        # 4-way test objective boxplot
        test_vals: Dict[str, np.ndarray] = {}
        for vname, rd in results.items():
            tv = rd.get("test_obj_values")
            if isinstance(tv, list) and len(tv) > 0:
                try:
                    test_vals[vname] = np.asarray(tv, dtype=np.float64)
                except Exception:
                    pass
        _fourway_test_boxplot(test_vals, sub / "fourway_test_obj_values_boxplot.png", params_text)

    print(f"[fourway] Wrote: {out_root}")


if __name__ == "__main__":
    main()

