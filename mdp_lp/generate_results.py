#!/usr/bin/env python3
"""
Consolidated script for generating comparison plots and result reports.

Subcommands:
  boxplot     Side-by-side DFL vs Two-stage test objective boxplot per (exp, cost, [beta2]).
  report      Full comparison report: reward overlay, losses, test boxplot, lambda histograms, INDEX.
  small_test  2-way comparison for a single (root, exp, cost, tag): overlays + lambda side-by-side.
  fourway     4-way (dfl no-SSA, dfl SSA, dfl_qpth SSA, twostage): copy PNGs + overlays + 4-way boxplot.

All paths default under mdp_lp/ (e.g. mdp_lp/results_elasticity, mdp_lp/results_elasticity_comparisons).

Usage:
  python -m mdp_lp.generate_results boxplot [base_dir] [epochs]
  python -m mdp_lp.generate_results report --epochs 100 --input_dir mdp_lp/results_elasticity --output_dir mdp_lp/results_elasticity_comparisons
  python -m mdp_lp.generate_results small_test --root mdp_lp/small_test --tag my_tag --epochs 20
  python -m mdp_lp.generate_results fourway --epochs 100 --n_agg 20 --results_root mdp_lp/results_elasticity --out_dir mdp_lp/fourway_results
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

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

Key = Tuple[str, str, Optional[float]]  # (experiment_type, cost_case, beta_2)


# ---------- Shared helpers ----------

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2))


def _safe_series(d: Dict[str, Any], k: str) -> Optional[np.ndarray]:
    v = d.get(k)
    if v is None or not isinstance(v, list) or len(v) == 0:
        return None
    return np.asarray(v, dtype=np.float64)


def _expected_avg_prices(d: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], str]:
    def _f(x: Any) -> Optional[float]:
        try:
            return float(x) if x is not None else None
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
        return _f(d.get("p_constant")), None, "E"
    p_mean, p_std = _f(d.get("p_lognormal_mean")), _f(d.get("p_lognormal_std"))
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
            if isinstance(x, (int, np.integer)):
                return str(int(x))
            if isinstance(x, (float, np.floating)):
                return f"{float(x):.{nd}f}"
            return str(x)
        except Exception:
            return str(x)
    avg_p, avg_pr, avg_kind = _expected_avg_prices(d)
    parts = []
    if avg_p is not None:
        parts.append(f"E[p]={fnum(avg_p,2)}" + (f" E[p_related]={fnum(avg_pr,2)}" if avg_pr is not None else ""))
    parts.append(f"foc={fnum(d.get('foc'),2)} voc={fnum(d.get('voc'),2)} hc={fnum(d.get('hc'),2)} bc={fnum(d.get('bc'),2)}")
    parts.append(f"b_max={fnum(d.get('b_max'),0)} s_max={fnum(d.get('s_max'),0)} q_max={fnum(d.get('q_max'),0)} d_max={fnum(d.get('d_max'),0)} beta={fnum(d.get('beta'),3)}")
    return "\n".join(parts)


def _stamp_fig(fig: plt.Figure, text: str) -> None:
    if not text:
        return
    fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=8.4, family="monospace",
             color="#222222", bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="#bdbdbd", linewidth=0.8, alpha=0.92))


# ---------- Boxplot (create_all_comparison_plots) ----------

def create_all_comparison_plots(epochs: int = 100, base_dir: Optional[str] = None) -> None:
    """Create side-by-side DFL vs Two-stage boxplot for each (exp, cost, [beta2])."""
    from mdp_lp.plot_elasticity import plot_comparison_boxplot
    base = Path(base_dir or "mdp_lp/results_elasticity")
    beta2_vals = (0.09, -0.09) if str(base) == "results2" else (0.5, -0.5)
    experiments = [
        ("a", "high_margin", None), ("a", "low_margin", None),
        ("b", "high_margin", None), ("b", "low_margin", None),
        ("c", "high_margin", beta2_vals[0]), ("c", "high_margin", beta2_vals[1]),
        ("c", "low_margin", beta2_vals[0]), ("c", "low_margin", beta2_vals[1]),
    ]
    epochs_dir = f"epochs_{epochs}"
    for exp_type, cost_case, beta_2 in experiments:
        if exp_type == "c":
            dfl_dir = base / f"exp_{exp_type}" / cost_case / "dfl" / f"beta2_{beta_2}"
            twostage_dir = base / f"exp_{exp_type}" / cost_case / "twostage" / f"beta2_{beta_2}"
            save_dir = base / f"exp_{exp_type}" / cost_case / f"beta2_{beta_2}"
        else:
            dfl_dir = base / f"exp_{exp_type}" / cost_case / "dfl"
            twostage_dir = base / f"exp_{exp_type}" / cost_case / "twostage"
            save_dir = base / f"exp_{exp_type}" / cost_case
        dfl_file = dfl_dir / epochs_dir / "results.json"
        ts_file = twostage_dir / epochs_dir / "results.json"
        if not dfl_file.exists():
            dfl_file = dfl_dir / "results.json"
        if not ts_file.exists():
            ts_file = twostage_dir / "results.json"
        if not dfl_file.exists() or not ts_file.exists():
            continue
        try:
            dfl_results = _read_json(dfl_file)
            ts_results = _read_json(ts_file)
            dfl_values = np.array(dfl_results["test_obj_values"])
            ts_values = np.array(ts_results["test_obj_values"])
            save_dir.mkdir(parents=True, exist_ok=True)
            plot_comparison_boxplot(dfl_values, ts_values, save_dir)
            label = f"{exp_type}/{cost_case}" + (f"/β₂={beta_2}" if beta_2 is not None else "")
            print(f"Created comparison plot: {label}")
        except Exception as e:
            print(f"Error creating comparison for {exp_type}/{cost_case}: {e}")


# ---------- Report (full comparison report) ----------

def _get_key(d: Dict[str, Any]) -> Key:
    exp, cost = d.get("experiment_type"), d.get("cost_case")
    if exp is None or cost is None:
        raise ValueError("Missing experiment_type/cost_case")
    beta2 = d.get("beta_2") if exp != "a" and exp != "b" else None
    if beta2 is not None:
        beta2 = float(beta2)
    return exp, cost, beta2


def _format_key(key: Key) -> str:
    exp, cost, beta2 = key
    return f"exp_{exp}/{cost}" if beta2 is None else f"exp_{exp}/{cost}/beta2_{beta2:+.2f}"


def _plot_reward_overlay(dfl: Dict, ts: Dict, out_path: Path) -> None:
    r_dfl = _safe_series(dfl, "train_reward_per_epoch")
    r_ts = _safe_series(ts, "train_reward_per_epoch")
    if r_dfl is None or r_ts is None:
        return
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    ax.plot(np.arange(1, len(r_dfl) + 1), r_dfl, label="DFL", linewidth=2.2, color="#2E59A8")
    ax.plot(np.arange(1, len(r_ts) + 1), r_ts, label="Two-stage", linewidth=2.2, color="#C84B31")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Train objective value"); ax.set_title("Train reward per epoch")
    ax.legend(frameon=False, loc="best")
    fig.subplots_adjust(bottom=0.30); _stamp_fig(fig, _param_text(dfl))
    fig.savefig(out_path, dpi=250); plt.close(fig)


def _plot_losses(dfl: Dict, ts: Dict, out_path: Path) -> None:
    dfl_loss = _safe_series(dfl, "train_dfl_loss_per_epoch")
    dfl_mse = _safe_series(dfl, "train_mse_per_epoch")
    ts_pred = _safe_series(ts, "train_twostage_loss_per_epoch")
    if dfl_loss is None and dfl_mse is None and ts_pred is None:
        return
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 7.2), sharex=True)
    if dfl_loss is not None:
        axes[0].plot(np.arange(1, len(dfl_loss) + 1), dfl_loss, color="purple", linewidth=2.0)
    axes[0].set_ylabel("DFL loss"); axes[0].set_title("Loss curves")
    if dfl_mse is not None:
        axes[1].plot(np.arange(1, len(dfl_mse) + 1), dfl_mse, color="green", linewidth=2.0)
    axes[1].set_ylabel("DFL MSE")
    if ts_pred is not None:
        axes[2].plot(np.arange(1, len(ts_pred) + 1), ts_pred, color="orange", linewidth=2.0)
    axes[2].set_ylabel("Two-stage pred loss"); axes[2].set_xlabel("Epoch")
    fig.subplots_adjust(bottom=0.14); _stamp_fig(fig, _param_text(dfl))
    fig.savefig(out_path, dpi=250); plt.close(fig)


def _plot_test_boxplot(dfl: Dict, ts: Dict, out_path: Path) -> None:
    v_dfl, v_ts = dfl.get("test_obj_values"), ts.get("test_obj_values")
    if not isinstance(v_dfl, list) or not isinstance(v_ts, list) or len(v_dfl) == 0 or len(v_ts) == 0:
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.boxplot([v_dfl, v_ts], tick_labels=["DFL", "Two-stage"])
    ax.set_ylabel("Test objective value"); ax.set_title("Test objective distribution")
    fig.subplots_adjust(bottom=0.32); _stamp_fig(fig, _param_text(dfl))
    fig.savefig(out_path, dpi=250); plt.close(fig)


def _copy_lambda_histograms(dfl_results_path: Path, ts_results_path: Path, out_dir: Path) -> None:
    for src, name in [(dfl_results_path.parent / "lambda_histograms.png", "lambda_histograms_dfl.png"),
                       (ts_results_path.parent / "lambda_histograms.png", "lambda_histograms_twostage.png")]:
        if src.exists():
            shutil.copyfile(src, out_dir / name)


def _cmd_report(args: argparse.Namespace) -> None:
    base = Path(args.input_dir)
    out_base = Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    epochs_dir = f"epochs_{args.epochs}"
    dfl_paths = sorted(base.glob(f"**/dfl/**/{epochs_dir}/results.json"))
    ts_paths = sorted(base.glob(f"**/twostage/**/{epochs_dir}/results.json"))
    dfl_map: Dict[Key, Path] = {}
    ts_map: Dict[Key, Path] = {}
    for p in dfl_paths:
        d = _read_json(p)
        if d.get("status") not in (None, "complete"):
            continue
        dfl_map[_get_key(d)] = p
    for p in ts_paths:
        d = _read_json(p)
        if d.get("status") not in (None, "complete"):
            continue
        ts_map[_get_key(d)] = p
    keys = sorted(set(dfl_map.keys()) & set(ts_map.keys()))
    if not keys:
        print("No matching DFL/twostage result pairs found.")
        return
    index_lines = ["## Elasticity DFL vs Two-stage comparisons", "", f"Input: `{base}`", f"Output: `{out_base}`", f"Epochs: `{args.epochs}`", "", "### Configs"]
    for key in keys:
        dfl_file, ts_file = dfl_map[key], ts_map[key]
        dfl, ts = _read_json(dfl_file), _read_json(ts_file)
        exp, cost, beta2 = key
        out_dir = out_base / f"exp_{exp}" / cost if beta2 is None else out_base / f"exp_{exp}" / cost / f"beta2_{beta2:+.2f}"
        out_dir.mkdir(parents=True, exist_ok=True)
        _copy_lambda_histograms(dfl_file, ts_file, out_dir)
        _plot_reward_overlay(dfl, ts, out_dir / "reward_overlay.png")
        _plot_losses(dfl, ts, out_dir / "losses.png")
        _plot_test_boxplot(dfl, ts, out_dir / "test_boxplot.png")
        index_lines.append(f"- `{_format_key(key)}` → `{out_dir.as_posix()}`")
    out_base.joinpath("INDEX.md").write_text("\n".join(index_lines) + "\n")
    print(f"Wrote comparison report to: {out_base}")


# ---------- Small test (tag-based 2-way) ----------

def _param_block(d: Dict[str, Any]) -> str:
    def f(x: Any, nd: int = 3) -> str:
        try:
            if x is None: return "None"
            if isinstance(x, (int, np.integer)): return str(int(x))
            if isinstance(x, (float, np.floating)): return f"{float(x):.{nd}f}"
            return str(x)
        except Exception: return str(x)
    avg_true = None
    if isinstance(d.get("test_lambda_true"), list) and len(d["test_lambda_true"]) > 0:
        avg_true = float(np.mean(np.asarray(d["test_lambda_true"], dtype=np.float64)))
    return "\n".join([
        f"tag={d.get('tag','')}", f"avg demand (mean lambda_true)={f(avg_true,3)}",
        f"p_constant={f(d.get('p_constant'),2)}  beta={f(d.get('beta'),3)}",
        f"foc={f(d.get('foc'),2)} voc={f(d.get('voc'),2)} hc={f(d.get('hc'),2)} bc={f(d.get('bc'),2)}",
        f"s_max(cap)={f(d.get('s_max'),0)}  b_max={f(d.get('b_max'),0)}  q_max={f(d.get('q_max'),0)}  d_max={f(d.get('d_max'),0)}",
    ])


def _paths_small_test(root: Path, exp: str, cost: str, tag: str, epochs: int) -> Tuple[Path, Path]:
    return (root / f"exp_{exp}" / cost / "dfl" / f"tag_{tag}" / f"epochs_{epochs}" / "results.json",
            root / f"exp_{exp}" / cost / "twostage" / f"tag_{tag}" / f"epochs_{epochs}" / "results.json")


def _plot_reward_overlay_stamp(dfl: Dict, ts: Dict, out: Path, stamp: str) -> None:
    y_dfl = _safe_series(dfl, "train_reward_per_epoch")
    y_ts = _safe_series(ts, "train_reward_per_epoch")
    if y_dfl is None and y_ts is None: return
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    if y_dfl is not None: ax.plot(np.arange(1, len(y_dfl) + 1), y_dfl, label="DFL (LP)", linewidth=2.2, color="#2E59A8")
    if y_ts is not None: ax.plot(np.arange(1, len(y_ts) + 1), y_ts, label="Two-stage", linewidth=2.2, color="#C84B31")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Train objective value"); ax.set_title("Train objective per epoch (overlay)")
    ax.legend(frameon=False, loc="best"); fig.subplots_adjust(bottom=0.30); _stamp_fig(fig, stamp)
    fig.savefig(out, dpi=250); plt.close(fig)


def _plot_pred_loss_overlay(dfl: Dict, ts: Dict, out: Path, stamp: str) -> None:
    y_dfl = _safe_series(dfl, "train_mse_per_epoch")
    y_ts = _safe_series(ts, "train_twostage_loss_per_epoch")
    if y_dfl is None and y_ts is None: return
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    if y_dfl is not None: ax.plot(np.arange(1, len(y_dfl) + 1), y_dfl, label="DFL MSE", linewidth=2.2, color="#1B9E77")
    if y_ts is not None: ax.plot(np.arange(1, len(y_ts) + 1), y_ts, label="Two-stage MSE", linewidth=2.2, color="#C84B31")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Prediction loss (MSE)"); ax.set_title("Prediction loss per epoch (overlay)")
    ax.legend(frameon=False, loc="best"); fig.subplots_adjust(bottom=0.30); _stamp_fig(fig, stamp)
    fig.savefig(out, dpi=250); plt.close(fig)


def _copy_and_combine_lambda_histograms(dfl_run_dir: Path, ts_run_dir: Path, out_dir: Path, stamp: str) -> None:
    dfl_img, ts_img = dfl_run_dir / "lambda_histograms.png", ts_run_dir / "lambda_histograms.png"
    if dfl_img.exists(): shutil.copy(dfl_img, out_dir / "lambda_histograms_dfl.png")
    if ts_img.exists(): shutil.copy(ts_img, out_dir / "lambda_histograms_twostage.png")
    if dfl_img.exists() and ts_img.exists():
        try:
            img_ts, img_dfl = plt.imread(str(ts_img)), plt.imread(str(dfl_img))
            fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
            axes[0].imshow(img_ts); axes[0].set_title("Two-stage: lambda_histograms.png"); axes[0].axis("off")
            axes[1].imshow(img_dfl); axes[1].set_title("DFL (LP): lambda_histograms.png"); axes[1].axis("off")
            fig.subplots_adjust(bottom=0.20, wspace=0.02); _stamp_fig(fig, stamp)
            fig.savefig(out_dir / "lambda_histograms_side_by_side.png", dpi=250); plt.close(fig)
        except Exception: pass


def _plot_test_obj_boxplot_stamp(dfl: Dict, ts: Dict, out: Path, stamp: str) -> None:
    dfl_v, ts_v = _safe_series(dfl, "test_obj_values"), _safe_series(ts, "test_obj_values")
    if dfl_v is None or ts_v is None: return
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    bp = ax.boxplot([ts_v, dfl_v], patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], ["#C84B31", "#2E59A8"]):
        patch.set_facecolor(c); patch.set_alpha(0.25); patch.set_edgecolor(c); patch.set_linewidth(1.5)
    for med in bp["medians"]: med.set_color("#111111"); med.set_linewidth(1.8)
    ax.set_xticklabels(["Two-stage", "DFL (LP)"]); ax.set_ylabel("Test objective value"); ax.set_title("Test objective distribution (2-way)")
    fig.subplots_adjust(bottom=0.30); _stamp_fig(fig, stamp); fig.savefig(out, dpi=250); plt.close(fig)


def _cmd_small_test(args: argparse.Namespace) -> None:
    root = Path(args.root)
    dfl_path, ts_path = _paths_small_test(root, args.exp, args.cost, args.tag, args.epochs)
    if not dfl_path.exists(): raise FileNotFoundError(f"Missing DFL results.json: {dfl_path}")
    if not ts_path.exists(): raise FileNotFoundError(f"Missing Two-stage results.json: {ts_path}")
    dfl, ts = _read_json(dfl_path), _read_json(ts_path)
    out = Path(args.output_dir) / f"tag_{args.tag}" / f"epochs_{args.epochs}"
    out.mkdir(parents=True, exist_ok=True)
    stamp = _param_block(ts)
    _plot_reward_overlay_stamp(dfl, ts, out / "reward_overlay.png", stamp)
    _plot_pred_loss_overlay(dfl, ts, out / "prediction_loss_overlay.png", stamp)
    _plot_test_obj_boxplot_stamp(dfl, ts, out / "test_objective_boxplot.png", stamp)
    _copy_and_combine_lambda_histograms(dfl_path.parent, ts_path.parent, out, stamp)
    print(f"[small_test] Wrote: {out}")


# ---------- Fourway ----------

def _copy_pngs(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    if src_dir.exists():
        for p in src_dir.glob("*.png"):
            try: shutil.copy(p, dst_dir / p.name)
            except Exception: pass


def _is_all_zero(x: Optional[np.ndarray], atol: float = 1e-12) -> bool:
    if x is None or x.size == 0: return True
    x = np.asarray(x, dtype=np.float64)[np.isfinite(np.asarray(x, dtype=np.float64))]
    return x.size == 0 or bool(np.all(np.abs(x) <= atol))


def _ensure_dfl_loss(rd: Dict[str, Any]) -> None:
    rw = _safe_series(rd, "train_reward_per_epoch")
    ms = _safe_series(rd, "train_mse_per_epoch")
    dl = _safe_series(rd, "train_dfl_loss_per_epoch")
    if rw is None or ms is None: return
    if dl is None or _is_all_zero(dl):
        w = rd.get("loss_weight", 1.0)
        try: w = float(w)
        except Exception: w = 1.0
        rd["train_dfl_loss_per_epoch"] = (-rw + w * ms).tolist()


def _write_variant_curve_plots(dst_dir: Path, vname: str, rd: Dict[str, Any]) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for key, ylabel, title, fname in [
        ("train_reward_per_epoch", "Train objective value", "Train objective per epoch", "train_reward_per_epoch.png"),
        ("train_mse_per_epoch", "MSE (Prediction Loss)", "DFL - Prediction Loss (MSE) per Epoch", "train_mse_per_epoch.png"),
        ("train_dfl_loss_per_epoch", "DFL Loss  (-v* + w·MSE)", "DFL - Full Loss per Epoch", "train_dfl_loss_per_epoch.png"),
    ]:
        rw = _safe_series(rd, key)
        if rw is None: continue
        fig, ax = plt.subplots(figsize=(7.0, 2.6))
        ax.plot(np.arange(1, len(rw) + 1), rw, linewidth=2.2, color="#2E59A8")
        ax.set_xlabel("Epochs"); ax.set_ylabel(ylabel); ax.set_title(title)
        fig.savefig(dst_dir / fname, dpi=300, bbox_inches="tight"); plt.close(fig)


def _overlay_rewards(series: Dict[str, np.ndarray], out_path: Path, params_text: str) -> None:
    if not series: return
    fig, ax = plt.subplots(figsize=(8.6, 3.0))
    colors = {"dfl_no_ssa": "#2E59A8", "dfl_ssa": "#1B9E77", "dfl_qpth_ssa": "#7B52AB", "twostage": "#C84B31"}
    labels = {"dfl_no_ssa": "DFL (LP) no-SSA", "dfl_ssa": "DFL (LP) + SSA", "dfl_qpth_ssa": "DFL (qpth) + SSA", "twostage": "Two-stage"}
    for key, y in series.items():
        ax.plot(np.arange(1, len(y) + 1), y, label=labels.get(key, key), linewidth=2.1, color=colors.get(key))
    ax.set_xlabel("Epoch"); ax.set_ylabel("Train objective value"); ax.set_title("Train reward per epoch (overlay)")
    ax.legend(frameon=False, loc="best", ncol=2); fig.subplots_adjust(bottom=0.30); _stamp_fig(fig, params_text)
    fig.savefig(out_path, dpi=250); plt.close(fig)


def _overlay_losses(dfl_losses: Dict[str, np.ndarray], mse_losses: Dict[str, np.ndarray], ts_loss: Optional[np.ndarray], out_path: Path, params_text: str) -> None:
    if not dfl_losses and not mse_losses and ts_loss is None: return
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2))
    colors_dfl = {"dfl_no_ssa": "#2E59A8", "dfl_ssa": "#1B9E77", "dfl_qpth_ssa": "#7B52AB"}
    labels_dfl = {"dfl_no_ssa": "DFL (LP) no-SSA", "dfl_ssa": "DFL (LP) + SSA", "dfl_qpth_ssa": "DFL (qpth) + SSA"}
    for key, y in dfl_losses.items():
        axes[0].plot(np.arange(1, len(y) + 1), y, label=labels_dfl.get(key, key), linewidth=2.0, color=colors_dfl.get(key))
    axes[0].set_title("DFL loss per epoch"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("DFL loss  (-v* + w·MSE)"); axes[0].legend(frameon=False, loc="best")
    for key, y in mse_losses.items():
        axes[1].plot(np.arange(1, len(y) + 1), y, label=f"{labels_dfl.get(key, key)} MSE", linewidth=1.9, color=colors_dfl.get(key, None), alpha=0.95)
    if ts_loss is not None:
        axes[1].plot(np.arange(1, len(ts_loss) + 1), ts_loss, label="Two-stage pred loss", linewidth=2.0, color="#C84B31")
    axes[1].set_title("Prediction loss per epoch"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss"); axes[1].legend(frameon=False, loc="best")
    fig.subplots_adjust(bottom=0.32, wspace=0.28); _stamp_fig(fig, params_text); fig.savefig(out_path, dpi=250); plt.close(fig)


def _fourway_test_boxplot(values: Dict[str, np.ndarray], out_path: Path, params_text: str) -> None:
    if not values: return
    order, labels = ["dfl_no_ssa", "dfl_ssa", "dfl_qpth_ssa", "twostage"], ["DFL (LP)\nno-SSA", "DFL (LP)\n+SSA", "DFL (qpth)\n+SSA", "Two-stage"]
    data, used_labels = [], []
    for k, lab in zip(order, labels):
        if k in values and values[k].size > 0:
            data.append(values[k]); used_labels.append(lab)
    if not data: return
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)
    palette = ["#2E59A8", "#1B9E77", "#7B52AB", "#C84B31"]
    for patch, c in zip(bp["boxes"], palette[: len(data)]):
        patch.set_facecolor(c); patch.set_alpha(0.25); patch.set_edgecolor(c); patch.set_linewidth(1.5)
    for med in bp["medians"]: med.set_color("#111111"); med.set_linewidth(1.8)
    ax.set_xticklabels(used_labels); ax.set_ylabel("Test objective value"); ax.set_title("Test objective distribution (4-way)")
    fig.subplots_adjust(bottom=0.30); _stamp_fig(fig, params_text); fig.savefig(out_path, dpi=250); plt.close(fig)


def _variant_paths(root: Path, exp: str, cost: str, beta2: Optional[float], epochs: int, n_agg: int, qp_reg: float) -> Dict[str, Path]:
    base = root / f"exp_{exp}" / cost
    if exp == "c":
        assert beta2 is not None
        base_dfl, base_qp, base_ts = base / "dfl" / f"beta2_{beta2}", base / "dfl_qpth" / f"beta2_{beta2}", base / "twostage" / f"beta2_{beta2}"
    else:
        base_dfl, base_qp, base_ts = base / "dfl", base / "dfl_qpth", base / "twostage"
    return {
        "dfl_no_ssa": base_dfl / f"epochs_{epochs}",
        "dfl_ssa": base_dfl / f"n_agg_{n_agg}" / f"epochs_{epochs}",
        "dfl_qpth_ssa": base_qp / f"n_agg_{n_agg}" / f"qp_reg_{qp_reg}" / f"epochs_{epochs}",
        "twostage": base_ts / f"epochs_{epochs}",
    }


def _cmd_fourway(args: argparse.Namespace) -> None:
    root, out_root = Path(args.results_root), Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    settings: list = [
        ("a", "high_margin", None), ("a", "low_margin", None), ("b", "high_margin", None), ("b", "low_margin", None),
        ("c", "high_margin", 0.09), ("c", "high_margin", -0.09), ("c", "low_margin", 0.09), ("c", "low_margin", -0.09),
    ]
    for exp, cost, beta2 in settings:
        sub = out_root / f"exp_{exp}" / cost
        if beta2 is not None: sub = sub / f"beta2_{beta2:+.2f}"
        sub.mkdir(parents=True, exist_ok=True)
        paths = _variant_paths(root, exp=exp, cost=cost, beta2=beta2, epochs=args.epochs, n_agg=args.n_agg, qp_reg=args.qp_reg)
        for vname, src_dir in paths.items():
            _copy_pngs(src_dir, sub / vname)
        results: Dict[str, Dict[str, Any]] = {}
        for vname, src_dir in paths.items():
            for fname in ["results.json", "results_backup2.json", "results_backup.json"]:
                if (src_dir / fname).exists():
                    try: results[vname] = _read_json(src_dir / fname); break
                    except Exception: pass
        for vname, rd in results.items():
            if vname.startswith("dfl"): _ensure_dfl_loss(rd)
            _write_variant_curve_plots(sub / vname, vname, rd)
        params_src = results.get("dfl_no_ssa") or results.get("dfl_ssa") or results.get("dfl_qpth_ssa") or results.get("twostage")
        params_text = _param_text(params_src) if params_src else ""
        reward_series = {}
        for k, rd in results.items():
            s = _safe_series(rd, "train_reward_per_epoch")
            if s is not None:
                reward_series[k] = s
        _overlay_rewards(reward_series, sub / "fourway_train_reward_overlay.png", params_text)
        dfl_losses = {k: _safe_series(rd, "train_dfl_loss_per_epoch") for k, rd in results.items() if k.startswith("dfl")}
        mse_losses = {k: _safe_series(rd, "train_mse_per_epoch") for k, rd in results.items() if k.startswith("dfl")}
        dfl_losses = {k: v for k, v in dfl_losses.items() if v is not None}
        mse_losses = {k: v for k, v in mse_losses.items() if v is not None}
        ts_loss = _safe_series(results["twostage"], "train_twostage_loss_per_epoch") if "twostage" in results else None
        _overlay_losses(dfl_losses, mse_losses, ts_loss, sub / "fourway_losses_overlay.png", params_text)
        test_vals = {}
        for vname, rd in results.items():
            tv = rd.get("test_obj_values")
            if isinstance(tv, list) and len(tv) > 0:
                try: test_vals[vname] = np.asarray(tv, dtype=np.float64)
                except Exception: pass
        _fourway_test_boxplot(test_vals, sub / "fourway_test_obj_values_boxplot.png", params_text)
    print(f"[fourway] Wrote: {out_root}")


# ---------- exp_b: side-by-side plots + gradient magnitudes ----------

def _cmd_exp_b(args: argparse.Namespace) -> None:
    """Generate side-by-side DFL vs Two-stage boxplots for exp_b and report ||dℓ/dw||, ||dv*/dw||."""
    from mdp_lp.plot_elasticity import plot_comparison_boxplot
    base = Path(args.base_dir or "mdp_lp")
    exp_b_dir = base / "exp_b" if (base / "exp_b").exists() else base
    if not exp_b_dir.exists():
        print(f"exp_b directory not found under {base}")
        return
    epochs_dir = f"epochs_{args.epochs}"
    report_lines = [
        "Experiment B: gradient magnitudes (last epoch)",
        "  ||dℓ/dw||  = magnitude_grad_pred_loss_last_epoch  (norm of gradient of prediction loss w.r.t. weights)",
        "  ||dv*/dw|| = magnitude_grad_vstar_last_epoch     (norm of gradient of v* w.r.t. weights)",
        "  mse_last_epoch = prediction loss (MSE) value at last epoch (not a gradient).",
        "  Loss weight (lw): to balance gradient magnitudes use  lw ≈ ||dv*/dw|| / ||dℓ/dw||  (from a run).",
        "  E.g. if ||dv*/dw||=500 and ||dℓ/dw||=1000 then lw≈0.5 gives similar-sized updates from v* and pred loss.",
        "",
        f"{'cost_case':<14} {'tag':<28} {'||dℓ/dw||':<14} {'||dv*/dw||':<14} {'lw_balance':<10}",
        "-" * 82,
    ]
    for cost in ("high_margin", "low_margin"):
        cost_dir = exp_b_dir / cost
        ts_path = cost_dir / "twostage" / epochs_dir / "results.json"
        if not ts_path.exists():
            ts_path = cost_dir / "twostage" / "results.json"
        if not ts_path.exists():
            continue
        ts_data = _read_json(ts_path)
        ts_values = np.array(ts_data.get("test_obj_values") or [])
        dfl_globs = list((cost_dir / "dfl").glob(f"*/{epochs_dir}/results.json"))
        if not dfl_globs:
            dfl_globs = list((cost_dir / "dfl").glob(f"{epochs_dir}/results.json"))
        for dfl_path in sorted(dfl_globs):
            dfl_data = _read_json(dfl_path)
            dfl_values = np.array(dfl_data.get("test_obj_values") or [])
            if len(dfl_values) == 0 or len(ts_values) == 0:
                continue
            # tag: from dfl/tag_xxx/epochs_100/results.json -> tag_xxx; from dfl/epochs_100/results.json -> default
            tag = (dfl_path.parent.parent.name if dfl_path.parent.parent.name != "dfl" else "default")
            save_dir = cost_dir / tag if tag != "default" else cost_dir
            save_dir.mkdir(parents=True, exist_ok=True)
            plot_comparison_boxplot(dfl_values, ts_values, cost_dir, filename=f"{tag}/comparison_boxplot.png" if tag != "default" else "comparison_boxplot.png")
            print(f"Created comparison boxplot: exp_b/{cost}/{tag or 'default'}")
            dl_dw = dfl_data.get("magnitude_grad_pred_loss_last_epoch")
            dv_dw = dfl_data.get("magnitude_grad_vstar_last_epoch")
            if dl_dw is None or dv_dw is None:
                vstar_ep = _safe_series(dfl_data, "train_grad_norm_vstar_per_epoch")
                pred_ep = _safe_series(dfl_data, "train_grad_norm_pred_loss_per_epoch")
                if vstar_ep is not None and len(vstar_ep) > 0:
                    dv_dw = float(vstar_ep[-1])
                if pred_ep is not None and len(pred_ep) > 0:
                    dl_dw = float(pred_ep[-1])
            if dl_dw is not None and dv_dw is not None:
                lw_bal = float(dv_dw) / float(dl_dw) if float(dl_dw) > 0 else 0.0
                report_lines.append(f"{cost:<14} {tag:<28} {float(dl_dw):<14.6g} {float(dv_dw):<14.6g} {lw_bal:<10.4f}")
    if len(report_lines) <= 6:
        report_lines.append("")
        report_lines.append("No gradient magnitudes found in exp_b DFL results.json.")
        report_lines.append("Re-run DFL training for exp_b (current train_elasticity.py logs train_grad_norm_*_per_epoch")
        report_lines.append("and magnitude_grad_*_last_epoch). Then run:  python -m mdp_lp.generate_results exp_b")
    report_path = exp_b_dir / "gradient_magnitudes_report.txt"
    report_path.write_text("\n".join(report_lines))
    print(f"Wrote {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comparison plots and result reports (all under mdp_lp/).")
    sub = parser.add_subparsers(dest="command", required=True)

    # boxplot
    p_box = sub.add_parser("boxplot", help="Side-by-side DFL vs Two-stage boxplot per (exp, cost, [beta2])")
    p_box.add_argument("base_dir", nargs="?", default=None, help="Base dir (default: mdp_lp/results_elasticity)")
    p_box.add_argument("epochs", nargs="?", type=int, default=100, help="Epochs (default: 100)")

    # report
    p_rep = sub.add_parser("report", help="Full comparison report (reward overlay, losses, test boxplot, lambda histograms)")
    p_rep.add_argument("--epochs", type=int, default=100)
    p_rep.add_argument("--input_dir", type=str, default="mdp_lp/results_elasticity")
    p_rep.add_argument("--output_dir", type=str, default="mdp_lp/results_elasticity_comparisons")

    # small_test
    p_sm = sub.add_parser("small_test", help="2-way comparison for one (root, exp, cost, tag)")
    p_sm.add_argument("--root", type=str, default="mdp_lp/small_test")
    p_sm.add_argument("--exp", type=str, default="a")
    p_sm.add_argument("--cost", type=str, default="high_margin")
    p_sm.add_argument("--tag", type=str, required=True)
    p_sm.add_argument("--epochs", type=int, default=20)
    p_sm.add_argument("--output_dir", type=str, default="mdp_lp/small_test/comparison")

    # exp_b
    p_eb = sub.add_parser("exp_b", help="Exp B: side-by-side DFL vs Two-stage boxplots + gradient magnitudes report")
    p_eb.add_argument("--base_dir", type=str, default="mdp_lp", help="Base dir containing exp_b (default: mdp_lp)")
    p_eb.add_argument("--epochs", type=int, default=100)

    # fourway
    p_fw = sub.add_parser("fourway", help="4-way: dfl no-SSA, dfl SSA, dfl_qpth SSA, twostage")
    p_fw.add_argument("--epochs", type=int, default=100)
    p_fw.add_argument("--n_agg", type=int, default=20)
    p_fw.add_argument("--qp_reg", type=float, default=0.5)
    p_fw.add_argument("--results_root", type=str, default="mdp_lp/results_elasticity")
    p_fw.add_argument("--out_dir", type=str, default="mdp_lp/fourway_results")

    args = parser.parse_args()
    if args.command == "boxplot":
        base = getattr(args, "base_dir", None)
        ep = getattr(args, "epochs", 100)
        if base is not None and (isinstance(base, int) or str(base).isdigit()):
            ep = int(base)
            base = None
        print(f"Creating comparison boxplots (epochs={ep}, base_dir={base or 'mdp_lp/results_elasticity'})...")
        create_all_comparison_plots(epochs=ep, base_dir=base)
    elif args.command == "report":
        _cmd_report(args)
    elif args.command == "small_test":
        _cmd_small_test(args)
    elif args.command == "exp_b":
        _cmd_exp_b(args)
    elif args.command == "fourway":
        _cmd_fourway(args)


if __name__ == "__main__":
    main()
