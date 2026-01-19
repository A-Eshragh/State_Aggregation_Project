#!/usr/bin/env python3
"""
Generate a clean 2-way comparison folder for a small_test setting.

Creates:
- reward_overlay.png
- prediction_loss_overlay.png  (DFL MSE vs Two-stage pred loss)
- lambda_comparison.png        (test lambda_true vs lambda_pred distributions)
- test_objective_boxplot.png   (test_obj_values for DFL vs Two-stage)

Usage:
  python -m mdp_lp.generate_small_test_comparison \
      --root mdp_lp/small_test \
      --exp a --cost high_margin --tag hypo_p100_cost0_lam5_cap50 --epochs 20
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


def _read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text())


def _safe_series(d: Dict[str, Any], key: str) -> Optional[np.ndarray]:
    v = d.get(key)
    if not isinstance(v, list) or len(v) == 0:
        return None
    return np.asarray(v, dtype=np.float64)


def _stamp(fig: plt.Figure, text: str) -> None:
    if not text:
        return
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


def _param_block(d: Dict[str, Any]) -> str:
    def f(x: Any, nd: int = 3) -> str:
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

    avg_true = None
    if isinstance(d.get("test_lambda_true"), list) and len(d["test_lambda_true"]) > 0:
        avg_true = float(np.mean(np.asarray(d["test_lambda_true"], dtype=np.float64)))
    lines = [
        f"tag={d.get('tag','')}",
        f"avg demand (mean lambda_true)={f(avg_true,3)}",
        f"p_constant={f(d.get('p_constant'),2)}  beta={f(d.get('beta'),3)}",
        f"foc={f(d.get('foc'),2)} voc={f(d.get('voc'),2)} hc={f(d.get('hc'),2)} bc={f(d.get('bc'),2)}",
        f"s_max(cap)={f(d.get('s_max'),0)}  b_max={f(d.get('b_max'),0)}  q_max={f(d.get('q_max'),0)}  d_max={f(d.get('d_max'),0)}",
    ]
    return "\n".join(lines)


def _plot_reward_overlay(dfl: Dict[str, Any], ts: Dict[str, Any], out: Path, stamp: str) -> None:
    y_dfl = _safe_series(dfl, "train_reward_per_epoch")
    y_ts = _safe_series(ts, "train_reward_per_epoch")
    if y_dfl is None and y_ts is None:
        return
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    if y_dfl is not None:
        ax.plot(np.arange(1, len(y_dfl) + 1), y_dfl, label="DFL (LP)", linewidth=2.2, color="#2E59A8")
    if y_ts is not None:
        ax.plot(np.arange(1, len(y_ts) + 1), y_ts, label="Two-stage", linewidth=2.2, color="#C84B31")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train objective value")
    ax.set_title("Train objective per epoch (overlay)")
    ax.legend(frameon=False, loc="best")
    fig.subplots_adjust(bottom=0.30)
    _stamp(fig, stamp)
    fig.savefig(out, dpi=250)
    plt.close(fig)


def _plot_pred_loss_overlay(dfl: Dict[str, Any], ts: Dict[str, Any], out: Path, stamp: str) -> None:
    y_dfl = _safe_series(dfl, "train_mse_per_epoch")
    y_ts = _safe_series(ts, "train_twostage_loss_per_epoch")
    if y_dfl is None and y_ts is None:
        return
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    if y_dfl is not None:
        ax.plot(np.arange(1, len(y_dfl) + 1), y_dfl, label="DFL MSE", linewidth=2.2, color="#1B9E77")
    if y_ts is not None:
        ax.plot(np.arange(1, len(y_ts) + 1), y_ts, label="Two-stage MSE", linewidth=2.2, color="#C84B31")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Prediction loss (MSE)")
    ax.set_title("Prediction loss per epoch (overlay)")
    ax.legend(frameon=False, loc="best")
    fig.subplots_adjust(bottom=0.30)
    _stamp(fig, stamp)
    fig.savefig(out, dpi=250)
    plt.close(fig)


def _copy_and_combine_lambda_histograms(
    dfl_run_dir: Path, ts_run_dir: Path, out_dir: Path, stamp: str
) -> None:
    """
    To avoid confusion and ensure an apples-to-apples visual match, we copy the exact
    per-run `lambda_histograms.png` generated during training, and also create a
    side-by-side combined image.
    """
    dfl_img = dfl_run_dir / "lambda_histograms.png"
    ts_img = ts_run_dir / "lambda_histograms.png"
    if not dfl_img.exists() or not ts_img.exists():
        return

    out_dfl = out_dir / "lambda_histograms_dfl.png"
    out_ts = out_dir / "lambda_histograms_twostage.png"
    try:
        shutil.copy(dfl_img, out_dfl)
        shutil.copy(ts_img, out_ts)
    except Exception:
        pass

    # Combined view (side-by-side)
    try:
        img_ts = plt.imread(str(ts_img))
        img_dfl = plt.imread(str(dfl_img))
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
        axes[0].imshow(img_ts)
        axes[0].set_title("Two-stage: lambda_histograms.png")
        axes[0].axis("off")
        axes[1].imshow(img_dfl)
        axes[1].set_title("DFL (LP): lambda_histograms.png")
        axes[1].axis("off")
        fig.subplots_adjust(bottom=0.20, wspace=0.02)
        _stamp(fig, stamp)
        fig.savefig(out_dir / "lambda_histograms_side_by_side.png", dpi=250)
        plt.close(fig)
    except Exception:
        # best-effort; copied images are the important part
        pass


def _plot_test_obj_boxplot(dfl: Dict[str, Any], ts: Dict[str, Any], out: Path, stamp: str) -> None:
    dfl_v = _safe_series(dfl, "test_obj_values")
    ts_v = _safe_series(ts, "test_obj_values")
    if dfl_v is None or ts_v is None:
        return
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    bp = ax.boxplot([ts_v, dfl_v], patch_artist=True, showfliers=False)
    colors = ["#C84B31", "#2E59A8"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.25)
        patch.set_edgecolor(c)
        patch.set_linewidth(1.5)
    for med in bp["medians"]:
        med.set_color("#111111")
        med.set_linewidth(1.8)
    ax.set_xticklabels(["Two-stage", "DFL (LP)"])
    ax.set_ylabel("Test objective value")
    ax.set_title("Test objective distribution (2-way)")
    fig.subplots_adjust(bottom=0.30)
    _stamp(fig, stamp)
    fig.savefig(out, dpi=250)
    plt.close(fig)


def _paths(root: Path, exp: str, cost: str, tag: str, epochs: int) -> Tuple[Path, Path]:
    dfl = root / f"exp_{exp}" / cost / "dfl" / f"tag_{tag}" / f"epochs_{epochs}" / "results.json"
    ts = root / f"exp_{exp}" / cost / "twostage" / f"tag_{tag}" / f"epochs_{epochs}" / "results.json"
    return dfl, ts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="mdp_lp/small_test")
    ap.add_argument("--exp", type=str, default="a")
    ap.add_argument("--cost", type=str, default="high_margin")
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out_dir", type=str, default="mdp_lp/small_test/comparison")
    args = ap.parse_args()

    root = Path(args.root)
    dfl_path, ts_path = _paths(root, args.exp, args.cost, args.tag, args.epochs)
    if not dfl_path.exists():
        raise FileNotFoundError(f"Missing DFL results.json: {dfl_path}")
    if not ts_path.exists():
        raise FileNotFoundError(f"Missing Two-stage results.json: {ts_path}")

    dfl = _read_json(dfl_path)
    ts = _read_json(ts_path)

    out = Path(args.out_dir) / f"tag_{args.tag}" / f"epochs_{args.epochs}"
    out.mkdir(parents=True, exist_ok=True)

    stamp = _param_block(ts)  # same config across runs; use any

    _plot_reward_overlay(dfl, ts, out / "reward_overlay.png", stamp)
    _plot_pred_loss_overlay(dfl, ts, out / "prediction_loss_overlay.png", stamp)
    _plot_test_obj_boxplot(dfl, ts, out / "test_objective_boxplot.png", stamp)

    # Lambda comparison: copy the original per-run plots + create a combined view.
    dfl_run_dir = dfl_path.parent
    ts_run_dir = ts_path.parent
    _copy_and_combine_lambda_histograms(dfl_run_dir, ts_run_dir, out, stamp)

    # Remove old generated file name if present from previous script versions.
    old = out / "lambda_comparison.png"
    if old.exists():
        try:
            old.unlink()
        except Exception:
            pass

    print(f"[small_test] Wrote: {out}")


if __name__ == "__main__":
    main()

