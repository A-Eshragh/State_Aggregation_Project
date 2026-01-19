#!/usr/bin/env python3
"""
Generate DFL vs Two-stage comparison plots + per-experiment metadata into a separate output directory.

This script does NOT modify any existing training/plotting pipelines. It only reads existing
`mdp_lp/results_elasticity/**/results.json` files and writes a new report directory.

Usage (from repo root):
    python -m mdp_lp.generate_elasticity_comparisons --epochs 100

Outputs:
    mdp_lp/results_elasticity_comparisons/...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Cleaner default styling (no external deps).
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


def _get_key(d: Dict[str, Any]) -> Key:
    exp = d.get("experiment_type")
    cost = d.get("cost_case")
    beta2 = d.get("beta_2", None)
    if exp is None or cost is None:
        raise ValueError("Missing experiment_type/cost_case in results.json")
    # beta_2 is absent for exp a/b
    if exp in ("a", "b"):
        beta2 = None
    else:
        # normalize float-ish values
        beta2 = float(beta2)
    return exp, cost, beta2


def _format_key(key: Key) -> str:
    exp, cost, beta2 = key
    if beta2 is None:
        return f"exp_{exp}/{cost}"
    return f"exp_{exp}/{cost}/beta2_{beta2:+.2f}"


def _safe_get_series(d: Dict[str, Any], k: str) -> Optional[np.ndarray]:
    v = d.get(k)
    if v is None:
        return None
    if not isinstance(v, list) or len(v) == 0:
        return None
    return np.asarray(v, dtype=np.float64)


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2))


def _expected_avg_prices(d: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], str]:
    """
    Compute average scenario price(s) used for demand generation (and in this code, also used as
    the per-unit selling price passed into the inventory reward).

    Prefers realized averages if present in results (future runs may log these):
      - train_avg_price, test_avg_price
      - train_avg_related_price, test_avg_related_price (exp c)

    Falls back to expectations under the sampling distribution.
    """

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

    # If we ever log realized averages, show those (train/test should match closely anyway).
    realized = _f(d.get("test_avg_price")) or _f(d.get("train_avg_price"))
    realized_related = _f(d.get("test_avg_related_price")) or _f(d.get("train_avg_related_price"))
    if realized is not None:
        return realized, realized_related, "avg"

    # Otherwise show expected value from params
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
    """
    Minimal parameter block: only experiment-defining parameters.
    (We intentionally omit training hyperparams like lr/hidden_dim/epochs/n/loss_w and
    identifiers like exp/cost since those are encoded by the output folder path.)
    """

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
            # json may have numbers as strings
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
    """
    Place params in a reserved bottom margin so it never covers the plotted data.
    """
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


def _stamp_params(ax: plt.Axes, text: str, *, loc: str = "upper_left") -> None:
    if loc == "upper_left":
        x, y, ha, va = 0.01, 0.99, "left", "top"
    elif loc == "upper_right":
        x, y, ha, va = 0.99, 0.99, "right", "top"
    elif loc == "lower_left":
        x, y, ha, va = 0.01, 0.01, "left", "bottom"
    else:
        x, y, ha, va = 0.99, 0.01, "right", "bottom"

    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=7.2,
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


def plot_reward_overlay(dfl: Dict[str, Any], ts: Dict[str, Any], out_path: Path) -> None:
    r_dfl = _safe_get_series(dfl, "train_reward_per_epoch")
    r_ts = _safe_get_series(ts, "train_reward_per_epoch")
    if r_dfl is None or r_ts is None:
        return

    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    ax.plot(np.arange(1, len(r_dfl) + 1), r_dfl, label="DFL", linewidth=2.2, color="#2E59A8")
    ax.plot(np.arange(1, len(r_ts) + 1), r_ts, label="Two-stage", linewidth=2.2, color="#C84B31")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train objective value")
    ax.set_title("Train reward per epoch")
    ax.legend(frameon=False, loc="best")

    # Reserve bottom margin for params (so it never covers the plot)
    fig.subplots_adjust(bottom=0.30)
    _stamp_params_fig(fig, _param_text(dfl))
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def plot_losses(dfl: Dict[str, Any], ts: Dict[str, Any], out_path: Path) -> None:
    # DFL
    dfl_loss = _safe_get_series(dfl, "train_dfl_loss_per_epoch")
    dfl_mse = _safe_get_series(dfl, "train_mse_per_epoch")
    # Two-stage
    ts_pred = _safe_get_series(ts, "train_twostage_loss_per_epoch")

    if dfl_loss is None and dfl_mse is None and ts_pred is None:
        return

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 7.2), sharex=True)

    if dfl_loss is not None:
        axes[0].plot(np.arange(1, len(dfl_loss) + 1), dfl_loss, color="purple", linewidth=2.0)
    axes[0].set_ylabel("DFL loss")
    # Add the loss expression as a smaller annotation under the y-axis label.
    # (Matplotlib doesn't support mixed-size ylabel text cleanly.)
    axes[0].text(
        -0.13,
        0.5,
        "(-v* + w·MSE)",
        transform=axes[0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=8,
        color="#666666",
    )
    axes[0].set_title("Loss curves")

    if dfl_mse is not None:
        axes[1].plot(np.arange(1, len(dfl_mse) + 1), dfl_mse, color="green", linewidth=2.0)
    axes[1].set_ylabel("DFL MSE")

    if ts_pred is not None:
        axes[2].plot(np.arange(1, len(ts_pred) + 1), ts_pred, color="orange", linewidth=2.0)
    axes[2].set_ylabel("Two-stage pred loss")
    axes[2].set_xlabel("Epoch")

    # Reserve bottom margin for params (so it never covers the plot)
    fig.subplots_adjust(bottom=0.14)
    _stamp_params_fig(fig, _param_text(dfl))
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def plot_test_boxplot(dfl: Dict[str, Any], ts: Dict[str, Any], out_path: Path) -> None:
    v_dfl = dfl.get("test_obj_values")
    v_ts = ts.get("test_obj_values")
    if not isinstance(v_dfl, list) or not isinstance(v_ts, list) or len(v_dfl) == 0 or len(v_ts) == 0:
        return

    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.boxplot([v_dfl, v_ts], tick_labels=["DFL", "Two-stage"])
    ax.set_ylabel("Test objective value")
    ax.set_title("Test objective distribution")
    fig.subplots_adjust(bottom=0.32)
    _stamp_params_fig(fig, _param_text(dfl))
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def _coerce_1d_float_array(x: Any) -> Optional[np.ndarray]:
    """
    Convert common JSON shapes into a 1D float array.
    - list[float] -> (n,)
    - list[list[float]] -> flattened (n_total,)
    """
    if x is None:
        return None
    if not isinstance(x, list) or len(x) == 0:
        return None
    try:
        arr = np.asarray(x, dtype=np.float64)
    except Exception:
        return None
    if arr.size == 0:
        return None
    return arr.reshape(-1)


def copy_lambda_histograms(*, dfl_results_path: Path, ts_results_path: Path, out_dir: Path) -> None:
    """
    Copy the per-run lambda histogram plots (already generated during training) into the
    comparison report folder.

    We do not generate any new histogram plots here; we just copy:
      - <run_dir>/lambda_histograms.png
    into:
      - <out_dir>/lambda_histograms_dfl.png
      - <out_dir>/lambda_histograms_twostage.png
    """
    dfl_img_path = dfl_results_path.parent / "lambda_histograms.png"
    ts_img_path = ts_results_path.parent / "lambda_histograms.png"
    if dfl_img_path.exists():
        shutil.copyfile(dfl_img_path, out_dir / "lambda_histograms_dfl.png")
    if ts_img_path.exists():
        shutil.copyfile(ts_img_path, out_dir / "lambda_histograms_twostage.png")


def plot_avg_selling_price(dfl: Dict[str, Any], ts: Dict[str, Any], out_path: Path) -> None:
    """
    Show average selling price(s) for the experiment's price generation.

    Notes:
    - For exp 'a' price is constant → avg = p_constant.
    - For exp 'b'/'c' prices are sampled as exp(N(log(mean), std)) so E[p] = mean * exp(0.5 * std^2).
    - If future runs log `train_avg_price` / `test_avg_price`, we prefer those.
    """
    exp = dfl.get("experiment_type")
    if exp not in ("a", "b", "c"):
        return

    def _f(x: Any) -> Optional[float]:
        try:
            if x is None:
                return None
            return float(x)
        except Exception:
            return None

    # Prefer realized averages if present (not currently logged in older runs).
    train_avg = _f(dfl.get("train_avg_price"))
    test_avg = _f(dfl.get("test_avg_price"))

    label_suffix = ""
    if train_avg is None or test_avg is None:
        # Fall back to theoretical expectation under the sampling distribution.
        if exp == "a":
            avg_p = _f(dfl.get("p_constant"))
        else:
            p_mean = _f(dfl.get("p_lognormal_mean"))
            p_std = _f(dfl.get("p_lognormal_std"))
            if p_mean is None or p_std is None:
                return
            avg_p = float(p_mean * np.exp(0.5 * (p_std**2)))
        train_avg = avg_p
        test_avg = avg_p
        label_suffix = " (expected)"

    bars = ["train", "test"]
    vals = [train_avg, test_avg]

    # Related price for exp c
    related_train = _f(dfl.get("train_avg_related_price"))
    related_test = _f(dfl.get("test_avg_related_price"))
    if exp == "c" and (related_train is None or related_test is None):
        p_mean = _f(dfl.get("p_related_lognormal_mean"))
        p_std = _f(dfl.get("p_related_lognormal_std"))
        if p_mean is not None and p_std is not None:
            avg_pr = float(p_mean * np.exp(0.5 * (p_std**2)))
            related_train = avg_pr
            related_test = avg_pr

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = np.arange(len(bars), dtype=np.float64)
    width = 0.35 if exp == "c" and related_train is not None and related_test is not None else 0.55

    ax.bar(x - width / 2, vals, width=width, color="#2E59A8", alpha=0.85, label=f"p{label_suffix}")
    if exp == "c" and related_train is not None and related_test is not None:
        ax.bar(
            x + width / 2,
            [related_train, related_test],
            width=width,
            color="#C84B31",
            alpha=0.85,
            label=f"p_related{label_suffix if label_suffix else ''}",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(bars)
    ax.set_ylabel("Average price")
    ax.set_title("Average selling price (scenario generation)")
    ax.legend(frameon=False, loc="best")

    fig.subplots_adjust(bottom=0.36)
    _stamp_params_fig(fig, _param_text(dfl))
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--input_dir",
        type=str,
        default="mdp_lp/results_elasticity",
        help="Directory containing experiment outputs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="mdp_lp/results_elasticity_comparisons",
        help="Directory to write comparison report outputs",
    )
    args = parser.parse_args()

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

    index_lines = [
        "## Elasticity DFL vs Two-stage comparisons",
        "",
        f"Input: `{base}`",
        f"Output: `{out_base}`",
        f"Epochs: `{args.epochs}`",
        "",
        "### Configs",
    ]

    for key in keys:
        dfl_file = dfl_map[key]
        ts_file = ts_map[key]
        dfl = _read_json(dfl_file)
        ts = _read_json(ts_file)

        exp, cost, beta2 = key
        if beta2 is None:
            out_dir = out_base / f"exp_{exp}" / cost
        else:
            out_dir = out_base / f"exp_{exp}" / cost / f"beta2_{beta2:+.2f}"
        out_dir.mkdir(parents=True, exist_ok=True)

        copy_lambda_histograms(dfl_results_path=dfl_file, ts_results_path=ts_file, out_dir=out_dir)
        plot_reward_overlay(dfl, ts, out_dir / "reward_overlay.png")
        plot_losses(dfl, ts, out_dir / "losses.png")
        plot_test_boxplot(dfl, ts, out_dir / "test_boxplot.png")

        index_lines.append(f"- `{_format_key(key)}` → `{out_dir.as_posix()}`")

    out_base.joinpath("INDEX.md").write_text("\n".join(index_lines) + "\n")
    print(f"Wrote comparison report to: {out_base}")


if __name__ == "__main__":
    main()

