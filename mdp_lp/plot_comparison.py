"""
Generate DFL vs two-stage comparison plots for a results root.
- Reward per epoch (DFL vs two-stage + optimal)
- Lambda histograms (DFL vs two-stage, train & test)
- Test objective boxplot (DFL vs two-stage side by side)
- Loss comparison (DFL MSE + full loss vs two-stage pred loss)
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_results(results_root: Path, cost_case: str, algo: str, epochs: int = 100) -> dict | None:
    """Load results.json from results_root/exp_b/{cost_case}/{algo}/epochs_{epochs}/."""
    path = results_root / "exp_b" / cost_case / algo / f"epochs_{epochs}" / "results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _hist_or_vline(ax, arr, *, bins=25, alpha=0.5, label="", color=""):
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if arr.size == 0:
        return
    if float(np.std(arr)) < 1e-6:
        ax.axvline(float(np.mean(arr)), color=color, linewidth=3, alpha=0.9, label=label)
        return
    ax.hist(arr, bins=bins, alpha=alpha, label=label, color=color)


def _last_epoch_lambda(data):
    """Get final-epoch train lambda from results (list of per-epoch arrays)."""
    pred_list = data.get("train_lambda_pred") or []
    true_list = data.get("train_lambda_true") or []
    if not pred_list or not true_list:
        return None, None
    pred = np.array(pred_list[-1]).reshape(-1)
    true = np.array(true_list[-1]).reshape(-1)
    return pred, true


def plot_reward_comparison(dfl: dict, ts: dict, cost_case: str, out_dir: Path) -> None:
    r_dfl = dfl.get("train_reward_per_epoch")
    r_ts = ts.get("train_reward_per_epoch")
    optimal = dfl.get("train_optimal_reward") or ts.get("train_optimal_reward")
    if not r_dfl or not r_ts:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(0, len(r_dfl)), r_dfl, linewidth=2, color="#2E59A8", label="DFL")
    ax.plot(range(0, len(r_ts)), r_ts, linewidth=2, color="#C84B31", label="Two-stage")
    if optimal is not None:
        ax.axhline(y=optimal, color="green", linestyle="--", linewidth=1.5, label="Optimal (train)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg discounted reward")
    ax.set_title(f"Exp B – {cost_case.replace('_', ' ').title()} – DFL vs Two-stage")
    ax.legend(loc="best", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / f"{cost_case}_reward_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_dir / f'{cost_case}_reward_comparison.png'}")


def plot_lambda_comparison(dfl: dict, ts: dict, cost_case: str, out_dir: Path) -> None:
    """2x2: rows = DFL / Two-stage, cols = Train / Test. Each cell: true (blue) vs predicted (red)."""
    dfl_train_p, dfl_train_t = _last_epoch_lambda(dfl)
    ts_train_p, ts_train_t = _last_epoch_lambda(ts)
    dfl_test_p = np.array(dfl.get("test_lambda_pred") or []).reshape(-1)
    dfl_test_t = np.array(dfl.get("test_lambda_true") or []).reshape(-1)
    ts_test_p = np.array(ts.get("test_lambda_pred") or []).reshape(-1)
    ts_test_t = np.array(ts.get("test_lambda_true") or []).reshape(-1)
    if (dfl_train_p is None and ts_train_p is None) and (len(dfl_test_p) == 0 and len(ts_test_p) == 0):
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    # DFL Train
    ax = axes[0, 0]
    if dfl_train_p is not None and dfl_train_t is not None:
        _hist_or_vline(ax, dfl_train_t, bins=25, alpha=0.5, label="True", color="blue")
        _hist_or_vline(ax, dfl_train_p, bins=25, alpha=0.5, label="Predicted", color="red")
    ax.set_title("DFL – Train λ")
    ax.set_xlabel("λ"); ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    # DFL Test
    ax = axes[0, 1]
    if len(dfl_test_p) and len(dfl_test_t):
        _hist_or_vline(ax, dfl_test_t, bins=25, alpha=0.5, label="True", color="blue")
        _hist_or_vline(ax, dfl_test_p, bins=25, alpha=0.5, label="Predicted", color="red")
    ax.set_title("DFL – Test λ")
    ax.set_xlabel("λ"); ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    # Two-stage Train
    ax = axes[1, 0]
    if ts_train_p is not None and ts_train_t is not None:
        _hist_or_vline(ax, ts_train_t, bins=25, alpha=0.5, label="True", color="blue")
        _hist_or_vline(ax, ts_train_p, bins=25, alpha=0.5, label="Predicted", color="red")
    ax.set_title("Two-stage – Train λ")
    ax.set_xlabel("λ"); ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    # Two-stage Test
    ax = axes[1, 1]
    if len(ts_test_p) and len(ts_test_t):
        _hist_or_vline(ax, ts_test_t, bins=25, alpha=0.5, label="True", color="blue")
        _hist_or_vline(ax, ts_test_p, bins=25, alpha=0.5, label="Predicted", color="red")
    ax.set_title("Two-stage – Test λ")
    ax.set_xlabel("λ"); ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle(f"Exp B – {cost_case.replace('_', ' ').title()} – λ Pred vs True", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / f"{cost_case}_lambda_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_dir / f'{cost_case}_lambda_comparison.png'}")


def plot_test_obj_comparison(dfl: dict, ts: dict, cost_case: str, out_dir: Path) -> None:
    """Side-by-side boxplot: DFL vs two-stage test_obj_values (LP optimal under predicted λ, one per test sample)."""
    v_dfl = dfl.get("test_obj_values")
    v_ts = ts.get("test_obj_values")
    if v_dfl is None:
        v_dfl = []
    if v_ts is None:
        v_ts = []
    v_dfl = np.asarray(v_dfl).reshape(-1)
    v_ts = np.asarray(v_ts).reshape(-1)
    if len(v_dfl) == 0 and len(v_ts) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [x for x in [v_dfl, v_ts] if len(x) > 0]
    labels = (["DFL", "Two-stage"]) if len(v_dfl) and len(v_ts) else (["DFL"] if len(v_dfl) else ["Two-stage"])
    bp = ax.boxplot(data, tick_labels=labels)
    # Annotate median (line in box) value above each box
    for i, vals in enumerate([v_dfl, v_ts]):
        if len(vals) == 0:
            continue
        med = float(np.median(vals))
        x_pos = i + 1
        q75 = float(np.percentile(vals, 75))
        q25 = float(np.percentile(vals, 25))
        y_pos = q75 + 0.05 * (q75 - q25) if q75 != q25 else q75 + 0.05 * (np.max(vals) - np.min(vals)) if np.max(vals) != np.min(vals) else q75 + 1
        ax.text(x_pos, y_pos, f"median = {med:.1f}", va="bottom", ha="center", fontsize=9)
    ax.set_ylabel("LP optimal value")
    ax.set_title(f"Exp B – {cost_case.replace('_', ' ').title()} – Test set: LP optimal under predicted λ\n(one value per test sample)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / f"{cost_case}_test_obj_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_dir / f'{cost_case}_test_obj_comparison.png'}")


def plot_loss_comparison(dfl: dict, ts: dict, cost_case: str, out_dir: Path) -> None:
    """Top: DFL MSE + two-stage pred loss on same axes. Bottom: DFL full loss only."""
    mse = dfl.get("train_mse_per_epoch")
    dfl_loss = dfl.get("train_dfl_loss_per_epoch")
    ts_loss = ts.get("train_twostage_loss_per_epoch")
    if not mse and not dfl_loss and not ts_loss:
        return
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    # Top: prediction loss comparison (DFL MSE vs two-stage pred loss)
    ax = axes[0]
    if mse:
        ax.plot(range(1, len(mse) + 1), mse, color="#1B9E77", linewidth=2, label="DFL pred (MSE)")
    if ts_loss:
        ax.plot(range(1, len(ts_loss) + 1), ts_loss, color="#C84B31", linewidth=2, label="Two-stage pred loss")
    ax.set_ylabel("Loss")
    ax.set_title("Prediction loss (MSE on λ)")
    ax.legend(loc="best", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Bottom: DFL full loss only
    ax = axes[1]
    if dfl_loss:
        ax.plot(range(1, len(dfl_loss) + 1), dfl_loss, color="#2E59A8", linewidth=2, label="DFL full loss")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    ax.set_title("DFL full loss (-v* + aug_lag + w·MSE)")
    ax.legend(loc="best", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.suptitle(f"Exp B – {cost_case.replace('_', ' ').title()} – Loss comparison", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / f"{cost_case}_loss_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_dir / f'{cost_case}_loss_comparison.png'}")


def plot_comparison(results_root: Path, cost_cases: list[str], epochs: int = 100) -> None:
    results_root = Path(results_root)
    out_dir = results_root / "comparison_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for cost_case in cost_cases:
        dfl = load_results(results_root, cost_case, "dfl", epochs=epochs)
        ts = load_results(results_root, cost_case, "twostage", epochs=epochs)
        if dfl is None or ts is None:
            continue
        plot_reward_comparison(dfl, ts, cost_case, out_dir)
        plot_lambda_comparison(dfl, ts, cost_case, out_dir)
        plot_test_obj_comparison(dfl, ts, cost_case, out_dir)
        plot_loss_comparison(dfl, ts, cost_case, out_dir)


def main():
    project_root = Path(__file__).resolve().parent.parent
    epochs = 100
    args_list = sys.argv[1:] if len(sys.argv) > 1 else []
    # Optional: last arg can be epochs (int) for custom run folders
    if args_list and args_list[-1].isdigit():
        epochs = int(args_list.pop())
    if args_list:
        roots = [Path(p) if Path(p).is_absolute() else project_root / p for p in args_list]
    else:
        roots = [project_root / "lagrangian_results", project_root / "l100"]

    cost_cases_lagrangian = ["high_margin", "low_margin"]
    cost_cases_l100 = ["high_margin", "low_margin", "normal_margin"]

    for root in roots:
        root = Path(root)
        if not root.exists():
            print(f"Skipping (not found): {root}")
            continue
        print(f"Comparison plots for {root} (epochs={epochs})")
        cost_cases = cost_cases_l100 if (root / "exp_b" / "normal_margin").exists() else cost_cases_lagrangian
        plot_comparison(root, cost_cases, epochs=epochs)


if __name__ == "__main__":
    main()
