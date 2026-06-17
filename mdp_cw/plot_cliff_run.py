"""
Per-run plots for cliff-walking experiments.

For a single run directory (containing `results.json`) this script produces:

    train_reward_per_epoch.png        reward-on-true-MDP curve + oracle line
    train_pred_loss_per_epoch.png     slip-CE curve
    train_dfl_loss_per_epoch.png      full DFL loss curve (DFL variants only)
    test_reward_boxplot.png           per-sample test reward
    test_regret_histogram.png         per-sample regret distribution
    test_oracle_vs_predicted.png      scatter (oracle, predicted)

Usage:

    python -m mdp_cw.plot_cliff_run --run_dir mdp_cw/results/dfl

It is also invoked automatically at the end of each training run if
`args.save_dir` is set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
def _load_results(run_dir: Path) -> dict:
    path = run_dir / "results.json"
    if not path.exists():
        # fall back to checkpoint when training was interrupted
        path = run_dir / "checkpoint.json"
    return json.loads(path.read_text())


def _algo_label(algo: str) -> str:
    return {"twostage": "Two-stage (CE)",
            "dfl": "DFL (LP)",
            "dfl_qpth": "DFL (qpth)"}.get(algo, algo)


# ---------------------------------------------------------------------------
def plot_train_reward(res: dict, out: Path, algo: str):
    curve = res.get("train_reward_per_epoch", [])
    if not curve:
        return
    curve = np.asarray(curve, dtype=float)
    xs = np.arange(len(curve))
    oracle = res.get("train_optimal_reward")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, curve, color="#2ca02c", linewidth=2, label=f"{_algo_label(algo)} train reward")
    if oracle is not None:
        ax.axhline(float(oracle), color="black", linestyle="--", alpha=0.7,
                   label=f"oracle ({float(oracle):.2f})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("reward on true MDP")
    ax.set_title(f"Train reward per epoch — {_algo_label(algo)}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_pred_loss(res: dict, out: Path, algo: str):
    curve = res.get("train_pred_loss_per_epoch", [])
    if not curve:
        return
    curve = np.asarray(curve, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(len(curve)), curve, color="#1f77b4", linewidth=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("slip-CE (prediction loss)")
    ax.set_title(f"Slip-distribution CE per epoch — {_algo_label(algo)}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_dfl_loss(res: dict, out: Path, algo: str):
    curve = res.get("train_dfl_loss_per_epoch", [])
    if not curve or algo not in ("dfl", "dfl_qpth"):
        return
    curve = np.asarray(curve, dtype=float)
    if np.allclose(curve, 0.0):
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(len(curve)), curve, color="#d62728", linewidth=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("DFL loss  (-E[v] + aug-lag)")
    ax.set_title(f"DFL loss per epoch — {_algo_label(algo)}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_test_reward_box(res: dict, out: Path, algo: str):
    pred = np.asarray(res.get("test_obj_values", []), dtype=float)
    oracle = np.asarray(res.get("test_oracle_values", []), dtype=float)
    if pred.size == 0:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    bp = ax.boxplot([oracle, pred], tick_labels=["oracle", _algo_label(algo)],
                    patch_artist=True)
    for patch, c in zip(bp["boxes"], ["#7f7f7f", "#2ca02c"]):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.set_ylabel("reward on true MDP")
    ax.set_title(f"Test reward distribution — {_algo_label(algo)}")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_regret_histogram(res: dict, out: Path, algo: str):
    regret = np.asarray(res.get("test_regret", []), dtype=float)
    if regret.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(regret, bins=20, color="#ff7f0e", alpha=0.85, edgecolor="black")
    ax.axvline(regret.mean(), color="black", linestyle="--",
               label=f"mean={regret.mean():.2f}")
    ax.set_xlabel("regret (oracle − predicted)")
    ax.set_ylabel("count")
    ax.set_title(f"Test regret histogram — {_algo_label(algo)}")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_oracle_vs_pred(res: dict, out: Path, algo: str):
    pred = np.asarray(res.get("test_obj_values", []), dtype=float)
    oracle = np.asarray(res.get("test_oracle_values", []), dtype=float)
    if pred.size == 0 or oracle.size == 0:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    lims = [min(pred.min(), oracle.min()), max(pred.max(), oracle.max())]
    ax.plot(lims, lims, color="black", linestyle="--", alpha=0.6, label="y = x")
    ax.scatter(oracle, pred, alpha=0.75, color="#2ca02c", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("oracle reward")
    ax.set_ylabel("predicted-policy reward on true MDP")
    ax.set_title(f"Oracle vs. predicted — {_algo_label(algo)}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def make_all_plots(run_dir: Path):
    run_dir = Path(run_dir)
    res = _load_results(run_dir)
    algo = str(res.get("algo", "dfl"))

    plot_train_reward(res, run_dir / "train_reward_per_epoch.png", algo)
    plot_pred_loss(res, run_dir / "train_pred_loss_per_epoch.png", algo)
    plot_dfl_loss(res, run_dir / "train_dfl_loss_per_epoch.png", algo)
    plot_test_reward_box(res, run_dir / "test_reward_boxplot.png", algo)
    plot_regret_histogram(res, run_dir / "test_regret_histogram.png", algo)
    plot_oracle_vs_pred(res, run_dir / "test_oracle_vs_predicted.png", algo)
    print(f"[plot_cliff_run] wrote per-run plots to {run_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True,
                   help="Run directory containing results.json / checkpoint.json.")
    args = p.parse_args()
    make_all_plots(Path(args.run_dir))


if __name__ == "__main__":
    main()
