from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_targets(targets_path: Path) -> tuple[dict, dict, dict]:
    data = load_json(targets_path)
    inventory = {k: v for k, v in data["inventory"]["targets"].items()}
    cliff_table = {k: v for k, v in data["cliff"]["targets"].items()}
    cliff_figure = {k: v for k, v in data["cliff_decision_figure"]["targets"].items()}
    return inventory, cliff_table, cliff_figure


def build_tables(results_root: Path, targets_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inventory_targets, cliff_targets, cliff_figure_targets = load_targets(targets_path)
    inventory = load_json(results_root / "inventory_numeric_grid_summary.json")
    cliff = load_json(results_root / "cliff" / "cliff_combined_summary.json")

    def inventory_row(label: str, key: str) -> dict:
        item = inventory[key]
        error = abs(item["mean_regret_pct"] - inventory_targets[label]["mean"])
        return {
            "method": label,
            "paper_mean": inventory_targets[label]["mean"],
            "paper_se": inventory_targets[label]["se"],
            "repro_mean": item["mean_regret_pct"],
            "repro_se": item["se_regret_pct"],
            "abs_error": error,
            "n": item["n"],
            "config": key,
            "status": "reproduced" if error <= 0.5 else "check",
        }

    def cliff_row(label: str, source: str, key: str) -> dict:
        item = cliff[source][key]
        target_mean = cliff_targets[label]["mean"]
        target_se = cliff_targets[label]["se"]
        error = abs(item["mean"] - target_mean)
        return {
            "method": label,
            "paper_mean": target_mean,
            "paper_se": target_se,
            "repro_mean": item["mean"],
            "repro_se": item["se"],
            "abs_error": error,
            "n": item["n"],
            "config": key,
            "status": "reproduced" if error <= max(0.25, target_se if target_se is not None else 0.25) else "check",
        }

    inventory_df = pd.DataFrame([
        inventory_row("TS", "ts"),
        inventory_row("QP", "qp_50"),
        inventory_row("LP", "lp"),
        inventory_row("Sketch", "sketch_r01"),
        inventory_row("SA M=10", "sa10_r02"),
        inventory_row("SA M=20", "sa20_r02"),
        inventory_row("SA M=30", "sa30_r02"),
    ])
    cliff_df = pd.DataFrame([
        cliff_row("TS", "baseline", "ts"),
        cliff_row("QP", "baseline", "qpth"),
        cliff_row("LP", "baseline", "dfl"),
        cliff_row("Sketch", "baseline", "sketch"),
        cliff_row("SA M=6", "sa", "M6_tau0p5_lw500p0"),
        cliff_row("SA M=12", "sa", "M12_tau0p5_lw500p0"),
        cliff_row("SA M=24", "sa", "M24_tau2p0_lw500p0"),
    ])

    cliff_figure_rows = []
    for label, target in cliff_figure_targets.items():
        cliff_figure_rows.append({
            "method": label,
            "paper_mean": target["mean"],
            "paper_se": target["se"],
            "repro_mean": np.nan,
            "repro_se": np.nan,
            "abs_error": np.nan,
            "n": 0,
            "config": "not available from the reproduced Table 3 run",
            "status": "not reproduced",
        })
    cliff_figure_df = pd.DataFrame(cliff_figure_rows)
    return inventory_df, cliff_df, cliff_figure_df


def plot_paper_vs_repro(df: pd.DataFrame, title: str, ylabel: str, ax) -> None:
    x = np.arange(len(df))
    width = 0.36
    paper_yerr = df["paper_se"] if df["paper_se"].notna().any() else None
    ax.bar(
        x - width / 2,
        df["paper_mean"],
        width,
        yerr=paper_yerr,
        capsize=4 if paper_yerr is not None else 0,
        color="#cfcfcf",
        edgecolor="black",
        linewidth=0.5,
        label="paper",
    )
    ax.bar(
        x + width / 2,
        df["repro_mean"],
        width,
        yerr=df["repro_se"],
        capsize=4,
        color="#4c78a8",
        edgecolor="black",
        linewidth=0.5,
        label="reproduction",
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(df["method"], rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()


def plot_cliff_decision_figure_comparison(df: pd.DataFrame, ax) -> None:
    x = np.arange(len(df))
    width = 0.36
    paper_y = df["paper_mean"].astype(float).to_numpy()
    has_repro = df["repro_mean"].notna().any()
    repro_y = df["repro_mean"].astype(float).to_numpy() if has_repro else None
    repro_yerr = df["repro_se"] if has_repro and df["repro_se"].notna().any() else None
    ax.bar(
        x - width / 2,
        paper_y,
        width=width,
        color="#cfcfcf",
        edgecolor="black",
        linewidth=0.5,
        label="paper",
    )
    if has_repro:
        ax.bar(
            x + width / 2,
            repro_y,
            width=width,
            yerr=repro_yerr,
            capsize=4 if repro_yerr is not None else 0,
            color="#4c78a8",
            edgecolor="black",
            linewidth=0.5,
            label="reproduction",
        )
    ax.set_title("Cliff Walking - Decision quality")
    ax.set_ylabel("Test regret")
    ax.set_xticks(x)
    ax.set_xticklabels(df["method"], rotation=25, ha="right")
    ax.set_ylim(0.6, 1.0)
    ax.set_yticks(np.arange(0.6, 1.01, 0.1))
    ax.grid(axis="y", alpha=0.25)
    yr = ax.get_ylim()[1] - ax.get_ylim()[0]
    label_groups = [(0 if not has_repro else -width / 2, paper_y)]
    if has_repro:
        label_groups.append((width / 2, repro_y))
    for offset, values in label_groups:
        for i, (method, value) in enumerate(zip(df["method"], values)):
            if pd.isna(value):
                continue
            shown_y = min(float(value), ax.get_ylim()[1])
            y_text = shown_y - 0.045 * yr if value > ax.get_ylim()[1] else shown_y + 0.015 * yr
            ax.text(i + offset, y_text, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    ax.legend()


def plot_cliff_table3_comparison(df: pd.DataFrame, ax) -> None:
    plot_paper_vs_repro(df, "Cliff Walking - Appendix Table 3", "Test regret", ax)


def _pad_curves(curves: list[list[float]]) -> np.ndarray:
    curves = [list(map(float, c)) for c in curves if c]
    if not curves:
        return np.zeros((0, 0), dtype=float)
    width = max(len(c) for c in curves)
    arr = np.full((len(curves), width), np.nan, dtype=float)
    for i, c in enumerate(curves):
        arr[i, : len(c)] = c
    return arr


def _curve_mean_se(curves: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    arr = _pad_curves(curves)
    if arr.size == 0:
        return np.asarray([]), np.asarray([])
    mean = np.nanmean(arr, axis=0)
    count = np.sum(~np.isnan(arr), axis=0)
    std = np.nanstd(arr, axis=0, ddof=1)
    se = np.where(count > 1, std / np.sqrt(count), 0.0)
    return mean, se


def _load_inventory_curves(inventory: dict, key: str) -> list[list[float]]:
    curves = []
    for rec in inventory[key].get("records", []):
        path = Path(rec.get("path", ""))
        if not path.exists():
            continue
        data = load_json(path)
        curve = data.get("train_reward_per_epoch", [])
        if curve:
            curves.append(curve)
    return curves


def _load_cliff_curves(cliff: dict, source: str, key: str) -> list[list[float]]:
    item = cliff[source][key]
    curves = []
    for row in item.get("rows", []):
        curve = row.get("train_reward_per_epoch", [])
        if curve:
            curves.append(curve)
    return curves


def plot_training_curves(results_root: Path, ax_inventory, ax_cliff) -> None:
    inventory = load_json(results_root / "inventory_numeric_grid_summary.json")
    cliff = load_json(results_root / "cliff" / "cliff_combined_summary.json")

    inventory_specs = [
        ("Two-stage", "ts", "#ff7f0e", "--"),
        ("DFL-QP", "qp_50", "#2ca02c", "-"),
        ("DFL-LP", "lp", "#1f77b4", "-"),
        ("DFL-Sketch", "sketch_r01", "#d62728", ":"),
        ("DFL-SA (M=10)", "sa10_r02", "#9467bd", "-"),
    ]
    cliff_specs = [
        ("Two-stage", "baseline", "ts", "#ff7f0e", "--"),
        ("DFL-QP", "baseline", "qpth", "#2ca02c", "-"),
        ("DFL-LP", "baseline", "dfl", "#1f77b4", "-"),
        ("DFL-Sketch", "baseline", "sketch", "#d62728", ":"),
        ("DFL-SA (M=24)", "sa", "M24_tau2p0_lw500p0", "#9467bd", "-"),
    ]

    for label, key, color, linestyle in inventory_specs:
        mean, se = _curve_mean_se(_load_inventory_curves(inventory, key))
        if len(mean) == 0:
            continue
        x = np.arange(len(mean))
        ax_inventory.plot(x, mean, label=label, color=color, linestyle=linestyle, linewidth=1.6)
        ax_inventory.fill_between(x, mean - se, mean + se, color=color, alpha=0.12, linewidth=0)

    for label, source, key, color, linestyle in cliff_specs:
        mean, se = _curve_mean_se(_load_cliff_curves(cliff, source, key))
        if len(mean) == 0:
            continue
        x = np.arange(len(mean))
        ax_cliff.plot(x, mean, label=label, color=color, linestyle=linestyle, linewidth=1.6)
        ax_cliff.fill_between(x, mean - se, mean + se, color=color, alpha=0.12, linewidth=0)

    ax_inventory.set_title("Inventory MDP - Training curves")
    ax_inventory.set_xlabel("Training epoch")
    ax_inventory.set_ylabel("Training reward")
    ax_inventory.grid(alpha=0.25)
    ax_inventory.legend(fontsize=8)

    ax_cliff.set_title("Cliff Walking - Training curves")
    ax_cliff.set_xlabel("Training epoch")
    ax_cliff.set_ylabel("Avg. episode reward")
    ax_cliff.grid(alpha=0.25)
    ax_cliff.legend(fontsize=8)


def write_figures(results_root: Path, targets_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    inventory_df, cliff_df, cliff_figure_df = build_tables(results_root, targets_path)

    inventory_df.to_csv(out_dir / "inventory_paper_vs_repro.csv", index=False)
    cliff_df.to_csv(out_dir / "cliff_table3_paper_vs_repro.csv", index=False)
    cliff_figure_df.to_csv(out_dir / "cliff_decision_figure_targets.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    plot_paper_vs_repro(
        inventory_df,
        "Inventory MDP: decision quality",
        "test regret (% oracle reward)",
        axes[0],
    )
    plot_cliff_decision_figure_comparison(cliff_figure_df, axes[1])
    fig.tight_layout()
    fig.savefig(out_dir / "decision_quality_figure_targets.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "decision_quality_paper_vs_reproduction.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "paper_vs_reproduction.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    plot_cliff_table3_comparison(cliff_df, ax)
    fig.tight_layout()
    fig.savefig(out_dir / "cliff_table3_paper_vs_reproduction.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    plot_training_curves(results_root, axes[0], axes[1])
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves_reproduction.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("notebooks/reproduction/repro/results"))
    parser.add_argument("--targets", type=Path, default=Path("notebooks/reproduction/repro/paper_targets.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("notebooks/reproduction/repro/figures"))
    args = parser.parse_args()
    write_figures(args.results_root, args.targets, args.out_dir)
    print(args.out_dir)


if __name__ == "__main__":
    main()
