#!/usr/bin/env python3
"""Paper-style Inventory reproduction runner for the notebook.

This file is a command orchestrator only. It calls the repository training
entry point `mdp_lp.run_elasticity_experiments` and the repository figure code
in root `make_figures.py`; it does not reimplement Inventory training,
LP/QP/Sketch solving, or plotting logic.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = Path("notebooks/reproduction/inventory_figure_runs")
DIM_TAG = "s30q30"


BASE_ARGS = [
    "-m",
    "mdp_lp.run_elasticity_experiments",
    "--experiment_type",
    "b",
    "--cost_case",
    "normal_margin",
    "--bc",
    "10",
    "--n_samples",
    "10",
    "--b_max",
    "5",
    "--s_max",
    "30",
    "--q_max",
    "30",
]


def method_spec(method: str) -> tuple[str, int | None, tuple[str, ...]]:
    if method == "twostage":
        return "twostage", None, ("--aug_lag_rho", "10000")
    if method == "dfl":
        return "dfl", None, ("--aug_lag_rho", "10")
    if method == "qpth":
        return "dfl_qpth", None, ("--qp_reg", "0.001", "--aug_lag_rho", "10000")
    if method == "sketch":
        return (
            "dfl_sketch",
            -1,
            (
                "--n_agg",
                "-1",
                "--n_sketches",
                "8",
                "--sketch_ratio",
                "0.1",
                "--sketch_mode",
                "gaussian",
                "--aug_lag_rho",
                "0.1",
                "--loss_weight",
                "1",
            ),
        )
    if method.startswith("sa"):
        n_agg = int(method[2:])
        return (
            "dfl_sketch",
            n_agg,
            (
                "--n_agg",
                str(n_agg),
                "--n_sketches",
                "8",
                "--sketch_ratio",
                "0.2",
                "--sketch_mode",
                "gaussian",
                "--aug_lag_rho",
                "0.1",
                "--loss_weight",
                "1",
            ),
        )
    raise ValueError(f"unknown method {method!r}")


def tag_for(method: str, seed: int) -> str:
    tag = f"seed_{seed}_{DIM_TAG}"
    if method == "sketch":
        return tag + "_K8_r0.1_gaussian"
    if method.startswith("sa"):
        return tag + "_K8_r0.2_gaussian"
    return tag


def result_dir(root: Path, method: str, seed: int, epochs: int) -> Path:
    algo, n_agg, _ = method_spec(method)
    tag = tag_for(method, seed)
    if algo == "dfl_sketch":
        base = root / "exp_b" / "normal_margin" / f"dfl_sketch_ngg_{n_agg}" / f"tag_{tag}"
    else:
        base = root / "exp_b" / "normal_margin" / algo / f"tag_{tag}"
    if algo == "dfl_qpth":
        base = base / "qp_reg_0.001"
    return base / f"epochs_{epochs}"


def result_path(root: Path, method: str, seed: int, epochs: int) -> Path:
    return result_dir(root, method, seed, epochs) / "results.json"


def run_one(
    root: Path,
    method: str,
    seed: int,
    epochs: int,
    pretrain_epochs: int,
    skip_existing: bool,
) -> tuple[str, int, str, float]:
    out = result_path(root, method, seed, epochs)
    if skip_existing and out.exists():
        return method, seed, "skip", 0.0

    algo, _, extra = method_spec(method)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        *BASE_ARGS,
        "--algo",
        algo,
        "--n_epochs",
        str(epochs),
        "--pretrain_epochs",
        str(pretrain_epochs),
        "--results_root",
        str(root),
        "--seed",
        str(seed),
        "--tag",
        tag_for(method, seed),
        *extra,
    ]
    t0 = time.time()
    with (out.parent / "run.log").open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    ok = proc.returncode == 0 and out.exists()
    return method, seed, "ok" if ok else "fail", time.time() - t0


def summarize(root: Path, methods: list[str], seeds: list[int], epochs: int) -> list[dict]:
    rows = []
    for method in methods:
        vals = []
        statuses = []
        for seed in seeds:
            path = result_path(root, method, seed, epochs)
            if not path.exists():
                statuses.append(f"{seed}:missing")
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            statuses.append(f"{seed}:{data.get('status', 'unknown')}")
            regret = data.get("test_mean_regret", data.get("regret_mean"))
            if regret is not None:
                vals.append(float(regret))
        rows.append(
            {
                "method": method,
                "n": len(vals),
                "mean_test_regret": sum(vals) / len(vals) if vals else None,
                "statuses": ",".join(statuses),
            }
        )
    return rows


def write_summary(root: Path, methods: list[str], seeds: list[int], epochs: int) -> None:
    rows = summarize(root, methods, seeds, epochs)
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = [
        "# Inventory Paper-Style Figure Reproduction Summary",
        "",
        "| method | n | mean_test_regret | statuses |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in rows:
        mean = "" if row["mean_test_regret"] is None else f"{row['mean_test_regret']:.4f}"
        lines.append(f"| {row['method']} | {row['n']} | {mean} | {row['statuses']} |")
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((root / "summary.md").read_text(encoding="utf-8"))


def render_inventory_pngs(root: Path, seeds: list[int]) -> None:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    import matplotlib.pyplot as plt
    import make_figures as paper_figures

    out_dir = root / "paper_style_figures"

    def save_png(draw, path: Path) -> None:
        prev = plt.rcParams.copy()
        plt.rcParams.update(paper_figures.RC)
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        draw(ax)
        if path.name == "inventory_training_curves.pdf":
            lows, highs = [], []
            for line in ax.lines:
                y = line.get_ydata()
                if len(y):
                    lows.append(float(min(y)))
                    highs.append(float(max(y)))
            if lows and highs:
                lo, hi = min(lows), max(highs)
                pad = max(5.0, 0.08 * (hi - lo))
                ax.set_ylim(lo - pad, hi + pad)
        elif path.name == "inventory_decision_quality.pdf":
            heights = [
                patch.get_height()
                for patch in ax.patches
                if patch.get_height() > 0
            ]
            if heights:
                lo = max(0.0, min(heights) * 0.85)
                hi = max(heights) * 1.15
                ax.set_ylim(lo, hi)
        fig.tight_layout()
        png_path = path.with_suffix(".png")
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        plt.rcParams.update(prev)
        print(f"Saved {png_path}")

    old_save = paper_figures._save
    try:
        paper_figures._save = save_png
        paper_figures.render_inventory(out_dir, root, seeds)
    finally:
        paper_figures._save = old_save


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument(
        "--methods",
        nargs="+",
        default=["twostage", "dfl", "qpth", "sketch", "sa10", "sa20", "sa30"],
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--pretrain_epochs", type=int, default=30)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--skip_existing", action="store_true", default=True)
    p.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    p.add_argument("--summary_only", action="store_true")
    p.add_argument("--no_figure", action="store_true")
    args = p.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        jobs = [
            (args.root, method, seed, args.epochs, args.pretrain_epochs, args.skip_existing)
            for method in args.methods
            for seed in args.seeds
        ]
        with futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            for fut in futures.as_completed([pool.submit(run_one, *job) for job in jobs]):
                print(fut.result(), flush=True)

    write_summary(args.root, args.methods, args.seeds, args.epochs)

    if not args.no_figure:
        render_inventory_pngs(args.root, args.seeds)


if __name__ == "__main__":
    main()
