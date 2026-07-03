#!/usr/bin/env python3
"""Run Cliff-Walking reproduction jobs through the repository CLI.

This script intentionally lives under notebooks/reproduction. It does not
modify or monkey-patch repository code; each job calls:

    python -m mdp_cw.run_cliff_experiments

The default parameter set follows the explicit Cliff-Walking entries in the
paper Table 2 where the current CLI exposes them. Optional data-distribution
arguments are kept visible because Table 2 does not fully specify the weather
regime mixture.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = Path("notebooks/reproduction/cliff_table2_reproduction")

PAPER_TARGETS = {
    "twostage": ("Two-stage", 14.75, 0.29),
    "dfl": ("DFL-LP", 14.14, 0.18),
    "dfl_sketch": ("DFL-Sketch", 11.34, 1.44),
    "dfl_qpth": ("DFL-QP", 9.53, 0.84),
}

TABLE2_BASE = (
    "--n_samples", "200",          # current CLI: train_frac=0.5 -> 100 train, 100 test
    "--train_frac", "0.5",
    "--batch_size", "32",
    "--hidden_dim", "64",
    "--lr", "3e-3",
    "--pretrain_epochs", "60",
    "--n_epochs", "50",
    "--beta", "0.95",
)

TABLE2_METHODS = {
    "twostage": ("twostage", ()),
    "dfl": ("dfl", ("--aug_lag_rho", "0.1")),
    "dfl_qpth": ("dfl_qpth", ("--aug_lag_rho", "0.1", "--qp_reg", "0.1")),
    "dfl_sketch": (
        "dfl_sketch",
        ("--aug_lag_rho", "1.0", "--n_sketches", "8", "--sketch_ratio", "0.2"),
    ),
    "ssa10": (
        "dfl_ssa",
        ("--aug_lag_rho", "0.1", "--n_agg", "10", "--sa_tau", "0.5"),
    ),
    "ssa20": (
        "dfl_ssa",
        ("--aug_lag_rho", "0.1", "--n_agg", "20", "--sa_tau", "0.5"),
    ),
    "ssa30": (
        "dfl_ssa",
        ("--aug_lag_rho", "0.1", "--n_agg", "30", "--sa_tau", "0.5"),
    ),
}

DATA_ROUTES = {
    "table2_cli_defaults": (),
    "readme_paper_v4": (
        "--regime_set", "v4",
        "--cliff_penalty", "-10",
        "--noise_std", "2.5",
        "--center_scale", "0.88",
        "--ambiguous_pairs", "1:2,2:3",
        "--loss_weight", "500",
    ),
    "dist_shift_recipe": (
        "--regime_set", "v4",
        "--center_scale", "0.88",
        "--ambiguous_pairs", "0:2",
        "--ambiguous_gap", "0.0",
        "--regime_weights", "15,1,1,1",
        "--test_regime_weights", "0,0,1,0",
        "--cliff_penalty", "-50.0",
        "--noise_std", "0.6",
        "--loss_weight", "500",
    ),
}


def result_path(root: Path, route: str, method: str, seed: int) -> Path:
    return root / route / f"{method}_s{seed}" / "results.json"


def run_one(
    root: Path,
    route: str,
    method: str,
    seed: int,
    skip_existing: bool,
) -> tuple[str, str, int, str, float]:
    out = result_path(root, route, method, seed)
    if skip_existing and out.exists():
        return route, method, seed, "skip", 0.0
    out.parent.mkdir(parents=True, exist_ok=True)

    algo, method_args = TABLE2_METHODS[method]
    cmd = [
        sys.executable,
        "-m",
        "mdp_cw.run_cliff_experiments",
        *TABLE2_BASE,
        *DATA_ROUTES[route],
        "--algo",
        algo,
        *method_args,
        "--seed",
        str(seed),
        "--save_dir",
        str(out.parent),
    ]

    (out.parent / "command.json").write_text(
        json.dumps({"cmd": cmd, "cwd": str(REPO)}, indent=2),
        encoding="utf-8",
    )
    t0 = time.time()
    with (out.parent / "run.log").open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    ok = proc.returncode == 0 and out.exists()
    return route, method, seed, "ok" if ok else "fail", time.time() - t0


def read_regret(path: Path) -> float | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    for key in ("regret_mean", "test_mean_regret"):
        if key in data:
            return float(data[key])
    return None


def summarize(root: Path, routes: list[str], methods: list[str]) -> list[dict]:
    rows = []
    try:
        config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
        seeds_to_check = [int(s) for s in config.get("seeds", [])]
    except Exception:
        seeds_to_check = []
    for route in routes:
        for method in methods:
            vals = []
            seeds = []
            failures = []
            if seeds_to_check:
                run_dirs = [(str(seed), result_path(root, route, method, seed).parent) for seed in seeds_to_check]
            else:
                run_dirs = [
                    (p.name.rsplit("_s", 1)[-1], p)
                    for p in sorted((root / route).glob(f"{method}_s[0-9]*"))
                    if p.name.rsplit("_s", 1)[0] == method
                ]
            for seed, run_dir in run_dirs:
                if not run_dir.exists():
                    failures.append(seed)
                    continue
                val = read_regret(run_dir / "results.json")
                if val is None or not np.isfinite(val):
                    failures.append(seed)
                    continue
                vals.append(val)
                seeds.append(seed)
            target = PAPER_TARGETS.get(method)
            row = {
                "route": route,
                "method": method,
                "label": target[0] if target else method,
                "n_ok": len(vals),
                "ok_seeds": ",".join(seeds),
                "failed_seeds": ",".join(failures),
                "paper_mean": target[1] if target else None,
                "paper_se": target[2] if target else None,
            }
            if vals:
                arr = np.asarray(vals, dtype=float)
                row.update({
                    "mean": float(arr.mean()),
                    "se": float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0,
                    "delta": float(arr.mean() - target[1]) if target else None,
                })
            else:
                row.update({"mean": None, "se": None, "delta": None})
            rows.append(row)
    return rows


def write_summary(root: Path, routes: list[str], methods: list[str]) -> None:
    rows = summarize(root, routes, methods)
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = [
        "# Cliff Table 2 Reproduction",
        "",
        "| route | method | n ok | failed | mean | SE | paper mean | delta |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        mean = "" if r["mean"] is None else f"{r['mean']:.4f}"
        se = "" if r["se"] is None else f"{r['se']:.4f}"
        paper = "" if r["paper_mean"] is None else f"{r['paper_mean']:.4f}"
        delta = "" if r["delta"] is None else f"{r['delta']:.4f}"
        lines.append(
            f"| {r['route']} | {r['label']} | {r['n_ok']} | {r['failed_seeds']} | "
            f"{mean} | {se} | {paper} | {delta} |"
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((root / "summary.md").read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--routes", nargs="+", default=["table2_cli_defaults"])
    p.add_argument("--methods", nargs="+", default=list(TABLE2_METHODS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--skip_existing", action="store_true", default=True)
    p.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--summary_only", action="store_true")
    args = p.parse_args()

    if args.fresh and args.root.exists() and not args.summary_only:
        shutil.rmtree(args.root)
    args.root.mkdir(parents=True, exist_ok=True)

    config = {
        "table2_base": TABLE2_BASE,
        "routes": {name: DATA_ROUTES[name] for name in args.routes},
        "methods": {name: TABLE2_METHODS[name] for name in args.methods},
        "seeds": args.seeds,
    }
    (args.root / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if not args.summary_only:
        jobs = [
            (args.root, route, method, seed, args.skip_existing)
            for route in args.routes
            for method in args.methods
            for seed in args.seeds
        ]
        with futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            for fut in futures.as_completed([pool.submit(run_one, *job) for job in jobs]):
                print(fut.result(), flush=True)
    write_summary(args.root, args.routes, args.methods)


if __name__ == "__main__":
    main()
