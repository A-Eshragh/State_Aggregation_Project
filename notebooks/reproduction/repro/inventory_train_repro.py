from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


NOTEBOOK_REPO = Path(__file__).resolve().parents[3]
REPO = NOTEBOOK_REPO
RESULTS_ROOT = NOTEBOOK_REPO / "notebooks" / "reproduction" / "repro" / "results" / "inventory"


def base_args() -> list[str]:
    return [
        "--experiment_type", "b",
        "--cost_case", "normal_margin",
        "--bc", "10",
        "--n_samples", "10",
        "--pretrain_epochs", "30",
        "--b_max", "5",
        "--s_max", "30",
        "--q_max", "30",
        "--n_epochs", "50",
        "--lr", "0.01",
        "--results_root", str(RESULTS_ROOT),
    ]


CONFIGS = {
    "ts": dict(algo="twostage", n_agg=-1, extra=["--aug_lag_rho", "10000.0"]),
    "lp": dict(algo="dfl", n_agg=-1, extra=["--aug_lag_rho", "10000.0"]),
    "qp_1e-3": dict(algo="dfl_qpth", n_agg=-1, extra=["--qp_reg", "0.001", "--aug_lag_rho", "10000.0"]),
    "qp_1e-2": dict(algo="dfl_qpth", n_agg=-1, extra=["--qp_reg", "0.01", "--aug_lag_rho", "10000.0"]),
    "qp_1e-1": dict(algo="dfl_qpth", n_agg=-1, extra=["--qp_reg", "0.1", "--aug_lag_rho", "10000.0"]),
    "qp_1": dict(algo="dfl_qpth", n_agg=-1, extra=["--qp_reg", "1.0", "--aug_lag_rho", "10000.0"]),
    "qp_10": dict(algo="dfl_qpth", n_agg=-1, extra=["--qp_reg", "10.0", "--aug_lag_rho", "10000.0"]),
    "qp_50": dict(algo="dfl_qpth", n_agg=-1, extra=["--qp_reg", "50.0", "--aug_lag_rho", "10000.0"]),
    "sketch_r01": dict(algo="dfl_sketch", n_agg=-1, extra=["--n_sketches", "8", "--sketch_ratio", "0.1", "--sketch_mode", "gaussian", "--aug_lag_rho", "10.0", "--loss_weight", "1"]),
    "sketch_r02": dict(algo="dfl_sketch", n_agg=-1, extra=["--n_sketches", "8", "--sketch_ratio", "0.2", "--sketch_mode", "gaussian", "--aug_lag_rho", "10.0", "--loss_weight", "1"]),
    "sa10_r02": dict(algo="dfl_sketch", n_agg=10, extra=["--n_sketches", "8", "--sketch_ratio", "0.2", "--sketch_mode", "gaussian", "--aug_lag_rho", "0.1", "--loss_weight", "1"]),
    "sa20_r02": dict(algo="dfl_sketch", n_agg=20, extra=["--n_sketches", "8", "--sketch_ratio", "0.2", "--sketch_mode", "gaussian", "--aug_lag_rho", "0.1", "--loss_weight", "1"]),
    "sa30_r02": dict(algo="dfl_sketch", n_agg=30, extra=["--n_sketches", "8", "--sketch_ratio", "0.2", "--sketch_mode", "gaussian", "--aug_lag_rho", "0.1", "--loss_weight", "1"]),
}


def result_path(config: str, seed: int) -> Path:
    cfg = CONFIGS[config]
    algo = cfg["algo"]
    n_agg = cfg["n_agg"]
    tag = f"grid_{config}_seed_{seed}_s30q30"
    base = RESULTS_ROOT / "exp_b" / "normal_margin"
    if algo == "dfl_sketch":
        p = base / f"dfl_sketch_ngg_{n_agg}" / f"tag_{tag}" / "epochs_50"
    elif algo == "dfl_qpth":
        qp_reg = cfg["extra"][cfg["extra"].index("--qp_reg") + 1]
        p = base / algo / f"tag_{tag}" / f"qp_reg_{qp_reg}" / "epochs_50"
    else:
        p = base / algo / f"tag_{tag}" / "epochs_50"
    return p / "results.json"


FORCE = False


def run_one(config: str, seed: int) -> tuple[str, int, str, float]:
    out = result_path(config, seed)
    if out.exists() and not FORCE:
        return config, seed, "skip", 0.0
    if FORCE and out.parent.exists():
        shutil.rmtree(out.parent)
    cfg = CONFIGS[config]
    tag = f"grid_{config}_seed_{seed}_s30q30"
    cmd = [
        sys.executable, "-m", "mdp_lp.run_elasticity_experiments",
        *base_args(),
        "--algo", cfg["algo"],
        "--n_agg", str(cfg["n_agg"]),
        "--seed", str(seed),
        "--tag", tag,
        *cfg["extra"],
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.parent / "run.log"
    t0 = time.time()
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, text=True)
    return config, seed, "ok" if proc.returncode == 0 and out.exists() else f"fail:{proc.returncode}", time.time() - t0


def summarize() -> None:
    rows = {}
    for config in CONFIGS:
        vals = []
        records = []
        for seed in range(1, 11):
            p = result_path(config, seed)
            if not p.exists():
                records.append({"seed": seed, "missing": True, "path": str(p)})
                continue
            d = json.loads(p.read_text())
            oracle = float(d["test_mean_oracle_reward"])
            regret = float(d["test_mean_regret"])
            pct = 100.0 * regret / oracle
            vals.append(pct)
            records.append({"seed": seed, "regret_pct": pct, "oracle": oracle, "regret": regret, "path": str(p)})
        if vals:
            import statistics
            rows[config] = {
                "n": len(vals),
                "mean_regret_pct": sum(vals) / len(vals),
                "std_regret_pct": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "se_regret_pct": (statistics.stdev(vals) / (len(vals) ** 0.5)) if len(vals) > 1 else 0.0,
                "min_regret_pct": min(vals),
                "max_regret_pct": max(vals),
                "values": vals,
                "records": records,
            }
        else:
            rows[config] = {"n": 0, "records": records}
    out = RESULTS_ROOT.parent / "inventory_numeric_grid_summary.json"
    out.write_text(json.dumps(rows, indent=2))
    print(out)
    for k, v in rows.items():
        if v["n"]:
            print(k, "n", v["n"], "mean", f'{v["mean_regret_pct"]:.4f}', "se", f'{v["se_regret_pct"]:.4f}', "min", f'{v["min_regret_pct"]:.4f}', "max", f'{v["max_regret_pct"]:.4f}')
        else:
            print(k, "n 0")


def main() -> None:
    global REPO, RESULTS_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(range(1, 11)))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--force", action="store_true", help="Regenerate runs even if results.json already exists.")
    args = parser.parse_args()
    REPO = args.repo_root.resolve()
    RESULTS_ROOT = args.results_root.resolve()
    global FORCE
    FORCE = bool(args.force)
    if not args.summary_only:
        jobs = [(c, s) for c in args.configs for s in args.seeds]
        failed = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_one, c, s) for c, s in jobs]
            for fut in as_completed(futs):
                result = fut.result()
                print(result, flush=True)
                if not str(result[2]).startswith(("ok", "skip")):
                    failed.append(result)
        if failed:
            raise RuntimeError(f"Inventory reproduction failed for {len(failed)} job(s): {failed[:5]}")
    summarize()


if __name__ == "__main__":
    main()
