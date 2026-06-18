#!/usr/bin/env python3
"""
Run a 3-way DFL comparison for mdp_lp (skipping the LP no-aggregation baseline):

1) DFL-QP (qpth), no aggregation
2) DFL-QP (qpth), WITH soft aggregation
3) DFL (LP + fixed-basis gradients), WITH soft aggregation

This is intentionally separate from `run_all_elasticity.py` to avoid changing
your existing DFL-vs-two-stage suite.
"""

import argparse
import os
import subprocess
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path


def run_experiment(exp_config):
    experiment_type, cost_case, algo, beta_2, n_agg, qp_reg = exp_config

    cmd = [
        sys.executable,
        "-m",
        "mdp_lp.run_elasticity_experiments",
        "--experiment_type",
        experiment_type,
        "--cost_case",
        cost_case,
        "--algo",
        algo,
        "--beta_0",
        "4.0",
        "--beta_1",
        "-0.3",
        "--n_epochs",
        "100",
        "--n_samples",
        "200",
    ]

    if experiment_type == "c":
        cmd.extend(["--beta_2", str(beta_2)])

    # Add 4-way knobs
    cmd.extend(["--n_agg", str(n_agg)])
    if algo == "dfl_qpth":
        cmd.extend(["--qp_reg", str(qp_reg)])

    log_dir = Path("mdp_lp/results_elasticity") / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    tag = f"exp_{experiment_type}_{cost_case}_{algo}"
    if experiment_type == "c":
        tag += f"_beta2_{beta_2}"
    if n_agg != -1:
        tag += f"_nagg_{n_agg}"
    if algo == "dfl_qpth":
        tag += f"_qpreg_{qp_reg}"
    log_file = log_dir / f"{tag}.log"

    print(f"[PID {os.getpid()}] Running: {tag}")
    with open(log_file, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)

    if result.returncode == 0:
        print(f"[PID {os.getpid()}] Completed: {tag}")
        return True, tag, None
    else:
        try:
            lines = log_file.read_text().splitlines()
            error_msg = "\n".join(lines[-20:])[:800] if lines else "Unknown error"
        except Exception:
            error_msg = "Could not read log file"
        print(f"[PID {os.getpid()}] Failed: {tag} - {error_msg}")
        return False, tag, error_msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_agg", type=int, default=20, help="Number of clusters for soft aggregation")
    parser.add_argument("--qp_reg", type=float, default=0.5, help="QP regularization coefficient for dfl_qpth")
    parser.add_argument("--workers", type=int, default=-1, help="Number of parallel workers (-1 = auto)")
    parser.add_argument(
        "--variants",
        type=str,
        default="all",
        choices=["all", "qpth_noagg", "qpth_sa", "lp_sa"],
        help="Which variants to run: all (default), qpth_noagg, qpth_sa, lp_sa",
    )
    args = parser.parse_args()

    # Requested order:
    #   (1) qpth no-agg, (2) qpth+SSA, (3) LP+SSA
    all_variants = [
        ("dfl_qpth", -1, args.qp_reg),
        ("dfl_qpth", args.n_agg, args.qp_reg),
        ("dfl", args.n_agg, 0.0),
    ]
    if args.variants == "all":
        variants = all_variants
    elif args.variants == "qpth_noagg":
        variants = [all_variants[0]]
    elif args.variants == "qpth_sa":
        variants = [all_variants[1]]
    else:  # lp_sa
        variants = [all_variants[2]]

    experiments = []
    for experiment_type, cost_case, beta_2 in [
        ("a", "high_margin", 0.0),
        ("a", "low_margin", 0.0),
        ("b", "high_margin", 0.0),
        ("b", "low_margin", 0.0),
        ("c", "high_margin", 0.09),
        ("c", "high_margin", -0.09),
        ("c", "low_margin", 0.09),
        ("c", "low_margin", -0.09),
    ]:
        for algo, n_agg, qp_reg in variants:
            experiments.append((experiment_type, cost_case, algo, beta_2, n_agg, qp_reg if qp_reg is not None else 0.0))

    num_cores = cpu_count()
    if args.workers == -1:
        num_workers = max(1, min(num_cores - 2, len(experiments)))
    else:
        num_workers = max(1, int(args.workers))

    print("=" * 70)
    print("RUNNING ALL ELASTICITY EXPERIMENTS (3-WAY DFL COMPARISON)")
    print(f"Workers: {num_workers}/{num_cores} cores")
    print(f"Aggregation clusters (n_agg): {args.n_agg}")
    print(f"QP regularization (qp_reg): {args.qp_reg}")
    print("=" * 70)

    with Pool(processes=num_workers) as pool:
        results = pool.map(run_experiment, experiments)

    success_count = sum(1 for success, _, _ in results if success)
    print("\n" + "=" * 70)
    print(f"EXPERIMENTS COMPLETE: {success_count}/{len(experiments)} successful")
    print("=" * 70)

    failures = [(tag, err) for success, tag, err in results if not success]
    if failures:
        print("\nFAILED EXPERIMENTS:")
        for tag, err in failures:
            print(f"  {tag}: {err}")


if __name__ == "__main__":
    main()

