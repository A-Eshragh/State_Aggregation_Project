#!/usr/bin/env python3
"""
Scan all results.json under project root; extract params and metrics.
Output: structured summary for DFL vs twostage comparison and best-setting recommendation.
"""
import json
from pathlib import Path
import sys

def main():
    root = Path(__file__).resolve().parent.parent
    results_files = list(root.rglob("results.json"))
    # Prefer non-backup (exclude old_results if you want only recent; include for full picture)
    rows = []
    for fp in sorted(results_files):
        try:
            data = json.loads(fp.read_text())
        except Exception as e:
            print(f"Skip {fp}: {e}", file=sys.stderr)
            continue
        # Path relative to root for run_id
        try:
            rel = fp.relative_to(root)
        except ValueError:
            rel = fp
        run_id = str(rel.parent)
        algo = data.get("algo") or "?"
        exp = data.get("experiment_type") or "?"
        cost = data.get("cost_case") or "?"
        n_samples = data.get("n_samples")
        s_max = data.get("s_max")
        p_std = data.get("p_lognormal_std")
        sigma_eps = data.get("sigma_epsilon")
        n_epochs = data.get("n_epochs")
        loss_weight = data.get("loss_weight")
        aug_rho = data.get("aug_lag_rho")
        beta = data.get("beta")

        train_opt = data.get("train_optimal_reward")
        train_rew = data.get("train_reward_per_epoch")
        if isinstance(train_rew, list) and len(train_rew) > 0:
            train_epoch0 = train_rew[0]
            train_final = train_rew[-1]
        else:
            train_epoch0 = train_final = None

        test_obj = data.get("test_obj_values")
        if test_obj is not None and len(test_obj) > 0:
            import numpy as np
            a = np.array(test_obj)
            test_mean = float(a.mean())
            test_median = float(np.median(a))
        else:
            test_mean = test_median = None

        rows.append({
            "path": run_id,
            "algo": algo,
            "exp": exp,
            "cost": cost,
            "n_samples": n_samples,
            "s_max": s_max,
            "p_lognormal_std": p_std,
            "sigma_epsilon": sigma_eps,
            "n_epochs": n_epochs,
            "loss_weight": loss_weight,
            "aug_lag_rho": aug_rho,
            "beta": beta,
            "train_optimal_reward": train_opt,
            "train_reward_epoch0": train_epoch0,
            "train_reward_final": train_final,
            "test_obj_mean": test_mean,
            "test_obj_median": test_median,
        })
    return rows

if __name__ == "__main__":
    rows = main()
    # Print CSV-style for parsing
    keys = ["path", "algo", "exp", "cost", "n_samples", "s_max", "p_lognormal_std", "sigma_epsilon",
            "n_epochs", "loss_weight", "aug_lag_rho", "beta",
            "train_optimal_reward", "train_reward_epoch0", "train_reward_final", "test_obj_mean", "test_obj_median"]
    print("\t".join(keys))
    for r in rows:
        print("\t".join(str(r.get(k, "")) for k in keys))
