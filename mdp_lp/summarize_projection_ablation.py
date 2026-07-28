"""
Summarize the Level-3 projection ablation runs (run_projection_ablation.sh).

For each (arm, tau): mean +/- SE test regret over seeds, paired per-seed wins
vs the projmdp arm at the same tau, training-time infeasible-solve fraction
(projlp), final train reward, and wall-clock.

Run from repo root:  python -m mdp_lp.summarize_projection_ablation
"""

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path("mdp_lp/results_projection_ablation/exp_b/normal_margin/dfl")
N_TRAIN_INSTANCES = 10  # n_samples 10 -> 40 generated, train_frac .25 -> 10 train


def load_runs():
    runs = []
    for tagdir in sorted(ROOT.glob("tag_*")):
        m = re.match(r"tag_(\w+)_tau([\d.]+)_seed(\d+)", tagdir.name)
        if not m:
            continue
        arm, tau, seed = m.group(1), float(m.group(2)), int(m.group(3))
        rdir = tagdir / "n_agg_10" / "epochs_50"
        try:
            res = json.load(open(rdir / "results.json"))
            ckpt = json.load(open(rdir / "checkpoint_final.json"))
        except FileNotFoundError:
            print(f"[warn] incomplete run: {tagdir.name}")
            continue
        infeas = ckpt.get("lp_infeasible_per_epoch", [])
        n_epochs = max(len(infeas), 1)
        runs.append(dict(
            arm=arm, tau=tau, seed=seed,
            regret=float(res["test_mean_regret"]),
            lam_mse=float(res["test_lambda_mse"]),
            train_reward_final=float(ckpt["train_reward_per_epoch"][-1]),
            wall=float(ckpt["train_time"][-1]) if ckpt.get("train_time") else np.nan,
            infeas_frac=float(sum(infeas)) / (n_epochs * N_TRAIN_INSTANCES),
        ))
    return runs


def main():
    runs = load_runs()
    if not runs:
        print("no completed runs found under", ROOT)
        return
    arms = sorted({(r["arm"], r["tau"]) for r in runs})
    base = {(r["tau"], r["seed"]): r["regret"]
            for r in runs if r["arm"] == "projmdp"}

    hdr = (f"{'arm':>8} {'tau':>5} {'n':>3} {'regret mean±SE':>18} "
           f"{'wins vs projmdp':>16} {'infeas%':>8} {'train rwd':>10} {'wall(s)':>8}")
    lines = [hdr, "-" * len(hdr)]
    for arm, tau in arms:
        sub = [r for r in runs if r["arm"] == arm and r["tau"] == tau]
        reg = np.array([r["regret"] for r in sub])
        se = reg.std(ddof=1) / np.sqrt(len(reg)) if len(reg) > 1 else 0.0
        wins = sum(1 for r in sub
                   if (r["tau"], r["seed"]) in base
                   and r["regret"] < base[(r["tau"], r["seed"])])
        nbase = sum(1 for r in sub if (r["tau"], r["seed"]) in base)
        infeas = 100 * np.mean([r["infeas_frac"] for r in sub])
        trwd = np.mean([r["train_reward_final"] for r in sub])
        wall = np.mean([r["wall"] for r in sub])
        winstr = "--" if arm == "projmdp" else f"{wins}/{nbase}"
        lines.append(f"{arm:>8} {tau:>5} {len(sub):>3} "
                     f"{reg.mean():>10.2f} ± {se:<5.2f} {winstr:>16} "
                     f"{infeas:>7.1f}% {trwd:>10.2f} {wall:>8.1f}")
    out = "\n".join(lines)
    print(out)
    (ROOT.parent.parent.parent / "summary_projection_ablation.txt").write_text(
        out + "\n")
    (ROOT.parent.parent.parent / "runs_projection_ablation.json").write_text(
        json.dumps(runs, indent=1))


if __name__ == "__main__":
    main()
