"""Quick ablation: BC-only floor + DFL with alpha=0 vs alpha=2.

Conditions (all share offline data from the same biased behavior policy):
  - BC-only         : pretrain features+head with CE; evaluate argmax(pi_bc).
  - DFL alpha=0     : full DFL pipeline but no BC bonus in the LP reward.
  - DFL alpha=2     : full DFL pipeline (current default).
  - Two-stage MBRL  : world-model + DQN; no BC anywhere (existing baseline).

Usage:
  python -m mdp_continuous.ablation_bc_alpha --seeds 1 2 3 --n_workers 6
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from .configs import CartPoleConfig
from .train_cartpole import run_neural_dfl, run_two_stage_mbrl, set_seed
from .cartpole_env import CartPoleDynamics, collect_trajectories
from .features import ActionIndexedFeatures
from .policy import evaluate_policy
from .make_figure1_cartpole import _BiasedPolicy


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "results" / "ablation_bc_alpha"


def _bc_only(seed, cfg, behavior, pretrain_epochs):
    """Train BC head on offline data, evaluate argmax(pi_bc) on real env."""
    set_seed(seed)
    device = torch.device("cpu")
    dynamics = CartPoleDynamics(cfg)
    A = 2

    features = ActionIndexedFeatures(
        state_dim=4, n_actions=A,
        n_features=cfg.n_features,
        hidden_dims=cfg.neural_hidden,
    ).to(device)
    k = features.d_per_action

    data = collect_trajectories(dynamics, behavior, cfg.n_trajectories,
                                cfg.max_steps, device)
    w_head = torch.randn(k, A, device=device, requires_grad=True)
    opt = torch.optim.Adam(list(features.parameters()) + [w_head], lr=cfg.lr)
    ce = torch.nn.CrossEntropyLoss()
    batch_size = 4096
    N = data["states"].shape[0]

    for _ in range(pretrain_epochs):
        opt.zero_grad()
        idx = torch.randperm(N)[:batch_size]
        f_s = features._state_features(data["states"][idx])
        logits = f_s @ w_head
        loss = ce(logits, data["actions"][idx].long())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(features.parameters(), max_norm=1.0)
        opt.step()

    def ce_policy(states):
        with torch.no_grad():
            return (features._state_features(states) @ w_head).argmax(dim=1)

    res = evaluate_policy(dynamics, ce_policy, cfg.n_eval_episodes,
                          cfg.max_steps, cfg.gamma, device)
    return float(res["avg_reward"]), float(res["avg_length"])


def _run_one(spec):
    cond = spec["cond"]
    seed = spec["seed"]
    torch.set_num_threads(1)
    set_seed(seed)
    cfg = CartPoleConfig(
        n_iterations=spec["n_iterations"],
        gamma=spec["gamma"],
        lr=spec["lr"],
        qp_eta=0.01,
        reward_type="cos_theta",
        n_features=spec["n_features"],
        n_trajectories=spec["n_trajectories"],
        n_epochs=spec["n_iterations"],
        ridge=1e-4,
        rho=10.0,
        n_eval_episodes=spec["n_eval_episodes"],
    )
    behavior = _BiasedPolicy(spec["td_weight"])
    t0 = time.time()

    if cond == "bc_only":
        r, l = _bc_only(seed, cfg, behavior, spec["pretrain_epochs"])
        return {
            "cond": cond, "seed": seed, "best_reward": r,
            "final_reward": r, "rewards": [r], "lengths": [l],
            "time": time.time() - t0,
        }

    if cond.startswith("dfl") or cond == "two_stage_a":
        alpha = spec["alpha_shape"]
        use_dfl_phase = (cond != "two_stage_a")
        out = run_neural_dfl(
            cfg, verbose=False,
            pretrain_epochs=spec["pretrain_epochs"],
            use_dfl_phase=use_dfl_phase,
            eval_policy_type="lp",
            behavior_policy=behavior,
            n_ce_steps=0,
            n_dfl_steps=5,
            dfl_lr=spec["dfl_lr"],
            solver="lp",
            alpha_shape=alpha,
        )
        rewards = [float(h["avg_reward"]) for h in out["history"]]
        lengths = [float(h["avg_length"]) for h in out["history"]]
        return {
            "cond": cond, "seed": seed,
            "best_reward": float(out["best_reward"]),
            "final_reward": rewards[-1] if rewards else float("nan"),
            "rewards": rewards, "lengths": lengths,
            "time": time.time() - t0,
        }

    if cond.startswith("mbrl"):
        bc_alpha = spec.get("bc_alpha", 0.0)
        out = run_two_stage_mbrl(
            cfg, verbose=False,
            world_epochs=400, world_lr=1e-3,
            n_q_steps_per_iter=128,
            rollouts_per_iter=0, rollout_length=1,
            n_initial_synthetic=0,
            q_hidden=(32, 32),
            behavior_policy=behavior,
            bc_alpha=bc_alpha,
            bc_pretrain_epochs=spec["pretrain_epochs"],
        )
        rewards = [float(h["avg_reward"]) for h in out["history"]]
        lengths = [float(h["avg_length"]) for h in out["history"]]
        return {
            "cond": cond, "seed": seed,
            "best_reward": float(out["best_reward"]),
            "final_reward": rewards[-1] if rewards else float("nan"),
            "rewards": rewards, "lengths": lengths,
            "time": time.time() - t0,
        }

    raise ValueError(cond)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--n_iterations", type=int, default=25)
    p.add_argument("--n_trajectories", type=int, default=100)
    p.add_argument("--n_features", type=int, default=128)
    p.add_argument("--n_eval_episodes", type=int, default=100)
    p.add_argument("--pretrain_epochs", type=int, default=30)
    p.add_argument("--td_weight", type=float, default=0.013)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dfl_lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--n_workers", type=int, default=6)
    p.add_argument("--out_tag", type=str, default="quick")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.out_tag}.json"

    base = {
        "n_iterations": args.n_iterations,
        "n_trajectories": args.n_trajectories,
        "n_features": args.n_features,
        "n_eval_episodes": args.n_eval_episodes,
        "pretrain_epochs": args.pretrain_epochs,
        "td_weight": args.td_weight,
        "lr": args.lr,
        "dfl_lr": args.dfl_lr,
        "gamma": args.gamma,
    }

    conditions = [
        ("bc_only",     {}),
        ("two_stage_a", {"alpha_shape": 2.0}),         # DFL minus gradient phase
        ("dfl_a0",      {"alpha_shape": 0.0}),         # DFL, no BC bonus
        ("dfl_a2",      {"alpha_shape": 2.0}),         # DFL, default BC bonus
        ("mbrl_b",      {"bc_alpha": 0.0}),            # MBRL no BC anywhere
        ("mbrl_c",      {"bc_alpha": 2.0}),            # MBRL with BC shaping
    ]
    jobs = []
    for cond, extras in conditions:
        for s in args.seeds:
            jobs.append({"cond": cond, "seed": int(s), **base, **extras})

    print(f"[ablation] {len(jobs)} jobs across {args.n_workers} workers")
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futs = {ex.submit(_run_one, j): j for j in jobs}
        for f in as_completed(futs):
            r = f.result()
            print(f"[done] cond={r['cond']:<8s} seed={r['seed']} "
                  f"final={r['final_reward']:.2f} "
                  f"best={r['best_reward']:.2f} "
                  f"time={r['time']:.1f}s")
            results.append(r)
    print(f"[ablation] total wall time {time.time() - t0:.1f}s")

    # Aggregate
    summary = {}
    for cond, _ in conditions:
        runs = [r for r in results if r["cond"] == cond]
        finals = np.array([r["final_reward"] for r in runs])
        bests = np.array([r["best_reward"] for r in runs])
        summary[cond] = {
            "n": int(len(runs)),
            "final_mean": float(finals.mean()) if len(finals) else float("nan"),
            "final_std":  float(finals.std(ddof=1))  if len(finals) > 1 else 0.0,
            "best_mean":  float(bests.mean())  if len(bests)  else float("nan"),
            "best_std":   float(bests.std(ddof=1))   if len(bests)  > 1 else 0.0,
            "per_seed":   [{"seed": r["seed"],
                            "final": r["final_reward"],
                            "best":  r["best_reward"]} for r in runs],
        }

    payload = {"args": vars(args), "summary": summary, "raw": results}
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[ablation] saved -> {out_path}")

    print("\n=== Summary (mean ± std across seeds) ===")
    print(f"{'condition':<10s} {'final mean':>12s} {'final std':>10s} "
          f"{'best mean':>11s} {'best std':>9s}  n")
    for cond, _ in conditions:
        s = summary[cond]
        print(f"{cond:<10s} {s['final_mean']:>12.2f} {s['final_std']:>10.2f} "
              f"{s['best_mean']:>11.2f} {s['best_std']:>9.2f}  {s['n']}")


if __name__ == "__main__":
    main()
