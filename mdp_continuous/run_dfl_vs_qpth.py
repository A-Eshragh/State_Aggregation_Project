"""
Multi-seed comparison runner for cartpole DFL methods.

Routes each (solver, seed) pair through the right runner from train_cartpole:
    lp / gurobi_qp / qpth  → run_neural_dfl with the matching solver
    two_stage              → run_neural_dfl with use_dfl_phase=False
                             (no world model — same features+pretrain as
                             DFL-LP, just skips the DFL gradient phase;
                             so iter-0 reward matches DFL-LP iter-0)
    two_stage_mbrl         → run_two_stage_mbrl  (legacy MBRL baseline)

Both two_stage variants write to the same canonical filename
(curve_two_stage_mbrl_*) so make_figures.py shows the latest as the
"Two-stage" curve/bar.

Output JSONs land under mdp_continuous/results/dfl_vs_qpth/ with the same
canonical filenames that make_figure1_cartpole._cache_key produces, so
make_figures.py picks them up unchanged.

Usage:
    python -m mdp_continuous.run_dfl_vs_qpth --seeds 1 2 3 4 5
    python -m mdp_continuous.run_dfl_vs_qpth --seeds 1 2 3 --solvers lp qpth
    python -m mdp_continuous.run_dfl_vs_qpth --seeds 1..20 --solvers two_stage
    python -m mdp_continuous.run_dfl_vs_qpth --seeds 1 --n_iterations 20  (quick)
"""
import argparse
import json
import statistics as st
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from .cartpole_env import CartPoleConfig
from .make_figure1_cartpole import make_behavior_policy
from .train_cartpole import run_neural_dfl, run_two_stage_mbrl, set_seed


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results" / "dfl_vs_qpth"


# Canonical cache prefix per solver — matches make_figure1_cartpole._cache_key
# so make_figures.py picks these JSONs up without renaming.
SOLVER_TO_PREFIX = {
    "lp":             "curve_neural",
    "gurobi_qp":      "curve_qpth",       # historical "DFL-QP" slot in panel
    "qpth":           "curve_qpth",       # real qpth library — same slot
    "sketch":         "curve_sketch",     # sketch-and-solve LP (this repo)
    "two_stage":      "curve_two_stage_mbrl",   # ← reuses MBRL filename so
                                                #   make_figures.py picks it
                                                #   up unchanged
    "two_stage_mbrl": "curve_two_stage_mbrl",
}


def _canonical_cache_key(solver: str, args) -> str:
    """Mirror make_figure1_cartpole._cache_key for the panel anchor files."""
    prefix = SOLVER_TO_PREFIX[solver]
    dfl_lr_tag = "auto" if args.dfl_lr is None else f"{args.dfl_lr}"
    qpth_tag = f"_qpeta{args.qp_eta}" if "qpth" in prefix else ""
    sketch_tag = ""
    if solver == "sketch":
        sketch_tag = (
            f"_K{args.n_sketches}_r{args.sketch_ratio}_m{args.sketch_mode}")
    return (
        f"{prefix}"
        f"_beh{args.behavior}_td{args.td_weight}_eps{args.eps}"
        f"_ntraj{args.n_trajectories}"
        f"_iters{args.n_iterations}"
        f"_nfeat{args.n_features}"
        f"_preTS{args.pretrain_epochs_ts}_preDFL{args.pretrain_epochs_dfl}"
        f"_nce{args.n_ce_steps}_ndfl{args.n_dfl_steps}"
        f"_dfllr{dfl_lr_tag}"
        f"_eval{args.eval_policy}"
        f"{qpth_tag}"
        f"{sketch_tag}"
    )


def _run_one(args_dict: dict) -> dict:
    """One (solver, seed) job. Picklable; runs in a worker process."""
    solver = args_dict["solver"]
    seed = args_dict["seed"]
    torch.set_num_threads(args_dict.get("torch_threads", 1))
    set_seed(seed)

    cfg = CartPoleConfig(
        n_iterations=args_dict["n_iterations"],
        gamma=args_dict["gamma"],
        lr=args_dict["lr"],
        qp_eta=args_dict["qp_eta"],
        reward_type="cos_theta",
        n_features=args_dict["n_features"],
        n_trajectories=args_dict["n_trajectories"],
        n_epochs=args_dict["n_iterations"],
        ridge=args_dict["ridge"],
        rho=args_dict["rho"],
        n_eval_episodes=args_dict["n_eval_episodes"],
    )
    behavior = make_behavior_policy(
        args_dict["behavior"], args_dict["eps"], args_dict["td_weight"])

    t0 = time.time()
    if solver == "two_stage":
        # Two-stage WITHOUT a world model: same ActionIndexedFeatures and
        # CE pretrain as DFL-LP, then skip the DFL gradient phase. The LP
        # is still solved every iter for evaluation, so iter-0 reward is
        # byte-identical to DFL-LP iter-0 (same seed, same pretrained
        # features, same LP solve).
        out = run_neural_dfl(
            cfg, verbose=False,
            bins_per_dim=args_dict["bins_per_dim"],
            tau=args_dict["tau"],
            use_dfl_phase=False,                       # ← skip DFL gradient
            pretrain_epochs=args_dict["pretrain_epochs_dfl"],
            eval_policy_type=args_dict["eval_policy"],
            behavior_policy=behavior,
            n_ce_steps=args_dict["n_ce_steps"],
            n_dfl_steps=args_dict["n_dfl_steps"],
            dfl_lr=args_dict["dfl_lr"],
            solver="lp",                               # eval via LP
        )
        elapsed = time.time() - t0
        rewards = [float(h["avg_reward"]) for h in out["history"]]
        return {
            "solver": solver,
            "seed": int(seed),
            "rewards": rewards,
            "total": float(elapsed),
            "solver_total": float(out["solver_total_time"]),
            "forward": float(out["solver_forward_time"]),
            "backward": float(out["backward_time"]),
        }

    if solver == "two_stage_mbrl":
        # MBRL: train world model + DQN. Kept for backward compatibility.
        out = run_two_stage_mbrl(
            cfg, verbose=False,
            bins_per_dim=args_dict["bins_per_dim"],
            tau=args_dict["tau"],
            world_epochs=args_dict.get("world_epochs", 400),
            world_lr=args_dict.get("world_lr", 1e-3),
            n_q_steps_per_iter=args_dict.get("mbrl_q_steps", 128),
            rollouts_per_iter=0,
            rollout_length=1,
            n_initial_synthetic=args_dict.get("mbrl_n_initial_synthetic", 0),
            q_hidden=tuple(args_dict.get("mbrl_q_hidden", (32, 32))),
        )
        elapsed = time.time() - t0
        rewards = [float(h["avg_reward"]) for h in out["history"]]
        return {
            "solver": solver,
            "seed": int(seed),
            "rewards": rewards,
            "total": float(elapsed),
            "solver_total": 0.0,   # MBRL has no LP/QP solver step
            "forward": 0.0,
            "backward": 0.0,
        }

    # DFL paths: lp / gurobi_qp / qpth / sketch
    if solver == "sketch":
        # Inject the sketched LP into train_cartpole's symbol table for
        # this worker only — we don't edit train_cartpole.py. The
        # sketched LP keeps the (1^T M) w = 1^T b anchor row through the
        # sketch (mirrors mdp_lp/solvers/sketched_lp.py), which is what
        # keeps the sketched problem bounded without resorting to QP
        # regularisation.
        from . import train_cartpole as tc
        from .sketched_solver import differentiable_sketched_lp

        n_sk = int(args_dict["n_sketches"])
        sk_ratio = float(args_dict["sketch_ratio"])
        sk_mode = str(args_dict["sketch_mode"])

        original_lp = tc.differentiable_reduced_lp

        def _sketched(M, c, b):
            return differentiable_sketched_lp(
                M, c, b,
                n_sketches=n_sk,
                sketch_ratio=sk_ratio,
                sketch_mode=sk_mode,
                sketch_seed=None,  # resample per call -> stochastic regulariser
            )

        tc.differentiable_reduced_lp = _sketched
        try:
            out = run_neural_dfl(
                cfg, verbose=False,
                bins_per_dim=args_dict["bins_per_dim"],
                tau=args_dict["tau"],
                pretrain_epochs=args_dict["pretrain_epochs_dfl"],
                eval_policy_type=args_dict["eval_policy"],
                behavior_policy=behavior,
                n_ce_steps=args_dict["n_ce_steps"],
                n_dfl_steps=args_dict["n_dfl_steps"],
                dfl_lr=args_dict["dfl_lr"],
                solver="lp",  # patched binding handles the sketching
            )
        finally:
            tc.differentiable_reduced_lp = original_lp
    else:
        out = run_neural_dfl(
            cfg, verbose=False,
            bins_per_dim=args_dict["bins_per_dim"],
            tau=args_dict["tau"],
            pretrain_epochs=args_dict["pretrain_epochs_dfl"],
            eval_policy_type=args_dict["eval_policy"],
            behavior_policy=behavior,
            n_ce_steps=args_dict["n_ce_steps"],
            n_dfl_steps=args_dict["n_dfl_steps"],
            dfl_lr=args_dict["dfl_lr"],
            solver=solver,
        )
    elapsed = time.time() - t0
    rewards = [float(h["avg_reward"]) for h in out["history"]]
    return {
        "solver": solver,
        "seed": int(seed),
        "rewards": rewards,
        "total": float(elapsed),
        "solver_total": float(out["solver_total_time"]),
        "forward": float(out["solver_forward_time"]),
        "backward": float(out["backward_time"]),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--solvers", type=str, nargs="+", default=["lp", "qpth"],
                   choices=["lp", "gurobi_qp", "qpth", "sketch",
                            "two_stage", "two_stage_mbrl"])
    p.add_argument("--n_workers", type=int, default=4)

    # Panel-3 anchor defaults — match make_figure1_cartpole / make_figure_3panel.
    p.add_argument("--n_iterations", type=int, default=60)
    p.add_argument("--n_features", type=int, default=1024)
    p.add_argument("--n_trajectories", type=int, default=100)
    p.add_argument("--n_eval_episodes", type=int, default=100)
    p.add_argument("--pretrain_epochs_ts", type=int, default=30,
                   help="Pretrain epochs for two-stage path; baked into the "
                        "canonical cache key for parity with make_figure1.")
    p.add_argument("--pretrain_epochs_dfl", type=int, default=30)
    p.add_argument("--n_ce_steps", type=int, default=0)
    p.add_argument("--n_dfl_steps", type=int, default=5)
    p.add_argument("--dfl_lr", type=float, default=None)

    # Sketch-specific knobs (only used when 'sketch' is in --solvers).
    p.add_argument("--n_sketches", type=int, default=4,
                   help="Number of independent row-sketches to ensemble-avg.")
    p.add_argument("--sketch_ratio", type=float, default=0.5,
                   help="k = round(sketch_ratio * m) sketched rows per replica.")
    p.add_argument("--sketch_mode", type=str, default="gaussian",
                   choices=["gaussian", "rademacher", "subsample"])

    # MBRL-specific knobs (only used when 'two_stage_mbrl' is in --solvers).
    p.add_argument("--world_epochs", type=int, default=400)
    p.add_argument("--world_lr", type=float, default=1e-3)
    p.add_argument("--mbrl_q_steps", type=int, default=128)
    p.add_argument("--mbrl_q_hidden", type=int, nargs="+", default=[32, 32])
    p.add_argument("--mbrl_n_initial_synthetic", type=int, default=0)

    p.add_argument("--behavior", type=str, default="biased",
                   choices=["expert", "noisy", "random",
                            "angle_only", "biased"])
    p.add_argument("--eps", type=float, default=0.5)
    p.add_argument("--td_weight", type=float, default=0.013)
    p.add_argument("--eval_policy", type=str, default="lp",
                   choices=["lp", "ce"])

    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--qp_eta", type=float, default=0.01)
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--rho", type=float, default=10.0)
    p.add_argument("--bins_per_dim", type=int, default=8)
    p.add_argument("--tau", type=float, default=0.5)

    p.add_argument("--results_dir", type=str, default=str(RESULTS_DIR),
                   help="Directory for per-solver JSON outputs. Default: "
                        "mdp_continuous/results/dfl_vs_qpth/")
    p.add_argument("--no_save", action="store_true",
                   help="Skip writing JSON results to disk.")
    return p.parse_args()


def main():
    args = parse_args()

    jobs = []
    for solver in args.solvers:
        for seed in args.seeds:
            jobs.append({
                "solver": solver, "seed": int(seed),
                "n_iterations": args.n_iterations,
                "n_features": args.n_features,
                "n_trajectories": args.n_trajectories,
                "n_eval_episodes": args.n_eval_episodes,
                "pretrain_epochs_ts": args.pretrain_epochs_ts,
                "pretrain_epochs_dfl": args.pretrain_epochs_dfl,
                "n_ce_steps": args.n_ce_steps,
                "n_dfl_steps": args.n_dfl_steps,
                "dfl_lr": args.dfl_lr,
                "behavior": args.behavior,
                "eps": args.eps,
                "td_weight": args.td_weight,
                "eval_policy": args.eval_policy,
                "gamma": args.gamma, "lr": args.lr,
                "qp_eta": args.qp_eta, "ridge": args.ridge, "rho": args.rho,
                "bins_per_dim": args.bins_per_dim, "tau": args.tau,
                "world_epochs": args.world_epochs,
                "world_lr": args.world_lr,
                "mbrl_q_steps": args.mbrl_q_steps,
                "mbrl_q_hidden": args.mbrl_q_hidden,
                "mbrl_n_initial_synthetic": args.mbrl_n_initial_synthetic,
                "n_sketches": args.n_sketches,
                "sketch_ratio": args.sketch_ratio,
                "sketch_mode": args.sketch_mode,
                "torch_threads": 1,
            })

    print(f"[run] {len(jobs)} jobs across {args.n_workers} workers "
          f"(solvers={args.solvers}, seeds={args.seeds})")
    print(f"[run] iters={args.n_iterations}  nfeat={args.n_features}  "
          f"behavior={args.behavior}  eval={args.eval_policy}")

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futures = {ex.submit(_run_one, j): j for j in jobs}
        for fut in as_completed(futures):
            r = fut.result()
            print(f"[done] solver={r['solver']:>10s} seed={r['seed']:>2d} "
                  f"best={max(r['rewards']):>6.2f} "
                  f"final={r['rewards'][-1]:>6.2f} "
                  f"total={r['total']:>6.1f}s "
                  f"solver={r['solver_total']:>5.1f}s")
            results.append(r)
    print(f"[run] {len(results)} jobs finished in {time.time() - t0:.1f}s")

    print()
    print(f"=== summary across {len(args.seeds)} seeds ===")
    hdr = (f"{'solver':>10s} {'total_mu':>9s} {'solver_mu':>10s} "
           f"{'best_mu':>8s} {'best_sd':>8s} {'final_mu':>9s}")
    print(hdr)

    results_dir = Path(args.results_dir)
    if not args.no_save:
        results_dir.mkdir(parents=True, exist_ok=True)

    for solver in args.solvers:
        rs = sorted([r for r in results if r["solver"] == solver],
                    key=lambda r: r["seed"])
        if not rs:
            continue
        bests = [max(r["rewards"]) for r in rs]
        finals = [r["rewards"][-1] for r in rs]
        totals = [r["total"] for r in rs]
        solvers_t = [r["solver_total"] for r in rs]
        print(f"{solver:>10s} {st.mean(totals):>8.1f}s "
              f"{st.mean(solvers_t):>9.1f}s "
              f"{st.mean(bests):>8.2f} "
              f"{st.pstdev(bests):>8.2f} "
              f"{st.mean(finals):>9.2f}")

        if not args.no_save:
            # Save under the canonical name make_figure1_cartpole would have
            # used → make_figures.py picks these up without renaming.
            payload = {
                "solver": solver,
                "seeds": [r["seed"] for r in rs],
                "rewards_per_seed": [r["rewards"] for r in rs],
                "times_per_seed": [r["total"] for r in rs],
                "solver_times_per_seed": [r["solver_total"] for r in rs],
                "forward_times_per_seed": [r["forward"] for r in rs],
                "backward_times_per_seed": [r["backward"] for r in rs],
            }
            cache_name = _canonical_cache_key(solver, args)
            out_path = results_dir / f"{cache_name}.json"
            with open(out_path, "w") as f:
                json.dump(payload, f)
            print(f"           saved → {out_path.name}")


if __name__ == "__main__":
    main()
