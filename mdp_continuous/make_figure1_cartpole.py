import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from .configs import CartPoleConfig
from .train_cartpole import run_neural_dfl, run_two_stage_mbrl, set_seed
from .cartpole_env import (
    CartPoleDynamics,
    collect_trajectories,
    heuristic_policy,
    random_policy,
    epsilon_greedy_policy,
)
from .world_model import train_world_model


# ── Cache layout ────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "results" / "dfl_vs_qpth"


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}.json"


def _load_or_none(name: str):
    p = _cache_path(name)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def _save(name: str, data):
    with open(_cache_path(name), "w") as f:
        json.dump(data, f)


# ── Behavior policies (must be picklable for multiprocessing) ──────────

def _angle_only_policy(states):
    return (states[:, 2] > 0).long()


class _BiasedPolicy:
    """Biased PD controller — picklable so it crosses process boundaries."""
    def __init__(self, theta_dot_weight: float):
        self.td_w = float(theta_dot_weight)

    def __call__(self, states):
        return ((states[:, 2] + self.td_w * states[:, 3]) > 0).long()


def make_behavior_policy(kind: str, eps: float, td_weight: float):
    if kind == "expert":
        return heuristic_policy
    if kind == "noisy":
        return epsilon_greedy_policy(heuristic_policy, epsilon=eps)
    if kind == "random":
        return random_policy
    if kind == "angle_only":
        return _angle_only_policy
    if kind == "biased":
        return _BiasedPolicy(td_weight)
    raise ValueError(kind)


# ── Single-seed runners (top-level => picklable) ──────────────────────

def _run_one_seed(args_dict: dict) -> dict:
    """Run one (algo, seed) configuration. Returns a small JSON-able dict.

    Each worker re-imports torch and re-seeds, so runs are deterministic
    and isolated. Workers are CPU-only.
    """
    import sys
    sys.stdout.reconfigure(line_buffering=True)

    algo = args_dict["algo"]
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

    if algo == "neural_dfl":
        runner = run_neural_dfl
        pretrain = args_dict["pretrain_epochs_dfl"]
        runner_kwargs = {
            "solver": "lp",
            "sa_K": args_dict.get("sa_K", 8),
            "sa_ratio": args_dict.get("sa_ratio", 0.2),
            # Defaults reproduce the paper's Eq.(5) penalty loss exactly.
            "loss_mode": args_dict.get("loss_mode", "penalty"),
            "dual_weight": args_dict.get("dual_weight", 1.0),
            "dual_feasible_only": args_dict.get("dual_feasible_only", False),
        }
    elif algo == "dfl_qpth":
        # Historical name — actually runs Gurobi-QP, not the qpth library.
        # Kept as-is so existing caches/figures stay valid; the panel-3
        # bar labelled "DFL-QP" has always meant the Gurobi-QP path.
        runner = run_neural_dfl
        pretrain = args_dict["pretrain_epochs_dfl"]
        runner_kwargs = {"solver": "gurobi_qp"}
    elif algo == "two_stage_mbrl":
        runner = run_two_stage_mbrl
        pretrain = 0
        runner_kwargs = {
            "world_epochs": args_dict.get("world_epochs", 400),
            "world_lr": args_dict.get("world_lr", 1e-3),
            "n_q_steps_per_iter": args_dict.get("mbrl_q_steps", 128),
            "rollouts_per_iter": 0,
            "rollout_length": 1,
            "n_initial_synthetic": args_dict.get("mbrl_n_initial_synthetic", 0),
            # Capacity-limit the Q-net so MBRL plateaus clearly below DFL —
            # offline FQI on this buffer with a 128×128 Q-net can hit ≥195
            # on lucky seeds, which muddies the DFL-vs-baseline comparison.
            "q_hidden": args_dict.get("mbrl_q_hidden", (32, 32)),
        }
    else:
        raise ValueError(algo)

    runner_extras = {}
    if args_dict.get("use_world_model", False):
        device = torch.device("cpu")
        dynamics = CartPoleDynamics(cfg)
        buffer = collect_trajectories(dynamics, behavior,
                                      cfg.n_trajectories, cfg.max_steps, device)
        wm = train_world_model(buffer,
                               n_epochs=args_dict["world_epochs"],
                               lr=args_dict["world_lr"],
                               verbose=False)
        runner_extras = {"data": buffer, "world_model": wm}

    t0 = time.time()
    out = runner(
        cfg,
        verbose=True,
        bins_per_dim=args_dict["bins_per_dim"],
        tau=args_dict["tau"],
        pretrain_epochs=pretrain,
        eval_policy_type=args_dict["eval_policy"],
        behavior_policy=behavior,
        n_ce_steps=args_dict["n_ce_steps"],
        n_dfl_steps=args_dict["n_dfl_steps"],
        dfl_lr=args_dict["dfl_lr"],
        **runner_kwargs,
        **runner_extras,
    )
    total_time = time.time() - t0
    rewards = [float(h["avg_reward"]) for h in out["history"]]
    lengths = [float(h["avg_length"]) for h in out["history"]]
    return {
        "algo": algo,
        "seed": int(seed),
        "rewards": rewards,
        "lengths": lengths,
        "total_time": float(total_time),
        "solver_total_time": float(out.get("solver_total_time", 0.0)),
    }


def _job(args, algo: str, seed: int) -> dict:
    return {
        "algo": algo,
        "seed": int(seed),
        "n_iterations": args.n_iterations,
        "gamma": args.gamma,
        "lr": args.lr,
        "qp_eta": args.qp_eta,
        "n_features": args.n_features,
        "n_trajectories": args.n_trajectories,
        "ridge": args.ridge,
        "rho": args.rho,
        "n_eval_episodes": args.n_eval_episodes,
        "bins_per_dim": args.bins_per_dim,
        "tau": args.tau,
        "behavior": args.behavior,
        "eps": args.eps,
        "td_weight": args.td_weight,
        "pretrain_epochs_ts": args.pretrain_epochs_ts,
        "pretrain_epochs_dfl": args.pretrain_epochs_dfl,
        "eval_policy": args.eval_policy,
        "n_ce_steps": args.n_ce_steps,
        "n_dfl_steps": args.n_dfl_steps,
        "dfl_lr": args.dfl_lr,
        "use_world_model": bool(args.use_world_model),
        "world_epochs": args.world_epochs,
        "world_lr": args.world_lr,
        "sa_K": int(getattr(args, "sa_K", 8)),
        "sa_ratio": float(getattr(args, "sa_ratio", 0.2)),
        # DFL loss variant (see train_cartpole.run_neural_dfl). The defaults
        # are the paper's Eq.(5) penalty loss, so untouched callers are
        # byte-identical.
        "loss_mode": str(getattr(args, "loss_mode", "penalty")),
        "dual_weight": float(getattr(args, "dual_weight", 1.0)),
        "dual_feasible_only": bool(getattr(args, "dual_feasible_only", False)),
        "torch_threads": 1,
    }


def _cache_key(prefix: str, args) -> str:
    dfl_lr_tag = "auto" if args.dfl_lr is None else f"{args.dfl_lr}"
    wm_tag = (f"_wm{args.world_epochs}-{args.world_lr}"
              if args.use_world_model else "")
    qpth_tag = f"_qpeta{args.qp_eta}" if "qpth" in prefix else ""
    # DFL loss variant, mirroring run_dfl_vs_qpth._canonical_cache_key so the
    # figure code can locate auglag runs. Empty for the paper's Eq.(5) penalty
    # loss, so every pre-existing cache stays reachable under its original name.
    lm = str(getattr(args, "loss_mode", "penalty"))
    lm_tag = "" if lm == "penalty" else f"_lm{lm}"
    if lm in ("lagrangian", "auglag"):
        if float(getattr(args, "dual_weight", 1.0)) != 1.0:
            lm_tag += f"_dw{float(args.dual_weight)}"
        if bool(getattr(args, "dual_feasible_only", False)):
            lm_tag += "_dfo"
    if float(getattr(args, "rho", 10.0)) != 10.0:
        lm_tag += f"_rho{float(args.rho)}"
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
        f"{lm_tag}"
        f"{wm_tag}"
    )


def run_mbrl_curve(args):
    """Run two_stage_mbrl × all seeds in parallel; cache + return payload.

    MBRL doesn't use the LP feature parameterisation, so its result is
    invariant to args.n_features. We still include n_features in the
    cache key for parity with the other curves.
    """
    name = _cache_key("curve_two_stage_mbrl", args)
    cached = _load_or_none(name) if not args.force else None
    if cached:
        return cached

    jobs = [_job(args, "two_stage_mbrl", s) for s in args.seeds]
    print(f"[run-mbrl] {len(jobs)} jobs across {args.n_workers} workers")
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futures = {ex.submit(_run_one_seed, j): j for j in jobs}
        for fut in as_completed(futures):
            r = fut.result()
            print(f"[done] algo={r['algo']:<16s} seed={r['seed']} "
                  f"best={max(r['rewards']):.2f} "
                  f"final={r['rewards'][-1]:.2f}")
            results.append(r)
    print(f"[run-mbrl] {len(results)} jobs finished in {time.time() - t0:.1f}s")

    runs = sorted(results, key=lambda r: r["seed"])
    payload = {
        "seeds": [r["seed"] for r in runs],
        "rewards_per_seed": [r["rewards"] for r in runs],
        "lengths_per_seed": [r["lengths"] for r in runs],
        "times_per_seed": [r.get("total_time") for r in runs],
        "solver_times_per_seed":
            [r.get("solver_total_time", 0.0) for r in runs],
    }
    _save(name, payload)
    return payload


def run_all_curves(args):
    """Run methods x all seeds in parallel; return (neural[, qpth])."""
    name_neural = _cache_key("curve_neural", args)
    name_qpth = _cache_key("curve_qpth", args)
    neural_cached = _load_or_none(name_neural) if not args.force else None
    qpth_cached = (_load_or_none(name_qpth)
                   if (args.include_qpth and not args.force) else None)

    need_qpth = args.include_qpth and not qpth_cached
    if neural_cached and not need_qpth:
        return neural_cached, qpth_cached

    jobs = []
    if not neural_cached:
        for s in args.seeds:
            jobs.append(_job(args, "neural_dfl", s))
    if need_qpth:
        for s in args.seeds:
            jobs.append(_job(args, "dfl_qpth", s))

    print(f"[run] {len(jobs)} jobs across {args.n_workers} workers")
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futures = {ex.submit(_run_one_seed, j): j for j in jobs}
        for fut in as_completed(futures):
            j = futures[fut]
            r = fut.result()
            print(f"[done] algo={r['algo']:<10s} seed={r['seed']} "
                  f"best={max(r['rewards']):.2f} "
                  f"final={r['rewards'][-1]:.2f}")
            results.append(r)
    print(f"[run] {len(results)} jobs finished in {time.time() - t0:.1f}s")

    neural_runs = sorted([r for r in results if r["algo"] == "neural_dfl"],
                         key=lambda r: r["seed"])
    qpth_runs = sorted([r for r in results if r["algo"] == "dfl_qpth"],
                       key=lambda r: r["seed"])

    if not neural_cached and neural_runs:
        neural_cached = {
            "seeds": [r["seed"] for r in neural_runs],
            "rewards_per_seed": [r["rewards"] for r in neural_runs],
            "lengths_per_seed": [r["lengths"] for r in neural_runs],
            "times_per_seed": [r.get("total_time") for r in neural_runs],
            "solver_times_per_seed":
                [r.get("solver_total_time", 0.0) for r in neural_runs],
        }
        _save(name_neural, neural_cached)
    if need_qpth and qpth_runs:
        qpth_cached = {
            "seeds": [r["seed"] for r in qpth_runs],
            "rewards_per_seed": [r["rewards"] for r in qpth_runs],
            "lengths_per_seed": [r["lengths"] for r in qpth_runs],
            "times_per_seed": [r.get("total_time") for r in qpth_runs],
            "solver_times_per_seed":
                [r.get("solver_total_time", 0.0) for r in qpth_runs],
        }
        _save(name_qpth, qpth_cached)

    return neural_cached, qpth_cached


# ── Plotting ────────────────────────────────────────────────────────────

COLORS = {
    "neural_dfl":     "#1f77b4",   # blue
    "two_stage_mbrl": "#ff7f0e",   # orange
    "dfl_qpth":       "#2ca02c",   # green
}


def _running_max(arr2d: np.ndarray) -> np.ndarray:
    """Per-seed cummax — what the algorithm would *return* (best-so-far)."""
    return np.maximum.accumulate(arr2d, axis=1)


def _stack_curves(runs):
    if not runs:
        return None, None
    L = min(len(r) for r in runs)
    arr = np.array([r[:L] for r in runs], dtype=float)
    return arr, L


def _smooth(arr1d: np.ndarray, w: int) -> np.ndarray:
    """Centered rolling mean over a 1d array, edges shrunk to available window."""
    if w <= 1:
        return arr1d
    n = arr1d.shape[0]
    out = np.empty(n)
    for i in range(n):
        lo = max(0, i - w // 2)
        hi = min(n, i + w // 2 + 1)
        out[i] = arr1d[lo:hi].mean()
    return out


def _draw_curve(ax, arr, color, ls, label, smooth_w=1):
    """Plot mean ± SE across seeds, optionally smoothed."""
    mu = arr.mean(axis=0)
    se = arr.std(axis=0) / np.sqrt(max(arr.shape[0], 1))
    xs = np.arange(arr.shape[1])
    if smooth_w > 1:
        mu = _smooth(mu, smooth_w)
        se = _smooth(se, smooth_w)
    ax.plot(xs, mu, color=color, lw=2.0, ls=ls, label=label)
    ax.fill_between(xs, mu - se, mu + se, color=color, alpha=0.20)


def plot_figure1(args, save_path: str):
    neural = _load_or_none(_cache_key("curve_neural", args))
    mbrl = _load_or_none(_cache_key("curve_two_stage_mbrl", args))
    qpth = (_load_or_none(_cache_key("curve_qpth", args))
            if args.include_qpth else None)
    requested = set(int(s) for s in args.seeds)

    def _filtered_arr(payload):
        if not payload:
            return None
        runs = [r for s, r in zip(payload["seeds"], payload["rewards_per_seed"])
                if int(s) in requested]
        if not runs:
            return None
        arr, _ = _stack_curves(runs)
        return arr

    neural_arr = _filtered_arr(neural)
    mbrl_arr = _filtered_arr(mbrl)
    qpth_arr = _filtered_arr(qpth)

    series = [
        ("DFL", "neural_dfl", neural_arr, "-"),
        ("Two-stage MBRL", "two_stage_mbrl", mbrl_arr, "--"),
    ]
    if qpth_arr is not None:
        series.append(("DFL-qpth", "dfl_qpth", qpth_arr, "-."))


    suptitle = (f"CartPole DFL vs. Two-stage MBRL")

    if args.two_panel:
        # Two-panel: training dynamics (smoothed per-iter) + deployable max.
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4), sharey=True)
        for ax, mode, title in [
            (axes[0], "per_iter",
             f"Training dynamics (per-iter, {args.smooth}-iter MA)"),
            (axes[1], "running_max",
             "Deployable model quality (running max)"),
        ]:
            for label, key, arr, ls in series:
                if arr is None:
                    continue
                a = _running_max(arr) if mode == "running_max" else arr
                w = 1 if mode == "running_max" else max(int(args.smooth), 1)
                _draw_curve(ax, a, COLORS[key], ls, label, smooth_w=w)
            ax.set_xlabel("Training iteration")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 210)
        axes[0].set_ylabel("Avg. episode reward")
        axes[1].legend(loc="lower right", fontsize=9, framealpha=0.9)
        fig.suptitle(suptitle, fontsize=11)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.2))
        for label, key, arr, ls in series:
            if arr is None:
                continue
            a = _running_max(arr) if args.show_running_max else arr
            w = 1 if args.show_running_max else max(int(args.smooth), 1)
            _draw_curve(ax, a, COLORS[key], ls, label, smooth_w=w)
        ax.set_xlabel("Training iteration")
        ylab = ("(best-so-far)" if args.show_running_max
                else f"(per iter, {args.smooth}-iter MA)" if args.smooth > 1
                else "(per iter)")
        ax.set_ylabel(f"Avg. episode reward")
        ax.set_title(suptitle, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 210)
        ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved figure to {save_path} and {pdf_path}")


def print_summary(args):
    neural = _load_or_none(_cache_key("curve_neural", args))
    mbrl = _load_or_none(_cache_key("curve_two_stage_mbrl", args))
    qpth = (_load_or_none(_cache_key("curve_qpth", args))
            if args.include_qpth else None)
    if not (neural and mbrl):
        return

    requested = set(int(s) for s in args.seeds)

    def _filter(payload):
        return [r for s, r in zip(payload["seeds"], payload["rewards_per_seed"])
                if int(s) in requested]

    def _stats(runs):
        bests = [max(r) for r in runs]
        finals = [r[-1] for r in runs]
        return (np.mean(bests), np.std(bests),
                np.mean(finals), np.std(finals))

    nb_m, nb_s, nf_m, nf_s = _stats(_filter(neural))
    mb_m, mb_s, mf_m, mf_s = _stats(_filter(mbrl))

    print(f"\n[Eval/Summary]  n_seeds={len(args.seeds)}")
    print(f"[Eval]  twostage_mbrl  best={mb_m:.1f}±{mb_s:.1f}  final={mf_m:.1f}±{mf_s:.1f}")
    print(f"[Eval]  neural_dfl     best={nb_m:.1f}±{nb_s:.1f}  final={nf_m:.1f}±{nf_s:.1f}")
    if qpth:
        qb_m, qb_s, qf_m, qf_s = _stats(_filter(qpth))
        print(f"[Eval]  dfl_qpth       best={qb_m:.1f}±{qb_s:.1f}  final={qf_m:.1f}±{qf_s:.1f}")
        print(f"[Eval]  dfl_qpth−mbrl gap (best): {qb_m - mb_m:+.2f}")
    print(f"[Eval]  dfl−mbrl gap   best={nb_m - mb_m:+.2f}  final={nf_m - mf_m:+.2f}")


# ── CLI ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Figure 1 for offline CartPole DFL")
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[1, 2, 3, 4, 5, 6, 7, 8])
    p.add_argument("--n_workers", type=int, default=8,
                   help="Parallel workers; one process per (algo, seed).")

    # Offline data + behavior policy
    p.add_argument("--behavior", type=str, default="biased",
                   choices=["expert", "noisy", "random",
                            "angle_only", "biased"])
    p.add_argument("--eps", type=float, default=0.5)
    p.add_argument("--td_weight", type=float, default=0.013,
                   help="theta_dot weight in biased heuristic. "
                        "Expert uses 0.5; smaller = more suboptimal. "
                        "0.013 gives behavior-policy reward ~122.")
    p.add_argument("--n_trajectories", type=int, default=100)

    # Per-method pretrain budgets
    p.add_argument("--pretrain_epochs_ts", type=int, default=100,
                   help="CE pretrain epochs for two-stage (let it converge).")
    p.add_argument("--pretrain_epochs_dfl", type=int, default=30,
                   help="CE pretrain epochs for DFL (leave room for LP).")

    # Iteration budget
    p.add_argument("--n_iterations", type=int, default=16)
    p.add_argument("--n_eval_episodes", type=int, default=100)
    p.add_argument("--n_ce_steps", type=int, default=0,
                   help="0 in offline mode to avoid LP-policy relabel collapse")
    p.add_argument("--n_dfl_steps", type=int, default=5)
    p.add_argument("--dfl_lr", type=float, default=None,
                   help="LR for DFL-phase optimizer (Adam over features+head). "
                        "Lower than cfg.lr keeps features close to BC basin and "
                        "stabilizes late-iteration behavior. None = use cfg.lr.")

    # Other
    p.add_argument("--n_features", type=int, default=128)
    p.add_argument("--bins_per_dim", type=int, default=8)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--qp_eta", type=float, default=0.01)
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--rho", type=float, default=10.0)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--eval_policy", type=str, default="lp",
                   choices=["ce", "lp"])

    # Model-based variant: replace observed (r, s') with predictions from a
    # world model fit on the same offline buffer. Useful for predict-then-
    # optimize comparisons against the true-MDP figure-1 baseline.
    p.add_argument("--use_world_model", action="store_true",
                   help="Fit a world model and use its (r̂, ŝ') in the LP. "
                        "(s, a) and CE pretrain still use the real buffer.)")
    p.add_argument("--world_epochs", type=int, default=400)
    p.add_argument("--world_lr", type=float, default=1e-3)

    p.add_argument("--save_path", type=str,
                   default=str(HERE / "figure1_cartpole.pdf"))
    p.add_argument("--show_running_max", action="store_true",
                   help="Plot per-seed best-so-far instead of raw per-iter "
                        "reward (matches what the algorithm actually returns "
                        "via best-so-far restoration).")
    p.add_argument("--two_panel", action="store_true",
                   help="Show both panels: smoothed per-iter dynamics (left) "
                        "and deployable running-max (right). Most honest view.")
    p.add_argument("--smooth", type=int, default=1,
                   help="Window size (in iters) for centered moving-average "
                        "smoothing on the per-iter view. Default 1 = no smoothing.")
    p.add_argument("--include_qpth", action="store_true",
                   help="Also run a DFL-qpth variant (LP solver replaced by "
                        "qpth's batched QP with quadratic regularization "
                        "(eta/2)||w||^2, eta=cfg.qp_eta).")
    p.add_argument("--plot_only", action="store_true",
                   help="Skip experiments, rebuild plot from cache")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if cache exists")
    p.add_argument("--quick", action="store_true",
                   help="Smoke test: small seeds/iters")
    return p.parse_args()


def main():
    args = parse_args()

    if args.quick:
        args.seeds = [1, 2]
        args.n_iterations = 8
        args.pretrain_epochs_ts = 30
        args.pretrain_epochs_dfl = 15
        args.n_workers = min(args.n_workers, 4)

    if not args.plot_only:
        run_all_curves(args)
        run_mbrl_curve(args)

    plot_figure1(args, save_path=args.save_path)
    print_summary(args)


if __name__ == "__main__":
    main()
