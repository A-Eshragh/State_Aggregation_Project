from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import types
from argparse import Namespace
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from resource_profile import ResourceMonitor


RAW_SEEDS = [360, 606, 689, 665, 790, 310, 582, 962, 246, 297]


def configure_cliff_grid(*, n_rows: int = 4, n_cols: int = 120) -> None:
    """Configure the Cliff MDP before importing data/train modules.

    The original paper Cliff task is 4x12 = 48 states. This experimental
    runner uses a wider 4x120 grid, i.e. 480 states, while keeping the same
    action space and slip model.
    """
    import mdp_cw.cliff_mdp as cm

    if n_rows != 4:
        raise ValueError("This runner currently keeps the original 4-row Cliff geometry.")
    if n_cols < 12:
        raise ValueError("n_cols should be at least 12.")

    cm.N_ROWS = int(n_rows)
    cm.N_COLS = int(n_cols)
    cm.N_STATES = cm.N_ROWS * cm.N_COLS
    cm.START_ROW, cm.START_COL = cm.N_ROWS - 1, 0
    cm.GOAL_ROW, cm.GOAL_COL = cm.N_ROWS - 1, cm.N_COLS - 1
    cm.CLIFF_COLS = list(range(1, cm.N_COLS - 1))
    cm.START_STATE = cm.state_idx(cm.START_ROW, cm.START_COL)
    cm.GOAL_STATE = cm.state_idx(cm.GOAL_ROW, cm.GOAL_COL)
    cm.CLIFF_STATES = [cm.state_idx(cm.START_ROW, c) for c in cm.CLIFF_COLS]
    cm.NEXT_IDX, cm.REW_PER_SLIP = cm._build_next_and_reward_tables()


def disable_per_run_plotting() -> None:
    module = types.ModuleType("mdp_cw.plot_cliff_run")
    module.make_all_plots = lambda save_dir: None
    sys.modules["mdp_cw.plot_cliff_run"] = module


def import_cliff_stack():
    from mdp_cw.prepare_cliff_data import CliffDataConfig, prepare_cliff_data
    from mdp_cw.train_cliff import train_eval_cliff

    return CliffDataConfig, prepare_cliff_data, train_eval_cliff


def method_config(method: str, n_agg: int, tau: float, noise: float, samples: int) -> dict:
    if method == "ts":
        return {"algo": "twostage", "ssa_objective": "none", "n_agg": -1}
    if method == "lp":
        return {"algo": "dfl", "ssa_objective": "none", "n_agg": -1}
    if method == "qp":
        return {"algo": "dfl_qpth", "ssa_objective": "none", "n_agg": -1}
    if method == "sketch":
        return {"algo": "dfl_sketch", "ssa_objective": "none", "n_agg": -1}
    if method == "sa":
        return {
            "algo": "dfl_ssa",
            "ssa_objective": "cw",
            "n_agg": int(n_agg),
            "sa_tau": float(tau),
            "stochastic_membership_samples": 1,
            "stochastic_membership_noise": 0.0,
        }
    if method == "stoch_sa":
        return {
            "algo": "dfl_ssa",
            "ssa_objective": "stochastic_softagg",
            "n_agg": int(n_agg),
            "sa_tau": float(tau),
            "stochastic_membership_samples": int(samples),
            "stochastic_membership_noise": float(noise),
        }
    raise ValueError(f"Unknown method: {method}")


def make_args(
    *,
    method: str,
    seed_index: int,
    out: Path,
    n_samples: int,
    n_epochs: int,
    pretrain_epochs: int,
    n_agg: int,
    tau: float,
    noise: float,
    samples: int,
    n_sketches: int,
    sketch_ratio: float,
) -> Namespace:
    cfg = method_config(method, n_agg, tau, noise, samples)
    return Namespace(
        algo=str(cfg["algo"]),
        n_samples=int(n_samples),
        data_feat_dim=8,
        feat_dim=32,
        noise_std=2.5,
        train_frac=0.5,
        batch_size=32,
        seed=RAW_SEEDS[seed_index],
        regime_set="v4",
        center_scale=0.88,
        ambiguous_pairs=[(0, 2)],
        ambiguous_gap=0.0,
        cliff_penalty=-50.0,
        regime_weights=[15.0, 1.0, 1.0, 1.0],
        test_regime_weights=[0.0, 0.0, 1.0, 0.0],
        hidden_dim=64,
        nn_feat_dim=32,
        lr=0.003,
        pretrain_lr=0.01,
        pretrain_epochs=int(pretrain_epochs),
        n_epochs=int(n_epochs),
        grad_clip=10.0,
        beta=0.95,
        qp_reg=0.1,
        qp_max_iter=20,
        loss_weight=500.0,
        aug_lag_rho=0.1,
        dfl_perturb_std=0.0,
        qpth_warmup_epochs=0,
        n_agg=int(cfg["n_agg"]),
        sa_tau=float(cfg.get("sa_tau", tau)),
        sa_r_embed=8,
        n_sketches=int(n_sketches),
        sketch_ratio=float(sketch_ratio),
        ssa_objective=str(cfg.get("ssa_objective", "none")),
        cw_agg_mode="dest_r_bias",
        agg_lr_ratio=1.0,
        lp_basis_ridge=0.0,
        hinge_weight_alpha=0.0,
        hard_mine_every=0,
        margin_lambda=0.0,
        sketch_use_qpth=False,
        stochastic_membership_samples=int(cfg.get("stochastic_membership_samples", 1)),
        stochastic_membership_noise=float(cfg.get("stochastic_membership_noise", 0.0)),
        save_dir=out,
    )


def make_data(args: Namespace):
    CliffDataConfig, prepare_cliff_data, _ = import_cliff_stack()
    config = CliffDataConfig(
        n_samples=args.n_samples,
        feat_dim=args.data_feat_dim,
        noise_std=args.noise_std,
        beta=args.beta,
        train_frac=args.train_frac,
        batch_size=args.batch_size,
        seed=args.seed,
        regime_set=args.regime_set,
        center_scale=args.center_scale,
        ambiguous_pairs=args.ambiguous_pairs,
        ambiguous_gap=args.ambiguous_gap,
        cliff_penalty=args.cliff_penalty,
        regime_weights=args.regime_weights,
        test_regime_weights=args.test_regime_weights,
    )
    return prepare_cliff_data(config)


def label_for(method: str, n_agg: int, tau: float, noise: float, samples: int) -> str:
    def f(x: float) -> str:
        return str(x).replace(".", "p").replace("-", "m")

    if method in {"ts", "lp", "qp", "sketch"}:
        return method
    if method == "sa":
        return f"sa_M{n_agg}_tau{f(tau)}"
    return f"stoch_M{n_agg}_tau{f(tau)}_k{samples}_noise{f(noise)}"


def run_one(args, *, method: str, seed_index: int, n_agg: int, tau: float, noise: float, samples: int) -> dict:
    label = label_for(method, n_agg, tau, noise, samples)
    out = args.root / label / f"seed_{seed_index}"
    result_path = out / "results.json"
    if result_path.exists() and not args.force:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["status"] = "skip"
        return result
    if args.force and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    train_args = make_args(
        method=method,
        seed_index=seed_index,
        out=out,
        n_samples=args.n_samples,
        n_epochs=args.n_epochs,
        pretrain_epochs=args.pretrain_epochs,
        n_agg=n_agg,
        tau=tau,
        noise=noise,
        samples=samples,
        n_sketches=args.n_sketches,
        sketch_ratio=args.sketch_ratio,
    )
    config_dump = {
        "grid": {"n_rows": args.n_rows, "n_cols": args.n_cols, "n_states": args.n_rows * args.n_cols},
        "method": method,
        "label": label,
        "train_args": vars(train_args),
    }
    (out / "large_cliff_config.json").write_text(json.dumps(config_dump, indent=2, default=str), encoding="utf-8")
    _, _, train_eval_cliff = import_cliff_stack()

    t0 = time.time()
    with ResourceMonitor(args.profile_interval_sec) as monitor:
        with (out / "run.log").open("w", encoding="utf-8") as f:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = f
            sys.stderr = f
            try:
                disable_per_run_plotting()
                train_loader, test_loader, meta = make_data(train_args)
                result = train_eval_cliff(train_args, train_loader, test_loader, meta)
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
    result["status"] = "ok"
    result["method"] = method
    result["label"] = label
    result["seed_index"] = seed_index
    result["raw_seed"] = train_args.seed
    result["n_states"] = int(args.n_rows * args.n_cols)
    result["wall_time_sec"] = time.time() - t0
    result["resource_profile"] = {
        **monitor.result().to_dict(),
        "runner": "cliff_large_state_impl",
        "n_actions": 4,
        "n_state_action_variables": int(args.n_rows * args.n_cols * 4),
    }
    result_path.write_text(json.dumps(jsonable(result), indent=2), encoding="utf-8")
    return result


def jsonable(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def summarize(root: Path) -> dict:
    groups: dict[str, list[dict]] = {}
    for p in sorted(root.glob("*/*/results.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        groups.setdefault(str(d["label"]), []).append(d)
    summary = {}
    for label, rows in groups.items():
        regrets = np.asarray([float(r["test_mean_regret"]) for r in rows], dtype=float)
        summary[label] = {
            "n": len(rows),
            "method": rows[0].get("method"),
            "n_states": int(rows[0].get("n_states", 0)),
            "regret_mean": float(regrets.mean()),
            "regret_se": float(regrets.std(ddof=1) / np.sqrt(len(regrets))) if len(regrets) > 1 else 0.0,
            "train_time_mean_sec": float(np.mean([float(r.get("train_time", [0])[-1]) for r in rows])),
            "solver_time_mean_sec": float(np.mean([float(np.sum(r.get("solver_time_sec_per_epoch", []))) for r in rows])),
            "sa_time_mean_sec": float(np.mean([float(np.sum(r.get("sa_time_sec_per_epoch", []))) for r in rows])),
            "rows": [
                {
                    "seed_index": int(r["seed_index"]),
                    "test_mean_regret": float(r["test_mean_regret"]),
                    "test_mean_pred_reward": float(r["test_mean_pred_reward"]),
                    "test_mean_oracle_reward": float(r["test_mean_oracle_reward"]),
                    "wall_time_sec": float(r.get("wall_time_sec", 0.0)),
                }
                for r in rows
            ],
        }
    return summary


def parse_csv_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def parse_csv_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO / "notebooks/reproduction/repro/results/cliff_large_state")
    parser.add_argument("--n-rows", type=int, default=4)
    parser.add_argument("--n-cols", type=int, default=120)
    parser.add_argument("--methods", nargs="+", default=["sa", "stoch_sa"])
    parser.add_argument("--seeds", type=parse_csv_ints, default=parse_csv_ints("0"))
    parser.add_argument("--n-aggs", type=parse_csv_ints, default=parse_csv_ints("24,48"))
    parser.add_argument("--taus", type=parse_csv_floats, default=parse_csv_floats("0.5"))
    parser.add_argument("--stoch-samples", type=int, default=2)
    parser.add_argument("--stoch-noises", type=parse_csv_floats, default=parse_csv_floats("0.15"))
    parser.add_argument("--n-sketches", type=int, default=8)
    parser.add_argument("--sketch-ratio", type=float, default=0.1)
    parser.add_argument("--profile-interval-sec", type=float, default=0.1)
    parser.add_argument("--n-samples", type=int, default=80)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    configure_cliff_grid(n_rows=args.n_rows, n_cols=args.n_cols)
    args.root.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        for method in args.methods:
            for seed_index in args.seeds:
                n_aggs = [-1] if method in {"ts", "lp", "qp", "sketch"} else args.n_aggs
                for n_agg in n_aggs:
                    for tau in args.taus:
                        noises = args.stoch_noises if method == "stoch_sa" else [0.0]
                        for noise in noises:
                            samples = args.stoch_samples if method == "stoch_sa" else 1
                            print(
                                f"[large-cliff] states={args.n_rows * args.n_cols} "
                                f"{method} seed={seed_index} M={n_agg} tau={tau} noise={noise}",
                                flush=True,
                            )
                            row = run_one(
                                args,
                                method=method,
                                seed_index=seed_index,
                                n_agg=n_agg,
                                tau=tau,
                                noise=noise,
                                samples=samples,
                            )
                            print(f"  {row['status']} regret={row.get('test_mean_regret')}", flush=True)

    summary = summarize(args.root)
    (args.root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
