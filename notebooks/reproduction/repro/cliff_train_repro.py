from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import types
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from qpth.qp import QPFunction, QPSolvers


REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mdp_cw.features_model import CliffTransitionModel, FeatureNet, TransitionHead
from mdp_cw.prepare_cliff_data import CliffDataConfig, prepare_cliff_data
from mdp_cw.train_cliff import (
    _build_gamma,
    _solve_with_algo,
    _start_state,
    eval_y_on_true_mdp,
    train_eval_cliff,
)
from mdp_lp.solvers.dense_lp_solver import DenseLPSolver


RESULTS_ROOT = REPO / "notebooks" / "reproduction" / "repro" / "results" / "cliff"
HISTORICAL_COMMIT = "d606c3f"
HISTORICAL_TREE = "mdp_cw/results_random_all5_k15_sketchstab_indexed/k15"
RAW_SEEDS = [360, 606, 689, 665, 790, 310, 582, 962, 246, 297]


def disable_per_run_plotting() -> None:
    """Keep the reproduction runner headless.

    train_eval_cliff saves JSON/checkpoints and then tries to create per-run
    diagnostic plots. The notebook creates the paper-facing figures separately,
    so the per-run plot hook is disabled here to avoid Tk backend failures in
    worker threads on Windows.
    """
    module = types.ModuleType("mdp_cw.plot_cliff_run")
    module.make_all_plots = lambda save_dir: None
    sys.modules["mdp_cw.plot_cliff_run"] = module


def parse_weights(text: str | None):
    text = (text or "").strip()
    return [float(x) for x in text.split(",")] if text else None


def parse_pairs(text: str | None):
    text = (text or "").strip()
    if not text:
        return None
    return [tuple(map(int, token.split(":"))) for token in text.split(",")]


def ensure_historical_tree(out_root: Path, repo_root: Path = REPO) -> Path:
    target = out_root / "historical_checkpoint_tree" / HISTORICAL_TREE
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tar_path = out_root / "historical_cliff_checkpoint_tree.tar"
    subprocess.run(["git", "archive", "-o", str(tar_path), HISTORICAL_COMMIT, HISTORICAL_TREE], cwd=repo_root, check=True)
    subprocess.run(["tar", "-xf", str(tar_path), "-C", str(out_root / "historical_checkpoint_tree")], cwd=repo_root, check=True)
    return target


def make_data(rc: dict):
    cfg = CliffDataConfig(
        n_samples=int(float(rc["n_samples"])),
        feat_dim=int(float(rc.get("data_feat_dim", rc.get("feat_dim", 8)))),
        noise_std=float(rc["noise_std"]),
        beta=float(rc["beta"]),
        train_frac=float(rc["train_frac"]),
        batch_size=int(float(rc["batch_size"])),
        seed=int(float(rc["seed"])),
        regime_set=rc.get("regime_set", "v1"),
        center_scale=float(rc.get("center_scale", 2.0)),
        ambiguous_pairs=parse_pairs(rc.get("ambiguous_pairs", "")),
        ambiguous_gap=float(rc.get("ambiguous_gap", 0.15)),
        cliff_penalty=float(rc.get("cliff_penalty", -100.0)),
        regime_weights=parse_weights(rc.get("regime_weights", "")),
        test_regime_weights=parse_weights(rc.get("test_regime_weights", "")),
    )
    return prepare_cliff_data(cfg)


def make_args(rc: dict) -> Namespace:
    return Namespace(
        algo=rc["algo"],
        beta=float(rc.get("beta", 0.95)),
        qp_reg=float(rc.get("qp_reg", 0.1)),
        n_sketches=int(float(rc.get("n_sketches", 4))),
        sketch_ratio=float(rc.get("sketch_ratio", 0.1)),
        sketch_mode=rc.get("sketch_mode", "gaussian"),
    )


def eval_checkpoint_run(run_dir: Path, loader_cache: dict, lp_solver, qp_solver) -> dict:
    rc = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    data_key = json.dumps(
        {
            k: rc.get(k, "")
            for k in [
                "n_samples", "data_feat_dim", "feat_dim", "noise_std", "beta",
                "train_frac", "batch_size", "seed", "regime_set", "center_scale",
                "ambiguous_pairs", "ambiguous_gap", "cliff_penalty",
                "regime_weights", "test_regime_weights",
            ]
        },
        sort_keys=True,
    )
    if data_key not in loader_cache:
        loader_cache[data_key] = make_data(rc)
    train_loader, test_loader, meta = loader_cache[data_key]

    w0, *_ = next(iter(train_loader))
    volatile_cells = list(meta.get("volatile_cells", []) or [])
    feature_net = FeatureNet(
        in_dim=w0.shape[-1],
        feat_dim=int(float(rc["nn_feat_dim"])),
        hidden=int(float(rc["hidden_dim"])),
        n_slots=1 + len(volatile_cells),
    )
    head = TransitionHead(feat_dim=int(float(rc["nn_feat_dim"])), volatile_cells=volatile_cells)
    model = CliffTransitionModel(feature_net, head, detach_features=True)
    feature_net.load_state_dict(torch.load(run_dir / "feature_net_state.pt", map_location="cpu", weights_only=True))
    head.load_state_dict(torch.load(run_dir / "head_state.pt", map_location="cpu", weights_only=True))
    model.eval()

    args = make_args(rc)
    pred_values, oracle_values = [], []
    with torch.no_grad():
        for batch in test_loader:
            w, _z, _slip, P_true, r_true, _y_true, v_true = batch
            P_pred, r_pred, _slip_pred = model(w)
            y_star, _ = _solve_with_algo(args, P_pred, r_pred, lp_solver, qp_solver)
            gamma = _build_gamma(w.shape[0], P_pred.shape[1], P_pred.device, P_pred.dtype, start_state=_start_state())
            v_pred, _ = eval_y_on_true_mdp(P_true, r_true, y_star, gamma, float(rc["beta"]))
            pred_values.extend(v_pred.tolist())
            oracle_values.extend(v_true.tolist())

    regret = np.asarray(oracle_values, dtype=float) - np.asarray(pred_values, dtype=float)
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    return {
        "seed_index": int(run_dir.name.rsplit("_s", 1)[1]),
        "raw_seed": int(float(rc["seed"])),
        "test_mean_regret": float(regret.mean()),
        "test_mean_oracle_reward": float(np.mean(oracle_values)),
        "test_mean_pred_reward": float(np.mean(pred_values)),
        "train_reward_per_epoch": checkpoint.get("train_reward_per_epoch", []),
    }


def summarize_values(rows: list[dict], key: str = "test_mean_regret") -> dict:
    vals = [float(r[key]) for r in rows]
    return {
        "n": len(vals),
        "mean": float(np.mean(vals)) if vals else None,
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "se": float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
        "min": float(np.min(vals)) if vals else None,
        "max": float(np.max(vals)) if vals else None,
        "values": vals,
        "rows": rows,
    }


def run_checkpoint_eval(out_root: Path, methods: list[str], n_seeds: int) -> dict:
    root = ensure_historical_tree(out_root)
    solver_obj = DenseLPSolver()

    def lp_solver(H, r, gamma):
        return solver_obj.solve(H, r, gamma)

    qp_solver = QPFunction(eps=1e-10, maxIter=20, solver=QPSolvers.PDIPM_BATCHED)
    loader_cache = {}
    summary = {}
    for method in methods:
        rows = []
        for seed in range(n_seeds):
            run_dir = root / f"{method}_s{seed}"
            if not run_dir.exists():
                continue
            print(f"[cliff checkpoint] {method} seed {seed}")
            rows.append(eval_checkpoint_run(run_dir, loader_cache, lp_solver, qp_solver))
        if rows:
            summary[method] = summarize_values(rows)
            print(method, summary[method]["mean"], "n=", summary[method]["n"])
    return summary


def sa_run_dir(out_root: Path, M: int, tau: float, idx: int, loss_weight: float) -> Path:
    tau_tag = str(tau).replace(".", "p")
    lw_tag = str(loss_weight).replace(".", "p")
    return out_root / "sa_runs" / f"M{M}_tau{tau_tag}_lw{lw_tag}" / f"ssa{M}_s{idx}"


def make_sa_args(out: Path, M: int, tau: float, idx: int, loss_weight: float) -> Namespace:
    seed = RAW_SEEDS[idx]
    return Namespace(
        algo="dfl_ssa",
        n_samples=200,
        data_feat_dim=8,
        feat_dim=32,
        noise_std=2.5,
        train_frac=0.5,
        batch_size=32,
        seed=seed,
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
        pretrain_epochs=60,
        n_epochs=50,
        grad_clip=10.0,
        beta=0.95,
        qp_reg=0.1,
        qp_max_iter=20,
        loss_weight=float(loss_weight),
        aug_lag_rho=0.1,
        dfl_perturb_std=0.0,
        qpth_warmup_epochs=0,
        n_agg=int(M),
        sa_tau=float(tau),
        sa_r_embed=8,
        n_sketches=4,
        sketch_ratio=0.1,
        ssa_objective="cw",
        cw_agg_mode="dest_r_bias",
        agg_lr_ratio=1.0,
        lp_basis_ridge=0.0,
        hinge_weight_alpha=0.0,
        hard_mine_every=0,
        margin_lambda=0.0,
        sketch_use_qpth=False,
        save_dir=out,
    )


def run_sa_one(out_root: Path, M: int, tau: float, idx: int, loss_weight: float, force: bool = False) -> tuple[int, float, int, str, float]:
    seed = RAW_SEEDS[idx]
    out = sa_run_dir(out_root, M, tau, idx, loss_weight)
    res = out / "results.json"
    if res.exists() and not force:
        return M, tau, idx, "skip", 0.0
    if force and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    args = make_sa_args(out, M, tau, idx, loss_weight)
    cfg = CliffDataConfig(
        n_samples=args.n_samples,
        feat_dim=args.data_feat_dim,
        noise_std=args.noise_std,
        beta=args.beta,
        train_frac=args.train_frac,
        batch_size=args.batch_size,
        seed=seed,
        regime_set=args.regime_set,
        center_scale=args.center_scale,
        ambiguous_pairs=args.ambiguous_pairs,
        ambiguous_gap=args.ambiguous_gap,
        cliff_penalty=args.cliff_penalty,
        regime_weights=args.regime_weights,
        test_regime_weights=args.test_regime_weights,
    )
    t0 = time.time()
    try:
        with (out / "run.log").open("w", encoding="utf-8") as f:
            original_stdout, original_stderr = sys.stdout, sys.stderr
            sys.stdout = f
            sys.stderr = f
            try:
                disable_per_run_plotting()
                train_loader, test_loader, meta = prepare_cliff_data(cfg)
                train_eval_cliff(args, train_loader, test_loader, meta)
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
        status = "ok" if res.exists() else "fail:no-results"
    except Exception as exc:
        with (out / "run.log").open("a", encoding="utf-8") as f:
            f.write(f"\n[cliff_train_repro] ERROR: {exc!r}\n")
        status = f"fail:{type(exc).__name__}"
    return M, tau, idx, status, time.time() - t0


def summarize_sa(out_root: Path) -> dict:
    rows = {}
    for root in sorted((out_root / "sa_runs").glob("M*_tau*")):
        vals = []
        records = []
        curves = []
        for p in sorted(root.glob("ssa*_s*/results.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            vals.append(float(d["test_mean_regret"]))
            records.append({"path": str(p), "test_mean_regret": float(d["test_mean_regret"])})
            ckpt = p.parent / "checkpoint.json"
            if ckpt.exists():
                curves.append(json.loads(ckpt.read_text(encoding="utf-8")).get("train_reward_per_epoch", []))
        if vals:
            rows[root.name] = summarize_values(
                [
                    {
                        "test_mean_regret": v,
                        "train_reward_per_epoch": curves[i] if i < len(curves) else [],
                    }
                    for i, v in enumerate(vals)
                ]
            )
            rows[root.name]["records"] = records
    return rows


def run_sa_grid(out_root: Path, specs: list[tuple[int, float, float]], seeds: list[int], workers: int, force: bool) -> dict:
    jobs = [(out_root, M, tau, idx, lw, force) for M, tau, lw in specs for idx in seeds]
    failed = []
    if workers != 1:
        print("[cliff SA] running sequentially because in-process training redirects stdout to per-run logs.")
    for job in jobs:
        result = run_sa_one(*job)
        print(result, flush=True)
        if not str(result[3]).startswith(("ok", "skip")):
            failed.append(result)
    if failed:
        raise RuntimeError(f"Cliff SA reproduction failed for {len(failed)} job(s): {failed[:5]}")
    return summarize_sa(out_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--methods", nargs="*", default=["ts", "qpth", "dfl", "sketch"])
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--sa-specs", nargs="*", default=["6:0.5:500", "12:0.5:500", "24:2.0:500"])
    parser.add_argument("--sa-seeds", nargs="*", type=int, default=list(range(10)))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Regenerate SA runs even if results.json already exists.")
    args = parser.parse_args()

    args.results_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.results_root / "cliff_checkpoint_eval_summary.json"
    sa_path = args.results_root / "cliff_sa_summary.json"

    if args.summary_only and checkpoint_path.exists() and sa_path.exists():
        checkpoint_summary = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        sa_summary = json.loads(sa_path.read_text(encoding="utf-8"))
    else:
        checkpoint_summary = run_checkpoint_eval(args.results_root, args.methods, args.n_seeds)
        checkpoint_path.write_text(json.dumps(checkpoint_summary, indent=2), encoding="utf-8")
        specs = []
        for spec in args.sa_specs:
            M, tau, lw = spec.split(":")
            specs.append((int(M), float(tau), float(lw)))
        sa_summary = run_sa_grid(args.results_root, specs, args.sa_seeds, args.workers, args.force)
        sa_path.write_text(json.dumps(sa_summary, indent=2), encoding="utf-8")

    combined = {"checkpoint": checkpoint_summary, "sa": sa_summary}
    out = args.results_root / "cliff_combined_summary.json"
    out.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
