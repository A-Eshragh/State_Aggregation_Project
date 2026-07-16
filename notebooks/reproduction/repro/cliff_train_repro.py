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

from mdp_cw.prepare_cliff_data import CliffDataConfig, prepare_cliff_data
from mdp_cw.train_cliff import train_eval_cliff


RESULTS_ROOT = REPO / "notebooks" / "reproduction" / "repro" / "results" / "cliff"
RAW_SEEDS = [360, 606, 689, 665, 790, 310, 582, 962, 246, 297]
METHOD_CONFIGS = {
    "ts": dict(algo="twostage", qp_max_iter=56, n_sketches=4, sketch_ratio=0.1),
    "qpth": dict(algo="dfl_qpth", qp_max_iter=56, n_sketches=4, sketch_ratio=0.1),
    "dfl": dict(algo="dfl", qp_max_iter=56, n_sketches=4, sketch_ratio=0.1),
    "sketch": dict(algo="dfl_sketch", qp_max_iter=56, n_sketches=4, sketch_ratio=0.1),
}


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


def base_run_dir(out_root: Path, method: str, idx: int) -> Path:
    return out_root / "baseline_runs" / method / f"{method}_s{idx}"


def make_base_args(out: Path, method: str, idx: int) -> Namespace:
    seed = RAW_SEEDS[idx]
    cfg = METHOD_CONFIGS[method]
    return Namespace(
        algo=cfg["algo"],
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
        qp_max_iter=int(cfg["qp_max_iter"]),
        loss_weight=500.0,
        aug_lag_rho=0.1,
        dfl_perturb_std=0.0,
        qpth_warmup_epochs=0,
        n_agg=12,
        sa_tau=0.5,
        sa_r_embed=8,
        n_sketches=int(cfg["n_sketches"]),
        sketch_ratio=float(cfg["sketch_ratio"]),
        sketch_mode="gaussian",
        scale_loss_weight_with_skew=False,
        loss_weight_skew_p_ice_ref=0.3,
        save_dir=out,
    )


def run_base_one(out_root: Path, method: str, idx: int, force: bool = False) -> tuple[str, int, str, float]:
    out = base_run_dir(out_root, method, idx)
    res = out / "results.json"
    if res.exists() and not force:
        return method, idx, "skip", 0.0
    if force and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    args = make_base_args(out, method, idx)
    cfg = CliffDataConfig(
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
    return method, idx, status, time.time() - t0


def summarize_base(out_root: Path, methods: list[str]) -> dict:
    summary = {}
    for method in methods:
        rows = []
        for p in sorted((out_root / "baseline_runs" / method).glob(f"{method}_s*/results.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            ckpt = p.parent / "checkpoint.json"
            curve = []
            if ckpt.exists():
                curve = json.loads(ckpt.read_text(encoding="utf-8")).get("train_reward_per_epoch", [])
            rows.append(
                {
                    "path": str(p),
                    "seed_index": int(p.parent.name.rsplit("_s", 1)[1]),
                    "raw_seed": RAW_SEEDS[int(p.parent.name.rsplit("_s", 1)[1])],
                    "test_mean_regret": float(d["test_mean_regret"]),
                    "test_mean_oracle_reward": float(d.get("test_mean_oracle_reward", np.nan)),
                    "test_mean_pred_reward": float(d.get("test_mean_pred_reward", np.nan)),
                    "train_reward_per_epoch": curve,
                }
            )
        if rows:
            rows.sort(key=lambda r: r["seed_index"])
            summary[method] = summarize_values(rows)
            print(method, summary[method]["mean"], "n=", summary[method]["n"])
    return summary


def run_base_grid(out_root: Path, methods: list[str], n_seeds: int, force: bool) -> dict:
    failed = []
    for method in methods:
        if method not in METHOD_CONFIGS:
            raise ValueError(f"Unknown Cliff method {method!r}; choose from {sorted(METHOD_CONFIGS)}")
        for idx in range(n_seeds):
            result = run_base_one(out_root, method, idx, force=force)
            print(result, flush=True)
            if not str(result[2]).startswith(("ok", "skip")):
                failed.append(result)
    if failed:
        raise RuntimeError(f"Cliff baseline reproduction failed for {len(failed)} job(s): {failed[:5]}")
    return summarize_base(out_root, methods)


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
    baseline_path = args.results_root / "cliff_baseline_summary.json"
    sa_path = args.results_root / "cliff_sa_summary.json"

    if args.summary_only and baseline_path.exists() and sa_path.exists():
        baseline_summary = json.loads(baseline_path.read_text(encoding="utf-8"))
        sa_summary = json.loads(sa_path.read_text(encoding="utf-8"))
    else:
        baseline_summary = run_base_grid(args.results_root, args.methods, args.n_seeds, args.force)
        baseline_path.write_text(json.dumps(baseline_summary, indent=2), encoding="utf-8")
        specs = []
        for spec in args.sa_specs:
            M, tau, lw = spec.split(":")
            specs.append((int(M), float(tau), float(lw)))
        sa_summary = run_sa_grid(args.results_root, specs, args.sa_seeds, args.workers, args.force)
        sa_path.write_text(json.dumps(sa_summary, indent=2), encoding="utf-8")

    combined = {"baseline": baseline_summary, "sa": sa_summary}
    out = args.results_root / "cliff_combined_summary.json"
    out.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
