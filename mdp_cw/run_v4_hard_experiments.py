"""
Launch TS + DFL on the **v4** regime library (richer volatile / ICE layout) with a
**harder feature** preset: higher noise, tighter cluster scale, ambiguous ICE pairs.

Default **larger training set**: ``n_samples=120`` (60 train / 60 test at train_frac=0.5).

Directory layout (keeps different data budgets separate)::

    {root}/n{n_samples}/k{K}/{ts,dfl,qpth,ssa}_s{seed}/

Algorithms (``--algos``): ``ts`` (twostage), ``dfl``, ``qpth`` (``dfl_qpth``),
``ssa`` (``dfl_ssa``). Default remains ``ts,dfl``. Backfill qpth/ssa at n=120::

    python -m mdp_cw.run_v4_hard_experiments --algos qpth,ssa --n_samples 120 \\
        --skew_k 7 --n_seeds 50 --concurrency 2 --scale_loss_weight_with_skew

Use ``--flat_layout`` to write ``{root}/k{K}/`` (matches older v4_hard trees).

**Data scaling ablation** (20 / 40 / 80 / 120 / 160)::

    python -m mdp_cw.run_v4_hard_experiments --n_samples_sweep 20,40,80,120,160 --skew_k 7 \\
        --n_seeds 50 --concurrency 2 --scale_loss_weight_with_skew

Summarize after::

    python -m mdp_cw.summarize_paired_runs --root mdp_cw/results_n120/n160 --k 7 --n_seeds 50

Skips a run if ``results.json`` already exists (idempotent).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _base_cmd(n_samples: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "mdp_cw.run_cliff_experiments",
        "--regime_set",
        "v4",
        "--cliff_penalty",
        "-10",
        "--noise_std",
        "2.5",
        "--center_scale",
        "0.88",
        "--ambiguous_pairs",
        "1:2,2:3",
        "--n_samples",
        str(int(n_samples)),
        "--n_epochs",
        "50",
        "--pretrain_epochs",
        "60",
        "--loss_weight",
        "500.0",
        "--aug_lag_rho",
        "1.0",
        "--qp_reg",
        "0.1",
        "--lr",
        "3e-3",
        "--batch_size",
        "32",
        "--train_frac",
        "0.5",
        "--hidden_dim",
        "64",
        "--nn_feat_dim",
        "32",
    ]


def _maybe_override_loss_weight(cmd: list[str], loss_weight: float | None) -> list[str]:
    if loss_weight is None:
        return cmd
    out: list[str] = []
    i = 0
    while i < len(cmd):
        if cmd[i] == "--loss_weight" and i + 1 < len(cmd):
            out.extend(["--loss_weight", str(float(loss_weight))])
            i += 2
            continue
        out.append(cmd[i])
        i += 1
    return out


def regime_weights_arg(k: int) -> list[str]:
    if k == 1:
        return []
    return ["--regime_weights", f"{k},1,1,1"]


def run_root(base: Path, n_samples: int, flat_layout: bool) -> Path:
    if flat_layout:
        return Path(base)
    return Path(base) / f"n{int(n_samples)}"


# Slug -> (run_cliff_experiments --algo value, results folder prefix)
_ALGO: dict[str, tuple[str, str]] = {
    "ts": ("twostage", "ts"),
    "dfl": ("dfl", "dfl"),
    "qpth": ("dfl_qpth", "qpth"),
    "ssa": ("dfl_ssa", "ssa"),
    "sketch": ("dfl_sketch", "sketch"),
}


def build_cmd(
    algo: str,
    k: int,
    seed: int,
    out_root: Path,
    scale_ce: bool,
    n_samples: int,
    qpth_reg: float = 0.06,
    ambiguous_pairs: str = "1:2,2:3",
    cliff_penalty: float = -10.0,
    test_regime_weights: str = "",
    loss_weight: float | None = None,
    ambiguous_gap: float | None = None,
    regime_set: str | None = None,
    aug_lag_rho: float | None = None,
    n_sketches: int | None = None,
    sketch_ratio: float | None = None,
    dfl_perturb_std: float | None = None,
    qpth_warmup_epochs: int | None = None,
    grad_clip: float | None = None,
) -> list[str]:
    if algo not in _ALGO:
        raise ValueError(f"Unknown algo slug {algo!r}; choose from {sorted(_ALGO)}")
    run_algo, folder = _ALGO[algo]
    cmd = _maybe_override_loss_weight(_base_cmd(n_samples), loss_weight) + regime_weights_arg(k)
    if ambiguous_pairs:
        cmd += ["--ambiguous_pairs", ambiguous_pairs]
    # Override cliff_penalty after base cmd so it takes precedence.
    cmd += ["--cliff_penalty", str(cliff_penalty)]
    if test_regime_weights:
        cmd += ["--test_regime_weights", test_regime_weights]
    # Optional overrides appended last so they take precedence over _base_cmd defaults.
    if ambiguous_gap is not None:
        cmd += ["--ambiguous_gap", str(ambiguous_gap)]
    if regime_set is not None:
        cmd += ["--regime_set", regime_set]
    if aug_lag_rho is not None:
        cmd += ["--aug_lag_rho", str(aug_lag_rho)]
    if scale_ce and algo != "ts":
        cmd.append("--scale_loss_weight_with_skew")
    cmd += [
        "--algo",
        run_algo,
        "--seed",
        str(seed),
        "--save_dir",
        str(out_root / f"k{k}" / f"{folder}_s{seed}"),
    ]
    # qp_reg controls how close qpth tracks the LP optimum; lower = closer to LP but harder solve.
    if algo == "qpth":
        cmd += ["--qp_reg", str(qpth_reg), "--qp_max_iter", "56"]
    # Sketch params: use overrides when provided, else fall back to defaults.
    if algo == "sketch":
        cmd += ["--n_sketches", str(n_sketches if n_sketches is not None else 4)]
        cmd += ["--sketch_ratio", str(sketch_ratio if sketch_ratio is not None else 0.1)]
    # PFYL perturbation for dfl (0 = disabled by default).
    if dfl_perturb_std is not None and dfl_perturb_std > 0.0:
        cmd += ["--dfl_perturb_std", str(dfl_perturb_std)]
    if qpth_warmup_epochs is not None and qpth_warmup_epochs > 0:
        cmd += ["--qpth_warmup_epochs", str(qpth_warmup_epochs)]
        if algo == "dfl":
            cmd += ["--qp_reg", str(qpth_reg)]
    if grad_clip is not None:
        cmd += ["--grad_clip", str(grad_clip)]
    return cmd


def collect_jobs(
    out_root: Path,
    k: int,
    n_seeds: int,
    scale_ce: bool,
    n_samples: int,
    algos: tuple[str, ...],
    qpth_reg: float = 0.06,
    ambiguous_pairs: str = "1:2,2:3",
    cliff_penalty: float = -10.0,
    test_regime_weights: str = "",
    loss_weight: float | None = None,
    ambiguous_gap: float | None = None,
    regime_set: str | None = None,
    aug_lag_rho: float | None = None,
    n_sketches: int | None = None,
    sketch_ratio: float | None = None,
    dfl_perturb_std: float | None = None,
    qpth_warmup_epochs: int | None = None,
    grad_clip: float | None = None,
) -> list[list[str]]:
    jobs: list[list[str]] = []
    for seed in range(n_seeds):
        for algo in algos:
            if algo not in _ALGO:
                raise ValueError(f"Unknown algo {algo!r} in --algos")
            folder = _ALGO[algo][1]
            out = out_root / f"k{k}" / f"{folder}_s{seed}" / "results.json"
            if out.exists():
                continue
            jobs.append(build_cmd(algo, k, seed, out_root, scale_ce, n_samples,
                                   qpth_reg=qpth_reg, ambiguous_pairs=ambiguous_pairs,
                                   cliff_penalty=cliff_penalty,
                                   test_regime_weights=test_regime_weights,
                                   loss_weight=loss_weight,
                                   ambiguous_gap=ambiguous_gap,
                                   regime_set=regime_set,
                                   aug_lag_rho=aug_lag_rho,
                                   n_sketches=n_sketches,
                                   sketch_ratio=sketch_ratio,
                                   dfl_perturb_std=dfl_perturb_std,
                                   qpth_warmup_epochs=qpth_warmup_epochs,
                                   grad_clip=grad_clip))
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("mdp_cw/results_n120"))
    p.add_argument("--skew_k", type=int, default=7, help="Training skew k:1:1:1 (k=1 uniform).")
    p.add_argument("--n_seeds", type=int, default=10)
    p.add_argument("--n_samples", type=int, default=120, help="Total MDP samples (half train, half test).")
    p.add_argument(
        "--flat_layout",
        action="store_true",
        help="Write under ROOT/k{K}/ (no n{N}/ prefix). Use for legacy trees.",
    )
    p.add_argument(
        "--n_samples_sweep",
        type=str,
        default="",
        help="Comma-separated n_samples values, e.g. '20,40,80,120,160'. Each uses ROOT/n{N}/.",
    )
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument(
        "--scale_loss_weight_with_skew",
        action="store_true",
        help="Append flag on DFL / qpth / ssa runs (recommended at high skew).",
    )
    p.add_argument(
        "--algos",
        type=str,
        default="ts,dfl",
        help="Comma-separated slugs: ts,dfl,qpth,ssa. Example backfill: qpth,ssa",
    )
    p.add_argument(
        "--qpth_reg",
        type=float,
        default=0.06,
        help="QP regularization for qpth (lower = closer to LP; try 0.005 or 0.01).",
    )
    p.add_argument(
        "--ambiguous_pairs",
        type=str,
        default="1:2,2:3",
        help="Ambiguous regime pairs, e.g. '1:2,2:3' (baseline) or '0:1,0:2,0:3' (all-ICE-vs-CLEAR).",
    )
    p.add_argument(
        "--cliff_penalty",
        type=float,
        default=-10.0,
        help="Cliff reward magnitude (default -10). Higher magnitude grows margins but risks DFL divergence.",
    )
    p.add_argument(
        "--test_regime_weights",
        type=str,
        default="",
        help="Test-only regime weights, e.g. '1,1,1,1' for uniform. Train uses --skew_k weights. "
             "Empty = same distribution as training.",
    )
    p.add_argument(
        "--loss_weight",
        type=float,
        default=None,
        help="Override CE/decision loss weight (default 500 from v4 hard preset). "
             "Try 100 with very steep cliff_penalty for stability.",
    )
    p.add_argument(
        "--ambiguous_gap",
        type=float,
        default=None,
        help="Cluster-center offset for ambiguous pairs (default 0.15 in run_cliff_experiments). "
             "Set to 0.0 to make ambiguous pair centers statistically identical.",
    )
    p.add_argument(
        "--regime_set",
        type=str,
        default=None,
        help="Override regime set (default v4 from base cmd). E.g. 'v5' for shared strip regimes.",
    )
    p.add_argument(
        "--aug_lag_rho",
        type=float,
        default=None,
        help="Override augmented Lagrangian rho (default 1.0 in base cmd). "
             "Lower (e.g. 0.1) softens LP constraint enforcement during DFL training.",
    )
    p.add_argument(
        "--n_sketches",
        type=int,
        default=None,
        help="Number of sketches for dfl_sketch (default 4). Try 16 for better approximation.",
    )
    p.add_argument(
        "--sketch_ratio",
        type=float,
        default=None,
        help="Row fraction per sketch for dfl_sketch (default 0.1). Try 0.5 or 0.8.",
    )
    p.add_argument(
        "--dfl_perturb_std",
        type=float,
        default=None,
        help="PFYL noise std on r_pred before LP solve (dfl only).",
    )
    p.add_argument(
        "--qpth_warmup_epochs",
        type=int,
        default=None,
        help="For dfl: use QPTH gradient for this many epochs before switching to fixed-basis LP. Try 25.",
    )
    p.add_argument(
        "--grad_clip",
        type=float,
        default=None,
        help="Override gradient clipping norm (default 10.0). Use 2.0 with low loss_weight to prevent divergence.",
    )
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    k = args.skew_k
    sweep = [int(x.strip()) for x in args.n_samples_sweep.split(",") if x.strip()]
    if sweep:
        n_list = sweep
    else:
        n_list = [int(args.n_samples)]
    algos = tuple(a.strip() for a in args.algos.split(",") if a.strip())

    all_failed: list[tuple[int, str]] = []
    for n_samples in n_list:
        out_root = run_root(args.root, n_samples, args.flat_layout)
        jobs = collect_jobs(
            out_root, k, args.n_seeds, args.scale_loss_weight_with_skew, n_samples, algos,
            qpth_reg=args.qpth_reg, ambiguous_pairs=args.ambiguous_pairs,
            cliff_penalty=args.cliff_penalty, test_regime_weights=args.test_regime_weights,
            loss_weight=args.loss_weight, ambiguous_gap=args.ambiguous_gap,
            regime_set=args.regime_set, aug_lag_rho=args.aug_lag_rho,
            n_sketches=args.n_sketches, sketch_ratio=args.sketch_ratio,
            dfl_perturb_std=args.dfl_perturb_std,
            qpth_warmup_epochs=args.qpth_warmup_epochs,
        )
        print(f"\n[{n_samples} samples] Queued {len(jobs)} runs under {out_root}/k{k}/", flush=True)
        if args.dry_run:
            for cmd in jobs[:3]:
                print(" ", " ".join(cmd))
            if len(jobs) > 3:
                print(f"  ... and {len(jobs) - 3} more")
            continue

        i = 0
        failed: list[tuple[int, str]] = []
        while i < len(jobs):
            batch = jobs[i : i + args.concurrency]
            procs = [(subprocess.Popen(cmd), cmd) for cmd in batch]
            for proc, cmd in procs:
                code = proc.wait()
                if code != 0:
                    failed.append((code, " ".join(cmd)))
            i += len(batch)
            print(f"  finished {min(i, len(jobs))}/{len(jobs)}", flush=True)

        if failed:
            all_failed.extend(failed)
            print(f"  {len(failed)} failures for n_samples={n_samples}", file=sys.stderr)

    if args.dry_run:
        return

    if all_failed:
        print(f"\n{len(all_failed)} total failed runs:", file=sys.stderr)
        for code, cmd in all_failed[:12]:
            print(f"  exit {code}: {cmd[:140]}...", file=sys.stderr)
        sys.exit(1)
    print("\nAll queued runs completed.")


if __name__ == "__main__":
    main()
