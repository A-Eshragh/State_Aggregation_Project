#!/usr/bin/env python3
"""Three report PNGs per sample budget: training reward, compute, and test regret.

Uses the **v4 hard** layout under ``{v4_root}/n{n}/k{k}/{ts,dfl,qpth,ssa}_s*`` (default ``v4_root`` = ``results_n120``).
Default **--ns** ``20,40,80,120,160`` writes one subfolder per present ``n{N}`` (e.g. ``.../report_three_panel/n160/``).
Default **k = 7** (skewed prior).

Each ``n{N}/`` output folder contains::

    report_01_train_reward_mean_se.png
    report_02_compute_solver_vs_backward.png
    report_03_test_mean_regret_bars.png

Usage::

    python -m mdp_cw.plot_report_three_panel_best_nk
    python -m mdp_cw.plot_report_three_panel_best_nk --ns 20,40,80,120,160 --k 7 \\
        --out_dir mdp_cw/final_results_best/report_three_panel
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

MDP_CW = Path(__file__).resolve().parent

# Match make_figures.py style exactly.
RC = {
    "font.family": "serif", "font.size": 20, "axes.labelsize": 18,
    "axes.titlesize": 20, "legend.fontsize": 13, "xtick.labelsize": 16,
    "ytick.labelsize": 18, "axes.linewidth": 1.2,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}

METHODS: Tuple[Tuple[str, str, str, str], ...] = (
    ("ts",     "TS",             "#ff7f0e", "--"),
    ("qpth",   "QP",             "#2ca02c", "-"),
    ("dfl",    "LP",             "#1f77b4", "-"),
    ("sketch", "Sketch",         "#d62728", ":"),
    ("ssa6",   "SA (M=6)",       "#9467bd", "-"),
    ("ssa12",  "SA (M=12)",      "#9467bd", "-"),
    ("ssa24",  "SA (M=24)",      "#9467bd", "-"),
    ("ssa",    "SA (M=12)",      "#9467bd", "-"),
)
DEFAULT_METHOD_KEYS = ("ts", "qpth", "dfl", "sketch", "ssa6", "ssa12", "ssa24")
TRAINING_METHOD_KEYS = ("ts", "qpth", "dfl", "sketch", "ssa12")
TRAINING_LEGEND_LABELS = {
    "ts": "Two-Stage",
    "qpth": "DFL-QP",
    "dfl": "DFL-LP",
    "sketch": "DFL-Sketch",
    "ssa12": "DFL-SA (M=12)",
}
TICK_LABELS = {
    "ssa6": "SA\nM=6",
    "ssa12": "SA\nM=12",
    "ssa24": "SA\nM=24",
    "ssa": "SA\nM=12",
}

def _select_methods(method_keys: Optional[Tuple[str, ...]] = None) -> Tuple[Tuple[str, str, str, str], ...]:
    keys = method_keys if method_keys is not None else DEFAULT_METHOD_KEYS
    allowed = {m[0]: m for m in METHODS}
    missing = [k for k in keys if k not in allowed]
    if missing:
        raise ValueError(f"Unknown method key(s): {missing}; choose from {sorted(allowed)}")
    return tuple(allowed[k] for k in keys)


def _load(root: Path, k: int, prefix: str, s: int) -> Optional[Dict]:
    p = root / f"k{k}" / f"{prefix}_s{s}" / "results.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def load_seeds(root: Path, k: int, prefix: str, n_seeds: int) -> List[Dict]:
    base = root / f"k{k}"
    if not base.is_dir():
        return []

    entries: List[Tuple[int, Path]] = []
    for run_dir in base.glob(f"{prefix}_s*"):
        if not run_dir.is_dir():
            continue
        m = re.search(r"_s(\d+)$", run_dir.name)
        if not m:
            continue
        res = run_dir / "results.json"
        if res.is_file():
            entries.append((int(m.group(1)), res))

    entries.sort(key=lambda t: t[0])
    if n_seeds > 0:
        entries = entries[:n_seeds]

    runs: List[Dict] = []
    for _, res in entries:
        try:
            runs.append(json.loads(res.read_text()))
        except Exception:
            continue
    return runs


def _stack_series(runs: List[Dict], key: str) -> Optional[np.ndarray]:
    rows = [
        r[key] for r in runs
        if key in r and isinstance(r[key], list) and len(r[key]) > 0
    ]
    if not rows:
        return None
    m = min(len(x) for x in rows)
    return np.array([x[:m] for x in rows], dtype=np.float64)


def _se(a: np.ndarray, axis: int = 0) -> np.ndarray:
    n = a.shape[axis]
    if n <= 1:
        return np.zeros_like(np.mean(a, axis=axis))
    return a.std(axis=axis, ddof=1) / np.sqrt(n)


def _run_label(root: Path, label: Optional[str] = None) -> str:
    return label if label is not None else root.name


def plot_reward_curves(
    ax: plt.Axes,
    root: Path,
    k: int,
    n_seeds: int,
    run_label: Optional[str] = None,
    method_keys: Optional[Tuple[str, ...]] = None,
) -> None:
    handles = []
    for prefix, label, color, ls in _select_methods(method_keys):
        runs = load_seeds(root, k, prefix, n_seeds)
        mat = _stack_series(runs, "train_reward_per_epoch")
        if mat is None:
            continue
        mu = mat.mean(axis=0)
        se = _se(mat, axis=0)
        x = np.arange(mu.shape[0])
        display_label = TRAINING_LEGEND_LABELS.get(prefix, label)
        ax.plot(x, mu, color=color, ls=ls, lw=2.0, label=display_label)
        ax.fill_between(x, mu - se, mu + se, color=color, alpha=0.2)
        handles.append(Line2D([0], [0], color=color, ls=ls, lw=2.0, label=display_label))
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Training reward")
    ax.set_title("Cliff Walking — Training curves")
    ax.legend(handles=handles, loc="lower right")
    ax.grid(True, alpha=0.25)


def _total_train_time(run: Dict) -> Optional[float]:
    """Return total training time (seconds) from a results dict.

    ``train_time`` is logged as cumulative seconds since training start, so
    the last entry is the total wall-clock training time for that seed.
    """
    tt = run.get("train_time")
    if tt and len(tt) > 0:
        return float(tt[-1])
    return None


def _mse(xs: np.ndarray):
    if xs.size == 0:
        return float("nan"), float("nan")
    if xs.size == 1:
        return float(xs[0]), 0.0
    return float(xs.mean()), float(xs.std(ddof=1) / np.sqrt(xs.size))


def plot_compute_total_time(
    ax: plt.Axes,
    root: Path,
    k: int,
    n_seeds: int,
    run_label: Optional[str] = None,
    method_keys: Optional[Tuple[str, ...]] = None,
) -> None:
    """Bars showing mean ± SE total training time per seed, matching make_figures.py cost panel."""
    keys, labels, means, ses, colors = [], [], [], [], []

    for prefix, label, color, _ls in _select_methods(method_keys):
        totals = [t for run in load_seeds(root, k, prefix, n_seeds)
                  if (t := _total_train_time(run)) is not None]
        v = np.array(totals, dtype=np.float64)
        if v.size == 0:
            continue
        mu, se = _mse(v)
        keys.append(prefix)
        labels.append(TICK_LABELS.get(prefix, label))
        means.append(mu)
        ses.append(se)
        colors.append(color)

    x = np.arange(len(labels))
    bar_w = 0.65
    means_a = np.array(means, dtype=np.float64)
    ses_a = np.array(ses, dtype=np.float64)

    ax.bar(
        x, means_a, yerr=ses_a, capsize=4, width=bar_w,
        color=colors, alpha=0.88,
        edgecolor="black", linewidth=0.6,
        error_kw=dict(ecolor="black", lw=0.8),
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Total training time (s)")
    ax.set_title("Cliff Walking — Compute cost")
    ax.grid(True, axis="y", alpha=0.3, ls=":")


def plot_regret_bars(
    ax: plt.Axes,
    root: Path,
    k: int,
    n_seeds: int,
    run_label: Optional[str] = None,
    method_keys: Optional[Tuple[str, ...]] = None,
) -> None:
    means, errs, labels, colors = [], [], [], []
    for prefix, label, color, _ls in _select_methods(method_keys):
        runs = load_seeds(root, k, prefix, n_seeds)
        vals = np.array(
            [float(r["test_mean_regret"]) for r in runs if "test_mean_regret" in r],
            dtype=np.float64,
        )
        if vals.size == 0:
            continue
        means.append(float(vals.mean()))
        errs.append(float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0)
        labels.append(TICK_LABELS.get(prefix, label))
        colors.append(color)

    n_bars = len(labels)
    x = np.arange(n_bars)
    bar_w = 0.52
    means_a = np.array(means, dtype=np.float64)
    errs_a = np.array(errs, dtype=np.float64)

    ax.bar(
        x, means_a, yerr=errs_a, capsize=4, width=bar_w,
        color=colors, alpha=0.88,
        edgecolor="black", linewidth=0.6,
        error_kw=dict(ecolor="black", lw=0.8),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test regret")
    ax.set_title("Cliff Walking — Decision quality")
    ax.grid(True, axis="y", alpha=0.25)

    # Zoom y when TS and DFL (or all bars) sit in a narrow band so differences are visible.
    tops = means_a + errs_a
    bottoms = np.maximum(means_a - errs_a, 0.0)
    span = float(tops.max() - bottoms.min())
    mid = float(0.5 * (tops.max() + bottoms.min()))
    if span > 0 and mid > 0 and span < 0.12 * mid:
        pad = max(0.04 * mid, 1.5 * span, 1e-5)
        ax.set_ylim(bottom=max(0.0, bottoms.min() - pad), top=tops.max() + 2.2 * pad)


def render_one_n(
    *,
    v4_root: Path,
    n: int,
    k: int,
    n_seeds: int,
    out_dir: Path,
) -> None:
    root = v4_root / f"n{n}"
    if not root.is_dir():
        raise FileNotFoundError(f"Missing {root}")
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save(name: str, draw) -> None:
        prev = plt.rcParams.copy()
        plt.rcParams.update(RC)
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        draw(ax)
        fig.tight_layout()
        p = out_dir / name
        fig.savefig(p, dpi=300, bbox_inches="tight")
        if p.suffix.lower() != ".pdf":
            fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        plt.rcParams.update(prev)
        rp = (p if p.is_absolute() else MDP_CW.parent / p).resolve()
        try:
            disp = rp.relative_to(MDP_CW.parent.resolve())
        except ValueError:
            disp = rp
        print(f"  saved {disp}")

    _save("report_01_train_reward_mean_se.png", lambda ax: plot_reward_curves(ax, root, k, n_seeds, method_keys=TRAINING_METHOD_KEYS))
    _save("report_02_compute_total_time.png", lambda ax: plot_compute_total_time(ax, root, k, n_seeds))
    _save("report_03_test_mean_regret_bars.png", lambda ax: plot_regret_bars(ax, root, k, n_seeds))


def _summary(
    root: Path,
    k: int,
    n_seeds: int,
    method_keys: Optional[Tuple[str, ...]] = None,
) -> Dict:
    methods: Dict[str, Dict[str, float]] = {}
    vals_by_prefix: Dict[str, np.ndarray] = {}
    selected = _select_methods(method_keys)
    for prefix, _label, _color, _ls in selected:
        runs = load_seeds(root, k, prefix, n_seeds)
        vals = np.array(
            [float(r["test_mean_regret"]) for r in runs if "test_mean_regret" in r],
            dtype=np.float64,
        )
        vals_by_prefix[prefix] = vals
        methods[prefix] = {
            "n": int(vals.size),
            "mean_regret": float(vals.mean()) if vals.size else float("nan"),
            "median_regret": float(np.median(vals)) if vals.size else float("nan"),
            "std_regret": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
        }

    paired: Dict[str, Dict[str, float]] = {}
    for prefix, _label, _color, _ls in selected:
        if prefix == "ts":
            continue
        ts_vals, other_vals = [], []
        for s in range(n_seeds):
            ts = _load(root, k, "ts", s)
            other = _load(root, k, prefix, s)
            if ts is None or other is None:
                continue
            if "test_mean_regret" not in ts or "test_mean_regret" not in other:
                continue
            ts_vals.append(float(ts["test_mean_regret"]))
            other_vals.append(float(other["test_mean_regret"]))
        d = np.asarray(ts_vals, dtype=np.float64) - np.asarray(other_vals, dtype=np.float64)
        paired[prefix] = {
            "paired_n": int(d.size),
            "median_ts_minus_other": float(np.median(d)) if d.size else float("nan"),
            "mean_ts_minus_other": float(d.mean()) if d.size else float("nan"),
            "wins_other": int((d > 0).sum()) if d.size else 0,
        }

    return {
        "root": str(root / f"k{k}"),
        "n_seeds": n_seeds,
        "methods": methods,
        "paired_ts_minus_other": paired,
    }


def write_summary(
    root: Path,
    k: int,
    n_seeds: int,
    out_dir: Path,
    method_keys: Optional[Tuple[str, ...]] = None,
) -> None:
    selected = _select_methods(method_keys)
    s = _summary(root, k, n_seeds, method_keys)
    (out_dir / f"summary_k{k}.json").write_text(json.dumps(s, indent=2) + "\n")

    lines = [f"root={s['root']}  n_seeds={n_seeds}"]
    for prefix, label, _color, _ls in selected:
        m = s["methods"][prefix]
        lines.append(
            f"{label:<10} n={m['n']} mean={m['mean_regret']:.5f} "
            f"median={m['median_regret']:.5f} std={m['std_regret']:.5f}"
        )
    for prefix, _label, _color, _ls in selected:
        if prefix == "ts":
            continue
        p = s["paired_ts_minus_other"][prefix]
        lines.append(
            f"TS-{prefix:<4} paired_n={p['paired_n']} "
            f"medianΔ={p['median_ts_minus_other']:+.5f} "
            f"meanΔ={p['mean_ts_minus_other']:+.5f} "
            f"wins={p['wins_other']}/{p['paired_n']}"
        )
    (out_dir / f"summary_k{k}.txt").write_text("\n".join(lines) + "\n")
    print(f"  saved {out_dir / f'summary_k{k}.json'}")
    print(f"  saved {out_dir / f'summary_k{k}.txt'}")


def render_flat_root(
    *,
    root: Path,
    k: int,
    n_seeds: int,
    out_dir: Path,
    n_label: Optional[str],
    method_keys: Optional[Tuple[str, ...]] = None,
) -> None:
    if not (root / f"k{k}").is_dir():
        raise FileNotFoundError(f"Missing {root / f'k{k}'}")
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save(name: str, draw) -> None:
        prev = plt.rcParams.copy()
        plt.rcParams.update(RC)
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        draw(ax)
        fig.tight_layout()
        p = out_dir / name
        fig.savefig(p, dpi=300, bbox_inches="tight")
        if p.suffix.lower() != ".pdf":
            fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        plt.rcParams.update(prev)
        print(f"  saved {p}")

    _save(
        "report_01_train_reward_mean_se.png",
        lambda ax: plot_reward_curves(ax, root, k, n_seeds, n_label, method_keys=TRAINING_METHOD_KEYS),
    )
    _save(
        "report_02_compute_total_time.png",
        lambda ax: plot_compute_total_time(ax, root, k, n_seeds, n_label, method_keys),
    )
    _save(
        "report_03_test_mean_regret_bars.png",
        lambda ax: plot_regret_bars(ax, root, k, n_seeds, n_label, method_keys),
    )
    write_summary(root, k, n_seeds, out_dir, method_keys)


def render_all_ns(
    *,
    v4_root: Path,
    ns: Tuple[int, ...],
    k: int,
    n_seeds: int,
    out_base: Path,
) -> Tuple[int, int]:
    """Return (written_count, skipped_count)."""
    written = 0
    skipped = 0
    for n in ns:
        root = v4_root / f"n{n}"
        if not root.is_dir():
            rp = (root if root.is_absolute() else MDP_CW.parent / root).resolve()
            try:
                disp = rp.relative_to(MDP_CW.parent.resolve())
            except ValueError:
                disp = rp
            print(f"  skip n={n}: missing {disp}")
            skipped += 1
            continue
        sub = out_base / f"n{n}"
        render_one_n(v4_root=v4_root, n=n, k=k, n_seeds=n_seeds, out_dir=sub)
        written += 1
    return written, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v4_root", type=Path, default=MDP_CW / "results_n120")
    ap.add_argument(
        "--ns",
        type=str,
        default="20,40,80,120,160",
        help="Comma-separated sample budgets (each must exist as v4_root/n{N}/).",
    )
    ap.add_argument("--k", type=int, default=7, help="Skew strength k (default: 7).")
    ap.add_argument("--n_seeds", type=int, default=50)
    ap.add_argument(
        "--flat_root",
        type=Path,
        default=None,
        help="Render a single flat ROOT/k{k}/ run instead of ROOT/n{N}/k{k}/.",
    )
    ap.add_argument(
        "--n_label",
        type=str,
        default=None,
        help="Display label for n in plot titles when using --flat_root.",
    )
    ap.add_argument(
        "--methods",
        type=str,
        default=",".join(DEFAULT_METHOD_KEYS),
        help="Comma-separated method keys to plot. Available: ts,dfl,qpth,ssa,sketch.",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=MDP_CW / "final_results_best" / "report_three_panel",
    )
    args = ap.parse_args()
    method_keys = tuple(x.strip() for x in args.methods.split(",") if x.strip())
    if args.flat_root is not None:
        render_flat_root(
            root=args.flat_root,
            k=args.k,
            n_seeds=args.n_seeds,
            out_dir=args.out_dir,
            n_label=args.n_label,
            method_keys=method_keys,
        )
        return 0

    ns_tuple = tuple(int(x.strip()) for x in args.ns.split(",") if x.strip())
    if not ns_tuple:
        raise SystemExit("empty --ns")
    w, sk = render_all_ns(
        v4_root=args.v4_root,
        ns=ns_tuple,
        k=args.k,
        n_seeds=args.n_seeds,
        out_base=args.out_dir,
    )
    print(f"report_three_panel: wrote {w} n-folder(s), skipped {sk} missing.")
    if w == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
