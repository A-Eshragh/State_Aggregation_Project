"""
Summarize the scalability sweep (run_scaling_sweep.sh).

For each (|S|, arm): mean +/- SE test regret over seeds, LP-layer wall time per
epoch (solver forward+backward; + aggregation time for the agg arm), total
training wall-clock, and run status. Runs that hit the wall budget still report
timing (from the periodic checkpoint.json) but no regret -> "> budget".

Also writes a two-panel rebuttal figure (runtime log-log + regret vs |S|).

Run from repo root:
  python -m mdp_lp.summarize_scaling
  python -m mdp_lp.summarize_scaling --results_root mdp_lp/results_scaling_probe \
      --log_dir mdp_lp/logs/scaling_probe
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

ARM_ORDER = ["full", "fulll20", "agg10", "aggratio", "aggl20",
             "agg10al", "aggratioal", "aggl20al", "twostage"]
ARM_LABEL = {
    "full": "Full-state DFL",
    "fulll20": "Full DFL (lw=20)",
    "agg10": "Ours (k=10)",
    "aggratio": "Ours (k=|S|/10)",
    "aggl20": "Ours (k=10, lw=20)",
    "agg10al": "Ours (k=10, aug-Lag)",
    "aggratioal": "Ours (k=|S|/10, aug-Lag)",
    "aggl20al": "Ours (k=10, lw=20, aug-Lag)",
    "twostage": "Two-stage",
}
# Categorical slots of the reference palette (adjacent-pairs validated, light
# mode). Color follows the arm, never its position in a panel. The aug-Lag
# arms are the headline "Ours"; blue marks the full proposed method.
ARM_COLOR = {"agg10": "#2a78d6", "full": "#eb6834", "twostage": "#1baf7a",
             "aggratio": "#eda100", "aggl20al": "#2a78d6",
             "agg10al": "#eda100", "aggratioal": "#e87ba4"}

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e6e3"
SPINE = "#b5b4ae"


def load_runs(results_root: Path, log_dir: Path):
    """One record per (size, arm, seed): regret/timing where available + status."""
    runs = []
    for status_file in sorted(log_dir.glob("S*_seed*.status")):
        m = re.match(r"S(\d+)_(\w+?)_seed(\d+)$", status_file.stem)
        if not m:
            continue
        size, arm, seed = int(m.group(1)), m.group(2), int(m.group(3))
        status = status_file.read_text().strip()
        tag = status_file.stem
        algo_dir = "twostage" if arm == "twostage" else "dfl"
        tag_dir = results_root / "exp_b" / "normal_margin" / algo_dir / f"tag_{tag}"

        rec = dict(size=size, arm=arm, seed=seed, status=status,
                   regret=np.nan, solver_s=np.nan, sa_s=np.nan, wall=np.nan)
        res_files = sorted(tag_dir.glob("**/results.json"))
        if res_files:
            res = json.load(open(res_files[0]))
            rec["regret"] = float(res["test_mean_regret"])
        # Timing: prefer the final checkpoint, fall back to the periodic one so
        # timed-out runs still contribute measured per-epoch cost.
        ckpt = None
        for name in ("checkpoint_final.json", "checkpoint.json"):
            hits = sorted(tag_dir.glob(f"**/{name}"))
            if hits:
                ckpt = json.load(open(hits[0]))
                break
        if ckpt:
            solver = ckpt.get("solver_time_sec_per_epoch") or []
            sa = ckpt.get("sa_time_sec_per_epoch") or []
            wall = ckpt.get("train_time") or []
            if solver:
                rec["solver_s"] = float(np.mean(solver))
            if sa:
                rec["sa_s"] = float(np.mean(sa))
            if wall:
                rec["wall"] = float(wall[-1])
        runs.append(rec)
    return runs


def aggregate(runs):
    """-> {(size, arm): dict} with means/SEs and status counts."""
    out = {}
    keys = sorted({(r["size"], r["arm"]) for r in runs})
    for size, arm in keys:
        grp = [r for r in runs if r["size"] == size and r["arm"] == arm]
        regs = np.array([r["regret"] for r in grp if np.isfinite(r["regret"])])
        # per-epoch LP-layer cost = solver (+ aggregation for the agg arm)
        lp = np.array([r["solver_s"] + (r["sa_s"] if np.isfinite(r["sa_s"]) else 0.0)
                       for r in grp if np.isfinite(r["solver_s"])])
        walls = np.array([r["wall"] for r in grp if np.isfinite(r["wall"])])
        n_to = sum(r["status"] == "timeout" for r in grp)
        n_fail = sum(r["status"].startswith("failed") for r in grp)
        n_run = sum(r["status"] == "running" for r in grp)
        out[(size, arm)] = dict(
            n_done=len(regs), n_timeout=n_to, n_failed=n_fail, n_running=n_run,
            regret_mean=regs.mean() if len(regs) else np.nan,
            regret_se=regs.std(ddof=1) / np.sqrt(len(regs)) if len(regs) > 1 else np.nan,
            lp_s_mean=lp.mean() if len(lp) else np.nan,
            wall_mean=walls.mean() if len(walls) else np.nan,
        )
    return out


def fmt_pm(mean, se):
    if not np.isfinite(mean):
        return "-"
    if np.isfinite(se):
        return f"{mean:.3f} ± {se:.3f}"
    return f"{mean:.3f}"


def make_table(agg):
    sizes = sorted({s for s, _ in agg})
    hdr = (f"{'|S|':>6} {'arm':>27} {'runs':>12} {'test regret':>18} "
           f"{'LP s/epoch':>11} {'train wall':>11}")
    lines = [hdr, "-" * len(hdr)]
    for size in sizes:
        for arm in ARM_ORDER:
            a = agg.get((size, arm))
            if a is None:
                continue
            status = f"{a['n_done']} done"
            if a["n_timeout"]:
                status += f", {a['n_timeout']} >budget"
            if a["n_failed"]:
                status += f", {a['n_failed']} fail"
            if a["n_running"]:
                status += f", {a['n_running']} run"
            regret = fmt_pm(a["regret_mean"], a["regret_se"])
            if a["n_done"] == 0 and a["n_timeout"]:
                regret = "> budget"
            lp = f"{a['lp_s_mean']:.2f}" if np.isfinite(a["lp_s_mean"]) else "-"
            wall = f"{a['wall_mean']:.0f}s" if np.isfinite(a["wall_mean"]) else "-"
            lines.append(f"{size:>6} {ARM_LABEL[arm]:>27} {status:>12} "
                         f"{regret:>18} {lp:>11} {wall:>11}")
        lines.append("")
    return "\n".join(lines)


def fit_exponent(sizes, ys):
    """Empirical log-log slope, for the '~|S|^p' annotation."""
    mask = np.isfinite(ys) & (ys > 0)
    if mask.sum() < 2:
        return None
    return float(np.polyfit(np.log(np.asarray(sizes)[mask]),
                            np.log(ys[mask]), 1)[0])


def make_figure(agg, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = sorted({s for s, _ in agg})
    fig, (ax_t, ax_r) = plt.subplots(1, 2, figsize=(9.2, 3.5), dpi=150)
    for ax in (ax_t, ax_r):
        ax.set_facecolor("white")
        ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(SPINE)
        ax.tick_params(colors=INK_2, labelsize=8)
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.minorticks_off()
        ax.set_xlabel("number of states |S|", color=INK, fontsize=9)

    # Left: LP-layer seconds per training epoch (twostage solves no LP while
    # training, so it has no line here).
    ax_t.set_yscale("log")
    for arm in ("full", "aggl20al", "aggratioal"):
        ys = np.array([agg.get((s, arm), {}).get("lp_s_mean", np.nan) for s in sizes])
        ax_t.plot(sizes, ys, color=ARM_COLOR[arm], linewidth=1.8, marker="o",
                  markersize=5.5, markeredgecolor="white", markeredgewidth=1.2,
                  zorder=3, label=ARM_LABEL[arm])
        p = fit_exponent(sizes, ys)
        if p is not None:
            last = np.where(np.isfinite(ys))[0]
            if len(last):
                i = last[-1]
                ax_t.annotate(f"$\\sim |S|^{{{p:.1f}}}$", (sizes[i], ys[i]),
                              xytext=(6, -2), textcoords="offset points",
                              color=INK_2, fontsize=8)
    ax_t.set_ylabel("LP layer sec / epoch (solve + backward)", color=INK, fontsize=9)
    ax_t.set_title("Training cost of the decision layer", color=INK, fontsize=10)
    ax_t.legend(frameon=False, fontsize=8, labelcolor=INK)

    # Right: decision quality (test regret, mean ± SE over seeds).
    for arm in ("full", "aggl20al", "agg10al", "twostage"):
        ys = np.array([agg.get((s, arm), {}).get("regret_mean", np.nan) for s in sizes])
        es = np.array([agg.get((s, arm), {}).get("regret_se", np.nan) for s in sizes])
        ax_r.errorbar(sizes, ys, yerr=np.where(np.isfinite(es), es, 0.0),
                      color=ARM_COLOR[arm], linewidth=1.8, marker="o",
                      markersize=5.5, markeredgecolor="white", markeredgewidth=1.2,
                      capsize=2.5, zorder=3, label=ARM_LABEL[arm])
        for s_, y_ in zip(sizes, ys):
            a = agg.get((s_, arm))
            if a and a["n_done"] == 1 and np.isfinite(y_):
                ax_r.annotate("1 seed", (s_, y_), xytext=(6, 4),
                              textcoords="offset points", color=INK_2, fontsize=7)
    ax_r.set_ylabel("test regret", color=INK, fontsize=9)
    ax_r.set_title("Decision quality", color=INK, fontsize=10)
    ax_r.legend(frameon=False, fontsize=8, labelcolor=INK)

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[figure] wrote {out_path.with_suffix('.png')} and .pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="mdp_lp/results_scaling")
    ap.add_argument("--log_dir", default="mdp_lp/logs/scaling_sweep")
    ap.add_argument("--no_fig", action="store_true")
    args = ap.parse_args()

    runs = load_runs(Path(args.results_root), Path(args.log_dir))
    if not runs:
        print(f"no runs found (status files under {args.log_dir})")
        return
    agg = aggregate(runs)
    table = make_table(agg)
    print(table)
    out = Path(args.results_root) / "scaling_summary.txt"
    out.write_text(table + "\n")
    print(f"[table] wrote {out}")
    if not args.no_fig:
        make_figure(agg, Path(args.results_root) / "scaling_fig")


if __name__ == "__main__":
    main()
