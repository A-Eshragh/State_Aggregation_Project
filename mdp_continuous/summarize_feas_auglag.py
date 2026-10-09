"""Summarize the CartPole rebuttal arms (DFL-Feas [34] and the aug-Lagrangian
DFL loss) against the paper's Eq.(5) penalty baseline.

All arms share every hyperparameter except the Phase-B loss and are run on the
same seed list, so comparisons are PAIRED: per-seed differences are the
meaningful statistic, not just the marginal means.

Usage:
    python -m mdp_continuous.summarize_feas_auglag
    python -m mdp_continuous.summarize_feas_auglag --results_dir <dir>
"""
import argparse
import json
import math
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DIR = HERE / "results" / "rebuttal_feas_auglag"

# (label, filename predicate) — order controls table order.
# NOTE the prefix must be "curve_neural_beh", not "curve_neural": the DFL-SA
# sweep writes "curve_neural_dfl_nfeat<d>_beh..." into the same directory, and a
# looser prefix silently matches those instead of the DFL-LP arm.
_LP = "curve_neural_beh"
ARMS = [
    ("Two-stage",               lambda n: n.startswith("curve_two_stage")),
    ("DFL-LP (penalty, paper)", lambda n: n.startswith(_LP) and "_lm" not in n),
    ("DFL-LP (lagrangian)",     lambda n: n.startswith(_LP) and "_lmlagrangian" in n),
    ("DFL-LP (auglag)",         lambda n: n.startswith(_LP) and "_lmauglag" in n),
    ("DFL-Sketch (penalty)",    lambda n: n.startswith("curve_sketch")
                                          and "_lm" not in n),
    ("DFL-Sketch (auglag)",     lambda n: n.startswith("curve_sketch")
                                          and "_lmauglag" in n),
    ("DFL-QP",                  lambda n: n.startswith("curve_qpth")),
    ("DFL-Feas [34] (exact y-)", lambda n: n.startswith("curve_feas")
                                           and "negplain" in n),
    ("DFL-Feas [34] (arm y-)",   lambda n: n.startswith("curve_feas")
                                           and "negarm" in n),
]

BASELINE = "DFL-LP (penalty, paper)"


def sem(xs):
    if len(xs) < 2:
        return float("nan")
    return st.stdev(xs) / math.sqrt(len(xs))


def load(results_dir: Path):
    found = {}
    for label, pred in ARMS:
        hits = [p for p in sorted(results_dir.glob("*.json")) if pred(p.name)]
        if not hits:
            continue
        if len(hits) > 1:
            # Prefer the canonical (least-suffixed) variant, not an arbitrary
            # tuning run: with dw/dfo/rho sweeps on disk, "newest" would silently
            # report whichever finished last.
            hits.sort(key=lambda p: (len(p.name), p.name))
            others = ", ".join(q.name.split("evallp")[-1].lstrip("_") or "(base)"
                               for q in hits[1:])
            print(f"[warn] {label}: {len(hits)} matching files; using "
                  f"{hits[0].name.split('evallp')[-1].lstrip('_') or '(base)'} "
                  f"— also present: {others}")
        p = hits[0]
        d = json.load(open(p))
        found[label] = {
            "path": p,
            "seeds": d["seeds"],
            # best-over-iterations is the metric the paper's panel reports
            "best": {s: max(r) for s, r in zip(d["seeds"], d["rewards_per_seed"])},
            "final": {s: r[-1] for s, r in zip(d["seeds"], d["rewards_per_seed"])},
            "time": d.get("times_per_seed", []),
            "solver_time": d.get("solver_times_per_seed", []),
            "feas_grad_hits": d.get("feas_grad_hits", []),
            "feas_grad_steps": d.get("feas_grad_steps", []),
            "dual_fallbacks": d.get("dual_fallbacks", []),
            "dual_solves": d.get("dual_solves", []),
        }
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default=str(DEFAULT_DIR))
    args = ap.parse_args()

    rd = Path(args.results_dir)
    arms = load(rd)
    if not arms:
        raise SystemExit(f"No result JSONs found in {rd}")

    base = arms.get(BASELINE)
    common = None
    for a in arms.values():
        s = set(a["seeds"])
        common = s if common is None else (common & s)
    common = sorted(common)
    print(f"CartPole (Section 6.3 setting) — {len(common)} paired seeds: {common}")
    print(f"results: {rd}")
    print()

    hdr = (f"{'arm':<26} {'best mean±SE':>16} {'final mean±SE':>16} "
           f"{'Δbest vs penalty':>18} {'wins':>7} {'time/seed':>10}")
    print(hdr)
    print("-" * len(hdr))
    for label, _ in ARMS:
        a = arms.get(label)
        if a is None:
            continue
        b = [a["best"][s] for s in common]
        f = [a["final"][s] for s in common]
        row = (f"{label:<26} {st.mean(b):>9.2f}±{sem(b):<6.2f} "
               f"{st.mean(f):>9.2f}±{sem(f):<6.2f} ")
        if base is not None and label != BASELINE:
            d = [a["best"][s] - base["best"][s] for s in common]
            wins = sum(1 for x in d if x > 0)
            row += f"{st.mean(d):>+11.2f}±{sem(d):<5.2f} {wins:>3}/{len(d):<3} "
        else:
            row += f"{'—':>18} {'—':>7} "
        t = a["time"]
        row += f"{(st.mean(t) if t else float('nan')):>9.1f}s"
        print(row)

    print()
    for label, _ in ARMS:
        a = arms.get(label)
        if a is None:
            continue
        steps = sum(a["feas_grad_steps"] or [])
        if steps:
            hits = sum(a["feas_grad_hits"])
            print(f"[{label}] gradient audit: {hits}/{steps} DFL steps with a "
                  f"non-zero feature gradient ({hits/steps:.1%})")
        dsolv = sum(a["dual_solves"] or [])
        if dsolv:
            dfb = sum(a["dual_fallbacks"])
            print(f"[{label}] true-LP dual solves: {dsolv}, slack fallback on "
                  f"{dfb} ({dfb/dsolv:.1%}) — multiplier then bounded only by the "
                  f"slack penalty")

    missing = [l for l, _ in ARMS if l not in arms]
    if missing:
        print()
        print("[warn] arms with no results file:", ", ".join(missing))


if __name__ == "__main__":
    main()
