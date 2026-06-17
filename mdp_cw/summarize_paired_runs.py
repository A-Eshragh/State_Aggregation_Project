"""
Paired TS vs DFL summary for a dose-style directory layout::

    ROOT/k{K}/ts_s{S}/results.json
    ROOT/k{K}/dfl_s{S}/results.json

Prints median regrets, median paired (TS−DFL), win count, Wilcoxon one-sided
(alternative: median paired diff > 0 ⇒ DFL lower regret).

Example::

    python -m mdp_cw.summarize_paired_runs --root mdp_cw/results_n120/n160 --k 7 --n_seeds 50

With qpth (paired seeds only)::

    python -m mdp_cw.summarize_paired_runs --root mdp_cw/results_k11_allambig --k 11 --n_seeds 50 --qpth
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats as st


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--k", type=int, default=7)
    p.add_argument("--n_seeds", type=int, default=50)
    p.add_argument(
        "--qpth",
        action="store_true",
        help="Also print paired TS−qpth (same seeds) when qpth_s{seed}/results.json exists.",
    )
    args = p.parse_args()
    root = Path(args.root)
    k = args.k
    n = args.n_seeds

    ts_r, dfl_r = [], []
    for s in range(n):
        tp = root / f"k{k}" / f"ts_s{s}" / "results.json"
        dp = root / f"k{k}" / f"dfl_s{s}" / "results.json"
        if not tp.exists() or not dp.exists():
            continue
        ts_r.append(float(json.loads(tp.read_text())["test_mean_regret"]))
        dfl_r.append(float(json.loads(dp.read_text())["test_mean_regret"]))

    if len(ts_r) < 3:
        print(f"Need ≥3 paired runs; found {len(ts_r)} under {root}/k{k}/")
        return

    ts_r = np.asarray(ts_r, dtype=float)
    dfl_r = np.asarray(dfl_r, dtype=float)
    d = ts_r - dfl_r
    print(f"root={root}  k={k}  paired_n={len(d)}")
    print(f"  median regret   TS: {np.median(ts_r):.5f}   DFL: {np.median(dfl_r):.5f}")
    print(f"  median paired (TS−DFL): {np.median(d):.5f}")
    print(f"  DFL wins (TS regret > DFL): {int((d > 0).sum())} / {len(d)}")
    try:
        w = st.wilcoxon(d, alternative="greater", zero_method="wilcox")
        print(f"  Wilcoxon one-sided p (DFL better): {w.pvalue:.4g}")
    except Exception as e:
        print(f"  Wilcoxon: {e}")

    if args.qpth:
        tsq, qpq = [], []
        for s in range(n):
            tp = root / f"k{k}" / f"ts_s{s}" / "results.json"
            qp = root / f"k{k}" / f"qpth_s{s}" / "results.json"
            if not tp.exists() or not qp.exists():
                continue
            tsq.append(float(json.loads(tp.read_text())["test_mean_regret"]))
            qpq.append(float(json.loads(qp.read_text())["test_mean_regret"]))
        if len(tsq) < 3:
            print(f"  qpth: need ≥3 paired runs; found {len(tsq)}")
            return
        tsq_arr = np.asarray(tsq)
        qpq_arr = np.asarray(qpq)
        dq = tsq_arr - qpq_arr
        print(f"  qpth paired_n={len(dq)}  median regrets TS: {np.median(tsq_arr):.5f} qpth: {np.median(qpq_arr):.5f}")
        print(f"  median paired (TS−qpth): {np.median(dq):.5f}")
        print(f"  qpth wins: {int((dq > 0).sum())} / {len(dq)}")
        try:
            wq = st.wilcoxon(dq, alternative="greater", zero_method="wilcox")
            print(f"  Wilcoxon one-sided p (qpth better): {wq.pvalue:.4g}")
        except Exception as e:
            print(f"  qpth Wilcoxon: {e}")


if __name__ == "__main__":
    main()
