#!/usr/bin/env bash
# CartPole, 10 seeds per method. Runs every arm, then re-derives the table from
# the caches and checks it, so a green run is the reproduction check.
# Seeds differ per row (DFL-LP 20, DFL-Sketch 5 at 30 iterations), so report
# them with the table.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RECOVERY_PY="/net/csefiles/siemens/warriors-ls-recovery/users/zzhao628/#113442817/anaconda3/envs/rl/bin/python"
if [ -x "$HOME/anaconda3/envs/rl/bin/python" ]; then
  PY="${PY:-$HOME/anaconda3/envs/rl/bin/python}"
else
  PY="${PY:-$RECOVERY_PY}"
fi
[ -x "$PY" ] || { echo "[fatal] no usable python at: $PY" >&2; exit 1; }

# Gurobi segfaults on deep stacks for large LPs.
ulimit -s 65536
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

WORKERS="${WORKERS:-10}"

SEEDS_MAIN="${SEEDS_MAIN:-1 2 3 4 5 6 7 8 9 10}"   # QP, Two-stage, Feas, SA
SEEDS_LP="${SEEDS_LP:-$(seq 1 20 | tr '\n' ' ')}"  # DFL-LP row: 20 seeds
SEEDS_SKETCH="${SEEDS_SKETCH:-4 5 6 7 8}"          # DFL-Sketch row: 5 seeds
ITERS="${ITERS:-60}"
ITERS_SKETCH="${ITERS_SKETCH:-30}"                 # DFL-Sketch row: 30 iters
SKETCH_K="${SKETCH_K:-8}"
SKETCH_R="${SKETCH_R:-0.2}"

AUGLAG_FLAGS="${AUGLAG_FLAGS:-}"
if [ -n "$AUGLAG_FLAGS" ]; then
  echo "[warn] AUGLAG_FLAGS='$AUGLAG_FLAGS' set — these runs write _lmauglag"
  echo "[warn] caches and do NOT reproduce the published table."
fi

# Figures need one extra run (the 60-iteration sketch arm). Off by default.
RENDER_FIGURES="${RENDER_FIGURES:-0}"

# VERIFY_ONLY=1 checks the cached results without running anything.
VERIFY_ONLY="${VERIFY_ONLY:-0}"
run() {
  if [ "$VERIFY_ONLY" != "0" ]; then
    echo "      [skip: VERIFY_ONLY] $*"
    return 0
  fi
  "$@"
}

echo "[reproduce] CartPole results table"
echo "[reproduce] workers=$WORKERS  loss=${AUGLAG_FLAGS:-penalty (Eq.5)}"
echo "[reproduce] seeds: main=[$SEEDS_MAIN]"
echo "[reproduce]        lp=[$SEEDS_LP]"
echo "[reproduce]        sketch=[$SEEDS_SKETCH] @ ${ITERS_SKETCH} iters"
echo "[reproduce] DFL-SA configs: d=30 -> K=4 r=0.8 ; d=60 -> K=8 r=0.5 ; d=90 -> K=4 r=0.8"
echo

echo "[1/8] DFL-LP  (lp, 20 seeds)"
run "$PY" -m mdp_continuous.run_dfl_vs_qpth \
    --solvers lp \
    --seeds $SEEDS_LP \
    --n_iterations "$ITERS" \
    --n_workers "$WORKERS" \
    $AUGLAG_FLAGS

echo "[2/8] DFL-QP + Two-stage  (qpth, two_stage)"
run "$PY" -m mdp_continuous.run_dfl_vs_qpth \
    --solvers qpth two_stage \
    --seeds $SEEDS_MAIN \
    --n_iterations "$ITERS" \
    --n_workers "$WORKERS"

echo "[3/8] DFL-Feas  (feas, Odece [34])"
run "$PY" -m mdp_continuous.run_dfl_vs_qpth \
    --solvers feas \
    --odece_neg_solver plain \
    --seeds $SEEDS_MAIN \
    --n_iterations "$ITERS" \
    --n_workers "$WORKERS"

echo "[4/8] DFL-Sketch  (sketch, ${ITERS_SKETCH} iters, K=$SKETCH_K r=$SKETCH_R)"
run "$PY" -m mdp_continuous.run_dfl_vs_qpth \
    --solvers sketch \
    --seeds $SEEDS_SKETCH \
    --n_iterations "$ITERS_SKETCH" \
    --n_sketches "$SKETCH_K" \
    --sketch_ratio "$SKETCH_R" \
    --n_workers "$WORKERS" \
    $AUGLAG_FLAGS

echo "[5/8] DFL-SA  d=30,90  (K=4, r=0.8)"
run "$PY" -m mdp_continuous.sweep_n_features \
    --seeds $SEEDS_MAIN \
    --n_features_list 30 90 \
    --n_iterations "$ITERS" \
    --n_workers "$WORKERS" \
    --sa_K 4 \
    --sa_ratio 0.8 \
    --show_running_max \
    --force \
    $AUGLAG_FLAGS

echo "[6/8] DFL-SA  d=60  (K=8, r=0.5)"
run "$PY" -m mdp_continuous.sweep_n_features \
    --seeds $SEEDS_MAIN \
    --n_features_list 60 \
    --n_iterations "$ITERS" \
    --n_workers "$WORKERS" \
    --sa_K 8 \
    --sa_ratio 0.5 \
    --show_running_max \
    --force \
    $AUGLAG_FLAGS

if [ "$RENDER_FIGURES" != "0" ]; then
  echo "[7/8] extra sketch run for the figure  (60 iters, seeds 1-8)"
  run "$PY" -m mdp_continuous.run_dfl_vs_qpth \
      --solvers sketch \
      --seeds 1 2 3 4 5 6 7 8 \
      --n_iterations 60 \
      --n_sketches "$SKETCH_K" \
      --sketch_ratio "$SKETCH_R" \
      --n_workers "$WORKERS"
  echo "[7/8] make_figures.py cartpole"
  run "$PY" make_figures.py cartpole
  ls -la "$ROOT/figures/cartpole_"*.pdf
else
  echo "[7/8] figures skipped (RENDER_FIGURES=1 to render)"
fi

# ── [8/8] Verify: re-derive the table from the caches and diff it ────────────
echo
echo "[8/8] verifying against the published table"
"$PY" - <<'PYEOF'
import json, math, sys
import statistics as st
from pathlib import Path

CACHE = Path("mdp_continuous/results/dfl_vs_qpth")

# Shared cache-key stems. run_dfl_vs_qpth puts _iters before _nfeat;
# sweep_n_features puts _nfeat first. Both are reproduced verbatim here.
def anchor(prefix, iters, tail=""):
    return (f"{prefix}_behbiased_td0.013_eps0.5_ntraj100_iters{iters}"
            f"_nfeat1024_preTS30_preDFL30_nce0_ndfl5_dfllrauto_evallp{tail}")

def sweep(d, K, r):
    return (f"curve_neural_dfl_nfeat{d}_behbiased_td0.013_eps0.5_ntraj100"
            f"_iters60_preTS30_preDFL30_nce0_ndfl5_dfllrauto_evallp"
            f"_saK{K}_sar{r}")

# (label, cache stem, expected mean, expected SE, expected solver s/ep, n_seeds)
ROWS = [
    ("DFL-SA (d = 90)", sweep(90, 4, 0.8),                     196.715,  2.292,  0.24, 10),
    ("DFL-SA (d = 60)", sweep(60, 8, 0.5),                     193.052,  3.723,  0.23, 10),
    ("DFL-QP",          anchor("curve_qpth", 60, "_qpeta0.01"), 189.681,  4.338, 10.10, 10),
    ("DFL-SA (d = 30)", sweep(30, 4, 0.8),                     184.534,  8.423,  0.15, 10),
    ("DFL-LP",          anchor("curve_neural", 60),             176.171,  5.444,  5.64, 20),
    ("DFL-Sketch",      anchor("curve_sketch", 30,
                               "_K8_r0.2_mgaussian"),          175.750,  9.138,  4.83,  5),
    ("DFL-Feas",        anchor("curve_feas", 60,
                               "_a0.5_mu0.0_negplain"),        159.970, 11.768, 12.18, 10),
    ("Two-stage",       anchor("curve_two_stage_mbrl", 60),     154.435, 11.477,  0.00, 10),
]

print()
print(f"{'method':<16} {'best reward':>21}  {'s/ep':>6}  {'seeds':>5}  "
      f"{'d(mean)':>8}  status")
print("-" * 74)

bad = 0
for label, stem, exp_mu, exp_se, exp_sep, exp_n in ROWS:
    path = CACHE / f"{stem}.json"
    if not path.exists():
        print(f"{label:<16} {'MISSING':>21}  {'':>6}  {'':>5}  {'':>8}  "
              f"FAIL  no cache: {path.name}")
        bad += 1
        continue
    p = json.loads(path.read_text())
    best = [max(r) for r in p["rewards_per_seed"]]
    n = len(best)
    mu = st.mean(best)
    se = st.stdev(best) / math.sqrt(n) if n > 1 else 0.0
    niter = len(p["rewards_per_seed"][0])
    stt = p.get("solver_times_per_seed") or [0.0] * n
    sep = st.mean(stt) / niter

    notes = []
    if n != exp_n:
        notes.append(f"n={n} (expected {exp_n})")
    # These runs are seeded, so a same-machine re-run is normally bit-identical;
    # accept anything inside one standard error as a reproduction.
    d_mu = mu - exp_mu
    ok = abs(d_mu) <= max(exp_se, 1e-6) and not notes
    if abs(d_mu) > max(exp_se, 1e-6):
        notes.append(f"mean off by {d_mu:+.3f} (> 1 SE = {exp_se:.3f})")
    if exp_sep > 0 and abs(sep - exp_sep) > max(0.25 * exp_sep, 0.02):
        notes.append(f"s/ep {sep:.2f} vs {exp_sep:.2f}")
        ok = False

    status = "OK  " if ok else "FAIL"
    if not ok:
        bad += 1
    exact = "  (exact)" if abs(d_mu) < 5e-4 else ""
    print(f"{label:<16} {mu:>10.3f} +/- {se:>6.3f}  {sep:>6.2f}  {n:>5}  "
          f"{d_mu:>+8.3f}  {status}{exact}"
          + ("  " + "; ".join(notes) if notes else ""))

print("-" * 74)
print("protocol audit (not uniform — see the header of this script):")
for label, stem, *_ in ROWS:
    path = CACHE / f"{stem}.json"
    if not path.exists():
        continue
    p = json.loads(path.read_text())
    print(f"  {label:<16} seeds={p['seeds']}  "
          f"iters={len(p['rewards_per_seed'][0])}")

if bad:
    print(f"\n[verify] {bad}/{len(ROWS)} rows did NOT reproduce.")
    sys.exit(1)
print(f"\n[verify] all {len(ROWS)} rows reproduce the published table.")
PYEOF

echo
echo "[reproduce] done."
