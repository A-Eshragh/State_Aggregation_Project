#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20}"
WORKERS="${WORKERS:-8}"
N_FEATURES_LIST="${N_FEATURES_LIST:-8 16 32 64 128 256 1024}"

echo "[reproduce] seeds=$SEEDS"
echo "[reproduce] workers=$WORKERS"
echo "[reproduce] n_features_list=$N_FEATURES_LIST"
echo


echo "[1/3] run_dfl_vs_qpth: lp, qpth, sketch, two_stage"
python -m mdp_continuous.run_dfl_vs_qpth \
    --solvers lp qpth sketch two_stage \
    --seeds $SEEDS \
    --n_workers "$WORKERS"

# 2. Panel-2 sweep: DFL-SA across feature dimensions.
echo "[2/3] sweep_n_features"
python -m mdp_continuous.sweep_n_features \
    --seeds $SEEDS \
    --n_features_list $N_FEATURES_LIST \
    --n_workers "$WORKERS"

# 3. Re-render PDFs from the (now-updated) caches.
echo "[3/3] make_figures.py cartpole"
python make_figures.py cartpole

echo
echo "[reproduce] done. Output PDFs:"
ls -la "$ROOT/figures/cartpole_"*.pdf
