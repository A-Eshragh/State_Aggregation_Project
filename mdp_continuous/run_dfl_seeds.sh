#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20}"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
WORKERS="${WORKERS:-10}"

echo "[reproduce] seeds=$SEEDS"
echo "[reproduce] workers=$WORKERS"
echo "[reproduce] DFL-SA configs: d=30,60 → K=8 r=0.5 ; d=90 → K=4 r=0.8"
echo


echo "[1/4] run_dfl_vs_qpth: lp, qpth, sketch, two_stage"
# python -m mdp_continuous.run_dfl_vs_qpth \
#     --solvers lp qpth sketch two_stage \
#     --seeds $SEEDS \
#     --n_workers "$WORKERS"

echo "[2/4] sweep_n_features  d=30,60"
python -m mdp_continuous.sweep_n_features \
    --seeds $SEEDS \
    --n_features_list 30 60 \
    --n_workers "$WORKERS" \
    --sa_K 8 \
    --sa_ratio 0.5 \
    --show_running_max \
    --force

echo "[3/4] sweep_n_features  d=90"
python -m mdp_continuous.sweep_n_features \
    --seeds $SEEDS \
    --n_features_list 90 \
    --n_workers "$WORKERS" \
    --sa_K 4 \
    --sa_ratio 0.8 \
    --show_running_max \
    --force

# 4. Re-render PDFs from the (now-updated) caches.
echo "[4/4] make_figures.py cartpole"
python make_figures.py cartpole

echo
echo "[reproduce] done. Output PDFs:"
ls -la "$ROOT/figures/cartpole_"*.pdf
