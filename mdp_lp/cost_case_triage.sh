#!/usr/bin/env bash
# Single-seed triage: does twostage become clearly worse than dfl in different cost cases?
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

for COST in low_margin normal_margin; do
  for SEED in 1 2 3; do
    for ALGO in twostage dfl; do
      echo "[triage] cost=$COST seed=$SEED algo=$ALGO"
      python -m mdp_lp.run_elasticity_experiments \
        --experiment_type b --cost_case "$COST" \
        --algo "$ALGO" --n_agg -1 \
        --n_epochs 50 --n_samples 20 --lr 0.01 \
        --pretrain_epochs 50 \
        --b_max 5 --s_max 50 --q_max 50 \
        --aug_lag_rho 10000.0 \
        --seed "$SEED" --tag "seed_${SEED}_s50q50_costtriage"
    done
  done
done
echo "[triage] Done."
