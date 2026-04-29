#!/usr/bin/env bash
# Make two-stage worse: undertrain it (n_epochs=5 instead of 50).
# Same M=56, N_SAMPLES=20, PRETRAIN=50, 10 seeds.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

for SEED in 1 2 3 4 5 6 7 8 9 10; do
  for N_EP in 5 10 20; do
    TAG="seed_${SEED}_s50q50_ts_ep${N_EP}"
    echo "[ts_undertrain] seed=$SEED  n_epochs=$N_EP"
    python -m mdp_lp.run_elasticity_experiments \
      --experiment_type b --cost_case high_margin \
      --algo twostage --n_agg -1 \
      --n_epochs "$N_EP" --n_samples 20 --lr 0.01 \
      --pretrain_epochs 50 \
      --b_max 5 --s_max 50 --q_max 50 \
      --aug_lag_rho 10000.0 \
      --seed "$SEED" --tag "$TAG"
  done
done
echo "[ts_undertrain] Done."
