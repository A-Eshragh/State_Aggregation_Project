#!/usr/bin/env bash
# Sweep n_samples ∈ {25, 50, 100} for twostage vs dfl, 3 seeds each.
# n_samples passed to runner is the TRAIN size; runner generates 4× total (1:3 train:test split).

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SEEDS=(1 2 3)
N_SAMPLES_LIST=(25 50)
EXPERIMENT_TYPE="b"
COST_CASE="normal_margin"
BC=10
PRETRAIN_EPOCHS=50
B_MAX=5
S_MAX=30
Q_MAX=30
DIM_TAG="s${S_MAX}q${Q_MAX}"
AUG_LAG_RHO_DFL=10000.0
RESULTS_ROOT="mdp_lp/results"

# NAME | ALGO | N_AGG | N_EPOCHS | LR
CONFIGS=(
  "twostage|twostage|-1|50|0.01"
  "dfl|dfl|-1|50|0.01"
)

for N_SAMPLES in "${N_SAMPLES_LIST[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for CFG in "${CONFIGS[@]}"; do
      IFS='|' read -r NAME ALGO N_AGG N_EPOCHS LR <<< "$CFG"

      TAG="seed_${SEED}_${DIM_TAG}_n${N_SAMPLES}"

      echo "=============================================================="
      echo "[run_nsamples_sweep] n_samples=$N_SAMPLES seed=$SEED | $NAME"
      echo "=============================================================="
      python -m mdp_lp.run_elasticity_experiments \
        --experiment_type "$EXPERIMENT_TYPE" \
        --cost_case "$COST_CASE" \
        --bc "$BC" \
        --algo "$ALGO" \
        --n_epochs "$N_EPOCHS" \
        --n_samples "$N_SAMPLES" \
        --lr "$LR" \
        --n_agg "$N_AGG" \
        --seed "$SEED" \
        --tag "$TAG" \
        --pretrain_epochs "$PRETRAIN_EPOCHS" \
        --b_max "$B_MAX" --s_max "$S_MAX" --q_max "$Q_MAX" \
        --results_root "$RESULTS_ROOT" \
        --aug_lag_rho "$AUG_LAG_RHO_DFL"
    done
  done
done

echo "[run_nsamples_sweep] Done. Results under ${RESULTS_ROOT}/exp_${EXPERIMENT_TYPE}/${COST_CASE}/..."
