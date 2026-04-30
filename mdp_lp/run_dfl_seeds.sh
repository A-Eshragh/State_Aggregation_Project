#!/usr/bin/env bash

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# SEEDS=(1 2 3 4 5 6 7 8 9 10)
SEEDS=(1 2 3 4 5)
# SEEDS=(5)
# SEEDS=(5 6 7 8 9 10)
EXPERIMENT_TYPE="b"
COST_CASE="normal_margin"
BC=10
N_SAMPLES=10
PRETRAIN_EPOCHS=50
# MDP dimensions: M = B_MAX + S_MAX + 1, A = Q_MAX + 1.
# Defaults (5/30/30) give M=36, A=31 — what the tuning was done at.
B_MAX=5
S_MAX=30
Q_MAX=30
DIM_TAG="s${S_MAX}q${Q_MAX}"
# Augmented-Lagrangian ρ (default for non-sketch algos uses runner default
# of 10000; sketch reuses the same default unless overridden via EXTRA4).
AUG_LAG_RHO_DFL=10000.0
AUG_LAG_RHO_SKETCH=0.0
RESULTS_ROOT="mdp_lp/results"


# NAME | ALGO | N_AGG | N_EPOCHS | LR | EXTRA1 | EXTRA2 | EXTRA3 | EXTRA4
#   dfl_sketch  : EXTRA1=n_sketches, EXTRA2=sketch_ratio, EXTRA3=sketch_mode, EXTRA4=ρ override
#   dfl_qpth    : EXTRA1=qp_reg
#   dfl/twostage: extras unused (ρ from AUG_LAG_RHO_DFL)
# Note: lr is per-config because dfl_sketch+n_agg requires lr=3e-5 (see header).
CONFIGS=(
#   "twostage|twostage|-1|50|0.01||||"
#   "dfl|dfl|-1|50|0.01||||"
#   "dfl_qpth|dfl_qpth|-1|50|0.01|10.0|||"
#   "dfl_sketch_noagg|dfl_sketch|-1|50|0.01|8|0.7|gaussian|"
#   "dfl_sketch_agg6|dfl_sketch|6|50|0.00003|8|0.7|gaussian|"
    "dfl_sketch_agg6|dfl_sketch|10|50|0.001|8|0.2|gaussian|"
)

for SEED in "${SEEDS[@]}"; do
  for CFG in "${CONFIGS[@]}"; do
    IFS='|' read -r NAME ALGO N_AGG N_EPOCHS LR EXTRA1 EXTRA2 EXTRA3 EXTRA4 <<< "$CFG"

    EXTRA_ARGS=()
    TAG="seed_${SEED}_${DIM_TAG}"
    if [[ "$ALGO" == "dfl_sketch" ]]; then
      N_SKETCHES="${EXTRA1:-4}"
      SKETCH_RATIO="${EXTRA2:-0.7}"
      SKETCH_MODE="${EXTRA3:-gaussian}"
      # Per-config ρ override (9th field) takes precedence over the sketch default.
      RHO="${EXTRA4:-$AUG_LAG_RHO_SKETCH}"
      EXTRA_ARGS+=(
        --n_sketches "$N_SKETCHES"
        --sketch_ratio "$SKETCH_RATIO"
        --sketch_mode "$SKETCH_MODE"
        --aug_lag_rho "$RHO"
      )
      TAG="${TAG}_K${N_SKETCHES}_r${SKETCH_RATIO}_${SKETCH_MODE}"
    elif [[ "$ALGO" == "dfl_qpth" ]]; then
      QP_REG="${EXTRA1:-10.0}"
      EXTRA_ARGS+=(--qp_reg "$QP_REG" --aug_lag_rho "$AUG_LAG_RHO_DFL")
      TAG="${TAG}_nagg_${N_AGG}_qpreg${QP_REG}"
    else
      EXTRA_ARGS+=(--aug_lag_rho "$AUG_LAG_RHO_DFL")
    fi

    echo "=============================================================="
    echo "[run_dfl_seeds] seed=$SEED | $NAME"
    echo "  algo=$ALGO n_agg=$N_AGG epochs=$N_EPOCHS lr=$LR ${EXTRA_ARGS[*]}"
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
      "${EXTRA_ARGS[@]}"
  done
done

echo "[run_dfl_seeds] Done. Results under ${RESULTS_ROOT}/exp_${EXPERIMENT_TYPE}/${COST_CASE}/..."
