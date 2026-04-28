#!/usr/bin/env bash

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SEEDS=(1 2 3)
EXPERIMENT_TYPE="b"
COST_CASE="high_margin"
N_SAMPLES=20
# Augmented Lagrangian penalty ρ for dfl_sketch runs (runner default is 10000.0).
AUG_LAG_RHO=1.0
PRETRAIN_EPOCHS=50

# Each entry: "name|algo|n_agg|n_epochs|lr[|n_sketches|sketch_ratio|sketch_mode]"
# The last three fields are only used when algo=dfl_sketch (defaults: K=4, ratio=0.7, mode=gaussian).
CONFIGS=(
  "single_elasticity_twostage|twostage|-1|200|0.01"
  "single_elasticity_dfl_qpth|dfl_qpth|-1|50|0.01"
  "single_elasticity_dfl|dfl|-1|50|0.01"
  "single_elasticity_dfl_state_agg|dfl|30|50|0.01"
  "single_elasticity_dfl_state_agg|dfl|20|50|0.01"
  "single_elasticity_dfl_state_agg|dfl|10|50|0.01"
  "single_elasticity_dfl_qpth_state_agg|dfl_qpth|10|50|0.01"
  "single_elasticity_dfl_qpth_state_agg|dfl_qpth|20|50|0.01"
  "single_elasticity_dfl_qpth_state_agg|dfl_qpth|30|50|0.01"
  "single_elasticity_dfl_sketch|dfl_sketch|-1|50|0.01|16|0.6|gaussian"
  "single_elasticity_dfl_sketch_state_agg|dfl_sketch|10|50|0.01|16|0.6|gaussian"
  "single_elasticity_dfl_sketch_state_agg|dfl_sketch|20|50|0.01|16|0.6|gaussian"
  "single_elasticity_dfl_sketch_state_agg|dfl_sketch|30|50|0.01|16|0.6|gaussian"
)

for SEED in "${SEEDS[@]}"; do
  for CFG in "${CONFIGS[@]}"; do
    IFS='|' read -r NAME ALGO N_AGG N_EPOCHS LR N_SKETCHES SKETCH_RATIO SKETCH_MODE <<< "$CFG"

    EXTRA_ARGS=()
    TAG="seed_${SEED}"
    if [[ "$ALGO" == "dfl_sketch" ]]; then
      N_SKETCHES="${N_SKETCHES:-4}"
      SKETCH_RATIO="${SKETCH_RATIO:-0.7}"
      SKETCH_MODE="${SKETCH_MODE:-gaussian}"
      EXTRA_ARGS+=(
        --n_sketches "$N_SKETCHES"
        --sketch_ratio "$SKETCH_RATIO"
        --sketch_mode "$SKETCH_MODE"
        --aug_lag_rho "$AUG_LAG_RHO"
      )
      # run_elasticity_experiments.py does not append an n_agg_* subfolder
      # for dfl_sketch, so different (n_agg, K, ratio, mode) runs would
      # collide in the same save_dir. Encode them into the tag to keep
      # outputs separate.
      TAG="${TAG}_nagg_${N_AGG}_K${N_SKETCHES}_r${SKETCH_RATIO}_${SKETCH_MODE}"
    fi

    echo "=============================================================="
    echo "[run_dfl_seeds] seed=$SEED | $NAME"
    echo "  algo=$ALGO n_agg=$N_AGG epochs=$N_EPOCHS lr=$LR ${EXTRA_ARGS[*]}"
    echo "=============================================================="
    python -m mdp_lp.run_elasticity_experiments \
      --experiment_type "$EXPERIMENT_TYPE" \
      --cost_case "$COST_CASE" \
      --algo "$ALGO" \
      --n_epochs "$N_EPOCHS" \
      --n_samples "$N_SAMPLES" \
      --lr "$LR" \
      --n_agg "$N_AGG" \
      --seed "$SEED" \
      --tag "$TAG" \
      --pretrain_epochs "$PRETRAIN_EPOCHS" \
      "${EXTRA_ARGS[@]}"
  done
done

echo "[run_dfl_seeds] Done. Results under mdp_lp/results_elasticity/exp_${EXPERIMENT_TYPE}/${COST_CASE}/..."
