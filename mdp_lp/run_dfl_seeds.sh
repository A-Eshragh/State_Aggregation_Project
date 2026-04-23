#!/usr/bin/env bash

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SEEDS=(1 2 3)
EXPERIMENT_TYPE="b"
COST_CASE="high_margin"
N_SAMPLES=20

# Each entry: "name|algo|n_agg|n_epochs|lr"
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
)

for SEED in "${SEEDS[@]}"; do
  for CFG in "${CONFIGS[@]}"; do
    IFS='|' read -r NAME ALGO N_AGG N_EPOCHS LR <<< "$CFG"
    echo "=============================================================="
    echo "[run_dfl_seeds] seed=$SEED | $NAME"
    echo "  algo=$ALGO n_agg=$N_AGG epochs=$N_EPOCHS lr=$LR"
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
      --tag "seed_${SEED}"
  done
done

echo "[run_dfl_seeds] Done. Results under mdp_lp/results_elasticity/exp_${EXPERIMENT_TYPE}/${COST_CASE}/..."
