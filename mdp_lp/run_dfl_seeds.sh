#!/usr/bin/env bash

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

<<<<<<< HEAD
SEEDS=(1 2 3 4 5 6 7 8 9 10)
EXPERIMENT_TYPE="b"
COST_CASE="high_margin"
N_SAMPLES=20
# Pretrain sweep (Apr 29) at M=36, sketch+n_agg=10 K=4 r=0.3 ρ=0:
#   pretrain=15:  7.00 ± 2.86%
#   pretrain=30:  5.24 ± 6.18%  (seed 3 catastrophic at 12.26%)
#   pretrain=50:  3.58 ± 2.11%  ← best mean AND variance
#   pretrain=100: 4.80 ± 3.79%  (overfit MSE, less room for DFL)
PRETRAIN_EPOCHS=50
# MDP dimensions: M = B_MAX + S_MAX + 1, A = Q_MAX + 1.
# Defaults (5/30/30) give M=36, A=31. Raised to (5/50/50) → M=56, A=51.
B_MAX=5
S_MAX=50
Q_MAX=50
DIM_TAG="s${S_MAX}q${Q_MAX}"
# Per-algo augmented-Lagrangian penalty:
#   dfl uses ρ_DFL=10000 (runner default; works at no-agg, no soft-agg bias since n_agg=-1).
#   dfl_sketch defaults to ρ_SKETCH=0; individual sketch CONFIGS can override via the
#   9th pipe-field. Picked from the 4-iteration tuning sweep (Apr 28):
#     no-agg sketch:    K=4 r=0.3 ρ=10000  → 5.41 ± 3.51% regret, 2.5 s/ep
#     n_agg=10 sketch:  K=4 r=0.3 ρ=0      → 7.40 ± 7.19% regret, 1.0 s/ep
#   (dfl no-agg ρ=10000 baseline: 2.98 ± 2.09% regret, 2.2 s/ep clean.)
AUG_LAG_RHO_DFL=10000.0
AUG_LAG_RHO_SKETCH=0.0


CONFIGS=(
    "single_elasticity_twostage|twostage|-1|50|0.01"
    "single_elasticity_dfl_qpth|dfl_qpth|-1|50|0.01"
    "single_elasticity_dfl|dfl|-1|50|0.01"
    "single_elasticity_dfl_state_agg|dfl|30|50|0.01"
    "single_elasticity_dfl_state_agg|dfl|20|50|0.01"
    "single_elasticity_dfl_state_agg|dfl|10|50|0.01"
    "single_elasticity_dfl_qpth_state_agg|dfl_qpth|10|50|0.01"
    "single_elasticity_dfl_qpth_state_agg|dfl_qpth|20|50|0.01"
    "single_elasticity_dfl_qpth_state_agg|dfl_qpth|30|50|0.01"
# Sketch CONFIGS use 9 fields: name|algo|n_agg|epochs|lr|K|ratio|mode|rho_override
    "single_elasticity_dfl_sketch|dfl_sketch|-1|50|0.01|4|0.3|gaussian|10000.0"
    "single_elasticity_dfl_sketch_state_agg|dfl_sketch|10|50|0.01|4|0.3|gaussian|0.0"
    "single_elasticity_dfl_sketch_state_agg|dfl_sketch|20|50|0.01|4|0.3|gaussian|0.0"
    "single_elasticity_dfl_sketch_state_agg|dfl_sketch|30|50|0.01|4|0.3|gaussian|0.0"
=======
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
>>>>>>> 8196c135130e99b57cc2090ef143ec96efa74bb3
)

for SEED in "${SEEDS[@]}"; do
  for CFG in "${CONFIGS[@]}"; do
<<<<<<< HEAD
    IFS='|' read -r NAME ALGO N_AGG N_EPOCHS LR EXTRA1 EXTRA2 EXTRA3 EXTRA4 <<< "$CFG"

    EXTRA_ARGS=()
    TAG="seed_${SEED}_${DIM_TAG}"
    if [[ "$ALGO" == "dfl_sketch" ]]; then
      N_SKETCHES="${EXTRA1:-4}"
      SKETCH_RATIO="${EXTRA2:-0.7}"
      SKETCH_MODE="${EXTRA3:-gaussian}"
      # Per-config ρ override (9th field) takes precedence over the sketch default.
      RHO="${EXTRA4:-$AUG_LAG_RHO_SKETCH}"
=======
    IFS='|' read -r NAME ALGO N_AGG N_EPOCHS LR N_SKETCHES SKETCH_RATIO SKETCH_MODE <<< "$CFG"

    EXTRA_ARGS=()
    TAG="seed_${SEED}"
    if [[ "$ALGO" == "dfl_sketch" ]]; then
      N_SKETCHES="${N_SKETCHES:-4}"
      SKETCH_RATIO="${SKETCH_RATIO:-0.7}"
      SKETCH_MODE="${SKETCH_MODE:-gaussian}"
>>>>>>> 8196c135130e99b57cc2090ef143ec96efa74bb3
      EXTRA_ARGS+=(
        --n_sketches "$N_SKETCHES"
        --sketch_ratio "$SKETCH_RATIO"
        --sketch_mode "$SKETCH_MODE"
<<<<<<< HEAD
        --aug_lag_rho "$RHO"
      )
      TAG="${TAG}_nagg_${N_AGG}_K${N_SKETCHES}_r${SKETCH_RATIO}_${SKETCH_MODE}"
    else
      EXTRA_ARGS+=(--aug_lag_rho "$AUG_LAG_RHO_DFL")
=======
        --aug_lag_rho "$AUG_LAG_RHO"
      )
      # run_elasticity_experiments.py does not append an n_agg_* subfolder
      # for dfl_sketch, so different (n_agg, K, ratio, mode) runs would
      # collide in the same save_dir. Encode them into the tag to keep
      # outputs separate.
      TAG="${TAG}_nagg_${N_AGG}_K${N_SKETCHES}_r${SKETCH_RATIO}_${SKETCH_MODE}"
>>>>>>> 8196c135130e99b57cc2090ef143ec96efa74bb3
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
<<<<<<< HEAD
      --b_max "$B_MAX" --s_max "$S_MAX" --q_max "$Q_MAX" \
=======
>>>>>>> 8196c135130e99b57cc2090ef143ec96efa74bb3
      "${EXTRA_ARGS[@]}"
  done
done

echo "[run_dfl_seeds] Done. Results under mdp_lp/results_elasticity/exp_${EXPERIMENT_TYPE}/${COST_CASE}/..."
