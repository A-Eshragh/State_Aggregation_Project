#!/usr/bin/env bash
# Inventory problem, |S| = 36, |A| = 31. Seeds 1-10, all methods.

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SEEDS=(1 2 3 4 5 6 7 8 9 10)
# SEEDS=(2)
# SEEDS=(6 7 8 9 10)
EXPERIMENT_TYPE="b"
COST_CASE="normal_margin"
BC=10
N_SAMPLES=10
PRETRAIN_EPOCHS=30
B_MAX=5
S_MAX=30
Q_MAX=30
DIM_TAG="s${S_MAX}q${Q_MAX}"
AUG_LAG_RHO_DFL=10000.0
AUG_LAG_RHO_NOAGG=10.0
AUG_LAG_RHO_SKETCH=0.1
RESULTS_ROOT="mdp_lp/results/${COST_CASE}_pretrain_${PRETRAIN_EPOCHS}"

# NAME|ALGO|N_AGG|N_EPOCHS|LR|EXTRA1|EXTRA2|EXTRA3|EXTRA4. The EXTRA fields are
# read per ALGO:
#   dfl_sketch : n_sketches | sketch_ratio | sketch_mode | aug_lag_rho override
#   dfl_qpth   : qp_reg     | -            | -           | aug_lag_rho override
#   dfl        : loss_mode  | aug_lag_rho  | -           | -
#   dfl_feas   : odece_alpha| odece_mse_weight (mu) | odece_margin | -
CONFIGS=(
    "twostage|twostage|-1|50|0.01||||"
    "dfl_qpth|dfl_qpth|-1|50|0.01|10.0|||"
    "dfl_auglag_r1000|dfl|-1|50|0.01|auglag|1000.0||"
    "dfl_sketch_agg30|dfl_sketch|30|50|0.01|8|0.2|gaussian|"
    "dfl_sketch_agg20|dfl_sketch|20|50|0.01|8|0.2|gaussian|"
    "dfl_sketch_agg10|dfl_sketch|10|50|0.01|8|0.2|gaussian|"
    "dfl_sketch_noagg|dfl_sketch|-1|50|0.01|8|0.2|gaussian|"
    "dfl_feas_a0_m0|dfl_feas|-1|50|0.01|0.0|0.0|0.01|"
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
      DEFAULT_RHO="$AUG_LAG_RHO_SKETCH"
      if [[ "$NAME" == "dfl_sketch_noagg" ]]; then
        DEFAULT_RHO="$AUG_LAG_RHO_NOAGG"
      fi
      # Per-config ρ override (9th field) takes precedence over the sketch default.
      RHO="${EXTRA4:-$DEFAULT_RHO}"
      EXTRA_ARGS+=(
        --n_sketches "$N_SKETCHES"
        --sketch_ratio "$SKETCH_RATIO"
        --sketch_mode "$SKETCH_MODE"
        --aug_lag_rho "$RHO"
        --loss_weight 1
      )
      TAG="${TAG}_K${N_SKETCHES}_r${SKETCH_RATIO}_${SKETCH_MODE}"
    elif [[ "$ALGO" == "dfl_qpth" ]]; then
      QP_REG="${EXTRA1:-10.0}"
      EXTRA_ARGS+=(--qp_reg "$QP_REG" --aug_lag_rho "${EXTRA4:-$AUG_LAG_RHO_DFL}")
      TAG="${TAG}_nagg_${N_AGG}_qpreg${QP_REG}"
    elif [[ "$ALGO" == "dfl" ]]; then
      LOSS_MODE="${EXTRA1:-penalty}"
      RHO="${EXTRA2:-$AUG_LAG_RHO_DFL}"
      EXTRA_ARGS+=(--loss_mode "$LOSS_MODE" --aug_lag_rho "$RHO")
      # Keep the plain penalty arm's tag as-is so it lands on the existing
      # paper tree (tag_seed_<s>_s30q30); only variants get a suffix.
      if [[ "$LOSS_MODE" != "penalty" || "$RHO" != "$AUG_LAG_RHO_DFL" ]]; then
        TAG="${TAG}_${LOSS_MODE}_r${RHO}"
      fi
    elif [[ "$ALGO" == "dfl_feas" ]]; then
      ODECE_ALPHA="${EXTRA1:-0.0}"
      ODECE_MU="${EXTRA2:-0.0}"
      ODECE_MARGIN="${EXTRA3:-0.01}"
      EXTRA_ARGS+=(
        --odece_alpha "$ODECE_ALPHA"
        --odece_mse_weight "$ODECE_MU"
        --odece_margin "$ODECE_MARGIN"
        --aug_lag_rho "$AUG_LAG_RHO_DFL"
      )
      TAG="${TAG}_feas_a${ODECE_ALPHA}_m${ODECE_MU}"
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
