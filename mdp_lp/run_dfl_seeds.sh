#!/usr/bin/env bash
#
# Figure-3 inventory setting (|S| = B_MAX + S_MAX + 1 = 36, |A| = Q_MAX + 1 = 31).
# Results below are the 10-seed (seeds 1-10) numbers produced by this script's
# config, read from ${RESULTS_ROOT} and mdp_lp/results_feas_compare (same config,
# verified byte-identical per-seed regret on the shared arms). regret% is the
# Figure-3 unit: 100 * test_mean_regret / test_mean_oracle_reward, mean +/- SE
# across seeds; raw is mean(test_regret); solver s/ep is the median across seeds
# of mean(solver_time_sec_per_epoch) (two-stage has no decision-aware solver).
#
#   method                     CONFIGS name         regret%          raw regret     solver s/ep
#   -------------------------------------------------------------------------------------------
#   DFL-QP (reg=10)            dfl_qpth          12.55 +- 0.99    56.88 +-  4.48       13.26
#   DFL-SA k=30 + sketch       dfl_sketch_agg30  12.86 +- 1.47    58.14 +-  6.50        0.56
#   DFL-LP (penalty, rho=1e4)  dfl               13.01 +- 1.19    58.90 +-  5.31        0.28
#   DFL-Sketch (no agg, r=0.2) dfl_sketch_noagg  13.06 +- 1.35    59.17 +-  6.08        0.53
#   DFL aug-Lag (rho=1000)     dfl_auglag_r1000  13.28 +- 1.21    60.15 +-  5.43        0.68
#   Two-stage                  twostage          13.35 +- 1.19    60.43 +-  5.29        0.00
#   DFL-Feas (alpha=0, mu=0)   dfl_feas_a0_m0    13.66 +- 1.39    61.83 +-  6.17        0.74
#   DFL-SA k=20 + sketch       dfl_sketch_agg20  13.98 +- 1.39    63.40 +-  6.38        0.39
#   DFL-SA k=10 + sketch       dfl_sketch_agg10  14.97 +- 2.32    67.73 +- 10.53        0.17
#
# DFL-SA arms are dfl_sketch with n_agg=k, K=8 sketches, r=0.2, gaussian.
# DFL-Feas is Odece [34] adapted; alpha=0 (OPL only) and mu=0 (no reward
# completion) is the faithful variant, i.e. the weakest of the feas family --
# alpha=0.5 with mu=1 scores 11.56% / 52.30 raw, ahead of every arm above.
# All methods sit within ~2.4 percentage points on a ~13% oracle gap, i.e. this
# setting is estimation-dominated; aggregation quality degrades monotonically as
# k shrinks (30 -> 20 -> 10) while solver time drops 0.56 -> 0.39 -> 0.17 s/epoch.

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

# Config format: NAME|ALGO|N_AGG|N_EPOCHS|LR|EXTRA1|EXTRA2|EXTRA3|EXTRA4
# The four EXTRA fields are interpreted per ALGO (see the dispatch below):
#   dfl_sketch : n_sketches | sketch_ratio | sketch_mode | aug_lag_rho override
#   dfl_qpth   : qp_reg     | -            | -           | aug_lag_rho override
#   dfl        : loss_mode  | aug_lag_rho  | -           | -
#   dfl_feas   : odece_alpha| odece_mse_weight (mu) | odece_margin | -
#   twostage   : -          | -            | -           | -
# Uncommenting the whole block below reproduces the 8-row table in the header.
CONFIGS=(
    "twostage|twostage|-1|50|0.01||||"
    "dfl_qpth|dfl_qpth|-1|50|0.01|10.0|||"
    "dfl|dfl|-1|50|0.01|penalty|10000.0||"
    "dfl_auglag_r1000|dfl|-1|50|0.01|auglag|1000.0||"
    "dfl_sketch_agg30|dfl_sketch|30|50|0.01|8|0.2|gaussian|"
    "dfl_sketch_agg20|dfl_sketch|20|50|0.01|8|0.2|gaussian|"
    "dfl_sketch_agg10|dfl_sketch|10|50|0.01|8|0.2|gaussian|"
    "dfl_sketch_noagg|dfl_sketch|-1|50|0.01|8|0.2|gaussian|"
    "dfl_feas_a0_m0|dfl_feas|-1|50|0.01|0.0|0.0|0.01|"
    # "dfl_sketch_agg5|dfl_sketch|5|50|0.01|16|0.2|gaussian|"
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
