#!/usr/bin/env bash
# Hyperparameter search for DFL-Sketch: vary K (n_sketches) and r (sketch_ratio).
# Goal: match or beat DFL-QPTH on seeds 1-4 (target ≤ 11.83% mean test regret).

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SEEDS=(1 2 3 4)
KS=(4 8 16 32)
RS=(0.2 0.4 0.6 0.8)

# Fixed settings — pretrain=30 per user request.
EXPERIMENT_TYPE="b"
COST_CASE="normal_margin"
BC=10
N_SAMPLES=10
PRETRAIN_EPOCHS=30
N_EPOCHS=50
LR=0.001
N_AGG=10
B_MAX=5
S_MAX=30
Q_MAX=30
DIM_TAG="s${S_MAX}q${Q_MAX}"
RESULTS_ROOT="mdp_lp/results_hpsearch"
SKETCH_MODE="gaussian"
AUG_LAG_RHO=0.0

JOBS_FILE="/tmp/hpsearch_jobs.txt"
> "$JOBS_FILE"
for SEED in "${SEEDS[@]}"; do
  for K in "${KS[@]}"; do
    for R in "${RS[@]}"; do
      TAG="seed_${SEED}_${DIM_TAG}_K${K}_r${R}_${SKETCH_MODE}"
      echo "$SEED $K $R $TAG" >> "$JOBS_FILE"
    done
  done
done

echo "[hpsearch] Total jobs: $(wc -l < $JOBS_FILE)"

run_one() {
  local SEED=$1 K=$2 R=$3 TAG=$4
  python -m mdp_lp.run_elasticity_experiments \
    --experiment_type "$EXPERIMENT_TYPE" --cost_case "$COST_CASE" --bc "$BC" \
    --algo dfl_sketch --n_epochs "$N_EPOCHS" --n_samples "$N_SAMPLES" \
    --lr "$LR" --n_agg "$N_AGG" --seed "$SEED" --tag "$TAG" \
    --pretrain_epochs "$PRETRAIN_EPOCHS" \
    --b_max "$B_MAX" --s_max "$S_MAX" --q_max "$Q_MAX" \
    --results_root "$RESULTS_ROOT" --aug_lag_rho "$AUG_LAG_RHO" \
    --n_sketches "$K" --sketch_ratio "$R" --sketch_mode "$SKETCH_MODE" \
    > /tmp/hpsearch_${SEED}_K${K}_r${R}.log 2>&1
  echo "[done] seed=$SEED K=$K r=$R"
}

export -f run_one
export EXPERIMENT_TYPE COST_CASE BC N_SAMPLES PRETRAIN_EPOCHS N_EPOCHS LR N_AGG \
       B_MAX S_MAX Q_MAX DIM_TAG RESULTS_ROOT SKETCH_MODE AUG_LAG_RHO

# Run 3 jobs in parallel — leaves headroom for the still-running sequential
# qpth job. Each dfl_sketch run is ~15-25s.
cat "$JOBS_FILE" | xargs -L1 -P3 bash -c 'run_one $@' --

echo "[hpsearch] Sketch search complete. Running qpth baseline..."

# Also run dfl_qpth at the same pretrain for seeds 1-4 (target to beat).
QPTH_JOBS="/tmp/hpsearch_qpth_jobs.txt"
> "$QPTH_JOBS"
for SEED in "${SEEDS[@]}"; do
  echo "$SEED" >> "$QPTH_JOBS"
done

run_qpth() {
  local SEED=$1
  python -m mdp_lp.run_elasticity_experiments \
    --experiment_type "$EXPERIMENT_TYPE" --cost_case "$COST_CASE" --bc "$BC" \
    --algo dfl_qpth --n_epochs "$N_EPOCHS" --n_samples "$N_SAMPLES" \
    --lr 0.01 --n_agg -1 --seed "$SEED" --tag "seed_${SEED}_${DIM_TAG}" \
    --pretrain_epochs "$PRETRAIN_EPOCHS" --qp_reg 10.0 \
    --b_max "$B_MAX" --s_max "$S_MAX" --q_max "$Q_MAX" \
    --results_root "$RESULTS_ROOT" --aug_lag_rho 10000.0 \
    > /tmp/hpsearch_qpth_${SEED}.log 2>&1
  echo "[done] qpth seed=$SEED"
}
export -f run_qpth

# qpth is ~3-4 min per run; run 2 in parallel so the wall-clock is ~8 min.
cat "$QPTH_JOBS" | xargs -L1 -P2 bash -c 'run_qpth $@' --

echo "[hpsearch] All jobs complete."
