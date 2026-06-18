#!/usr/bin/env bash
set -euo pipefail

# Distribution-shift recipe used for the results_seeds1_10_sa10_20_30 bundle.
# This is the setup where TS is much worse (high regret) due to train/test prior shift.
#
# Train prior: regime_weights = 15,1,1,1
# Test prior:  test_regime_weights = 0,0,1,0
#
# Usage:
#   bash mdp_cw/run_dist_shift_seeds1_10_sa10_20_30.sh
#
# Optional overrides:
#   OUT_ROOT=mdp_cw/results_seeds1_10_sa10_20_30 K=15 SEEDS="1 2 3 4 5 6 7 8 9 10" \
#   bash mdp_cw/run_dist_shift_seeds1_10_sa10_20_30.sh

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

OUT_ROOT="${OUT_ROOT:-mdp_cw/results_seeds1_10_sa10_20_30_v2}"
K="${K:-15}"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"

OUT_DIR="${OUT_ROOT}/k${K}"
mkdir -p "$OUT_DIR"

BASE=(
  --regime_set v4
  --n_samples 200
  --train_frac 0.5
  --center_scale 0.88
  --ambiguous_pairs 0:2
  --ambiguous_gap 0.0
  --regime_weights "15,1,1,1"
  --test_regime_weights "0,0,1,0"
  --cliff_penalty -50.0
  --noise_std 0.6
  --pretrain_epochs 60
  --n_epochs 50
  --lr 3e-3
)

run_method_parallel() {
  local prefix="$1"
  shift
  echo "[run] ${prefix}"
  for seed in $SEEDS; do
    local run_dir="${OUT_DIR}/${prefix}_s${seed}"
    mkdir -p "$run_dir"
    python -m mdp_cw.run_cliff_experiments \
      "${BASE[@]}" \
      --seed "$seed" \
      --save_dir "$run_dir" \
      "$@" \
      > "${run_dir}/run.log" 2>&1 &
  done
  wait
}

# Two-stage
run_method_parallel "ts" \
  --algo twostage \
  --loss_weight 500

# DFL-LP
run_method_parallel "dfl" \
  --algo dfl \
  --loss_weight 500 \
  --lp_basis_ridge 1e-6

# DFL-QP (qpth)
run_method_parallel "qpth" \
  --algo dfl_qpth \
  --loss_weight 500 \
  --qp_reg 0.1 \
  --qp_max_iter 20

# DFL-Sketch (K=8, r=0.3)
run_method_parallel "sketch_k8r03" \
  --algo dfl_sketch \
  --loss_weight 100 \
  --aug_lag_rho 1.0 \
  --n_sketches 8 \
  --sketch_ratio 0.3 \
  --lp_basis_ridge 1e-6

# DFL-SA (M=10,20,30) with cw_sketch objective.
for M in 10 20 30; do
  run_method_parallel "ssa${M}" \
    --algo dfl_ssa \
    --loss_weight 500 \
    --lp_basis_ridge 1e-6 \
    --ssa_objective cw_sketch \
    --n_agg "$M" \
    --n_sketches 8 \
    --sketch_ratio 0.2
done

echo "[done] runs saved under ${OUT_DIR}"
echo "[done] render plots with:"
echo "python mdp_cw/make_figures.py --root \"${OUT_ROOT}\" --k \"${K}\" --n_seeds 10 --out_dir \"mdp_cw/selected_plots_seeds1_10_sa10_20_30\""
