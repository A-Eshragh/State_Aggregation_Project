#!/usr/bin/env bash
set -euo pipefail

# New-approach (same-dist) SA run: use the working SA setup.
# This is NOT the distribution-shift recipe.
#
# Setup matches the recent "new approach" runs:
#   regime_set=v6, train_frac=0.5, regime_weights=7,1,1,1, no test_regime_weights shift
#   SA objective: cw_local, n_agg=24
#
# Usage:
#   bash mdp_cw/run_new_approach_sa_working.sh
#
# Optional overrides:
#   OUT_ROOT=mdp_cw/results_new_approach_sa_working K=15 SEEDS="1 2 3 4 5 6 7 8 9 10" \
#   bash mdp_cw/run_new_approach_sa_working.sh

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

OUT_ROOT="${OUT_ROOT:-mdp_cw/results_new_approach_sa_working}"
K="${K:-15}"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
OUT_DIR="${OUT_ROOT}/k${K}"
mkdir -p "$OUT_DIR"

BASE=(
  --regime_set v6
  --n_samples 200
  --train_frac 0.5
  --center_scale 0.88
  --ambiguous_pairs 0:3
  --ambiguous_gap 0.0
  --regime_weights "7,1,1,1"
  --cliff_penalty -100.0
  --noise_std 0.6
  --pretrain_epochs 60
  --n_epochs 50
  --lr 3e-3
  --loss_weight 200.0
)

echo "[run] SA working config (new approach): cw_local, M=24"
for seed in $SEEDS; do
  run_dir="${OUT_DIR}/ssa24_s${seed}"
  mkdir -p "$run_dir"
  python -m mdp_cw.run_cliff_experiments \
    "${BASE[@]}" \
    --algo dfl_ssa \
    --ssa_objective cw_local \
    --n_agg 24 \
    --seed "$seed" \
    --save_dir "$run_dir" \
    > "${run_dir}/run.log" 2>&1 &
done
wait

echo "[done] SA runs saved under ${OUT_DIR}"
