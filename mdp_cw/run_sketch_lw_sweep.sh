#!/usr/bin/env bash
# The earlier cliff figure's DFL-Sketch ran with loss_weight=500, where the CE
# term was 98.8% of the objective and the sketch gradient only 1.2%. fig3_v7
# uses loss_weight=0.0 (volatile slot driven by the DFL term alone), which is
# where the sketch collapses. Sweep loss_weight in the v7 setting to see what
# the CE-carried regime buys here.
set -u
cd "$(dirname "$0")/.."
ulimit -s 65536
PY="${PY:-/net/csefiles/siemens/warriors-ls-recovery/users/zzhao628/#113442817/anaconda3/envs/rl/bin/python3}"
OUT=${1:-mdp_cw/results_sk_lw}
mkdir -p "$OUT/logs"
run() {
  local tag=$1 lw=$2 seed=$3
  local dir="$OUT/${tag}_s${seed}"
  [ -f "$dir/results.json" ] && { echo "skip $dir"; return; }
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MDPLP_LP_WORKERS=6 "$PY" -m mdp_cw.run_cliff_experiments \
    --algo dfl_sketch --n_samples 200 --feat_dim 8 --noise_std 0.1 --train_frac 0.5 \
    --batch_size 32 --seed "$seed" --regime_set v7 --center_scale 0.88 \
    --ambiguous_pairs 0:3 --ambiguous_gap 0.0 --cliff_penalty -5.0 \
    --regime_weights 7,0,0,1 --test_regime_weights 7,0,0,1 --stratify_regimes \
    --hidden_dim 64 --nn_feat_dim 32 --lr 0.001 --pretrain_lr 0.01 \
    --pretrain_epochs 60 --n_epochs 140 --grad_clip 10.0 --beta 0.98 \
    --loss_mode auglag --aug_lag_rho 1.0 --loss_weight "$lw" \
    --ambient_ce_weight 50.0 --head_ce_warmup_epochs 60 \
    --n_sketches 8 --sketch_ratio 0.2 --sketch_mode gaussian \
    --save_dir "$dir" > "$OUT/logs/${tag}_s${seed}.log" 2>&1
  echo "done $dir"
}
for s in ${SEEDS:-0 1 2}; do
  run lw1   1.0   "$s" &
  [ "${LW1ONLY:-0}" = 1 ] || run lw50  50.0  "$s" &
  [ "${LW1ONLY:-0}" = 1 ] || run lw500 500.0 "$s" &
done
wait
echo "complete -> $OUT"
