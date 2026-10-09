#!/usr/bin/env bash
# DFL-Sketch at sketch_ratio=1.0: k == S, so sketched_lp drops the redundant
# total-mass anchor row (use_anchor = k < M). The K sketched systems are then
# invertible remixings of the SAME LP, so each is exact -- but Gurobi may return
# a different vertex of the degenerate optimal face for each, and averaging them
# is precisely the intended basis-flip smoothing. Ratios 0.85-0.95 keep the
# near-redundant anchor and abort with "Gurobi failed to solve LP".
set -u
cd "$(dirname "$0")/.."
ulimit -s 65536
# /nethome/zzhao628/anaconda3 -> /scratch/zzhao628/anaconda3 can go missing; use a stable path
PY="${PY:-/net/csefiles/siemens/warriors-ls-recovery/users/zzhao628/#113442817/anaconda3/envs/rl/bin/python3}"
OUT=${1:-mdp_cw/results_sk_full}
mkdir -p "$OUT/logs"
run() {
  local tag=$1 ratio=$2 nsk=$3 seed=$4
  local dir="$OUT/${tag}_s${seed}"
  [ -f "$dir/results.json" ] && { echo "skip $dir"; return; }
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MDPLP_LP_WORKERS=6 "$PY" -m mdp_cw.run_cliff_experiments \
    --algo dfl_sketch --n_samples 200 --feat_dim 8 --noise_std 0.1 --train_frac 0.5 \
    --batch_size 32 --seed "$seed" --regime_set v7 --center_scale 0.88 \
    --ambiguous_pairs 0:3 --ambiguous_gap 0.0 --cliff_penalty -5.0 \
    --regime_weights 7,0,0,1 --test_regime_weights 7,0,0,1 --stratify_regimes \
    --hidden_dim 64 --nn_feat_dim 32 --lr 0.001 --pretrain_lr 0.01 \
    --pretrain_epochs 60 --n_epochs 140 --grad_clip 10.0 --beta 0.98 \
    --loss_mode auglag --aug_lag_rho 1.0 --loss_weight 0.0 \
    --ambient_ce_weight 50.0 --head_ce_warmup_epochs 60 \
    --n_sketches "$nsk" --sketch_ratio "$ratio" --sketch_mode gaussian \
    --save_dir "$dir" > "$OUT/logs/${tag}_s${seed}.log" 2>&1
  echo "done $dir"
}
for s in 0 1 2; do
  run full_k8  1.0  8 "$s" &
  run full_k16 1.0 16 "$s" &
done
wait
echo "complete -> $OUT"
