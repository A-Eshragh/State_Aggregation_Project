#!/usr/bin/env bash
# DFL-Feas (Odece) sensitivity on cliff v7. The main row uses alpha=0.5, the
# mdp_lp default, which was never validated here: it gives 2.767 on all 10
# seeds (= the CE-warmup policy) even though the Odece loss itself descends
# 45.6 -> 37.6. Probe the IPL/OPL balance and the known-objective completion
# term before calling the baseline a failure.
#   alpha -> 0 : pure OPL (true optimum should satisfy predicted constraints)
#   alpha -> 1 : pure IPL (predicted residual at y^- pushed toward true sign)
#   mse        : mu * ||r_pred - r_true||^2, the only term touching r_pred
set -u
cd "$(dirname "$0")/.."
ulimit -s 65536
DEFAULT_PY="/net/csefiles/siemens/warriors-ls-recovery/users/zzhao628/#113442817/anaconda3/envs/rl/bin/python3"
if [ -x "$DEFAULT_PY" ]; then PY="${PY:-$DEFAULT_PY}"; else PY="${PY:-python3}"; fi
OUT=${1:-mdp_cw/results_feas_tune}; mkdir -p "$OUT/logs"
run() {
  local tag=$1 alpha=$2 mse=$3 seed=$4
  local dir="$OUT/${tag}_s${seed}"
  [ -f "$dir/results.json" ] && { echo "skip $dir"; return; }
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MDPLP_LP_WORKERS=6 "$PY" -m mdp_cw.run_cliff_experiments \
    --algo dfl_feas --n_samples 200 --feat_dim 8 --noise_std 0.1 --train_frac 0.5 \
    --batch_size 32 --seed "$seed" --regime_set v7 --center_scale 0.88 \
    --ambiguous_pairs 0:3 --ambiguous_gap 0.0 --cliff_penalty -5.0 \
    --regime_weights 7,0,0,1 --test_regime_weights 7,0,0,1 --stratify_regimes \
    --hidden_dim 64 --nn_feat_dim 32 --lr 0.001 --pretrain_lr 0.01 \
    --pretrain_epochs 60 --n_epochs 140 --grad_clip 10.0 --beta 0.98 \
    --loss_weight 0.0 --ambient_ce_weight 50.0 --head_ce_warmup_epochs 60 \
    --odece_alpha "$alpha" --odece_margin 0.01 --odece_tol 1e-4 \
    --odece_mse_weight "$mse" \
    --save_dir "$dir" > "$OUT/logs/${tag}_s${seed}.log" 2>&1
  echo "done $dir"
}
for s in 0 1 2; do
  run a01     0.1 0.0 "$s" &
  run a09     0.9 0.0 "$s" &
  run a05mse  0.5 1.0 "$s" &
done
wait
echo "complete -> $OUT"
