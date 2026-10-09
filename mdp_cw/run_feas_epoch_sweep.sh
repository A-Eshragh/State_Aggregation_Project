#!/usr/bin/env bash
# Training-budget sweep for DFL-Feas (Odece) on cliff v7, mirroring the
# results_epochs sweep that DFL-LP's n_epochs=62 was selected from.
#
# head_ce_warmup_epochs=60, so n_epochs=60 is the CONTROL: the DFL phase never
# activates and the run is pure CE. Anything above 60 has (n_epochs - 60) Odece
# epochs. alpha held at 0.5 (the main-row value).
set -u
cd "$(dirname "$0")/.."
ulimit -s 65536
DEFAULT_PY="/net/csefiles/siemens/warriors-ls-recovery/users/zzhao628/#113442817/anaconda3/envs/rl/bin/python3"
if [ -x "$DEFAULT_PY" ]; then PY="${PY:-$DEFAULT_PY}"; else PY="${PY:-python3}"; fi
OUT=${1:-mdp_cw/results_feas_epochs}; mkdir -p "$OUT/logs"
JOBS="${JOBS:-10}"
throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n 2>/dev/null || sleep 5; done; }
run() {
  local ep=$1 seed=$2
  local dir="$OUT/ep${ep}_s${seed}"
  [ -f "$dir/results.json" ] && { echo "skip $dir"; return; }
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MDPLP_LP_WORKERS=6 "$PY" -m mdp_cw.run_cliff_experiments \
    --algo dfl_feas --n_samples 200 --feat_dim 8 --noise_std 0.1 --train_frac 0.5 \
    --batch_size 32 --seed "$seed" --regime_set v7 --center_scale 0.88 \
    --ambiguous_pairs 0:3 --ambiguous_gap 0.0 --cliff_penalty -5.0 \
    --regime_weights 7,0,0,1 --test_regime_weights 7,0,0,1 --stratify_regimes \
    --hidden_dim 64 --nn_feat_dim 32 --lr 0.001 --pretrain_lr 0.01 \
    --pretrain_epochs 60 --n_epochs "$ep" --grad_clip 10.0 --beta 0.98 \
    --loss_weight 0.0 --ambient_ce_weight 50.0 --head_ce_warmup_epochs 60 \
    --odece_alpha 0.5 --odece_margin 0.01 --odece_tol 1e-4 --odece_mse_weight 0.0 \
    --save_dir "$dir" > "$OUT/logs/ep${ep}_s${seed}.log" 2>&1
  echo "done $dir"
}
for ep in 60 62 65 70 80 100 140 200; do
  for s in 0 1 2; do throttle; run "$ep" "$s" & done
done
wait
echo "complete -> $OUT"
