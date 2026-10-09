#!/usr/bin/env bash
# Sweep the sketch row-keep ratio for DFL-Sketch on the v7 gate-trap setting.
#
# Every DFL-Sketch run in the paper so far used sketch_ratio=0.2, which keeps
# only round(0.2*48)+1 = 11 of the 48 flow-conservation rows. At that ratio the
# sketched LP is a 10.6-unit relaxation of the true occupancy LP and its
# solution barely responds to the predicted MDP (||dy||/||dy_LP|| = 0.004), so
# the DFL gradient is ~0 and training collapses to the CE-warmup policy.
# Everything else is held at the results_final7/sk_* config.
#
# Usage:  bash mdp_cw/run_sketch_ratio_sweep.sh [outdir]
set -u
cd "$(dirname "$0")/.."
ulimit -s 65536

OUT=${1:-mdp_cw/results_sk_tune}
PY=${PY:-python3}
mkdir -p "$OUT/logs"

run() {  # run <tag> <ratio> <n_sketches> <seed> [mode]
  local tag=$1 ratio=$2 nsk=$3 seed=$4 mode=${5:-gaussian}
  local dir="$OUT/${tag}_s${seed}"
  [ -f "$dir/results.json" ] && { echo "skip $dir (done)"; return; }
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $PY -m mdp_cw.run_cliff_experiments \
    --algo dfl_sketch \
    --n_samples 200 --feat_dim 8 --noise_std 0.1 --train_frac 0.5 \
    --batch_size 32 --seed "$seed" \
    --regime_set v7 --center_scale 0.88 \
    --ambiguous_pairs 0:3 --ambiguous_gap 0.0 \
    --cliff_penalty -5.0 \
    --regime_weights 7,0,0,1 --test_regime_weights 7,0,0,1 --stratify_regimes \
    --hidden_dim 64 --nn_feat_dim 32 \
    --lr 0.001 --pretrain_lr 0.01 --pretrain_epochs 60 --n_epochs 140 \
    --grad_clip 10.0 --beta 0.98 \
    --loss_mode auglag --aug_lag_rho 1.0 \
    --loss_weight 0.0 --ambient_ce_weight 50.0 --head_ce_warmup_epochs 60 \
    --n_sketches "$nsk" --sketch_ratio "$ratio" --sketch_mode "$mode" \
    --save_dir "$dir" > "$OUT/logs/${tag}_s${seed}.log" 2>&1
  echo "done $dir"
}

# Dense (gaussian/rademacher) sketches mix all 48 flow rows into k random
# combinations; at k near S the resulting basis is near-singular, so Gurobi
# either fails outright or the basis-sensitivity backward explodes
# (||grad_P|| ~ 1e5-1e6).  'subsample' keeps k GENUINE flow-conservation rows,
# so the sketched LP is a real (leaky) occupancy LP: bounded gradients, higher
# objective fidelity, and reliable solves.  Sweep the ratio under that mode.
SEEDS=${SEEDS:-"0 1 2"}
MODE=${MODE:-subsample}
for s in $SEEDS; do
  run sub060 0.60 8 "$s" "$MODE" &
  run sub080 0.80 8 "$s" "$MODE" &
  run sub090 0.90 8 "$s" "$MODE" &
done
wait
echo "sweep complete -> $OUT"
