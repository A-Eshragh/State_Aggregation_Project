#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
ulimit -s 65536                 # Gurobi segfaults on this LP with the default stack
PY=${PY:-python3}               # override if python3 is not the project env
TIMING=${TIMING:-0}             # 1 = sequential, for wall-clock measurement
LP_WORKERS=${LP_WORKERS:-$([ "$TIMING" = 1 ] && echo 8 || echo 6)}

SEEDS="0 1 2 3 4 5 6 7 8 9"
[ "$TIMING" = 1 ] && SEEDS=${SEEDS_TIMING:-0}   # one seed is enough for timing

# Instance + encoder. Identical for every method: this is what makes it paired.
COMMON="--n_samples 200 --feat_dim 8 --noise_std 0.1 --train_frac 0.5 --batch_size 32
        --regime_set v7 --center_scale 0.88 --ambiguous_pairs 0:3 --ambiguous_gap 0.0
        --cliff_penalty -5.0 --regime_weights 7,0,0,1 --test_regime_weights 7,0,0,1
        --stratify_regimes --hidden_dim 64 --nn_feat_dim 32
        --lr 0.001 --pretrain_lr 0.01 --pretrain_epochs 60 --grad_clip 10.0 --beta 0.98"

DFL="--n_epochs 140 --loss_mode auglag --aug_lag_rho 1.0
     --loss_weight 0.0 --ambient_ce_weight 50.0 --head_ce_warmup_epochs 60"
SA="$DFL --sa_tau 0.5 --sa_r_embed 8 --n_sketches 8 --sketch_ratio 0.6"

# name|output dir|seeds|method-specific args      (no spaces around the pipes)
CONFIGS=(
"twostage|results_tsconv/ts600|$SEEDS|--algo twostage --n_epochs 600 --loss_mode penalty --loss_weight 1.0"
"lp|results_epochs/lp62|$SEEDS|--algo dfl --n_epochs 62 --loss_mode auglag --aug_lag_rho 1000.0 --loss_weight 0.1 --head_ce_warmup_epochs 60"
"qp|results_final7/qpe0002|2 3 4 5 6 7 8 9 10 11|--algo dfl_qpth $DFL --qp_reg 0.0002 --qp_max_iter 56"
"feas|results_feas60/feas|$SEEDS|--algo dfl_feas $DFL --n_epochs 60 --odece_alpha 0.5"
"sketch_lw1|results_sk_lw/lw1|$SEEDS|--algo dfl_sketch $DFL --loss_weight 1.0 --n_sketches 8 --sketch_ratio 0.2"
"ssa12t2|results_ssa_m/ssa12t2|$SEEDS|--algo dfl_ssa $SA --n_agg 12 --sa_tau 2.0"
"ssa18t2k7|results_ssa_m/ssa18t2k7|$SEEDS|--algo dfl_ssa $SA --n_agg 18 --sa_tau 2.0 --sketch_ratio 0.389"
"ssa24t2k7|results_ssa_m/ssa24t2k7|$SEEDS|--algo dfl_ssa $SA --n_agg 24 --sa_tau 2.0 --sketch_ratio 0.292"
)

for CFG in "${CONFIGS[@]}"; do
  IFS='|' read -r NAME DIR SDS ARGS <<< "${CFG//$'\n'/ }"
  DIR=mdp_cw/$DIR
  [ $# -gt 0 ] && [[ " $* " != *" $NAME "* ]] && continue
  mkdir -p "$(dirname "$DIR")/logs"
  [ "$TIMING" = 1 ] && SDS=$SEEDS
  for S in $SDS; do
    [ -f "${DIR}_s$S/results.json" ] && continue
    echo "[run] ${DIR}_s$S"
    if [ "$TIMING" = 1 ]; then
      OMP_NUM_THREADS=4 MDPLP_LP_WORKERS=$LP_WORKERS "$PY" -m mdp_cw.run_cliff_experiments \
        $COMMON $ARGS --seed "$S" --save_dir "${DIR}_s$S" \
        > "$(dirname "$DIR")/logs/$(basename "$DIR")_s$S.log" 2>&1
    else
      OMP_NUM_THREADS=4 MDPLP_LP_WORKERS=$LP_WORKERS "$PY" -m mdp_cw.run_cliff_experiments \
        $COMMON $ARGS --seed "$S" --save_dir "${DIR}_s$S" \
        > "$(dirname "$DIR")/logs/$(basename "$DIR")_s$S.log" 2>&1 &
    fi
  done
  wait                          # one method at a time: at most 10 runs in parallel
done

"$PY" mdp_cw/plot_fig3_v7.py --out mdp_cw/selected_plots/fig3_v7.png
"$PY" - <<'EOF'
import json, os, numpy as np
S = [("Two-Stage (600 ep)", "results_tsconv/ts600", range(10)),
     ("DFL-LP", "results_epochs/lp62", range(10)),
     ("DFL-SA (M=12)", "results_ssa_m/ssa12t2", range(10)),
     ("DFL-SA (M=18)", "results_ssa_m/ssa18t2k7", range(10)),
     ("DFL-SA (M=24)", "results_ssa_m/ssa24t2k7", range(10)),
     ("DFL-QP", "results_final7/qpe0002", range(2, 12)),
     ("DFL-Feas (0 Odece ep)", "results_feas60/feas", range(10)),
     ("DFL-Sketch (lw=1)", "results_sk_lw/lw1", range(10))]
rows = []
for name, base, seeds in S:
    v = [json.load(open(p))["test_mean_regret"]
         for p in (f"mdp_cw/{base}_s{s}/results.json" for s in seeds) if os.path.isfile(p)]
    if v:
        a = np.array(v)
        rows.append((a.mean(), name, a.std(ddof=1) / np.sqrt(a.size), (a < 1.45).sum(), a.size))
print(f"\n{'method':24s} {'mean +/- SEM':>17} {'escape':>9}   (escape = regret < 1.45)")
for m, name, sem, esc, n in sorted(rows):
    print(f"{name:24s} {m:8.3f} +/- {sem:5.3f} {esc:6d}/{n}")
EOF
