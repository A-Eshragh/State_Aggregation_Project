#!/usr/bin/env bash
# v8 multi-gate pilot: does decision quality IMPROVE with aggregation size M?
#
# v7 has one hazard cell, so the optimal detour is a single global binary choice
# that M=12 already represents; there M only trades regularization strength and
# regret rises monotonically with M (1.69 / 2.16 / 2.54 at M=12/18/24).
# v8 moves the deadly gate between columns 3, 6 and 9 by regime, so the policy
# must localise the hazard and resolution should start to pay.
#
# Prior 17:1:1:1 puts the per-gate hazard fraction at 0.05, the peak of the
# analytic trap scan (TS-ceiling regret 4.21; 7:1:1:1 disarms to 1.11 because
# the compound risk becomes visible in the posterior mean).
#
#   bash mdp_cw/run_v8_pilot.sh              # 3 seeds x {TS, SA M=12, SA M=24}
#   SEEDS="0 1 2 3 4" bash mdp_cw/run_v8_pilot.sh
set -u
cd "$(dirname "$0")/.."
ulimit -s 65536
PY=${PY:-python3}
SEEDS=${SEEDS:-"0 1 2"}
OUT=mdp_cw/results_v8_pilot

# Identical to run_dfl_seed.sh COMMON except the regime set and prior. Keeping
# ambiguous_pairs 0:3 leaves clear-vs-east ambiguous (the trap) while west/mid
# stay identifiable, so localising the hazard is learnable and resolution can
# matter.
COMMON="--n_samples 200 --feat_dim 8 --noise_std 0.1 --train_frac 0.5 --batch_size 32
        --regime_set v8 --center_scale 0.88 --ambiguous_pairs 0:3 --ambiguous_gap 0.0
        --cliff_penalty -5.0 --regime_weights 17,1,1,1 --test_regime_weights 17,1,1,1
        --stratify_regimes --hidden_dim 64 --nn_feat_dim 32
        --lr 0.001 --pretrain_lr 0.01 --pretrain_epochs 60 --grad_clip 10.0 --beta 0.98"

DFL="--n_epochs 140 --loss_mode auglag --aug_lag_rho 1.0
     --loss_weight 0.0 --ambient_ce_weight 50.0 --head_ce_warmup_epochs 60"
SA="$DFL --sa_tau 0.5 --sa_r_embed 8 --n_sketches 8 --sketch_ratio 0.6"

CONFIGS=(
"ts|$OUT/ts|--algo twostage --n_epochs 600 --loss_mode penalty --loss_weight 1.0"
"ssa12|$OUT/ssa12|--algo dfl_ssa $SA --n_agg 12"
"ssa24|$OUT/ssa24|--algo dfl_ssa $SA --n_agg 24"
)

mkdir -p "$OUT/logs"
for CFG in "${CONFIGS[@]}"; do
  IFS='|' read -r NAME DIR ARGS <<< "${CFG//$'\n'/ }"
  [ $# -gt 0 ] && [[ " $* " != *" $NAME "* ]] && continue
  for S in $SEEDS; do
    [ -f "${DIR}_s$S/results.json" ] && continue
    echo "[run] ${DIR}_s$S"
    OMP_NUM_THREADS=4 MDPLP_LP_WORKERS=6 "$PY" -m mdp_cw.run_cliff_experiments \
      $COMMON $ARGS --seed "$S" --save_dir "${DIR}_s$S" \
      > "$OUT/logs/$(basename "$DIR")_s$S.log" 2>&1 &
  done
  wait
done

"$PY" - <<'EOF'
import json, os, numpy as np
OUT = "mdp_cw/results_v8_pilot"
for name in ("ts", "ssa12", "ssa24"):
    v = []
    for s in range(20):
        p = f"{OUT}/{name}_s{s}/results.json"
        if os.path.isfile(p):
            v.append(json.load(open(p))["test_mean_regret"])
    if v:
        a = np.array(v)
        print(f"{name:6s} n={a.size} mean={a.mean():.3f} std={a.std(ddof=1) if a.size>1 else 0:.3f}  {[round(x,3) for x in a]}")
EOF
