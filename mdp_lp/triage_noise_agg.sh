#!/usr/bin/env bash
# Triage at M=56 (s=q=50) on 3 seeds with two changes:
#   1. sigma_epsilon=0.7 (vs default 0.3) → noisier λ targets, harder for two-stage MSE
#   2. agg_pretrain_epochs=30 for sketch+agg → W is pretrained before DFL
# Configs: twostage, dfl no-agg, sketch agg M ∈ {10, 20, 30}
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

COMMON=(--experiment_type b --cost_case high_margin
        --n_epochs 50 --n_samples 20 --lr 0.01
        --pretrain_epochs 50
        --b_max 5 --s_max 50 --q_max 50
        --sigma_epsilon 0.5)

for SEED in 1 2 3; do
  echo "=============================================================="; echo "[triage] seed=$SEED twostage"; echo "=============================================================="
  python -m mdp_lp.run_elasticity_experiments "${COMMON[@]}" \
    --algo twostage --n_agg -1 \
    --seed $SEED --tag "seed_${SEED}_s50q50_se05_triage"

  echo "=============================================================="; echo "[triage] seed=$SEED dfl no-agg"; echo "=============================================================="
  python -m mdp_lp.run_elasticity_experiments "${COMMON[@]}" \
    --algo dfl --n_agg -1 --aug_lag_rho 10000.0 \
    --seed $SEED --tag "seed_${SEED}_s50q50_se05_triage"

  for NA in 10 20 30; do
    echo "=============================================================="; echo "[triage] seed=$SEED sketch M=$NA (agg_pretrain=30)"; echo "=============================================================="
    python -m mdp_lp.run_elasticity_experiments "${COMMON[@]}" \
      --algo dfl_sketch --n_agg $NA \
      --aug_lag_rho 0.0 \
      --n_sketches 4 --sketch_ratio 0.3 --sketch_mode gaussian \
      --agg_pretrain_epochs 30 \
      --seed $SEED --tag "seed_${SEED}_s50q50_se05_triage_aggpre30_nagg_${NA}_K4_r0.3_gaussian"
  done
done
echo "[triage] Done."
