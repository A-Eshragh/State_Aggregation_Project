#!/usr/bin/env bash
# Hyperparameter tuning for sketch + n_agg=10 at M=56:
# Goal: better mean regret AND lower variance than baseline (K=4 r=0.3 ρ=0 = 10.17 ± 14.16%).
# Two candidates, both ρ=0 (avoiding soft-agg bias):
#   A: K=8 r=0.3 ρ=0   — more averaging, same sketch ratio
#   B: K=4 r=0.5 ρ=0   — same K, less sketch bias
#   C: K=8 r=0.5 ρ=0   — both levers (most principled)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

COMMON=(--experiment_type b --cost_case high_margin --algo dfl_sketch
        --n_epochs 50 --n_samples 20 --lr 0.01
        --pretrain_epochs 30
        --b_max 5 --s_max 50 --q_max 50
        --n_agg 10 --aug_lag_rho 0.0
        --sketch_mode gaussian)

for SEED in 1 2 3 4 5; do
  for SPEC in "8|0.3" "4|0.5" "8|0.5"; do
    IFS='|' read -r K R <<< "$SPEC"
    TAG="seed_${SEED}_s50q50_nagg_10_K${K}_r${R}_gaussian_tune"
    echo "=============================================================="
    echo "[tune] seed=$SEED  K=$K  r=$R"
    echo "=============================================================="
    python -m mdp_lp.run_elasticity_experiments "${COMMON[@]}" \
      --n_sketches "$K" --sketch_ratio "$R" \
      --seed "$SEED" --tag "$TAG"
  done
done
echo "[tune] Done."
