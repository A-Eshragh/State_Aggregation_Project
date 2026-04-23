#!/usr/bin/env bash

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SEEDS=(1 2 3)
N_ITERATIONS=30
PRETRAIN_EPOCHS=30
N_EVAL_EPISODES=50
EVAL_POLICY="ce"
N_TRAJECTORIES=1
N_FEATURES=128

# Each entry: "name|algo|n_iterations|pretrain_epochs|lr"
CONFIGS=(
  "cartpole_twostage|two_stage|${N_ITERATIONS}|${PRETRAIN_EPOCHS}|1e-3"
  "cartpole_neural_dfl|neural_dfl|${N_ITERATIONS}|${PRETRAIN_EPOCHS}|1e-3"
)

for SEED in "${SEEDS[@]}"; do
  for CFG in "${CONFIGS[@]}"; do
    IFS='|' read -r NAME ALGO N_ITERS N_PRE LR <<< "$CFG"
    echo "=============================================================="
    echo "[run_dfl_seeds] seed=$SEED | $NAME"
    echo "  algo=$ALGO n_iter=$N_ITERS pretrain=$N_PRE lr=$LR eval=$EVAL_POLICY"
    echo "  n_trajectories=$N_TRAJECTORIES n_features=$N_FEATURES n_eval_episodes=$N_EVAL_EPISODES"
    echo "=============================================================="
    python -m mdp_continuous.run_cartpole_experiments \
      --algo "$ALGO" \
      --seed "$SEED" \
      --tag "seed_${SEED}" \
      --n_iterations "$N_ITERS" \
      --pretrain_epochs "$N_PRE" \
      --n_eval_episodes "$N_EVAL_EPISODES" \
      --n_trajectories "$N_TRAJECTORIES" \
      --n_features "$N_FEATURES" \
      --eval_policy "$EVAL_POLICY" \
      --lr "$LR"
  done
done

echo "[run_dfl_seeds] Done. Results under mdp_continuous/results_cartpole/..."
