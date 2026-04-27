#!/usr/bin/env bash
# python -m mdp_continuous.make_figure1_cartpole \
    # --seeds 1 2 3 4 5 6 7 8 \
    # --behavior biased --td_weight 0.013 \
    # --n_trajectories 100 --n_iterations 25 \
    # --pretrain_epochs_ts 30 --pretrain_epochs_dfl 30 \
    # --n_ce_steps 0 --n_dfl_steps 5 \
    # --eval_policy lp --dfl_lr 3e-4 \
    # --n_workers 8 --show_running_max

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SEEDS=(1 2 3 4 5 6 7 8 9 10)
N_ITERATIONS=25
N_EVAL_EPISODES=100
EVAL_POLICY="lp"
N_TRAJECTORIES=100
N_FEATURES=128
N_CE_STEPS=0
N_DFL_STEPS=5
PRETRAIN_TS=30
PRETRAIN_DFL=30
BEHAVIOR="biased"
TD_WEIGHT=0.013
LR=1e-3
DFL_LR=3e-4
MAX_PARALLEL=${MAX_PARALLEL:-8}

# Build the job list — one (algo,seed) per line for xargs.
JOBS=()
for SEED in "${SEEDS[@]}"; do
  JOBS+=("two_stage|${SEED}|${PRETRAIN_TS}")
  JOBS+=("neural_dfl|${SEED}|${PRETRAIN_DFL}")
done

echo "[run_dfl_seeds] launching ${#JOBS[@]} jobs across ${MAX_PARALLEL} workers"
echo "[run_dfl_seeds] behavior=$BEHAVIOR  td_weight=$TD_WEIGHT  n_traj=$N_TRAJECTORIES"
echo "[run_dfl_seeds] iters=$N_ITERATIONS  pre_ts=$PRETRAIN_TS  pre_dfl=$PRETRAIN_DFL"
echo "[run_dfl_seeds] n_ce_steps=$N_CE_STEPS  n_dfl_steps=$N_DFL_STEPS"
echo "[run_dfl_seeds] eval_policy=$EVAL_POLICY  lr=$LR  dfl_lr=$DFL_LR"

# Each worker invocation runs one (algo,seed). Stdout is line-buffered per-job.
run_one() {
  local spec="$1"
  IFS='|' read -r ALGO SEED N_PRE <<< "$spec"
  local LOG="mdp_continuous/results_cartpole/${ALGO}/tag_seed_${SEED}/run.log"
  mkdir -p "$(dirname "$LOG")"
  echo "[start] algo=$ALGO seed=$SEED pretrain=$N_PRE"
  python -m mdp_continuous.run_cartpole_experiments \
    --algo "$ALGO" \
    --seed "$SEED" \
    --tag "seed_${SEED}" \
    --n_iterations "$N_ITERATIONS" \
    --pretrain_epochs "$N_PRE" \
    --n_eval_episodes "$N_EVAL_EPISODES" \
    --n_trajectories "$N_TRAJECTORIES" \
    --n_features "$N_FEATURES" \
    --eval_policy "$EVAL_POLICY" \
    --n_ce_steps "$N_CE_STEPS" \
    --n_dfl_steps "$N_DFL_STEPS" \
    --lr "$LR" \
    --dfl_lr "$DFL_LR" \
    > "$LOG" 2>&1
  local STATUS=$?
  if [ $STATUS -eq 0 ]; then
    grep -E '\[run_cartpole\] (final_reward|saved)' "$LOG" \
      | sed "s/^/[done] algo=$ALGO seed=$SEED  /"
  else
    echo "[FAIL] algo=$ALGO seed=$SEED (see $LOG)"
  fi
  return $STATUS
}
export -f run_one
export N_ITERATIONS N_EVAL_EPISODES EVAL_POLICY N_TRAJECTORIES N_FEATURES \
       N_CE_STEPS N_DFL_STEPS LR DFL_LR

printf '%s\n' "${JOBS[@]}" \
  | xargs -P "$MAX_PARALLEL" -I {} bash -c 'run_one "$@"' _ {}

echo "[run_dfl_seeds] All jobs done. Results under mdp_continuous/results_cartpole/"
echo "[run_dfl_seeds] Generate figure1_cartpole.pdf with:"
echo "  python -m mdp_continuous.make_figure1_cartpole \\"
echo "      --behavior $BEHAVIOR --td_weight $TD_WEIGHT \\"
echo "      --n_trajectories $N_TRAJECTORIES --n_iterations $N_ITERATIONS \\"
echo "      --pretrain_epochs_ts $PRETRAIN_TS --pretrain_epochs_dfl $PRETRAIN_DFL \\"
echo "      --n_ce_steps $N_CE_STEPS --eval_policy $EVAL_POLICY --dfl_lr $DFL_LR \\"
echo "      --show_running_max"
