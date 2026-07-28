#!/usr/bin/env bash
# Follow-up to run_projection_ablation.sh: prediction-anchor x LP-layer 2x2
# at tau=0.1, 5 seeds each => 20 runs.
#   projmdplw20  aggregated-MDP LP + loss_weight 20 (canonical exp_b anchor)
#   projlplw20   exact Galerkin projected LP + loss_weight 20
#   projmdp      unanchored original arm (loss_weight 0)
#   projlp       unanchored original arm (loss_weight 0)
# The unanchored arms already exist if run_projection_ablation.sh was run;
# set SKIP_EXISTING=1 to keep them and only run the missing arms.
# Summary: python -m mdp_lp.summarize_projection_ablation (invoked at the end).
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RESULTS_ROOT="mdp_lp/results_projection_ablation"
SEEDS=(1 2 3 4 5)
TAU=0.1
COMMON=(--experiment_type b --cost_case normal_margin --bc 10 --algo dfl
        --n_epochs 50 --n_samples 10 --pretrain_epochs 30 --n_agg 10 --lr 1e-3
        --results_root "$RESULTS_ROOT" --sa_tau "$TAU")

JOBS_FILE="$(mktemp)"
for SEED in "${SEEDS[@]}"; do
  for ARM in projmdplw20 projlplw20 projmdp projlp; do
    echo "$ARM $SEED" >> "$JOBS_FILE"
  done
done
echo "[lw20 ablation] total jobs: $(wc -l < "$JOBS_FILE")"

export RESULTS_ROOT TAU
export COMMON_STR="${COMMON[*]}"
export SKIP_EXISTING="${SKIP_EXISTING:-0}"

run_wrapper() {
  local ARM SEED
  read -r ARM SEED <<< "$1"
  local TAG="${ARM}_tau${TAU}_seed${SEED}"
  local EXTRA=""
  case "$ARM" in
    projmdplw20) EXTRA="--sa_layer projmdp --loss_weight 20" ;;
    projlplw20)  EXTRA="--sa_layer projlp --loss_weight 20" ;;
    projmdp)     EXTRA="--sa_layer projmdp" ;;
    projlp)      EXTRA="--sa_layer projlp" ;;
  esac
  local DONE_MARKER="$RESULTS_ROOT/exp_b/normal_margin/dfl/tag_${TAG}/n_agg_10/epochs_50/results.json"
  if [[ "$SKIP_EXISTING" == "1" && -f "$DONE_MARKER" ]]; then
    echo "[skip] $TAG (results.json exists)"
    return 0
  fi
  # shellcheck disable=SC2086
  python -m mdp_lp.run_elasticity_experiments \
    $COMMON_STR --seed "$SEED" --tag "$TAG" $EXTRA \
    > "/tmp/ablation_${TAG}.log" 2>&1 \
    && echo "[done] $TAG" || echo "[FAILED] $TAG (see /tmp/ablation_${TAG}.log)"
}
export -f run_wrapper

xargs -a "$JOBS_FILE" -d '\n' -I{} -P "${ABLATION_JOBS:-10}" bash -c 'run_wrapper "$@"' _ {}
rm -f "$JOBS_FILE"
echo "[lw20 ablation] all runs finished"

python -m mdp_lp.summarize_projection_ablation
