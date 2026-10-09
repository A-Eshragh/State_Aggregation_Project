#!/usr/bin/env bash
# Level-3 ablation for the diagonal-projection rebuttal experiment:
#   projmdp (aggregated-MDP LP, ours) vs projlp (exact Galerkin projected LP)
#   vs hard (straight-through hard memberships) vs fixedw (fixed random soft W).
# tau=0.5 runs all four arms; tau=0.1 adds the regime where projlp is mostly
# feasible. 5 seeds each => 30 runs.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RESULTS_ROOT="mdp_lp/results_projection_ablation"
SEEDS=(1 2 3 4 5)
COMMON=(--experiment_type b --cost_case normal_margin --bc 10 --algo dfl
        --n_epochs 50 --n_samples 10 --pretrain_epochs 30 --n_agg 10 --lr 1e-3
        --results_root "$RESULTS_ROOT")

JOBS_FILE="$(mktemp)"
for SEED in "${SEEDS[@]}"; do
  for TAU in 0.5 0.1; do
    echo "projmdp $TAU $SEED" >> "$JOBS_FILE"
    echo "projlp $TAU $SEED" >> "$JOBS_FILE"
  done
  echo "hard 0.5 $SEED" >> "$JOBS_FILE"
  echo "fixedw 0.5 $SEED" >> "$JOBS_FILE"
done
echo "[ablation] total jobs: $(wc -l < "$JOBS_FILE")"

run_one() {
  local ARM=$1 TAU=$2 SEED=$3
  local TAG="${ARM}_tau${TAU}_seed${SEED}"
  local EXTRA=()
  case "$ARM" in
    projmdp) EXTRA=(--sa_layer projmdp) ;;
    projlp)  EXTRA=(--sa_layer projlp) ;;
    hard)    EXTRA=(--sa_layer hard) ;;
    fixedw)  EXTRA=(--sa_layer projmdp --agg_fixed_W --agg_fixed_W_seed "$SEED") ;;
  esac
  python -m mdp_lp.run_elasticity_experiments \
    "${COMMON[@]}" --sa_tau "$TAU" --seed "$SEED" --tag "$TAG" "${EXTRA[@]}" \
    > "/tmp/ablation_${TAG}.log" 2>&1 \
    && echo "[done] $TAG" || echo "[FAILED] $TAG (see /tmp/ablation_${TAG}.log)"
}
export -f run_one
export RESULTS_ROOT
export COMMON_STR="${COMMON[*]}"

# re-expand COMMON inside the subshell via the exported string
run_wrapper() {
  local ARM TAU SEED
  read -r ARM TAU SEED <<< "$1"
  local TAG="${ARM}_tau${TAU}_seed${SEED}"
  local EXTRA=""
  case "$ARM" in
    projmdp) EXTRA="--sa_layer projmdp" ;;
    projlp)  EXTRA="--sa_layer projlp" ;;
    hard)    EXTRA="--sa_layer hard" ;;
    fixedw)  EXTRA="--sa_layer projmdp --agg_fixed_W --agg_fixed_W_seed $SEED" ;;
  esac
  # shellcheck disable=SC2086
  python -m mdp_lp.run_elasticity_experiments \
    $COMMON_STR --sa_tau "$TAU" --seed "$SEED" --tag "$TAG" $EXTRA \
    > "/tmp/ablation_${TAG}.log" 2>&1 \
    && echo "[done] $TAG" || echo "[FAILED] $TAG (see /tmp/ablation_${TAG}.log)"
}
export -f run_wrapper

xargs -a "$JOBS_FILE" -d '\n' -I{} -P "${ABLATION_JOBS:-6}" bash -c 'run_wrapper "$@"' _ {}
rm -f "$JOBS_FILE"
echo "[ablation] all runs finished"
