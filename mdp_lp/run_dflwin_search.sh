#!/usr/bin/env bash
# Hyperparam search: find a regime where DFL (esp. aggregated "Ours") BEATS
# two-stage, per DFL_VS_TWOSTAGE_ANALYSIS.md's winning sanity regime:
#   exp b, high/low margin, b_max=20 s_max=30 (|S|=51), n_samples=20,
#   beta0=4 beta1=-0.3, p_logmean=10, p_std=0.5, sigma_eps=0.1,
#   loss_weight=100, aug_lag_rho=10, 100 epochs, lr default (1e-2).
# Stage A (default): 5 arms x {high,low}_margin x 3 seeds at |S|=51.
# Stage B (override SIZES/ARMS): scale the winning arm vs twostage.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/nethome/zzhao628/anaconda3/envs/rl/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-mdp_lp/results_dflwin}"
LOG_DIR="${LOG_DIR:-mdp_lp/logs/dflwin}"
SIZES=(${SIZES:-51})          # |S| = B_MAX + s_max + 1, B_MAX=20
SEEDS=(${SEEDS:-1 2 3})
ARMS=(${ARMS:-twostage full_lw100 agg_lw100 aggal_lw100 aggal_lw20})
COSTS=(${COSTS:-high_margin low_margin})
N_EPOCHS="${N_EPOCHS:-100}"
PRETRAIN="${PRETRAIN:-30}"
N_AGG="${N_AGG:-10}"
B_MAX=20
TIMEOUT_SEC="${TIMEOUT_SEC:-43200}"
JOBS="${JOBS:-6}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1

mkdir -p "$LOG_DIR"
export PYTHON RESULTS_ROOT LOG_DIR N_EPOCHS PRETRAIN N_AGG B_MAX TIMEOUT_SEC

run_wrapper() {
  ulimit -s 65536 2>/dev/null || true   # gurobipy large-model stack segfault
  local SIZE ARM COST SEED
  read -r SIZE ARM COST SEED <<< "$1"
  local TAG="S${SIZE}_${ARM}_${COST}_seed${SEED}"
  local S_MAX=$((SIZE - B_MAX - 1))
  local ALGO EXTRA=""
  local AGG="--n_agg $N_AGG --sa_layer projmdp --sa_tau 0.5"
  case "$ARM" in
    twostage)    ALGO=twostage ;;
    full_lw100)  ALGO=dfl; EXTRA="--loss_weight 100 --aug_lag_rho 10" ;;
    agg_lw100)   ALGO=dfl; EXTRA="$AGG --loss_weight 100 --aug_lag_rho 10" ;;
    aggal_lw100) ALGO=dfl; EXTRA="$AGG --loss_weight 100 --aug_lag_rho 10 --loss_mode auglag" ;;
    aggal_lw20)  ALGO=dfl; EXTRA="$AGG --loss_weight 20 --aug_lag_rho 10 --loss_mode auglag" ;;
    aggal_k20)   ALGO=dfl; EXTRA="--n_agg 20 --sa_layer projmdp --sa_tau 0.5 --loss_weight 100 --aug_lag_rho 10 --loss_mode auglag" ;;
    aggal_k30)   ALGO=dfl; EXTRA="--n_agg 30 --sa_layer projmdp --sa_tau 0.5 --loss_weight 100 --aug_lag_rho 10 --loss_mode auglag" ;;
    aggal_rho100) ALGO=dfl; EXTRA="$AGG --loss_weight 100 --aug_lag_rho 100 --loss_mode auglag" ;;
    aggal_tau03) ALGO=dfl; EXTRA="--n_agg 10 --sa_layer projmdp --sa_tau 0.3 --loss_weight 100 --aug_lag_rho 10 --loss_mode auglag" ;;
    *) echo "[dflwin] unknown arm $ARM"; return 0 ;;
  esac
  local LOG="$LOG_DIR/${TAG}.log"
  echo "running" > "$LOG_DIR/${TAG}.status"
  # shellcheck disable=SC2086
  timeout "$TIMEOUT_SEC" /usr/bin/time -v "$PYTHON" -m mdp_lp.run_elasticity_experiments \
    --experiment_type b --cost_case "$COST" --algo "$ALGO" \
    --b_max "$B_MAX" --s_max "$S_MAX" --beta_0 4.0 --beta_1 -0.3 \
    --p_lognormal_mean 10.0 --p_lognormal_std 0.5 --sigma_epsilon 0.1 \
    --n_epochs "$N_EPOCHS" --n_samples 20 --pretrain_epochs "$PRETRAIN" \
    --seed "$SEED" --tag "$TAG" \
    --results_root "$RESULTS_ROOT" $EXTRA > "$LOG" 2>&1
  local RC=$?
  if [ "$RC" -eq 0 ]; then
    echo "done" > "$LOG_DIR/${TAG}.status";    echo "[done] $TAG"
  elif [ "$RC" -eq 124 ]; then
    echo "timeout" > "$LOG_DIR/${TAG}.status"; echo "[TIMEOUT] $TAG"
  else
    echo "failed:$RC" > "$LOG_DIR/${TAG}.status"; echo "[FAILED rc=$RC] $TAG (see $LOG)"
  fi
}
export -f run_wrapper

JOBS_FILE="$(mktemp)"
for ((i=${#SIZES[@]}-1; i>=0; i--)); do
  SIZE="${SIZES[$i]}"
  for ARM in "${ARMS[@]}"; do
    for COST in "${COSTS[@]}"; do
      for SEED in "${SEEDS[@]}"; do
        echo "$SIZE $ARM $COST $SEED" >> "$JOBS_FILE"
      done
    done
  done
done
echo "[dflwin] jobs: $(wc -l < "$JOBS_FILE") (P=$JOBS)"
xargs -a "$JOBS_FILE" -d '\n' -I{} -P "$JOBS" bash -c 'run_wrapper "$@"' _ {}
rm -f "$JOBS_FILE"
echo "[dflwin] all runs finished"
