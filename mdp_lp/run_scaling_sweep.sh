#!/usr/bin/env bash
# Scalability sweep for the rebuttal ("all benchmarks use small |S|"):
# scale the inventory MDP state space via s_max (n_states = b_max + s_max + 1,
# b_max=5 and q_max=30 fixed => |A|=31) and compare, at each |S|:
#   full     : --algo dfl                 (full-state differentiable LP)
#   agg10    : --algo dfl --n_agg 10      (ours: soft aggregation, fixed k=10)
#   twostage : --algo twostage            (prediction-only baseline)
# Protocol otherwise identical to run_projection_ablation.sh (exp b,
# normal_margin, bc=10, 50 epochs, n_samples 10, pretrain 30, lr 1e-3).
#
# Everything is overridable via env vars, e.g. a 1-epoch feasibility probe:
#   SIZES="36 250 1000 2000" SEEDS="1" ARMS="full agg10" N_EPOCHS=1 PRETRAIN=1 \
#   TIMEOUT_SEC=7200 RESULTS_ROOT=mdp_lp/results_scaling_probe \
#   LOG_DIR=mdp_lp/logs/scaling_probe bash mdp_lp/run_scaling_sweep.sh
#
# Per-run wall budget is TIMEOUT_SEC; runs that exceed it are recorded as
# "timeout" in $LOG_DIR/<tag>.status (=> "> budget" entries in the table).
# Sizes >= BIG_MIN_S run in a low-parallelism queue (memory-heavy); both
# queues run concurrently. Summarize with: python -m mdp_lp.summarize_scaling
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/nethome/zzhao628/anaconda3/envs/rl/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-mdp_lp/results_scaling}"
LOG_DIR="${LOG_DIR:-mdp_lp/logs/scaling_sweep}"
SIZES=(${SIZES:-36 100 250 500 1000 2000})
SEEDS=(${SEEDS:-1 2 3})
ARMS=(${ARMS:-full agg10 twostage})
N_EPOCHS="${N_EPOCHS:-50}"
PRETRAIN="${PRETRAIN:-30}"
N_AGG="${N_AGG:-10}"
B_MAX=5   # default; |S| = B_MAX + s_max + 1
TIMEOUT_SEC="${TIMEOUT_SEC:-43200}"
BIG_MIN_S="${BIG_MIN_S:-1000}"
JOBS_SMALL="${JOBS_SMALL:-6}"
JOBS_BIG="${JOBS_BIG:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
# Unbuffered + faulthandler so crashed runs (e.g. segfaults at large |S|)
# leave their progress prints and a traceback in the log.
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1

mkdir -p "$LOG_DIR"
export PYTHON RESULTS_ROOT LOG_DIR N_EPOCHS PRETRAIN N_AGG B_MAX TIMEOUT_SEC

run_wrapper() {
  # gurobipy 12.0.2 segfaults (C-stack overflow, SIGSEGV in model teardown)
  # with large dense-built models on the default 8MB stack; |S|>=1000 needs
  # this. Thread stacks inherit the soft limit, so worker threads are covered.
  ulimit -s 65536 2>/dev/null || true
  local SIZE ARM SEED
  read -r SIZE ARM SEED <<< "$1"
  local TAG="S${SIZE}_${ARM}_seed${SEED}"
  local S_MAX=$((SIZE - B_MAX - 1))
  local ALGO EXTRA=""
  case "$ARM" in
    full)     ALGO=dfl ;;
    agg10)    ALGO=dfl; EXTRA="--n_agg $N_AGG --sa_layer projmdp --sa_tau 0.5" ;;
    aggratio) ALGO=dfl; local K=$((SIZE / 10)); [ "$K" -lt 4 ] && K=4
              EXTRA="--n_agg $K --sa_layer projmdp --sa_tau 0.5" ;;
    aggl20)   ALGO=dfl; EXTRA="--n_agg $N_AGG --sa_layer projmdp --sa_tau 0.5 --loss_weight 20" ;;
    agg10al)  ALGO=dfl; EXTRA="--n_agg $N_AGG --sa_layer projmdp --sa_tau 0.5 --loss_mode auglag" ;;
    aggratioal) ALGO=dfl; local K=$((SIZE / 10)); [ "$K" -lt 4 ] && K=4
              EXTRA="--n_agg $K --sa_layer projmdp --sa_tau 0.5 --loss_mode auglag" ;;
    aggl20al) ALGO=dfl; EXTRA="--n_agg $N_AGG --sa_layer projmdp --sa_tau 0.5 --loss_weight 20 --loss_mode auglag" ;;
    twostage) ALGO=twostage ;;
    *) echo "[scaling] unknown arm $ARM"; return 0 ;;
  esac
  local LOG="$LOG_DIR/${TAG}.log"
  echo "running" > "$LOG_DIR/${TAG}.status"
  # shellcheck disable=SC2086
  timeout "$TIMEOUT_SEC" /usr/bin/time -v "$PYTHON" -m mdp_lp.run_elasticity_experiments \
    --experiment_type b --cost_case normal_margin --bc 10 --algo "$ALGO" \
    --n_epochs "$N_EPOCHS" --n_samples 10 --pretrain_epochs "$PRETRAIN" --lr 1e-3 \
    --s_max "$S_MAX" --seed "$SEED" --tag "$TAG" \
    --results_root "$RESULTS_ROOT" $EXTRA > "$LOG" 2>&1
  local RC=$?
  if [ "$RC" -eq 0 ]; then
    echo "done" > "$LOG_DIR/${TAG}.status";    echo "[done] $TAG"
  elif [ "$RC" -eq 124 ]; then
    echo "timeout" > "$LOG_DIR/${TAG}.status"; echo "[TIMEOUT >$TIMEOUT_SEC s] $TAG"
  else
    echo "failed:$RC" > "$LOG_DIR/${TAG}.status"
    echo "[FAILED rc=$RC] $TAG (see $LOG)"
  fi
}
export -f run_wrapper

JOBS_SMALL_FILE="$(mktemp)"; JOBS_BIG_FILE="$(mktemp)"
# Largest sizes first inside each queue so the slowest runs start immediately.
for ((i=${#SIZES[@]}-1; i>=0; i--)); do
  SIZE="${SIZES[$i]}"
  for ARM in "${ARMS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      if [ "$SIZE" -ge "$BIG_MIN_S" ]; then
        echo "$SIZE $ARM $SEED" >> "$JOBS_BIG_FILE"
      else
        echo "$SIZE $ARM $SEED" >> "$JOBS_SMALL_FILE"
      fi
    done
  done
done
echo "[scaling] big jobs (|S|>=$BIG_MIN_S, P=$JOBS_BIG): $(wc -l < "$JOBS_BIG_FILE")"
echo "[scaling] small jobs (P=$JOBS_SMALL): $(wc -l < "$JOBS_SMALL_FILE")"

PIDS=()
if [ -s "$JOBS_BIG_FILE" ]; then
  xargs -a "$JOBS_BIG_FILE" -d '\n' -I{} -P "$JOBS_BIG" bash -c 'run_wrapper "$@"' _ {} &
  PIDS+=($!)
fi
if [ -s "$JOBS_SMALL_FILE" ]; then
  xargs -a "$JOBS_SMALL_FILE" -d '\n' -I{} -P "$JOBS_SMALL" bash -c 'run_wrapper "$@"' _ {} &
  PIDS+=($!)
fi
wait "${PIDS[@]}"
rm -f "$JOBS_SMALL_FILE" "$JOBS_BIG_FILE"
echo "[scaling] all runs finished"
