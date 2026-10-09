#!/usr/bin/env bash
# Inventory problem, |S| = 100, |A| = 31. Seeds 1-10, all arms.
# Finished runs are skipped. Results go to $RESULTS_ROOT.
# Same regime as |S|=51, but --loss_weight is set per arm (LW_BASE * LW_SCALE)
# instead of 100 everywhere, so report it with the row.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

PY="${PY:-/net/csefiles/siemens/warriors-ls-recovery/users/zzhao628/#113442817/anaconda3/envs/rl/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-mdp_lp/results_dflwin_lwbal}"
LOG_DIR="${LOG_DIR:-mdp_lp/logs/dflwin_lwbal}"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
COSTS="${COSTS:-high_margin}"
SIZE="${SIZE:-100}"                  # |S| = B_MAX + s_max + 1
B_MAX="${B_MAX:-20}"                 # matches the existing |S|=100 runs
Q_MAX="${Q_MAX:-30}"
LW_SCALE="${LW_SCALE:-1.0}"          # multiplies every arm's LW_BASE
ODECE_MARGIN="${ODECE_MARGIN:-0.01}" # DFL-Feas only; its loss_weight is 0
GRAD_CLIP="${GRAD_CLIP:-10.0}"
ARMS="${ARMS:-twostage fullal_r10lw100 aggal_lw100 aggal_k20 aggal_k30 feasfaith sketchal_g02 qpal_r1}"
JOBS_CHEAP="${JOBS_CHEAP:-4}"
JOBS_BIG="${JOBS_BIG:-2}"
BIG_ARMS_RE="${BIG_ARMS_RE:-^(fullal_|qpal_)}"
TIMEOUT_SEC="${TIMEOUT_SEC:-86400}"

S_MAX=$(( SIZE - B_MAX - 1 ))
if (( S_MAX < 1 )); then echo "SIZE must exceed $((B_MAX+1))"; exit 1; fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
mkdir -p "$LOG_DIR"

# Regime: identical for every arm.
REGIME="--experiment_type b --b_max $B_MAX --s_max $S_MAX --q_max $Q_MAX \
--beta_0 4.0 --beta_1 -0.3 \
--p_lognormal_mean 10.0 --p_lognormal_std 0.5 \
--sigma_epsilon 0.1 \
--n_samples 20 --n_epochs 100 --pretrain_epochs 30 --lr 0.01 \
--grad_clip $GRAD_CLIP \
--results_root $RESULTS_ROOT"

# Per-arm measured 1:1 loss_weight, scaled by $LW_SCALE.
lw_base() {
  case "$1" in
    fullal_r10lw100) echo 5.4  ;;
    aggal_lw100)     echo 2.6  ;;
    aggal_k20)       echo 3.5  ;;
    aggal_k30)       echo 2.7  ;;
    sketchal_g02)    echo 30.2 ;;
    qpal_r1)         echo 57.7 ;;
    *)               echo 0    ;;   # twostage / feasfaith carry no anchor
  esac
}

arm_args() {   # $1 = arm, $2 = resolved loss_weight
  local lw="$2"
  case "$1" in
    twostage)         echo "--algo twostage" ;;
    # DFL-LP, augmented Lagrangian
    fullal_r10lw100)  echo "--algo dfl --loss_mode auglag --aug_lag_rho 10 --loss_weight $lw" ;;
    # DFL-SA at M in {10,20,30}
    aggal_lw100)      echo "--algo dfl --loss_mode auglag --aug_lag_rho 10 --loss_weight $lw --n_agg 10 --sa_layer projmdp --sa_tau 0.5" ;;
    aggal_k20)        echo "--algo dfl --loss_mode auglag --aug_lag_rho 10 --loss_weight $lw --n_agg 20 --sa_layer projmdp --sa_tau 0.5" ;;
    aggal_k30)        echo "--algo dfl --loss_mode auglag --aug_lag_rho 10 --loss_weight $lw --n_agg 30 --sa_layer projmdp --sa_tau 0.5" ;;
    # DFL-Feas (Odece), faithful alpha=0 mu=0; no anchor -> tune the margin
    feasfaith)        echo "--algo dfl_feas --odece_alpha 0.0 --odece_mse_weight 0.0 --odece_margin $ODECE_MARGIN" ;;
    # DFL-Sketch: K=8 random row sketches, ~20% of the flow constraints
    sketchal_g02)     echo "--algo dfl_sketch --n_sketches 8 --sketch_ratio 0.2 --sketch_mode gaussian --loss_mode auglag --aug_lag_rho 10 --loss_weight $lw" ;;
    # DFL-QP: qpth, qp_reg = 1.0
    qpal_r1)          echo "--algo dfl_qpth --qp_reg 1.0 --loss_mode auglag --aug_lag_rho 10 --loss_weight $lw" ;;
    *) echo "" ;;
  esac
}

run_one() {
  local size arm cost seed
  read -r size arm cost seed <<< "$1"
  local base lw
  base="$(lw_base "$arm")"
  lw="$(awk -v b="$base" -v s="$LW_SCALE" 'BEGIN{printf "%.4g", b*s}')"

  # LW_SCALE is in the tag so runs never collide.
  local tag="S${size}bal${LW_SCALE}_${arm}_${cost}_seed${seed}"
  local extra; extra="$(arm_args "$arm" "$lw")"
  if [[ -z "$extra" ]]; then echo "[skip] unknown arm $arm"; return 0; fi
  if find "$RESULTS_ROOT" -path "*tag_${tag}/*results.json" -print -quit 2>/dev/null | grep -q .; then
    echo "[have] $tag"; return 0
  fi

  local log="$LOG_DIR/${tag}.log"
  echo "running" > "$LOG_DIR/${tag}.status"
  ulimit -s 65536 2>/dev/null || true   # gurobipy large-model teardown SIGSEGV
  # shellcheck disable=SC2086
  timeout "$TIMEOUT_SEC" "$PY" -m mdp_lp.run_elasticity_experiments \
      $REGIME --cost_case "$cost" --seed "$seed" --tag "$tag" $extra > "$log" 2>&1
  local rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "done" > "$LOG_DIR/${tag}.status"
    echo "[OK] $tag (lw=$lw) $(grep -oE 'Mean regret: +[0-9.]+' "$log" | tail -1)"
  elif [ "$rc" -eq 124 ]; then
    echo "timeout" > "$LOG_DIR/${tag}.status"; echo "[TIMEOUT >${TIMEOUT_SEC}s] $tag"
  else
    echo "failed:$rc" > "$LOG_DIR/${tag}.status"; echo "[FAIL rc=$rc] $tag (see $log)"
  fi
}
export -f run_one arm_args lw_base
export PY RESULTS_ROOT LOG_DIR REGIME LW_SCALE ODECE_MARGIN TIMEOUT_SEC

CHEAP_FILE="$(mktemp)"; BIG_FILE="$(mktemp)"
trap 'rm -f "$CHEAP_FILE" "$BIG_FILE"' EXIT
for cost in $COSTS; do
  for arm in $ARMS; do
    for seed in $SEEDS; do
      if [[ "$arm" =~ $BIG_ARMS_RE ]]; then
        echo "$SIZE $arm $cost $seed" >> "$BIG_FILE"
      else
        echo "$SIZE $arm $cost $seed" >> "$CHEAP_FILE"
      fi
    done
  done
done

echo "[lwbal] |S|=$SIZE (b_max=$B_MAX s_max=$S_MAX q_max=$Q_MAX) LW_SCALE=$LW_SCALE"
for arm in $ARMS; do
  b="$(lw_base "$arm")"
  printf '[lwbal]   %-18s loss_weight = %s\n' "$arm" \
    "$(awk -v b="$b" -v s="$LW_SCALE" 'BEGIN{printf "%.4g", b*s}')"
done
echo "[lwbal] cheap jobs (P=$JOBS_CHEAP): $(wc -l < "$CHEAP_FILE")   big jobs (P=$JOBS_BIG): $(wc -l < "$BIG_FILE")"

PIDS=()
if [ -s "$BIG_FILE" ]; then
  xargs -a "$BIG_FILE" -d '\n' -I{} -P "$JOBS_BIG" bash -c 'run_one "$0"' {} & PIDS+=($!)
fi
if [ -s "$CHEAP_FILE" ]; then
  xargs -a "$CHEAP_FILE" -d '\n' -I{} -P "$JOBS_CHEAP" bash -c 'run_one "$0"' {} & PIDS+=($!)
fi
[ ${#PIDS[@]} -gt 0 ] && wait "${PIDS[@]}"
echo "[lwbal] DONE. Results under $RESULTS_ROOT ; logs $LOG_DIR"
