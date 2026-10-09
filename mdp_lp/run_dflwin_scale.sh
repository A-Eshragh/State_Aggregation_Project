#!/usr/bin/env bash
# The |S|=51 "dflwin" table (run_dflwin_10seeds.sh) re-run at larger state
# spaces -- default |S| in {100, 200}. Same arms, same regime, same 8 seeds.
#
#   bash mdp_lp/run_dflwin_scale.sh                       # |S|=100,200 high_margin
#   SIZES=100 bash mdp_lp/run_dflwin_scale.sh             # one size
#   SCALE_MODE=ratio bash mdp_lp/run_dflwin_scale.sh      # see "how |S| grows"
#   ARMS="twostage aggal_lw100" SIZES=200 bash mdp_lp/run_dflwin_scale.sh
#
# Idempotent: any run whose results.json exists is skipped. Per-run wall budget
# is TIMEOUT_SEC; overruns are recorded as "timeout" in $LOG_DIR/<tag>.status
# and do NOT block the rest of the fleet.
#
# ---------------------------------------------------------------------------
# READ THIS BEFORE LAUNCHING -- measured cost, seed 1, high_margin, 100 epochs
# ---------------------------------------------------------------------------
# Solver s/epoch and total wall, from results.json of completed runs:
#
#            |S|=51            |S|=100           |S|=200 (EXTRAPOLATED)
#   twostage    0.00  0.02h       0.00  0.03h       0.00   ~0.1h
#   aggal_*     0.60  0.06h       0.50  0.08h       ~1.5    ~0.2h
#   sketchal    5.01  0.28h       --                ~80     ~2h
#   fullal      5.07  0.30h      81.59  4.52h      ~1200   ~34h   <-- see below
#   qpal_r1    23.30  0.77h       --               ~5500  ~150h   <-- see below
#
# The full-state solver cost went 2.38 -> 99.78 s/epoch from |S|=51 to 100 --
# a 42x jump for a 2x size increase (empirical exponent ~5.5). The aggregated
# arms did NOT move (0.60 -> 0.50 s/epoch) because their LP is M x M regardless
# of |S|; that is the paper's whole point.
#
# CONSEQUENCE: at |S|=200 the full-state arms (fullal_r10lw100, qpal_r1) are
# not affordable at 8 seeds -- one qpal seed alone is ~150h. They are excluded
# from ARMS_BIG below by default. Raise BIG_MAX_SIZE only if you have the
# budget, and expect TIMEOUT_SEC to fire.
#
# ---------------------------------------------------------------------------
# HOW |S| GROWS (SCALE_MODE) -- this is a real experimental choice, not a knob
# ---------------------------------------------------------------------------
# |S| = b_max + s_max + 1, and only their SUM is pinned by |S|. Two modes:
#
#   smax  (default; reproduces the existing |S|=100 runs)
#         b_max fixed at 20, all growth goes into s_max.
#         |S|=100 -> b=20 s=79 ;  |S|=200 -> b=20 s=179
#         At |S|=51 the winning cell had b/|S| = 20/51 = 0.39; this mode drives
#         that to 0.20 at |S|=100 and 0.10 at |S|=200, i.e. it does not hold
#         the regime fixed -- it drifts into a storage-dominated problem whose
#         extra states are rarely visited. Rarely-visited states give the
#         occupancy LP a near-degenerate optimal basis, and the DFL gradient is
#         basis-sensitivity based (solvers/differentiable_lp.py falls back to
#         lstsq on a singular basis). Two-stage never differentiates the LP, so
#         this degradation is asymmetric: it costs DFL and not two-stage.
#
#   ratio (recommended when the question is "does the |S|=51 result hold?")
#         b_max/|S| held at the 20/51 = 0.392 of the winning cell.
#         |S|=100 -> b=39 s=60 ;  |S|=200 -> b=78 s=121
#         This scales the state space while keeping the backlog/storage balance
#         -- and hence the reachable-state fraction -- close to the cell the
#         regime was tuned in.
#
# NOTE the existing |S|=100 results on disk are all SCALE_MODE=smax. Runs from
# the two modes are NOT comparable and are tagged differently to keep them apart.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

# NB: /nethome/zzhao628/anaconda3 is a dangling symlink to an unmounted
# /scratch; the surviving interpreter is the warriors-ls-recovery path.
PY="${PY:-/net/csefiles/siemens/warriors-ls-recovery/users/zzhao628/#113442817/anaconda3/envs/rl/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-mdp_lp/results_dflwin}"
LOG_DIR="${LOG_DIR:-mdp_lp/logs/dflwin_scale}"
SIZES="${SIZES:-100 200}"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8}"
COSTS="${COSTS:-high_margin}"
SCALE_MODE="${SCALE_MODE:-smax}"      # smax | ratio
Q_MAX="${Q_MAX:-30}"                  # |A| = Q_MAX + 1, fixed across sizes

# The |S|=51 table, minus the two rows dropped from it (full_lw100 DFL-LP
# penalty, feascomp DFL-Feas +completion). Kept in sync with the arm_args
# block of run_dflwin_10seeds.sh -- edit BOTH if an arm changes.
ARMS="${ARMS:-twostage fullal_r10lw100 aggal_lw100 aggal_k20 aggal_k30 feasfaith sketchal_g02 qpal_r1}"

# Arms whose LP stays small regardless of |S| -> cheap, run wide.
ARMS_CHEAP_RE="${ARMS_CHEAP_RE:-^(twostage|aggal_|feasfaith)}"
# Full-state arms are dropped above this size (see cost table). Set to a large
# number to force them in.
BIG_MAX_SIZE="${BIG_MAX_SIZE:-100}"

JOBS_CHEAP="${JOBS_CHEAP:-4}"
JOBS_BIG="${JOBS_BIG:-2}"
TIMEOUT_SEC="${TIMEOUT_SEC:-86400}"   # 24h per run

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1

mkdir -p "$LOG_DIR"

# ---- regime: identical to run_dflwin_10seeds.sh except b_max/s_max ----------
# exp b, beta0=4 beta1=-0.3, p_logmean=10 p_std=0.5, sigma_eps=0.1,
# n_samples=20 (-> 80 generated, train_frac .25 = 20 train / 60 test),
# 100 epochs, pretrain 30, lr 1e-2.
# A plain string, not an array: bash cannot export arrays, and run_one executes
# inside an xargs subshell.
REGIME_COMMON="--experiment_type b --q_max $Q_MAX \
--beta_0 4.0 --beta_1 -0.3 \
--p_lognormal_mean 10.0 --p_lognormal_std 0.5 \
--sigma_epsilon 0.1 \
--n_samples 20 \
--n_epochs 100 \
--pretrain_epochs 30 \
--lr 0.01 \
--results_root $RESULTS_ROOT"

geometry() {   # $1 = |S| ; echoes "b_max s_max"
  local size="$1" b
  case "$SCALE_MODE" in
    smax)  b=20 ;;
    ratio) b=$(( (size * 20 + 25) / 51 )) ;;   # hold b/|S| = 20/51, rounded
    *)     echo "unknown SCALE_MODE $SCALE_MODE" >&2; return 1 ;;
  esac
  (( b < 1 )) && b=1
  local s=$(( size - b - 1 ))
  if (( s < 1 )); then echo "|S|=$size too small for b_max=$b" >&2; return 1; fi
  echo "$b $s"
}

arm_args() {   # per-arm flags; MUST match run_dflwin_10seeds.sh
  case "$1" in
    twostage)         echo "--algo twostage" ;;
    # DFL-LP, augmented Lagrangian (penalty + linear dual term)
    fullal_r10lw100)  echo "--algo dfl --loss_mode auglag --aug_lag_rho 10 --loss_weight 100" ;;
    # DFL-SA: soft state aggregation at M in {10,20,30} + augmented Lagrangian
    aggal_lw100)      echo "--algo dfl --loss_mode auglag --aug_lag_rho 10 --loss_weight 100 --n_agg 10 --sa_layer projmdp --sa_tau 0.5" ;;
    aggal_k20)        echo "--algo dfl --loss_mode auglag --aug_lag_rho 10 --loss_weight 100 --n_agg 20 --sa_layer projmdp --sa_tau 0.5" ;;
    aggal_k30)        echo "--algo dfl --loss_mode auglag --aug_lag_rho 10 --loss_weight 100 --n_agg 30 --sa_layer projmdp --sa_tau 0.5" ;;
    # DFL-Feas = Odece [34] adapted, faithful variant (alpha=0, mu=0)
    feasfaith)        echo "--algo dfl_feas --odece_alpha 0.0 --odece_mse_weight 0.0" ;;
    # DFL-Sketch: K=8 random row sketches of the flow constraints, ~20% of rows
    sketchal_g02)     echo "--algo dfl_sketch --n_sketches 8 --sketch_ratio 0.2 --sketch_mode gaussian --loss_mode auglag --aug_lag_rho 10 --loss_weight 100" ;;
    # DFL-QP: qpth, qp_reg selected as 1.0 by sweep at |S|=51
    qpal_r1)          echo "--algo dfl_qpth --qp_reg 1.0 --loss_mode auglag --aug_lag_rho 10 --loss_weight 100" ;;
    *) echo "" ;;
  esac
}

run_one() {
  # One packed "size arm cost seed" string: xargs -I{} hands the whole line as
  # a single argument.
  local size arm cost seed
  read -r size arm cost seed <<< "$1"
  local geo b s
  geo="$(geometry "$size")" || return 0
  read -r b s <<< "$geo"

  # SCALE_MODE=smax keeps the historical tag so existing |S|=100 results are
  # recognised and skipped; ratio gets its own namespace.
  local tag
  if [[ "$SCALE_MODE" == "smax" ]]; then
    tag="S${size}_${arm}_${cost}_seed${seed}"
  else
    tag="S${size}r_${arm}_${cost}_seed${seed}"
  fi

  local extra; extra="$(arm_args "$arm")"
  if [[ -z "$extra" ]]; then echo "[skip] unknown arm $arm"; return 0; fi
  if find "$RESULTS_ROOT" -path "*tag_${tag}/*results.json" -print -quit 2>/dev/null | grep -q .; then
    echo "[have] $tag"; return 0
  fi

  local log="$LOG_DIR/${tag}.log"
  echo "running" > "$LOG_DIR/${tag}.status"
  # gurobipy 12.0.2 SIGSEGVs tearing down large dense models on the default
  # 8MB stack; worker threads inherit this soft limit.
  ulimit -s 65536 2>/dev/null || true
  # shellcheck disable=SC2086
  timeout "$TIMEOUT_SEC" "$PY" -m mdp_lp.run_elasticity_experiments \
      $REGIME_COMMON --b_max "$b" --s_max "$s" \
      --cost_case "$cost" --seed "$seed" --tag "$tag" $extra > "$log" 2>&1
  local rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "done" > "$LOG_DIR/${tag}.status"
    echo "[OK] $tag $(grep -oE 'Mean regret: +[0-9.]+' "$log" | tail -1)"
  elif [ "$rc" -eq 124 ]; then
    echo "timeout" > "$LOG_DIR/${tag}.status"; echo "[TIMEOUT >${TIMEOUT_SEC}s] $tag"
  else
    echo "failed:$rc" > "$LOG_DIR/${tag}.status"; echo "[FAIL rc=$rc] $tag (see $log)"
  fi
}
export -f run_one geometry arm_args
export PY RESULTS_ROOT LOG_DIR SCALE_MODE Q_MAX TIMEOUT_SEC REGIME_COMMON

CHEAP_FILE="$(mktemp)"; BIG_FILE="$(mktemp)"
trap 'rm -f "$CHEAP_FILE" "$BIG_FILE"' EXIT

# Build the job lists (cost is the outer loop so a single-cost run is contiguous).
for cost in $COSTS; do
  for size in $SIZES; do
    for arm in $ARMS; do
      if [[ ! "$arm" =~ $ARMS_CHEAP_RE ]] && (( size > BIG_MAX_SIZE )); then
        echo "[excluded] $arm at |S|=$size (> BIG_MAX_SIZE=$BIG_MAX_SIZE; see cost table)"
        continue
      fi
      for seed in $SEEDS; do
        if [[ "$arm" =~ $ARMS_CHEAP_RE ]]; then
          echo "$size $arm $cost $seed" >> "$CHEAP_FILE"
        else
          echo "$size $arm $cost $seed" >> "$BIG_FILE"
        fi
      done
    done
  done
done

echo "[dflwin_scale] mode=$SCALE_MODE sizes='$SIZES' seeds='$SEEDS' costs='$COSTS'"
for size in $SIZES; do
  geo="$(geometry "$size")" && read -r b s <<< "$geo" \
    && echo "[dflwin_scale]   |S|=$size -> b_max=$b s_max=$s q_max=$Q_MAX (|A|=$((Q_MAX+1)))"
done
echo "[dflwin_scale] cheap jobs (P=$JOBS_CHEAP): $(wc -l < "$CHEAP_FILE" 2>/dev/null || echo 0)"
echo "[dflwin_scale] big   jobs (P=$JOBS_BIG): $(wc -l < "$BIG_FILE" 2>/dev/null || echo 0)"

PIDS=()
if [ -s "$BIG_FILE" ]; then
  xargs -a "$BIG_FILE" -d '\n' -I{} -P "$JOBS_BIG" \
    bash -c 'run_one "$0"' {} & PIDS+=($!)
fi
if [ -s "$CHEAP_FILE" ]; then
  xargs -a "$CHEAP_FILE" -d '\n' -I{} -P "$JOBS_CHEAP" \
    bash -c 'run_one "$0"' {} & PIDS+=($!)
fi
[ ${#PIDS[@]} -gt 0 ] && wait "${PIDS[@]}"
echo "[dflwin_scale] DONE. Results under $RESULTS_ROOT ; logs $LOG_DIR"
