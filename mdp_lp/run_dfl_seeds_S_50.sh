#!/usr/bin/env bash
# 10-seed run of every arm in the decision-sensitive inventory regime (|S|=51).
#
# Regime = the "dflwin" cell (see run_dflwin_search.sh): b_max=20, s_max=30,
# beta0=4, beta1=-0.3, p_logmean=10, p_std=0.5, sigma_eps=0.1, n_samples=20
# (-> 80 generated, 20 train / 60 test), 100 epochs, lr 1e-2, pretrain 30.
# Unlike the canonical Fig.3 config this cell is DECISION-dominated: prediction
# accuracy and decision quality decouple, so DFL can separate from two-stage.
#
# Idempotent: a run whose results.json already exists is skipped, so this can be
# used to top up the arms that currently stop at 8 seeds.
#
#   bash mdp_lp/run_dflwin_10seeds.sh                    # high_margin, seeds 1-10
#   COSTS=low_margin bash mdp_lp/run_dflwin_10seeds.sh   # the other cost cell
#   SEEDS="9 10" ARMS="feascomp feasfaith aggal" bash mdp_lp/run_dflwin_10seeds.sh
#   JOBS=6 bash mdp_lp/run_dflwin_10seeds.sh             # more concurrency
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

PY="${PY:-/nethome/zzhao628/anaconda3/envs/rl/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-mdp_lp/results_dflwin}"
LOG_DIR="${LOG_DIR:-mdp_lp/logs/dflwin_10seeds}"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
COSTS="${COSTS:-high_margin}"
# Default table arms. Dropped from the table (definitions kept below so the
# existing results.json files stay reproducible): full_lw100 (DFL-LP penalty)
# and feascomp (DFL-Feas +completion) -- DFL-Feas is the alpha=0, mu=0 variant.
ARMS="${ARMS:-twostage fullal_r10lw100 aggal_lw100 aggal_k20 aggal_k30 feasfaith sketchal_g02 qpal_r1}"
JOBS="${JOBS:-3}"          # concurrent runs; each is capped at 2 BLAS threads below
SIZE="${SIZE:-51}"         # |S| = B_MAX + s_max + 1 ; B_MAX fixed at 20
B_MAX=20
S_MAX=$(( SIZE - B_MAX - 1 ))
if (( S_MAX < 1 )); then echo "SIZE must exceed $((B_MAX+1))"; exit 1; fi

# Keep a side fleet from starving other jobs on the shared box (an unthrottled
# torch run grabs ~34 cores). Raise only if the machine is idle.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
ulimit -s 65536 2>/dev/null || true     # gurobipy large-model stack segfault

mkdir -p "$LOG_DIR"

# ---- regime: identical for every arm (only the loss differs) -----------------
REGIME=(
  --experiment_type b
  --b_max "$B_MAX" --s_max "$S_MAX" --q_max 30
  --beta_0 4.0 --beta_1 -0.3
  --p_lognormal_mean 10.0 --p_lognormal_std 0.5
  --sigma_epsilon 0.1
  --n_samples 20                 # -> 80 generated, train_frac .25 = 20 train / 60 test
  --n_epochs 100
  --pretrain_epochs 30
  --lr 0.01
  --results_root "$RESULTS_ROOT"
)
AGG=(--n_agg 10 --sa_layer projmdp --sa_tau 0.5)

arm_args() {   # echo the per-arm flags
  case "$1" in
    twostage)         echo "--algo twostage" ;;
    # DFL-LP, penalty surrogate (paper Eq. 5)
    full_lw100)       echo "--algo dfl --loss_mode penalty --aug_lag_rho 10 --loss_weight 100" ;;
    # DFL-LP, augmented Lagrangian (penalty + linear dual term)
    fullal_r10lw100)  echo "--algo dfl --loss_mode auglag  --aug_lag_rho 10 --loss_weight 100" ;;
    # Ours / DFL-SA: soft state aggregation + augmented Lagrangian, at M=10
    # (headline), 20 and 30. Only the number of aggregated states differs;
    # sa_layer=projmdp and sa_tau=0.5 are shared via $AGG.
    aggal_lw100)      echo "--algo dfl --loss_mode auglag  --aug_lag_rho 10 --loss_weight 100 ${AGG[*]}" ;;
    aggal_k20)        echo "--algo dfl --loss_mode auglag  --aug_lag_rho 10 --loss_weight 100 --n_agg 20 --sa_layer projmdp --sa_tau 0.5" ;;
    aggal_k30)        echo "--algo dfl --loss_mode auglag  --aug_lag_rho 10 --loss_weight 100 --n_agg 30 --sa_layer projmdp --sa_tau 0.5" ;;
    # DFL-QP: qpth on the occupancy LP with quadratic regularizer qp_reg.
    # SELECTED SETTING = qpal_r1 (qp_reg 1.0, auglag). qp_reg had no tuning
    # history (a single seed-1 run at the 10.0 default), so it was swept over
    # {0.01, 0.1, 1.0, 10.0}; seed-1 regret was 127.1 / 50.5 / 16.3 / 24.3 ->
    # interior optimum at 1.0. NOTE the 8x sensitivity: unlike the other arms,
    # this row is the best of a sweep, so report the qp_reg alongside it.
    # The r001/r01/r10 arms are kept only to reproduce that sweep.
    qpal_r1)          echo "--algo dfl_qpth --qp_reg 1.0  --loss_mode auglag --aug_lag_rho 10 --loss_weight 100" ;;
    qpal_r001)        echo "--algo dfl_qpth --qp_reg 0.01 --loss_mode auglag --aug_lag_rho 10 --loss_weight 100" ;;
    qpal_r01)         echo "--algo dfl_qpth --qp_reg 0.1  --loss_mode auglag --aug_lag_rho 10 --loss_weight 100" ;;
    qpal_r10)         echo "--algo dfl_qpth --qp_reg 10.0 --loss_mode auglag --aug_lag_rho 10 --loss_weight 100" ;;
    # DFL-Sketch: DFL-LP with random row sketching of the flow constraints,
    # K=8 independent sketches averaged, each keeping ~20% of the rows.
    # No aggregation (n_agg=-1), so it pairs with fullal_r10lw100.
    sketchal_g02)     echo "--algo dfl_sketch --n_sketches 8 --sketch_ratio 0.2 --sketch_mode gaussian --loss_mode auglag --aug_lag_rho 10 --loss_weight 100" ;;
    # DFL-Feas = Odece [34] adapted; +completion grants it reward supervision
    feascomp)         echo "--algo dfl_feas --odece_alpha 0.5 --odece_mse_weight 1.0" ;;
    feasfaith)        echo "--algo dfl_feas --odece_alpha 0.0 --odece_mse_weight 0.0" ;;
    *) echo "" ;;
  esac
}

gate() { while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n; done; }

run_one() {
  local arm="$1" cost="$2" seed="$3"
  local tag="S${SIZE}_${arm}_${cost}_seed${seed}"
  local extra; extra="$(arm_args "$arm")"
  if [[ -z "$extra" ]]; then echo "[skip] unknown arm $arm"; return 0; fi
  if find "$RESULTS_ROOT" -path "*tag_${tag}/*results.json" -print -quit 2>/dev/null | grep -q .; then
    echo "[have] $tag"; return 0
  fi
  local log="$LOG_DIR/${tag}.log" st=OK
  # shellcheck disable=SC2086
  "$PY" -m mdp_lp.run_elasticity_experiments "${REGIME[@]}" \
      --cost_case "$cost" --seed "$seed" --tag "$tag" $extra > "$log" 2>&1 || st=FAIL
  echo "[$st] $tag $(grep -oE 'Mean regret: +[0-9.]+' "$log" | tail -1)"
}

for cost in $COSTS; do
  for seed in $SEEDS; do
    for arm in $ARMS; do
      gate; run_one "$arm" "$cost" "$seed" &
    done
  done
done
wait
echo "[dflwin_10seeds] DONE. Results under $RESULTS_ROOT ; logs $LOG_DIR"
echo "Summarize with:  python -m mdp_lp.summarize_dflwin  (or the ad-hoc parser on tag S51_<arm>_<cost>_seed<N>)"
