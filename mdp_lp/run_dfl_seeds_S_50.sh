#!/usr/bin/env bash
# Inventory problem, |S| = 51, |A| = 31. Seeds 1-10, all arms.
# Finished runs are skipped. Results go to $RESULTS_ROOT.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

PY="${PY:-/nethome/zzhao628/anaconda3/envs/rl/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-mdp_lp/results_dflwin}"
LOG_DIR="${LOG_DIR:-mdp_lp/logs/dflwin_10seeds}"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
COSTS="${COSTS:-high_margin}"
ARMS="${ARMS:-twostage fullal_r10lw100 aggal_lw100 aggal_k20 aggal_k30 feasfaith sketchal_g02 qpal_r1}"
JOBS="${JOBS:-3}"          # concurrent runs; each is capped at 2 BLAS threads below
SIZE="${SIZE:-51}"         # |S| = B_MAX + s_max + 1 ; B_MAX fixed at 20
B_MAX=20
S_MAX=$(( SIZE - B_MAX - 1 ))
if (( S_MAX < 1 )); then echo "SIZE must exceed $((B_MAX+1))"; exit 1; fi

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
    # DFL-LP, augmented Lagrangian (penalty + linear dual term)
    fullal_r10lw100)  echo "--algo dfl --loss_mode auglag  --aug_lag_rho 10 --loss_weight 100" ;;
    # Ours / DFL-SA at M = 10, 20, 30. --agg_pretrain_epochs differs per arm
    aggal_lw100)      echo "--algo dfl --loss_mode auglag  --aug_lag_rho 10 --loss_weight 100 ${AGG[*]} --agg_pretrain_epochs 10" ;;
    aggal_k20)        echo "--algo dfl --loss_mode auglag  --aug_lag_rho 10 --loss_weight 100 --n_agg 20 --sa_layer projmdp --sa_tau 0.5" ;;
    aggal_k30)        echo "--algo dfl --loss_mode auglag  --aug_lag_rho 10 --loss_weight 100 --n_agg 30 --sa_layer projmdp --sa_tau 0.5 --agg_pretrain_epochs 20" ;;
    # DFL-QP: qpth on the occupancy LP, quadratic regularizer qp_reg = 1.0.
    qpal_r1)          echo "--algo dfl_qpth --qp_reg 1.0  --loss_mode auglag --aug_lag_rho 10 --loss_weight 100" ;;
    # DFL-Sketch: DFL-LP with random row sketching of the flow constraints.
    sketchal_g02)     echo "--algo dfl_sketch --n_sketches 8 --sketch_ratio 0.2 --sketch_mode gaussian --loss_mode auglag --aug_lag_rho 10 --loss_weight 100" ;;
    # DFL-Feas = Odece [34] adapted, faithful alpha=0 mu=0
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
echo "Read test_mean_regret from results.json (see Readme.md)."
