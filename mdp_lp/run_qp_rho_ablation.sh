#!/usr/bin/env bash
#
# rho-sensitivity ablation for DFL-QP on the small inventory instance
# (|S| = 36, |A| = 31) -- the Table-1 / Fig-3 config, matched arm-for-arm to the
# existing DFL-LP rho sweep under mdp_lp/results_feas_compare/ (tags
# penalty_r{1,10,100,1000}_seed{1..5}).  rho = 1e4 already exists in the paper
# tree (mdp_lp/results/normal_margin_pretrain_30, 13.180 +- 1.101 on seeds 1-5),
# so only the four smaller values are run here.
#
# Everything except --aug_lag_rho, --tag and --results_root is byte-identical to
# the dfl_qpth arm of run_dfl_seeds.sh.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY='/net/csefiles/siemens/warriors-ls-recovery/users/zzhao628/#113442817/anaconda3/envs/rl/bin/python'

SEEDS="${SEEDS:-1 2 3 4 5}"
RHOS="${RHOS:-1.0 10.0 100.0 1000.0}"
RESULTS_ROOT="${RESULTS_ROOT:-mdp_lp/results_qp_rho}"
LOGDIR="${LOGDIR:-mdp_lp/results_qp_rho/logs}"
JOBS="${JOBS:-20}"
mkdir -p "$LOGDIR"

# keep each qpth run from grabbing the whole box (128 logical cores, shared)
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3

run() {  # args: rho seed
  local rho="$1" seed="$2"
  local tag="qpr${rho}_seed${seed}"
  local log="${LOGDIR}/${tag}.log"
  "$PY" -m mdp_lp.run_elasticity_experiments \
      --experiment_type b \
      --cost_case normal_margin \
      --bc 10 \
      --algo dfl_qpth \
      --n_epochs 50 \
      --n_samples 10 \
      --lr 0.01 \
      --n_agg -1 \
      --seed "$seed" \
      --tag "$tag" \
      --pretrain_epochs 30 \
      --b_max 5 --s_max 30 --q_max 30 \
      --results_root "$RESULTS_ROOT" \
      --qp_reg 10.0 \
      --aug_lag_rho "$rho" \
      > "$log" 2>&1
  echo "[qp_rho] done $tag"
}

gate() { while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n; done; }

for rho in $RHOS; do
  for seed in $SEEDS; do
    gate; run "$rho" "$seed" &
  done
done
wait

echo "[qp_rho] ALL DONE. Results under ${RESULTS_ROOT} ; logs in ${LOGDIR}"
