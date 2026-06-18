#!/usr/bin/env bash
# Exp B main run for paper: 100 samples, s_max=50, sigma_epsilon=2, p_lognormal_std=1.
# DFL (loss_weight=30, fixed) vs Two-stage for all 3 cost cases.

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RESULTS_ROOT="fixed_lag_100_se2_lw30"
EPOCHS=100
N_SAMPLES=100
S_MAX=50
BETA0=4.0
BETA1=-0.3
SIGMA_EPS=2.0
P_STD=1.0

echo "[exp_b_100_50_se2_lw30] n_samples=$N_SAMPLES s_max=$S_MAX epochs=$EPOCHS sigma_epsilon=$SIGMA_EPS p_lognormal_std=$P_STD results_root=$RESULTS_ROOT"

for cost in high_margin normal_margin low_margin; do
  echo "[exp_b_100_50_se2_lw30] --- $cost ---"

  echo "[exp_b_100_50_se2_lw30] DFL (loss_weight=30)..."
  python -m mdp_lp.run_elasticity_experiments \
    --experiment_type b --cost_case "$cost" --algo dfl \
    --loss_weight 30 --s_max "$S_MAX" --beta_0 "$BETA0" --beta_1 "$BETA1" \
    --sigma_epsilon "$SIGMA_EPS" --p_lognormal_std "$P_STD" \
    --n_epochs "$EPOCHS" --n_samples "$N_SAMPLES" \
    --results_root "$RESULTS_ROOT"

  echo "[exp_b_100_50_se2_lw30] Two-stage..."
  python -m mdp_lp.run_elasticity_experiments \
    --experiment_type b --cost_case "$cost" --algo twostage \
    --s_max "$S_MAX" --beta_0 "$BETA0" --beta_1 "$BETA1" \
    --sigma_epsilon "$SIGMA_EPS" --p_lognormal_std "$P_STD" \
    --n_epochs "$EPOCHS" --n_samples "$N_SAMPLES" \
    --results_root "$RESULTS_ROOT"
done

echo "[exp_b_100_50_se2_lw30] Done. Results under $RESULTS_ROOT/exp_b/"

