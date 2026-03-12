#!/usr/bin/env bash
# Sanity (log-MSE DFL): 20 samples, s_max=30, sigma_epsilon=2, p_lognormal_std=1.
# Exp B, high/normal/low margin, DFL (lw=30, log-MSE) and twostage, 100 epochs.

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RESULTS_ROOT="sanity_20_30_se2_lw30_100_logmse_v2"
EPOCHS=100
N_SAMPLES=20
S_MAX=30
BETA0=4.0
BETA1=-0.3
SIGMA_EPS=2.0
P_STD=1.0

echo "[sanity_20_30_se2_lw30_100_logmse] n_samples=$N_SAMPLES s_max=$S_MAX epochs=$EPOCHS sigma_epsilon=$SIGMA_EPS p_lognormal_std=$P_STD results_root=$RESULTS_ROOT"

for cost in high_margin normal_margin low_margin; do
  echo "[sanity_20_30_se2_lw30_100_logmse] --- $cost ---"

  echo "[sanity_20_30_se2_lw30_100_logmse] DFL (loss_weight=30, log-MSE)..."
  python -m mdp_lp.run_elasticity_experiments \
    --experiment_type b --cost_case "$cost" --algo dfl \
    --loss_weight 30 --s_max "$S_MAX" --beta_0 "$BETA0" --beta_1 "$BETA1" \
    --sigma_epsilon "$SIGMA_EPS" --p_lognormal_std "$P_STD" \
    --n_epochs "$EPOCHS" --n_samples "$N_SAMPLES" \
    --results_root "$RESULTS_ROOT"

  echo "[sanity_20_30_se2_lw30_100_logmse] twostage..."
  python -m mdp_lp.run_elasticity_experiments \
    --experiment_type b --cost_case "$cost" --algo twostage \
    --s_max "$S_MAX" --beta_0 "$BETA0" --beta_1 "$BETA1" \
    --sigma_epsilon "$SIGMA_EPS" --p_lognormal_std "$P_STD" \
    --n_epochs "$EPOCHS" --n_samples "$N_SAMPLES" \
    --results_root "$RESULTS_ROOT"
done

echo "[sanity_20_30_se2_lw30_100_logmse] Done. Results under $RESULTS_ROOT/exp_b/"

