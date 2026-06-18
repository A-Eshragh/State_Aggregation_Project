#!/usr/bin/env bash
# Exp B with wider price spread: p_lognormal_std=2 (vs 0.5 in ps05 runs).
# Same sanity-style settings: n_samples=20, s_max=30, sigma_epsilon=0.1, etc.
# Use this to check if more p variance helps DFL learn beyond the mean.

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RESULTS_ROOT="sanity_20_30_se01_ps2"
EPOCHS=50
N_SAMPLES=20
S_MAX=30
B_MAX=20
BETA0=4.0
BETA1=-0.3
P_LOGMEAN=10.0
P_LOGSTD=2.0
SIGMA_EPS=0.1

echo "[run_exp_b_ps2] results_root=$RESULTS_ROOT | p_lognormal_std=$P_LOGSTD | n_samples=$N_SAMPLES s_max=$S_MAX"

for COST in high_margin low_margin; do
  echo "[run_exp_b_ps2] Exp B $COST DFL..."
  python -m mdp_lp.run_elasticity_experiments \
    --experiment_type b --cost_case "$COST" --algo dfl \
    --b_max "$B_MAX" --s_max "$S_MAX" --beta_0 "$BETA0" --beta_1 "$BETA1" \
    --p_lognormal_mean "$P_LOGMEAN" --p_lognormal_std "$P_LOGSTD" \
    --sigma_epsilon "$SIGMA_EPS" \
    --n_epochs "$EPOCHS" --n_samples "$N_SAMPLES" \
    --results_root "$RESULTS_ROOT"

  echo "[run_exp_b_ps2] Exp B $COST twostage..."
  python -m mdp_lp.run_elasticity_experiments \
    --experiment_type b --cost_case "$COST" --algo twostage \
    --b_max "$B_MAX" --s_max "$S_MAX" --beta_0 "$BETA0" --beta_1 "$BETA1" \
    --p_lognormal_mean "$P_LOGMEAN" --p_lognormal_std "$P_LOGSTD" \
    --sigma_epsilon "$SIGMA_EPS" \
    --n_epochs "$EPOCHS" --n_samples "$N_SAMPLES" \
    --results_root "$RESULTS_ROOT"
done

echo "[run_exp_b_ps2] Done. Results under $RESULTS_ROOT/exp_b/"
