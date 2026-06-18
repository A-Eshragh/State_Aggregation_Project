#!/usr/bin/env bash
# Exp B: manual p = [1, 2, ..., n_samples]. No backlog (b_max=0). No prediction loss (loss_weight=0 for DFL).
# p: sample i gets price p_i = i+1 (so 1,2,...,20 for n_samples=20). No two close p.

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RESULTS_ROOT="sanity_fixed_p"
EPOCHS=50
N_SAMPLES=20
S_MAX=30
B_MAX=0
BETA0=4.0
BETA1=-0.3
SIGMA_EPS=0.1

echo "[run_exp_b_fixed_p] results_root=$RESULTS_ROOT | p = [1..$N_SAMPLES] | b_max=$B_MAX (no backlog) | no loss_weight"

for COST in high_margin low_margin; do
  echo "[run_exp_b_fixed_p] Exp B $COST DFL..."
  python -m mdp_lp.run_elasticity_experiments \
    --experiment_type b --cost_case "$COST" --algo dfl \
    --b_max "$B_MAX" --s_max "$S_MAX" --beta_0 "$BETA0" --beta_1 "$BETA1" \
    --sigma_epsilon "$SIGMA_EPS" --fixed_p --loss_weight 0 \
    --n_epochs "$EPOCHS" --n_samples "$N_SAMPLES" \
    --results_root "$RESULTS_ROOT"

  echo "[run_exp_b_fixed_p] Exp B $COST twostage..."
  python -m mdp_lp.run_elasticity_experiments \
    --experiment_type b --cost_case "$COST" --algo twostage \
    --b_max "$B_MAX" --s_max "$S_MAX" --beta_0 "$BETA0" --beta_1 "$BETA1" \
    --sigma_epsilon "$SIGMA_EPS" --fixed_p \
    --n_epochs "$EPOCHS" --n_samples "$N_SAMPLES" \
    --results_root "$RESULTS_ROOT"
done

echo "[run_exp_b_fixed_p] Done. Results under $RESULTS_ROOT/exp_b/"
