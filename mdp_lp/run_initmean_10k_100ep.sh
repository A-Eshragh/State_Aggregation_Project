#!/usr/bin/env bash
# Exact params from initmean_10k_attempt2; 100 epochs; high_margin + low_margin; DFL + twostage.

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RESULTS_ROOT="sanity_initmean_10k_100ep"
EPOCHS=100
N_SAMPLES=20
S_MAX=30
B_MAX=5
Q_MAX=50
D_MAX=50
BETA0=4.0
BETA1=-0.3
P_LOGMEAN=10.0
P_LOGSTD=1.0
SIGMA_EPS=0.1
TRAIN_FRAC=0.5
LOSS_WEIGHT=0.0
AUG_RHO=10000.0
LR=0.001
HIDDEN_DIM=32
BETA_DISCOUNT=0.95
GRAD_CLIP=10.0
FOC=0
VOC=0.5

echo "[run_initmean_10k_100ep] results_root=$RESULTS_ROOT | epochs=$EPOCHS | aug_lag_rho=$AUG_RHO | loss_weight=$LOSS_WEIGHT"

for COST in high_margin low_margin; do
  echo "[run_initmean_10k_100ep] Exp B $COST DFL..."
  python -m mdp_lp.run_elasticity_experiments \
    --experiment_type b --cost_case "$COST" --algo dfl \
    --b_max "$B_MAX" --s_max "$S_MAX" --q_max "$Q_MAX" --d_max "$D_MAX" \
    --beta_0 "$BETA0" --beta_1 "$BETA1" \
    --p_lognormal_mean "$P_LOGMEAN" --p_lognormal_std "$P_LOGSTD" \
    --sigma_epsilon "$SIGMA_EPS" \
    --foc "$FOC" --voc "$VOC" \
    --n_epochs "$EPOCHS" --n_samples "$N_SAMPLES" --train_frac "$TRAIN_FRAC" \
    --loss_weight "$LOSS_WEIGHT" --aug_lag_rho "$AUG_RHO" \
    --hyperparam '{"lr": 0.001, "hidden_dim": 32}' \
    --beta "$BETA_DISCOUNT" --grad_clip "$GRAD_CLIP" \
    --results_root "$RESULTS_ROOT"

  echo "[run_initmean_10k_100ep] Exp B $COST twostage..."
  python -m mdp_lp.run_elasticity_experiments \
    --experiment_type b --cost_case "$COST" --algo twostage \
    --b_max "$B_MAX" --s_max "$S_MAX" --q_max "$Q_MAX" --d_max "$D_MAX" \
    --beta_0 "$BETA0" --beta_1 "$BETA1" \
    --p_lognormal_mean "$P_LOGMEAN" --p_lognormal_std "$P_LOGSTD" \
    --sigma_epsilon "$SIGMA_EPS" \
    --foc "$FOC" --voc "$VOC" \
    --n_epochs "$EPOCHS" --n_samples "$N_SAMPLES" --train_frac "$TRAIN_FRAC" \
    --aug_lag_rho "$AUG_RHO" \
    --hyperparam '{"lr": 0.001, "hidden_dim": 32}' \
    --beta "$BETA_DISCOUNT" --grad_clip "$GRAD_CLIP" \
    --results_root "$RESULTS_ROOT"
done

echo "[run_initmean_10k_100ep] Done. Results under $RESULTS_ROOT/exp_b/"
