#!/usr/bin/env bash
# Exp B, no backlog and no fixed ordering cost. High margin hc=1, low margin hc=50.
# beta_0=4, beta_1=-0.3, s_max=50. DFL uses fixed loss_weight=20 (balanced with v* term, no adaptation). 100 epochs.

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
RESULTS_ROOT="mdp_lp/exp_b_clean"
EPOCHS=100
N_SAMPLES=100
BETA0=4.0
BETA1=-0.3
S_MAX=50

echo "[run_exp_b_clean] results_root=$RESULTS_ROOT | no backlog (b_max=0), no foc (foc=0) | high hc=1, low hc=50"

# High margin: b_max=0, foc=0, hc=1, loss_weight=20
echo "[run_exp_b_clean] Exp B high_margin DFL (lw=20)..."
python -m mdp_lp.run_elasticity_experiments \
  --experiment_type b --cost_case high_margin --algo dfl \
  --foc 0 --b_max 0 --hc 1 --loss_weight 20 --s_max "$S_MAX" --beta_0 "$BETA0" --beta_1 "$BETA1" \
  --n_epochs "$EPOCHS" --n_samples "$N_SAMPLES" \
  --results_root "$RESULTS_ROOT"

echo "[run_exp_b_clean] Exp B high_margin twostage..."
python -m mdp_lp.run_elasticity_experiments \
  --experiment_type b --cost_case high_margin --algo twostage \
  --foc 0 --b_max 0 --hc 1 --s_max "$S_MAX" --beta_0 "$BETA0" --beta_1 "$BETA1" \
  --n_epochs "$EPOCHS" --n_samples "$N_SAMPLES" \
  --results_root "$RESULTS_ROOT"

# Low margin: b_max=0, foc=0, hc=50, loss_weight=20
echo "[run_exp_b_clean] Exp B low_margin DFL (lw=20)..."
python -m mdp_lp.run_elasticity_experiments \
  --experiment_type b --cost_case low_margin --algo dfl \
  --foc 0 --b_max 0 --hc 50 --loss_weight 20 --s_max "$S_MAX" --beta_0 "$BETA0" --beta_1 "$BETA1" \
  --n_epochs "$EPOCHS" --n_samples "$N_SAMPLES" \
  --results_root "$RESULTS_ROOT"

echo "[run_exp_b_clean] Exp B low_margin twostage..."
python -m mdp_lp.run_elasticity_experiments \
  --experiment_type b --cost_case low_margin --algo twostage \
  --foc 0 --b_max 0 --hc 50 --s_max "$S_MAX" --beta_0 "$BETA0" --beta_1 "$BETA1" \
  --n_epochs "$EPOCHS" --n_samples "$N_SAMPLES" \
  --results_root "$RESULTS_ROOT"

echo "[run_exp_b_clean] Done. Results under $RESULTS_ROOT/exp_b/"
echo "Plots and gradient report: python -m mdp_lp.generate_results exp_b --base_dir $RESULTS_ROOT --epochs $EPOCHS"
