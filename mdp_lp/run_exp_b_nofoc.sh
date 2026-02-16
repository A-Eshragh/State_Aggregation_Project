#!/usr/bin/env bash
# Exp B rerun: save to separate dir; no fixed cost (foc=0); high margin = no backlog (b_max=0), low margin = keep backlog (b_max=50).
# Match original exp_b: beta_0=4, beta_1=-0.3, s_max=50, n_samples=100. DFL + twostage each cost. 100 epochs.

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
RESULTS_ROOT="mdp_lp/exp_b_nofoc"
EPOCHS=100
N_SAMPLES=100
# Same elasticity as original exp_b (otherwise demand is tiny and reward goes negative)
BETA0=4.0
BETA1=-0.3
S_MAX=50

echo "[run_exp_b_nofoc] results_root=$RESULTS_ROOT epochs=$EPOCHS beta_0=$BETA0 beta_1=$BETA1 s_max=$S_MAX"

# High margin: no backlog (b_max=0), no fixed cost (foc=0), s_max=50 to match original
echo "[run_exp_b_nofoc] Exp B high_margin DFL (foc=0 b_max=0)..."
python -m mdp_lp.run_elasticity_experiments \
  --experiment_type b --cost_case high_margin --algo dfl \
  --foc 0 --b_max 0 --s_max "$S_MAX" --beta_0 "$BETA0" --beta_1 "$BETA1" \
  --n_epochs "$EPOCHS" --n_samples "$N_SAMPLES" \
  --results_root "$RESULTS_ROOT"

echo "[run_exp_b_nofoc] Exp B high_margin twostage (foc=0 b_max=0)..."
python -m mdp_lp.run_elasticity_experiments \
  --experiment_type b --cost_case high_margin --algo twostage \
  --foc 0 --b_max 0 --s_max "$S_MAX" --beta_0 "$BETA0" --beta_1 "$BETA1" \
  --n_epochs "$EPOCHS" --n_samples "$N_SAMPLES" \
  --results_root "$RESULTS_ROOT"

# Low margin: keep backlog (b_max=50 default), no fixed cost (foc=0), same beta/s_max
echo "[run_exp_b_nofoc] Exp B low_margin DFL (foc=0, backlog kept)..."
python -m mdp_lp.run_elasticity_experiments \
  --experiment_type b --cost_case low_margin --algo dfl \
  --foc 0 --s_max "$S_MAX" --beta_0 "$BETA0" --beta_1 "$BETA1" \
  --n_epochs "$EPOCHS" --n_samples "$N_SAMPLES" \
  --results_root "$RESULTS_ROOT"

echo "[run_exp_b_nofoc] Exp B low_margin twostage (foc=0, backlog kept)..."
python -m mdp_lp.run_elasticity_experiments \
  --experiment_type b --cost_case low_margin --algo twostage \
  --foc 0 --s_max "$S_MAX" --beta_0 "$BETA0" --beta_1 "$BETA1" \
  --n_epochs "$EPOCHS" --n_samples "$N_SAMPLES" \
  --results_root "$RESULTS_ROOT"

echo "[run_exp_b_nofoc] Done. Results under $RESULTS_ROOT/exp_b/"
echo "Generate side-by-side plots and gradient report:"
echo "  python -m mdp_lp.generate_results exp_b --base_dir $RESULTS_ROOT --epochs $EPOCHS"
