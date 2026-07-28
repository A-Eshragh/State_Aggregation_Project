#!/usr/bin/env bash
# CartPole auglag hyperparameter tuning — SMALL seed count on purpose.
#
# Goal: find a setting where the aug-Lagrangian loss at least matches DFL-LP
# (Eq.5 penalty). Measured pathology at the default (dual_weight=1, rho=10):
#   |linear| ~ 2-20  vs  |c^T w*| ~ 5  and  rho*||viol||^2 ~ 0.003,
#   with 72% of true-LP solves hitting Gurobi's slack fallback (multipliers
#   bounded only by the 1e3 slack penalty).
# So the sweep damps the linear term (dual_weight), restricts it to genuine LP
# duals (dual_feasible_only), or raises the quadratic term to match (rho).
#
# NOTE dual_weight=0 is NOT swept: it reduces to the penalty loss identically,
# so it would "match" trivially and prove nothing.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SEEDS="${SEEDS:-1 2 3 4 5}"
WORKERS="${WORKERS:-15}"
OUT="${OUT:-mdp_continuous/results/rebuttal_feas_auglag}"
# $HOME/anaconda3 -> /scratch/zzhao628/anaconda3, which no longer exists; the
# rl env currently lives only on the recovery mount. Prefer whichever resolves.
RECOVERY_PY="/net/csefiles/siemens/warriors-ls-recovery/users/zzhao628/#113442817/anaconda3/envs/rl/bin/python"
if [ -x "$HOME/anaconda3/envs/rl/bin/python" ]; then
  PY="${PY:-$HOME/anaconda3/envs/rl/bin/python}"
else
  PY="${PY:-$RECOVERY_PY}"
fi
[ -x "$PY" ] || { echo "[fatal] no usable python at: $PY" >&2; exit 1; }

ulimit -s 65536
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p "$OUT"

run () {
  echo
  echo "==================== auglag $* ===================="
  "$PY" -m mdp_continuous.run_dfl_vs_qpth \
      --solvers lp --loss_mode auglag \
      --seeds $SEEDS --n_workers "$WORKERS" --results_dir "$OUT" "$@"
}

run --dual_weight 0.001
run --dual_weight 0.01
run --dual_weight 0.1
run --dual_feasible_only
run --rho 1000
run --rho 100000

echo
echo "[done] tuning results in $OUT"
