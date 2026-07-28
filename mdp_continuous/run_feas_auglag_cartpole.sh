#!/usr/bin/env bash
# CartPole (Section 6.3, paper setting) — rebuttal arms:
#   DFL-Feas / Odece [34] and the proposed aug-Lagrangian DFL loss.
#
# Everything except the Phase-B loss is held fixed at the paper's defaults
# (iters=60, nfeat=1024, behavior=biased, eps=0.5, td=0.013, eval=lp, rho=10,
# n_ce_steps=0, n_dfl_steps=5, pretrain=30), so iteration 0 is identical across
# arms and the seeds are paired.
#
# Results go to results/rebuttal_feas_auglag/ so the paper's cached JSONs under
# results/dfl_vs_qpth/ are left untouched.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20}"
WORKERS="${WORKERS:-20}"
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

# Gurobi segfaults on deep stacks for large LPs; see project memory.
ulimit -s 65536
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

mkdir -p "$OUT"

run () {
  echo
  echo "==================== $* ===================="
  "$PY" -m mdp_continuous.run_dfl_vs_qpth \
      --seeds $SEEDS --n_workers "$WORKERS" --results_dir "$OUT" "$@"
}

# 1. DFL-LP with the paper's Eq.(5) penalty loss — paired baseline. Re-run
#    rather than reused: the cached JSON predates the arm's switch from a dense
#    LP to sketch-and-solve, so only its iteration 0 still matches.
run --solvers lp

# 2. Proposed unbiased aug-Lagrangian loss (penalty + linear dual term).
run --solvers lp --loss_mode auglag

# 3. Lagrangian-only (linear term, no quadratic penalty) — isolates the term.
run --solvers lp --loss_mode lagrangian

# 4. DFL-Feas / Odece [34], faithful: y^- solves the predicted problem exactly.
run --solvers feas --odece_neg_solver plain

# 5. DFL-Feas, charitable variant: y^- reuses this arm's sketched LP layer.
run --solvers feas --odece_neg_solver arm

# 6. Two-stage reference (no DFL phase).
run --solvers two_stage

echo
echo "[done] results in $OUT"
