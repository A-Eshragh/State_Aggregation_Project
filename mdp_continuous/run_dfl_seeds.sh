#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# $HOME/anaconda3 -> /scratch/zzhao628/anaconda3, which no longer exists; the
# rl env currently lives only on the recovery mount. Prefer whichever resolves.
RECOVERY_PY="/net/csefiles/siemens/warriors-ls-recovery/users/zzhao628/#113442817/anaconda3/envs/rl/bin/python"
if [ -x "$HOME/anaconda3/envs/rl/bin/python" ]; then
  PY="${PY:-$HOME/anaconda3/envs/rl/bin/python}"
else
  PY="${PY:-$RECOVERY_PY}"
fi
[ -x "$PY" ] || { echo "[fatal] no usable python at: $PY" >&2; exit 1; }

# Gurobi segfaults on deep stacks for large LPs.
ulimit -s 65536
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

# SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20}"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
WORKERS="${WORKERS:-10}"

# ── DFL loss variant for the DFL arms ────────────────────────────────────────
# The DFL arms now train with the augmented-Lagrangian loss instead of the
# Eq.(5) penalty loss:
#     L = c^T w*  +  rho||b - M w*||^2  +  dual_weight * nu^T(b - M w*)
#
# HONEST STATUS OF THIS DEFAULT — read before trusting the numbers.
#   * The EXACT aug-Lagrangian (dual_weight=1, rho=10) is WORSE than the Eq.(5)
#     penalty loss on DFL-LP over 20 paired seeds:
#         Delta best = -11.24 +/- 6.42 SE,  Delta final = -53.49 +/- 16.54 SE
#         (6/20 seeds win).
#     Cause (measured): the linear term runs +1.9..+20.2 against |c^T w*| ~ 5
#     and rho||viol||^2 ~ 0.003, because 72% of true-LP solves hit Gurobi's
#     slack-penalised fallback — the polytope {w : Mw=b, w>=0} is empty at
#     pretrained features — and those multipliers are bounded only by the 1e3
#     slack penalty.
#   * --dual_feasible_only applies the linear term ONLY on steps where the true
#     LP was genuinely feasible, so the multiplier is a real LP dual. On a
#     5-seed sweep it had the best 'best' reward and the lowest variance
#     (191.34 +/- 4.38, vs 187.71 +/- 6.98 for the penalty loss).
#   * BUT that 5-seed sweep CANNOT rank these configs: the exact-AL Delta best
#     was +4.20 on those 5 seeds and -11.24 on 20 seeds — the sign flips. Every
#     5-seed contrast is within ~1 SE of zero. And --dual_feasible_only skips
#     the linear term on ~93% of steps, so it sits close to the penalty loss by
#     construction.
#   * rho is NOT the fix: rho=1000 was the worst config tested
#     (Delta final -70.92 +/- 23.96).
# So this default is the best-supported option among those tested, NOT an
# established win. Confirm on 20 seeds before putting it in the paper.
# Set AUGLAG_FLAGS="" to fall back to the original Eq.(5) penalty loss.
AUGLAG_FLAGS="${AUGLAG_FLAGS:---loss_mode auglag --dual_feasible_only}"

# Mirror of AUGLAG_FLAGS for make_figures.py, so step [5/5] reads the caches
# these runs write rather than the old penalty-loss ones.
if [ -n "$AUGLAG_FLAGS" ]; then
  export CARTPOLE_LOSS_MODE=auglag
  export CARTPOLE_DUAL_FEASIBLE_ONLY=1
fi

echo "[reproduce] seeds=$SEEDS"
echo "[reproduce] workers=$WORKERS"
echo "[reproduce] DFL loss: ${AUGLAG_FLAGS:-penalty (Eq.5)}"
echo "[reproduce] DFL-SA configs: d=30,60 → K=8 r=0.5 ; d=90 → K=4 r=0.8"
echo

# ── [1/5] DFL-LP and DFL-Sketch, aug-Lagrangian loss ─────────────────────────
# Both solve a SKETCHED reduced LP, so b - M w* is genuinely non-zero and the
# loss variant actually changes training.
echo "[1/5] run_dfl_vs_qpth: lp, sketch  (aug-Lagrangian)"
"$PY" -m mdp_continuous.run_dfl_vs_qpth \
    --solvers lp sketch \
    --seeds $SEEDS \
    --n_workers "$WORKERS" \
    $AUGLAG_FLAGS

# ── [2/5] DFL-QP and Two-stage ───────────────────────────────────────────────
# DFL-QP is deliberately NOT given AUGLAG_FLAGS. The QP enforces Mw=b exactly,
# so the violation is identically zero — measured at nfeat=1024: aug_lag=0.0000
# and linear=+/-0.0000. All four loss modes collapse to the same loss for this
# arm, so passing --loss_mode auglag would only rename its cache file while
# training identically.
# Two-stage runs no DFL phase at all, so the loss variant is irrelevant there.
echo "[2/5] run_dfl_vs_qpth: qpth, two_stage  (loss variant does not apply)"
"$PY" -m mdp_continuous.run_dfl_vs_qpth \
    --solvers qpth two_stage \
    --seeds $SEEDS \
    --n_workers "$WORKERS"

# ── [3/5] DFL-Feas / Odece [34] baseline ─────────────────────────────────────
# Constrained-DFL baseline requested by the reviewer. Solver-free: both LP
# solves are detached and the gradient reaches the features only through the
# constraint residual M(theta)w - b(theta) plus the objective-completion MSE.
# --odece_neg_solver plain is the faithful reading of [34] (the predicted
# problem is solved exactly); 'arm' is a charitable variant that reuses the
# sketched layer for y^-. loss_mode does not apply — Odece replaces the loss.
echo "[3/5] run_dfl_vs_qpth: feas  (DFL-Feas / Odece [34])"
"$PY" -m mdp_continuous.run_dfl_vs_qpth \
    --solvers feas \
    --odece_neg_solver plain \
    --seeds $SEEDS \
    --n_workers "$WORKERS"

# ── [4/5] DFL-SA sweeps, aug-Lagrangian loss ─────────────────────────────────
# DFL-SA also runs on the sketched LP, so the loss variant bites here too.
echo "[4/5] sweep_n_features  d=30,60  (aug-Lagrangian)"
"$PY" -m mdp_continuous.sweep_n_features \
    --seeds $SEEDS \
    --n_features_list 30 60 \
    --n_workers "$WORKERS" \
    --sa_K 8 \
    --sa_ratio 0.5 \
    --show_running_max \
    --force \
    $AUGLAG_FLAGS

echo "[4/5] sweep_n_features  d=90  (aug-Lagrangian)"
"$PY" -m mdp_continuous.sweep_n_features \
    --seeds $SEEDS \
    --n_features_list 90 \
    --n_workers "$WORKERS" \
    --sa_K 4 \
    --sa_ratio 0.8 \
    --show_running_max \
    --force \
    $AUGLAG_FLAGS

# ── [5/5] Re-render PDFs from the (now-updated) caches ───────────────────────
# CARTPOLE_LOSS_MODE / CARTPOLE_DUAL_FEASIBLE_ONLY (exported above) make the
# figure read the aug-Lagrangian caches. The penalty-loss caches are NOT
# overwritten — both variants coexist under different filenames — so unsetting
# those variables re-renders the original paper figure.
echo "[5/5] make_figures.py cartpole"
"$PY" make_figures.py cartpole

echo
echo "[reproduce] done. Output PDFs:"
ls -la "$ROOT/figures/cartpole_"*.pdf

echo
echo "[reproduce] NOTE: DFL-Feas is run and cached but is NOT yet a series in"
echo "[reproduce] make_figures.py cartpole — adding it needs a new curve/bar in"
echo "[reproduce] render_cartpole(). Inspect its numbers directly with:"
echo "[reproduce]   $PY -m mdp_continuous.summarize_feas_auglag --results_dir mdp_continuous/results/dfl_vs_qpth"
