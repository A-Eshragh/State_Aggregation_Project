#!/usr/bin/env bash
# Augmented-Lagrangian LOSS ablation (Inventory, DFL-LP, exact v*).
#
# 2x2 loss design  { linear term off/on } x { quadratic penalty off/on }:
#   regret      : -r^T y                                  (a)
#   penalty     : -r^T y + rho||H_true y - gamma||^2      (b)  = current paper Eq.(5)
#   lagrangian  : -r^T y + v*^T(H_true y - gamma)         (c)
#   auglag      : penalty + linear term                   (d)  = proposed unbiased loss
# plus a rho-sweep on penalty vs auglag.
#
# Isolation: writes ONLY under results_auglag_ablation/ and /tmp/auglag_*.log,
# disjoint from the projection ablation (results_projection_ablation/, /tmp/ablation_*).
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY=/nethome/zzhao628/anaconda3/envs/rl/bin/python

# ---- sweep knobs (edit or pass as env) ----------------------------------------
LW="${LW:-20}"                         # loss_weight (0 = pure DFL, isolates loss bias)
SEEDS="${SEEDS:-0 1 2}"
RHOS="${RHOS:-0.1 1 10 100 1000}"     # rho-sweep for penalty & auglag
RHO_CORE=10                            # rho for regret/lagrangian tagging (unused by them)
# distinct results dir per loss_weight so lw=20 and lw=0 runs never overwrite
RESULTS_ROOT="mdp_lp/results_auglag_ablation_lw${LW}"
# ---- fixed inventory config (exp_b high_margin, matches run_exp_b_clean.sh) ----
COMMON=(--experiment_type b --cost_case high_margin --algo dfl
        --foc 0 --b_max 0 --hc 1 --loss_weight "$LW"
        --s_max 50 --beta_0 4.0 --beta_1 -0.3
        --n_epochs 50 --n_samples 100 --n_agg -1
        --results_root "$RESULTS_ROOT")

run() {  # args: mode rho seed
  local mode="$1" rho="$2" seed="$3"
  local tag="${mode}_rho${rho}_seed${seed}"
  local log="/tmp/auglag_lw${LW}_${tag}.log"
  echo "[auglag_ablation] $tag"
  "$PY" -m mdp_lp.run_elasticity_experiments "${COMMON[@]}" \
      --aug_lag_rho "$rho" --loss_mode "$mode" --tag "$tag" --seed "$seed" \
      > "$log" 2>&1
  grep -iE 'Mean regret|Lambda MSE' "$log" | sed "s/^/    [$tag] /" || true
}

JOBS="${JOBS:-8}"                       # max concurrent runs (0/1 = sequential)
gate() { while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n; done; }

for seed in $SEEDS; do
  # rho-independent modes: run once per seed
  gate; run regret     "$RHO_CORE" "$seed" &
  gate; run lagrangian "$RHO_CORE" "$seed" &
  # rho-dependent modes: sweep rho
  for rho in $RHOS; do
    gate; run penalty "$rho" "$seed" &
    gate; run auglag  "$rho" "$seed" &
  done
done
wait

echo "[auglag_ablation] DONE. Results under $RESULTS_ROOT ; per-run logs /tmp/auglag_*.log"
