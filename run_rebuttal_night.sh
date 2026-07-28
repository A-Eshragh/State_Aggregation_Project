#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# NeurIPS rebuttal: overnight driver.
#
# Runs the five requested scripts STRICTLY ONE AT A TIME. That is deliberate:
# every "computation time" number these experiments report (train_time,
# solver_time_sec_per_epoch, sa_time_sec_per_epoch) is measured with
# time.time(), i.e. WALL CLOCK. Two of these scripts running at once would
# inflate each other's timings, and the timing comparisons that go in the paper
# are always *within* one script (arm A vs arm B in the same table). So the
# only safe arrangement is: while one script runs, none of the others do.
#
# |S|=100 is the exception: it is ~58 h of work, so it cannot share the night
# with the others sequentially. It runs as a SECOND session pinned to a disjoint
# set of cores, so the two never compete for the same CPU:
#
#   session A (chain):  CPUSET=0-79    STAGES="cw cont lp s50"
#   session B (S=100):  CPUSET=80-127  STAGES="s100"
#
#   bash run_rebuttal_night.sh                 # default chain, no s100
#   STAGES="s100" CPUSET=80-127 bash run_rebuttal_night.sh
#   bash run_rebuttal_night.sh --preflight     # validate launch paths, run nothing
#
# Logs + status:  logs/rebuttal_night_<stamp>/
# ---------------------------------------------------------------------------
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"

# The only surviving interpreter with numpy/torch/gurobipy/matplotlib.
# $HOME/anaconda3 is a dangling symlink to an unmounted /scratch, and the base
# env on the recovery mount has no gurobipy.
PY="${PY:-/net/csefiles/siemens/warriors-ls-recovery/users/zzhao628/#113442817/anaconda3/envs/rl/bin/python}"
PY_BIN="$(dirname "$PY")"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/rebuttal_night_$STAMP}"
STAGES="${STAGES:-cw cont lp s50}"
CPUSET="${CPUSET:-}"          # e.g. 0-79 ; empty = no pinning
mkdir -p "$LOG_DIR"

# Prefix that confines a stage to $CPUSET. Pinning is what makes the two
# sessions' wall-clock timings independent of each other.
PIN=()
if [ -n "$CPUSET" ]; then PIN=(taskset -c "$CPUSET"); fi

SUMMARY="$LOG_DIR/SUMMARY.txt"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$SUMMARY"; }

preflight() {
  local ok=0
  [ -x "$PY" ] || { echo "FAIL: no interpreter at $PY"; ok=1; }
  "$PY" -c "import numpy,torch,gurobipy,matplotlib" 2>/dev/null \
    || { echo "FAIL: $PY missing one of numpy/torch/gurobipy/matplotlib"; ok=1; }
  for f in mdp_cw/run_dfl_seed.sh mdp_continuous/run_dfl_seeds.sh \
           mdp_lp/run_dfl_seeds.sh mdp_lp/run_dfl_seeds_S_50.sh \
           mdp_lp/run_dfl_seeds_S_100.sh; do
    [ -f "$f" ] || { echo "FAIL: missing $f"; ok=1; }
  done
  # mdp_lp/run_dfl_seeds.sh calls bare `python`; confirm the PATH shim works.
  PATH="$PY_BIN:$PATH" python -c "import gurobipy" 2>/dev/null \
    || { echo "FAIL: PATH shim does not give gurobipy to bare 'python'"; ok=1; }
  local free_g; free_g=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  [ "${free_g:-0}" -ge 10 ] || echo "WARN: only ${free_g}G free on this filesystem"
  [ "$ok" -eq 0 ] && echo "preflight OK  (interpreter: $PY)"
  return $ok
}

# run_stage <name> <human label> -- <command...>
run_stage() {
  local name="$1" label="$2"; shift 3
  local log="$LOG_DIR/${name}.log"
  say "START  $name  ($label)"
  echo "running" > "$LOG_DIR/${name}.status"
  local t0; t0=$(date +%s)
  ( ulimit -s 65536 2>/dev/null; "${PIN[@]}" "$@" ) > "$log" 2>&1
  local rc=$? t1; t1=$(date +%s)
  local mins=$(( (t1-t0)/60 ))
  if [ "$rc" -eq 0 ]; then
    echo "done" > "$LOG_DIR/${name}.status"
    say "OK     $name  in ${mins} min"
  else
    echo "failed:$rc" > "$LOG_DIR/${name}.status"
    say "FAIL   $name  rc=$rc after ${mins} min  -> $log"
  fi
  # Never abort the chain on one stage's failure: the remaining stages are
  # independent and the night is the scarce resource.
  return 0
}

if [ "${1:-}" = "--preflight" ]; then preflight; exit $?; fi

preflight | tee -a "$SUMMARY"
preflight >/dev/null 2>&1 || { say "ABORT: preflight failed"; exit 1; }

say "chain: $STAGES   (strictly serial, one stage at a time)"
say "cpus:  ${CPUSET:-<unpinned, all 128>}"
say "logs:  $ROOT/$LOG_DIR"

for st in $STAGES; do
case "$st" in

  # ~50 min. Only ssa18 + ssa24 are missing (20 runs); the other 6 methods
  # already have all 10 seeds and are skipped. Note the file is run_dfl_seed.sh
  # (singular), and its default PY=python3 has no gurobipy -- hence PY= here.
  # Heaviest CPU footprint of the five: 10 concurrent runs x (4 OMP + 6 LP
  # workers). Fine because nothing else of ours is running.
  cw)   run_stage cw "mdp_cw ssa18+ssa24, ~50 min" -- \
          env PY="$PY" bash mdp_cw/run_dfl_seed.sh ;;

  # ~1 h. No cache-skip in run_dfl_vs_qpth.py and the sweeps pass --force, so
  # all 5 steps re-run from scratch and rewrite the auglag caches + PDFs.
  cont) run_stage cont "mdp_continuous full 5-step, ~1 h" -- \
          env PY="$PY" bash mdp_continuous/run_dfl_seeds.sh ;;

  # ~3.2 h. Two things to know: (1) it calls bare `python`, which resolves to
  # the base env with no gurobipy -- the PATH prepend fixes that; (2) there is
  # NO skip logic anywhere in run_elasticity_experiments.py, so it re-runs and
  # OVERWRITES all 90 runs (89 already exist). Deliberately NOT thread-capped:
  # it is internally serial, so letting torch use the box reproduces the
  # conditions the existing solver-s/ep table was measured under.
  lp)   run_stage lp "mdp_lp normal_margin, all 90 runs serial, ~3.2 h" -- \
          env PATH="$PY_BIN:$PATH" bash mdp_lp/run_dfl_seeds.sh ;;

  # ~2.5 h. Idempotent; 18 of 80 runs remain (8 of them qpal_r1 at ~48 min ea).
  # Its default PY points at the dangling $HOME/anaconda3 symlink -- override.
  s50)  run_stage s50 "mdp_lp |S|=51 top-up, 18 runs, ~2.5 h" -- \
          env PY="$PY" bash mdp_lp/run_dfl_seeds_S_50.sh ;;

  # Runs as its own session on a disjoint cpuset. Nothing is cached (0 of 64).
  # The script's own JOBS_BIG=2 would serialise the 16 big runs into ~58 h; we
  # raise it to 16 so every big run goes at once and the wall clock collapses to
  # the SLOWEST SINGLE RUN. That is the irreducible limit -- parallelism cannot
  # shorten one run:
  #     cheap arms  -> ~1.8 h  (feasfaith is the slowest at |S|=100)
  #     fullal      -> ~4.5 h  (measured)      => done by morning
  #     qpal_r1     -> ~12 h   (0.79/0.28 = 2.8x fullal at |S|=51, x 4.5 h)
  # So everything EXCEPT qpal_r1 lands by morning; qpal_r1 lands early afternoon.
  # Idempotent, so it can be stopped and resumed without losing finished runs.
  s100) run_stage s100 "mdp_lp |S|=100, 64 runs, wide fan-out" -- \
          env PY="$PY" JOBS_BIG="${JOBS_BIG:-16}" JOBS_CHEAP="${JOBS_CHEAP:-8}" \
          bash mdp_lp/run_dfl_seeds_S_100.sh ;;

  *) say "SKIP unknown stage '$st'" ;;
esac
done

say "CHAIN DONE"
echo
echo "===================== STATUS ====================="
for f in "$LOG_DIR"/*.status; do
  [ -e "$f" ] || continue
  printf '  %-6s %s\n' "$(basename "$f" .status)" "$(cat "$f")"
done
echo "=================================================="
