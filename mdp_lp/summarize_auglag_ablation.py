"""
Summarize the augmented-Lagrangian LOSS ablation (Inventory, DFL-LP, exact v*).

Reads every results.json under mdp_lp/results_auglag_ablation/, groups runs by
(loss_mode, aug_lag_rho) over seeds, and prints:
  1) the 2x2 loss table at rho=RHO_CORE  (regret / penalty / lagrangian / auglag)
  2) a rho-sweep table for penalty vs auglag.

Metrics: test mean regret (lower better), and lambda-bias |lambda_hat - lambda_true|
(the direct, sensitive signal of the minimizer bias the linear term removes).

Tolerates a partially-complete sweep. Writes summary_auglag_ablation.{txt,json}.
Usage:  python -m mdp_lp.summarize_auglag_ablation
"""
import json
import glob
import math
import sys
from collections import defaultdict

# results dir: pass as argv[1] (e.g. mdp_lp/results_auglag_ablation_lw0), else default
ROOT = sys.argv[1] if len(sys.argv) > 1 else "mdp_lp/results_auglag_ablation"
RHO_CORE = 10.0
MODES = ["regret", "penalty", "lagrangian", "auglag"]


def _mean_std(xs):
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(v)


def load():
    """Return {(mode, rho): {'regret': [...], 'bias': [...], 'mse': [...], 'seeds': [...]}}"""
    recs = defaultdict(lambda: {"regret": [], "sbias": [], "mae": [], "seeds": []})
    files = sorted(glob.glob(f"{ROOT}/**/results.json", recursive=True))
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if d.get("status") != "complete" or d.get("algo") != "dfl":
            continue
        mode = d.get("loss_mode", "penalty")
        rho = float(d.get("aug_lag_rho", float("nan")))
        lp = d.get("test_lambda_pred") or []
        lt = d.get("test_lambda_true") or []
        if not lp:
            continue
        sbias = sum(a - b for a, b in zip(lp, lt)) / len(lp)     # signed systematic bias
        mae = sum(abs(a - b) for a, b in zip(lp, lt)) / len(lp)  # variance-dominated
        key = (mode, rho)
        recs[key]["regret"].append(d.get("test_mean_regret", float("nan")))
        recs[key]["sbias"].append(sbias)
        recs[key]["mae"].append(mae)
        recs[key]["seeds"].append(d.get("seed"))
    return recs, len(files)


def fmt(recs, key):
    r = recs.get(key)
    if not r or not r["regret"]:
        return f"{'--':>16} {'--':>16} {'(0 seeds)':>10}"
    rm, rs = _mean_std(r["regret"])
    bm, bs = _mean_std(r["sbias"])
    n = len(r["regret"])
    return f"{rm:>7.3f} ± {rs:<6.3f} {bm:>+7.3f} ± {bs:<6.3f} {'('+str(n)+' seeds)':>10}"


def main():
    recs, nfiles = load()
    out = []

    def emit(s=""):
        print(s)
        out.append(s)

    emit(f"Augmented-Lagrangian loss ablation  (Inventory / DFL-LP / exact v*)")
    emit(f"parsed {nfiles} results.json under {ROOT}")
    emit("")
    emit("=" * 74)
    emit(f"(1) 2x2 loss table at rho={RHO_CORE:g}    [test regret (lower=better) | SIGNED bias mean(lam_pred - lam*)]")
    emit("-" * 74)
    emit(f"{'mode':>12} | {'test regret':>16} {'signed lam-bias':>16} {'':>10}")
    emit("-" * 74)
    for mode in MODES:
        # regret/lagrangian are rho-independent (tagged at RHO_CORE); penalty/auglag at RHO_CORE
        emit(f"{mode:>12} | {fmt(recs, (mode, RHO_CORE))}")
    emit("=" * 74)
    emit("")
    emit("(2) rho-sweep   [test regret ; SIGNED bias]   penalty vs auglag")
    emit("-" * 74)
    rhos = sorted({rho for (m, rho) in recs if m in ("penalty", "auglag")})
    emit(f"{'rho':>10} | {'penalty regret':>16} {'penalty sbias':>14} | {'auglag regret':>16} {'auglag sbias':>14}")
    emit("-" * 74)
    for rho in rhos:
        pr = recs.get(("penalty", rho), {})
        ar = recs.get(("auglag", rho), {})
        prm, _ = _mean_std(pr.get("regret", []))
        pbm, _ = _mean_std(pr.get("sbias", []))
        arm, _ = _mean_std(ar.get("regret", []))
        abm, _ = _mean_std(ar.get("sbias", []))
        emit(f"{rho:>10g} | {prm:>16.3f} {pbm:>+14.3f} | {arm:>16.3f} {abm:>+14.3f}")
    emit("=" * 74)

    txt = ROOT + "/summary_auglag_ablation.txt"
    js = ROOT + "/summary_auglag_ablation.json"
    with open(txt, "w") as fh:
        fh.write("\n".join(out) + "\n")
    serial = {f"{m}|rho={rho:g}": {k: v for k, v in rec.items()} for (m, rho), rec in recs.items()}
    with open(js, "w") as fh:
        json.dump(serial, fh, indent=2)
    print(f"\nwrote {txt} and {js}")


if __name__ == "__main__":
    main()
