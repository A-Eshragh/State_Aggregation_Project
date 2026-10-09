"""
Column 3 of Figure 1 the PAPER's way: Aug. Lagrangian + Sketch.

At each lambda on the dense grid, solve the predicted full-space LP K times
with Gaussian ROW SKETCHES of the constraints (paper Sec. "Sketch-and-solve"):

    max r_pred^T y   s.t.  (S_i H_pred) y = S_i gamma_pred,
                           1^T y = 1/(1-beta)   (mass conservation),  y >= 0

with S_i in R^{k x S}, k = round(alpha * S), entries N(0, 1/k). The K sketches
are sampled ONCE and held fixed across the lambda sweep (common random numbers,
as in sketch_smoothing_demo.py) so the averaged solution ybar(lambda) is a
deterministic, nearly-smooth function of lambda. The column-3 loss is the SAME
full aug-Lagrangian as column 2, evaluated at ybar:

    L(l) = -r_true^T ybar + v*^T (H_true ybar - gamma) + RHO ||H_true ybar - gamma||^2

Reuses lam grid, v_star and columns 1-2 terms from fig1_sa_learned.npz.
Sweeps alpha in ALPHAS at K=K_SKETCH; saves every curve to fig1_sketch.npz.

Usage:  python -m mdp_lp.plot_fig1_sketch_col
"""
import multiprocessing as mp
import numpy as np
import torch
from scipy.optimize import linprog

from mdp_lp.plot_fig1_sa_learned import (build_mdp, LAMBDA_TRUE, RHO, BETA,
                                         XLO, XHI, DLAM)

NPZ_IN = "mdp_lp/plot_results/fig1_sa_learned.npz"
NPZ_OUT = "mdp_lp/plot_results/fig1_sketch.npz"
K_SKETCH = 512
ALPHAS = [0.9]
SEED = 0
NPROC = 96

_G = {}   # read-only globals inherited by fork()ed pool workers


def solve_sketched(H_s, r, g_s, mass):
    """max r^T y s.t. H_s y = g_s, 1^T y = mass, y >= 0. Returns y or None."""
    A_eq = np.vstack([H_s, np.ones((1, H_s.shape[1]))])
    b_eq = np.concatenate([g_s, [mass]])
    res = linprog(c=-r, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if not res.success:
        res = linprog(c=-r, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs-ipm")
    return res.x if res.success else None


def main():
    # Own dense grid from DLAM (cols 1-2 may be on a coarser grid; the replot
    # handles per-column grids). Only v_star (grid-independent) comes from cache.
    lam = np.round(np.arange(XLO, XHI + 1e-9, DLAM), 6)
    v_star = np.load(NPZ_IN)["v_star"].astype(np.float64)

    mdp_t, g_t = build_mdp(LAMBDA_TRUE)
    Ht = mdp_t.build_H(mdp_t.P)[0].detach().numpy().astype(np.float64)
    rt = mdp_t.r.reshape(1, -1)[0].detach().numpy().astype(np.float64)
    gt = g_t[0].detach().numpy().astype(np.float64)
    S = Ht.shape[0]
    mass = 1.0 / (1.0 - BETA)
    print(f"S={S}, n_vars={Ht.shape[1]}, mass 1^T y = {mass:g}; "
          f"{len(lam)} lambda points, K={K_SKETCH}, alphas={ALPHAS}")

    # predicted-MDP tensors, built once per lambda
    preds = []
    for i, lv in enumerate(lam):
        mdp_p, g_p = build_mdp(lv)
        preds.append((mdp_p.build_H(mdp_p.P)[0].detach().numpy().astype(np.float64),
                      mdp_p.r.reshape(1, -1)[0].detach().numpy().astype(np.float64),
                      g_p[0].detach().numpy().astype(np.float64)))
        if (i + 1) % 50 == 0:
            print(f"  built {i+1}/{len(lam)} MDPs", flush=True)

    rng = np.random.default_rng(SEED)
    sketches = {}
    for alpha in ALPHAS:
        k = max(1, int(round(alpha * S)))
        sketches[alpha] = rng.standard_normal((K_SKETCH, k, S)) / np.sqrt(k)

    _G.update(preds=preds, sketches=sketches, mass=mass,
              Ht=Ht, rt=rt, gt=gt, v_star=v_star)
    tasks = [(alpha, i) for alpha in ALPHAS for i in range(len(lam))]
    print(f"solving {len(tasks)} (alpha, lambda) tasks on {NPROC} workers...")
    results = {}
    with mp.Pool(NPROC) as pool:
        for n_done, (alpha, i, Li, nf) in enumerate(
                pool.imap_unordered(_solve_point, tasks, chunksize=2), 1):
            results[(alpha, i)] = (Li, nf)
            if n_done % 100 == 0:
                print(f"  {n_done}/{len(tasks)} tasks done", flush=True)

    out = {"lam": lam, "K": K_SKETCH, "alphas": np.array(ALPHAS)}
    for alpha in ALPHAS:
        comp = np.array([results[(alpha, i)][0] for i in range(len(lam))])  # (N,3)
        n_fail = sum(results[(alpha, i)][1] for i in range(len(lam)))
        dfl_s, lin_s, pen_s = comp[:, 0], comp[:, 1], comp[:, 2]
        out[f"dfl_a{alpha:g}"] = dfl_s
        out[f"lin_a{alpha:g}"] = lin_s
        out[f"pen_a{alpha:g}"] = pen_s
        L = dfl_s + lin_s + RHO * pen_s
        good = np.isfinite(L)
        ls = lam[good][int(np.argmin(L[good]))]
        g = np.gradient(L[good], lam[good])
        r_ = np.std(np.diff(g))
        print(f"alpha={alpha} (rho={RHO:g}): lambda* = {ls:.3f} (shift {ls-LAMBDA_TRUE:+.3f}), "
              f"grad roughness = {r_:.3f}, LP failures = {n_fail}")
    np.savez(NPZ_OUT, **out)
    print(f"saved {NPZ_OUT}")


def _solve_point(task):
    """Returns (alpha, i, (dfl, lin, pen) components at ybar, n_fail):
    dfl = -r_true^T ybar, lin = v*^T c, pen = ||c||^2, c = H_true ybar - gamma."""
    alpha, i = task
    Hp, rp, gp = _G["preds"][i]
    sk = _G["sketches"][alpha]
    acc, valid, n_fail = 0.0, 0, 0
    for s in range(sk.shape[0]):
        y = solve_sketched(sk[s] @ Hp, rp, sk[s] @ gp, _G["mass"])
        if y is not None:
            acc = acc + y
            valid += 1
        else:
            n_fail += 1
    comp = (np.nan, np.nan, np.nan)
    if valid:
        ybar = acc / valid
        c = _G["Ht"] @ ybar - _G["gt"]
        comp = (-(_G["rt"] @ ybar), _G["v_star"] @ c, c @ c)
    return alpha, i, comp, n_fail


if __name__ == "__main__":
    main()
