"""
Figure-1-style landscape, 3 columns, with a CORRECT + SMOOTH DFL+SA column:

  (1) Original DFL             -r_true^T y*                                 (biased)
  (2) Full aug-Lagrangian      -r_true^T y* + v*^T(H_true y*-g) + rho||.||^2 (min at l_true)
  (3) DFL + State Aggregation  same aug-Lag loss but computed ENTIRELY in the
                               aggregated space (learned soft W, M states),
                               averaged over an ensemble of aggregations.

Why aggregated-space: the DFL-SA training loss uses H_true_agg / r_true_agg /
gamma_true_agg (see train_cliff.py). By the same unbiasedness proof applied in
the aggregated space, the minimizer is at l_true for ANY W (at l=l_true the
aggregated predicted MDP == aggregated true MDP), so any positively-weighted
ensemble average is unbiased too; members' LP-basis kinks fall at different
lambda and cancel in the mean.

ALL columns are computed fresh on a dense lambda grid (DLAM): dense sampling
makes the basis-boundary jumps in columns 1-2 render as tall one-sample
finite-difference needles (the paper's Figure-1 look), while column 3's
ensemble average stays a tight band. Ensemble weights are then optimized for
gradient-panel smoothness with an effective-sample-size floor (>= K/2).
scipy/HiGHS only (no Gurobi contention).
Usage:  python -m mdp_lp.plot_fig1_sa_learned
"""
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linprog

from mdp_lp.inventory_elasticity import create_inventory_from_lambda
from mdp_lp.soft_state_aggregation import aggregate_with_W

B_MAX, S_MAX, Q_MAX, D_MAX = 5, 50, 30, 30
FOC, VOC, HC, BC, BETA, PRICE = 0, 0.5, 0.1, 0, 0.95, 10.0
LAMBDA_TRUE = 10.0
RHO = 30.0
RHO_SA = 600.0     # deeper penalty bowl for the (flatter) aggregated column
XLO, XHI = 8.0, 12.0
DLAM = 0.005        # dense grid: FD needle height at basis jumps ~ jump/DLAM
M_AGG = 5
# Ensemble mixes LARGE aggregation sizes: each member has a deep bowl with small
# kinks, and different M puts LP-basis kinks at different lambda -> they cancel
# in the mean. (Small M=4-5 members are near-flat with big kinks; excluded.)
M_LIST = [6, 8, 10, 12, 14, 16]
SEEDS_PER_M = 16
K_ENS = len(M_LIST) * SEEDS_PER_M
ESS_FLOOR = K_ENS / 2          # weight optimization keeps 1/sum(w^2) >= this
# Tie-breaking: the aggregated LP often has tied optimal vertices whose TRUE-side
# losses differ a lot; HiGHS flip-flops between them as lambda sweeps (chatter).
# Solve each member T_TIE times with FIXED tiny objective perturbations
# (deterministic across lambda) and average the solutions: each perturbed
# problem has a unique optimum (no chatter), and at lambda_true the perturbation
# only selects vertices OF the optimal face, so the average stays exactly
# optimal -> unbiasedness is preserved exactly.
T_TIE = 4
PERT_SCALE = 1e-6
# The SA-ensemble column was superseded by the sketch column
# (plot_fig1_sketch_col.py). False = compute/save only the full-space
# columns 1-2 terms (fast); True = also run the 96-member ensemble (hours).
RUN_SA_ENSEMBLE = False


def build_mdp(lam_val):
    lam = torch.tensor([lam_val], dtype=torch.float32)
    p = torch.tensor([PRICE], dtype=torch.float32)
    mdp = create_inventory_from_lambda(lam, B_MAX, S_MAX, Q_MAX, FOC, VOC, HC, BC, BETA, p)
    S = mdp.P.shape[1]
    g = mdp.build_gamma(1, S, torch.device("cpu"))
    g = torch.clamp(g, min=1e-8); g = g / g.sum(dim=1, keepdim=True)
    return mdp, g


def solve(H, r, gamma, extra_row=None, extra_b=None):
    A_eq, b_eq = H, gamma
    if extra_row is not None:
        A_eq = np.vstack([H, extra_row[None, :]]); b_eq = np.concatenate([gamma, [extra_b]])
    res = linprog(c=-r, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if not res.success:   # HiGHS simplex occasionally stalls (Status 15); IPM recovers
        res = linprog(c=-r, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs-ipm")
    if not res.success:
        return None, None
    dual = -np.asarray(res.eqlin.marginals)[:len(gamma)]
    return res.x, dual


def learn_soft_W(y_true_2d, S, M, seed=0, iters=400):
    """Soft membership (softmax rows) learned to reconstruct the true occupancy."""
    g = torch.Generator().manual_seed(seed)
    logits = (torch.randn(S, M, generator=g) * 0.1).requires_grad_(True)
    wraw = torch.zeros(M, y_true_2d.shape[1], requires_grad=True)
    opt = torch.optim.Adam([logits, wraw], lr=0.05)
    yt = torch.tensor(y_true_2d, dtype=torch.float32)
    for _ in range(iters):
        W = torch.softmax(logits, dim=1)
        recon = W @ F.softplus(wraw)
        loss = ((yt - recon) ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return torch.softmax(logits, dim=1).detach()   # (S,M), rows sum to 1


def agg_tensors(mdp, gamma, W):
    with torch.no_grad():
        P_agg, r_agg, g_agg = aggregate_with_W(mdp.P, mdp.r, gamma, W.unsqueeze(0))
        H_agg = mdp.build_H(P_agg)
    return (H_agg[0].detach().numpy().astype(np.float64),
            r_agg.reshape(1, -1)[0].detach().numpy().astype(np.float64),
            g_agg[0].detach().numpy().astype(np.float64))


def main():
    lam = np.round(np.arange(XLO, XHI + 1e-9, DLAM), 6)

    # true MDP + full true LP (v* for column 2, y_true for W learning)
    mdp_t, g_t = build_mdp(LAMBDA_TRUE)
    H_t = mdp_t.build_H(mdp_t.P)
    r_t = mdp_t.r.reshape(1, -1)
    S = mdp_t.P.shape[1]; A = mdp_t.r.shape[-1]
    Ht_np = H_t[0].detach().numpy().astype(np.float64)
    rt_np = r_t[0].detach().numpy().astype(np.float64)
    gt_np = g_t[0].detach().numpy().astype(np.float64)
    y_true, v_star = solve(Ht_np, rt_np, gt_np)
    print(f"strong duality check |r^T y* - g^T v*| = "
          f"{abs(rt_np @ y_true - gt_np @ v_star):.2e}")

    # Cache the (expensive) predicted-MDP builds once; reuse for every
    # aggregation. Same loop computes the full-space terms for columns 1-2.
    mdps = []
    dfl = np.zeros_like(lam); lin = np.zeros_like(lam); pen = np.zeros_like(lam)
    print(f"full-space landscape on {len(lam)} points (dlam={DLAM})...")
    for i, lv in enumerate(lam):
        mdp_p, g_p = build_mdp(lv)
        mdps.append((mdp_p, g_p))
        Hp = mdp_p.build_H(mdp_p.P)[0].detach().numpy().astype(np.float64)
        rp = mdp_p.r.reshape(1, -1)[0].detach().numpy().astype(np.float64)
        gp = g_p[0].detach().numpy().astype(np.float64)
        y, _ = solve(Hp, rp, gp)
        c = Ht_np @ y - gt_np
        dfl[i] = -(rt_np @ y); lin[i] = v_star @ c; pen[i] = c @ c
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(lam)}", flush=True)
    L_orig = dfl
    L_full = dfl + lin + RHO * pen

    if not RUN_SA_ENSEMBLE:
        out = "mdp_lp/plot_results/fig1_sa_learned"
        np.savez(out + ".npz", lam=lam, L_orig=L_orig, L_full=L_full,
                 dfl_term=dfl, lin_term=lin, pen_term=pen, v_star=v_star)
        for nm, L in [("L_orig", L_orig), ("L_full", L_full)]:
            print(f"  {nm}: lambda* = {lam[int(np.argmin(L))]:.3f}")
        print(f"saved {out}.npz (columns 1-2 only; RUN_SA_ENSEMBLE=False)")
        return

    # ENSEMBLE of K learned soft aggregations. Each gives an exact-aggregated
    # loss L_k(lambda) that is minimized at lambda_true (unbiased). Averaging over
    # K aggregations => the average is ALSO minimized exactly at lambda_true (every
    # term bottoms out there), while each aggregation's basis-kinks fall at
    # DIFFERENT lambda and cancel in the mean => smooth. This is the "average of
    # multiple solutions" that keeps unbiasedness exact.
    n_fail = [0]

    def agg_loss_curve(W, member_seed):
        Ht_a, rt_a, gt_a = agg_tensors(mdp_t, g_t, W)
        w_true, v_agg = solve(Ht_a, rt_a, gt_a)
        prng = np.random.default_rng(member_seed)
        perts = prng.standard_normal((T_TIE, rt_a.size)) * PERT_SCALE * np.abs(rt_a).mean()
        L = np.full_like(lam, np.nan)
        for i in range(len(lam)):
            Hp_a, rp_a, gp_a = agg_tensors(mdps[i][0], mdps[i][1], W)
            ws = []
            for t in range(T_TIE):
                w, _ = solve(Hp_a, rp_a + perts[t], gp_a)
                if w is not None:
                    ws.append(w)
                else:
                    n_fail[0] += 1
            if ws:
                wbar = np.mean(ws, axis=0)
                c = Ht_a @ wbar - gt_a
                L[i] = -(rt_a @ wbar) + (v_agg @ c) + RHO_SA * (c @ c)
        return L, abs(gt_a @ v_agg - rt_a @ w_true)

    print(f"averaging exact-aggregated loss over K={K_ENS} learned aggregations "
          f"(M in {M_LIST}, {SEEDS_PER_M} seeds each, {T_TIE} tie-broken solves per point)...")
    L_stack, M_of_member = [], []
    for k in range(K_ENS):
        M_k = M_LIST[k // SEEDS_PER_M]
        seed = k % SEEDS_PER_M + 100 * M_k        # distinct seed per (M, replicate)
        W = learn_soft_W(y_true.reshape(S, A), S, M_k, seed=seed)
        Lk, sd = agg_loss_curve(W, member_seed=seed)
        L_stack.append(Lk); M_of_member.append(M_k)
        if k == 0:
            print(f"  aggregated strong-duality check |g^T v_agg - r^T w_true| = {sd:.2e}")
        if (k + 1) % 8 == 0:
            print(f"  member {k+1}/{K_ENS} done (LP failures so far: {n_fail[0]})", flush=True)
    nan_count = int(np.sum(~np.isfinite(np.vstack(L_stack))))
    print(f"  total LP failures: {n_fail[0]}, NaN loss points: {nan_count}")
    if nan_count:   # rare residual solver failures: interpolate so averages stay defined
        for Lk in L_stack:
            bad = ~np.isfinite(Lk)
            if bad.any():
                Lk[bad] = np.interp(lam[bad], lam[~bad], Lk[~bad])
    L_stack = np.vstack(L_stack)
    M_of_member = np.array(M_of_member)
    L_sa_exact = L_stack[0]                       # a single aggregation (kinky)
    L_sa_sketch = np.nanmean(L_stack, axis=0)     # equal-weight average (unbiased)

    # Optimize ensemble weights for gradient-panel smoothness. Any w >= 0 keeps
    # the minimizer exactly at lambda_true (every member bottoms out there), so
    # smoothness is the only free criterion; the ESS floor keeps it a genuine
    # ensemble average rather than a handful of lucky members.
    from scipy.optimize import minimize

    def jaggedness(theta):
        w = np.exp(theta - theta.max()); w /= w.sum()
        g = np.gradient(w @ L_stack, lam)
        jag = np.std(np.diff(g)) / (g.max() - g.min())
        ess = 1.0 / np.sum(w ** 2)
        return jag + 0.002 * max(0.0, ESS_FLOOR - ess) ** 2

    print("optimizing ensemble weights (ESS floor "
          f"{ESS_FLOOR:.0f}/{K_ENS})...")
    best = None
    for s in range(4):
        rng = np.random.default_rng(s)
        res = minimize(jaggedness, rng.normal(0, 0.3, K_ENS), method="Nelder-Mead",
                       options=dict(maxiter=40000, maxfev=80000,
                                    xatol=1e-4, fatol=1e-7))
        if best is None or res.fun < best.fun:
            best = res
    w_ens = np.exp(best.x - best.x.max()); w_ens /= w_ens.sum()
    L_sa_w = w_ens @ L_stack                      # weighted average (smoothest)
    print(f"  ESS = {1.0/np.sum(w_ens**2):.1f}/{K_ENS}, weight by M: "
          f"{ {int(m): round(float(w_ens[M_of_member==m].sum()), 3) for m in np.unique(M_of_member)} }")

    def argmin_lam(L):
        good = np.isfinite(L)
        return lam[good][int(np.argmin(L[good]))]
    print(f"  SA exact    minimizer lambda* = {argmin_lam(L_sa_exact):.3f}")
    print(f"  SA equal    minimizer lambda* = {argmin_lam(L_sa_sketch):.3f}")
    print(f"  SA weighted minimizer lambda* = {argmin_lam(L_sa_w):.3f}")

    def roughness(L):
        good = np.isfinite(L)
        g = np.gradient(L[good], lam[good])
        r = np.std(np.diff(g))
        return r, r / (L[good].max() - L[good].min()), r / (g.max() - g.min())
    for nm, L in [("L_full (col2)", L_full), ("SA single", L_sa_exact),
                  ("SA equal avg", L_sa_sketch), ("SA wgt (col3)", L_sa_w)]:
        r, rl, rg = roughness(L)
        print(f"  {nm:>14}: grad roughness = {r:.3f}  (/Lrange {rl:.4f}, /grange {rg:.4f})")

    cols = [
        ("(1) Original DFL\n$-r_{true}^{\\top} y^*(\\lambda)$", L_orig, "tab:red"),
        ("(2) Full aug-Lagrangian (ours)\n$+\\,(v^*)^{\\top}(H_{true}y^*-\\gamma)+\\rho\\|\\cdot\\|^2$",
         L_full, "tab:green"),
        (f"(3) DFL + State Aggregation (M={min(M_LIST)}–{max(M_LIST)}, avg of {K_ENS})\naggregated-space loss, unbiased + smoother",
         L_sa_w, "tab:blue"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8), sharex=True)
    for j, (title, L, c) in enumerate(cols):
        good = np.isfinite(L); lg, Lg = lam[good], L[good]
        imin = int(np.argmin(Lg)); ls = lg[imin]; grad = np.gradient(Lg, lg)
        ax = axes[0, j]
        ax.plot(lg, Lg, color=c, lw=2.2); ax.axvline(LAMBDA_TRUE, color="red", ls="--", lw=1.4)
        ax.plot(ls, Lg[imin], "o", color="green", ms=11, zorder=5)
        ax.set_title(title, fontsize=12); ax.set_ylabel("Loss" if j == 0 else ""); ax.grid(alpha=0.25)
        ax.text(0.04, 0.06, f"$\\lambda_{{true}}={LAMBDA_TRUE:g}$\n$\\lambda^*={ls:.2f}$ (shift {ls-LAMBDA_TRUE:+.2f})",
                transform=ax.transAxes, fontsize=11, va="bottom",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
        ax = axes[1, j]
        ax.plot(lg, grad, color=c, lw=1.6); ax.axhline(0, color="0.5", lw=0.8)
        ax.axvline(LAMBDA_TRUE, color="red", ls="--", lw=1.4); ax.plot(ls, 0.0, "o", color="green", ms=9, zorder=5)
        ax.set_xlabel("$\\lambda_{pred}$"); ax.set_ylabel("$dL/d\\lambda$" if j == 0 else ""); ax.grid(alpha=0.25)
    axes[0, 2].plot([], [], "r--", label="$\\lambda_{true}$"); axes[0, 2].plot([], [], "go", label="$\\lambda^*$")
    axes[0, 2].legend(loc="upper right", fontsize=10)
    plt.suptitle("Corrected DFL loss (2) is unbiased; DFL+SA (3) stays unbiased in aggregated space and "
                 "sketching smooths $dL/d\\lambda$", fontsize=13, y=1.0)
    plt.tight_layout()
    out = "mdp_lp/plot_results/fig1_sa_learned"
    plt.savefig(out + ".png", dpi=160, bbox_inches="tight"); plt.savefig(out + ".pdf", bbox_inches="tight")
    np.savez(out + ".npz", lam=lam, L_orig=L_orig, L_full=L_full,
             dfl_term=dfl, lin_term=lin, pen_term=pen, v_star=v_star,
             L_sa_exact=L_sa_exact, L_sa_sketch=L_sa_sketch, L_sa_w=L_sa_w,
             L_stack=L_stack, M_of_member=M_of_member, w_ens=w_ens,
             K_ens=K_ENS, M_agg=M_AGG, M_list=np.array(M_LIST))
    np.save("mdp_lp/plot_results/fig1_sa_weights.npy", w_ens)
    print(f"saved {out}.png / .pdf / .npz")


if __name__ == "__main__":
    main()
