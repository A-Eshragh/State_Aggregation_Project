"""
Level 1-2 rebuttal experiment: how much does the diagonal projection
(WᵀW -> diag(mu), i.e. the projected/aggregated-MDP LP actually trained)
change the occupancy and the gradients, versus the exact Galerkin projected LP?

Real inventory MDP (paper settings): S=36 (b_max=5, s_max=30), A=31 (q_max=30),
beta=0.95, normal_margin costs (foc=0, voc=0.5, hc=2), bc=10,
p ~ LogNormal(log 5, 0.6^2), lambda = exp(3.5 - log p + 0.3*eps).

Level 1 (occupancy): per (instance, W-source, tau): solve both LPs, lift both
  to the original S*A space (mass-corrected lift on the MDP side), and record
  relative occupancy gap, decision-value gap, policy disagreement, true-flow
  residuals, Gram off-diagonal mass rho, basis condition numbers, failures.
Level 2 (gradients): at off-optimum predictions lambda_pred = scale*lambda_true,
  record dL/dlambda for both layers (sign agreement, ratio) and the cosine
  similarity of dL/d(aggregation parameters); plus a Figure-1-style
  lambda-landscape of loss and gradient for both layers.

Run from repo root:  python -m mdp_lp.exp_projection_gap
Outputs: mdp_lp/results_projection_gap/{records.json, summary.txt, *.pdf/png}
"""

import json
import math
from pathlib import Path

import numpy as np
import torch

from mdp_lp.inventory_elasticity import create_inventory_from_lambda
from mdp_lp.soft_state_aggregation import (
    SoftAggDense,
    aggregate_with_W,
    galerkin_lp_coefficients,
)
from mdp_lp.solvers.dense_lp_solver import solve_lp_gurobi
from mdp_lp.solvers.differentiable_lp import DifferentiableLPSolver

# --- paper / checkpoint-provenance settings -------------------------------
B_MAX, S_MAX, Q_MAX = 5, 30, 30           # S=36, A=31
FOC, VOC, HC, BC = 0.0, 0.5, 2.0, 10.0    # normal_margin + hpsearch bc=10
BETA = 0.95
S, A = B_MAX + S_MAX + 1, Q_MAX + 1
TAUS = [0.1, 0.25, 0.5, 1.0, 2.0]
MS_RANDOM = [10, 20, 30]
N_INSTANCES = 20
SCALES = [0.8, 0.9, 1.1, 1.2]
LAM_TRUE_LANDSCAPE, PRICE_LANDSCAPE = 10.0, 5.0

OUT_DIR = Path("mdp_lp/results_projection_gap")

TRAINED_CKPTS = [
    "mdp_lp/results_hpsearch/exp_b/normal_margin/dfl_sketch_ngg_10/"
    f"tag_seed_{s}_s30q30_K8_r0.8_gaussian/epochs_50/agg_state_final.pt"
    for s in (1, 2, 3)
]


def build_mdp(lam, price):
    """lam, price: (1,) tensors. Returns (P, r, gamma, mdp). Differentiable in lam."""
    mdp = create_inventory_from_lambda(
        lam, B_MAX, S_MAX, Q_MAX, FOC, VOC, HC, BC, BETA, price)
    gamma = mdp.build_gamma(1, S, torch.device("cpu"))
    gamma = gamma / gamma.sum(dim=1, keepdim=True)
    return mdp.P, mdp.r, gamma, mdp


def sample_instances(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    p = torch.exp(torch.randn(n, generator=g) * 0.6 + math.log(5.0))
    lam = torch.exp(3.5 - torch.log(p) + 0.3 * torch.randn(n, generator=g))
    return lam.clamp(2.0, 25.0), p


def load_w_sources():
    """Return list of (name, agg_module, M)."""
    sources = []
    for seed in range(3):
        for M in MS_RANDOM:
            torch.manual_seed(seed)
            agg = SoftAggDense(n_states=S, n_actions=A, n_clusters=M, r_embed=8)
            sources.append((f"random_s{seed}_M{M}", agg, M))
    for path in TRAINED_CKPTS:
        pth = Path(path)
        if not pth.exists():
            print(f"[warn] missing checkpoint {path}")
            continue
        payload = torch.load(pth, map_location="cpu", weights_only=False)
        M = int(payload["n_clusters"])
        agg = SoftAggDense(n_states=int(payload["n_states"]),
                           n_actions=int(payload["n_actions"]),
                           n_clusters=M, r_embed=int(payload["r_embed"]))
        sd = {k: v for k, v in payload["state_dict"].items()
              if k in dict(agg.named_parameters())}
        missing, unexpected = agg.load_state_dict(sd, strict=False)
        assert not missing, f"missing keys {missing}"
        tag = pth.parts[-3].split("_s30q30")[0]  # tag_seed_k
        sources.append((f"trained_{tag}_M{M}", agg, M))
    return sources


# --- solves ---------------------------------------------------------------
def solve_projected_nograd(P, r, gamma, W):
    P_agg, r_agg, gamma_agg = aggregate_with_W(P, r, gamma, W)
    from mdp_lp.mdp import MDP
    H = MDP(BETA).build_H(P_agg)
    r_flat = r_agg.reshape(1, -1)
    y, basis, _ = solve_lp_gurobi(H, r_flat, gamma_agg)
    return y, r_flat, H, basis


def solve_exact_nograd(P, r, gamma, W):
    A_mat, r_t, b = galerkin_lp_coefficients(P, r, gamma, W, BETA)
    w, basis, _ = solve_lp_gurobi(A_mat, r_t, b)
    return w, r_t, A_mat, basis


def lift_projected(y_agg_flat, W, mass):
    M = W.shape[2]
    y_scaled = (y_agg_flat.view(1, M, A) / mass[:, :, None]).reshape(1, M * A)
    return SoftAggDense.disaggregate_y(y_scaled, W, A)


def lift_exact(w_flat, W):
    return SoftAggDense.disaggregate_y(w_flat, W, A)


def gram_offdiag(W):
    g = torch.einsum("bsi,bsm->bim", W, W)[0]
    return ((g - torch.diag(torch.diag(g))).abs().sum() / g.abs().sum()).item()


def basis_cond(H, basis):
    H_B = H[0][:, basis[0]].detach().numpy()
    return float(np.linalg.cond(H_B))


def policy_disagreement(y1, y2):
    """Occupancy-weighted fraction of states whose greedy action differs."""
    a1 = y1.view(S, A).argmax(dim=1)
    a2 = y2.view(S, A).argmax(dim=1)
    ws = y1.view(S, A).sum(dim=1).clamp_min(0)
    return float((ws * (a1 != a2).float()).sum() / ws.sum())


def flow_residual(y_lift, H_full, gamma):
    return float((H_full[0] @ y_lift[0] - gamma[0]).norm())


# --- Level 1 --------------------------------------------------------------
def run_level1(sources, lams, prices):
    records = []
    from mdp_lp.mdp import MDP
    for i in range(len(lams)):
        lam, price = lams[i:i + 1], prices[i:i + 1]
        with torch.no_grad():
            P, r, gamma, mdp = build_mdp(lam, price)
            H_full = mdp.build_H(P)
            r_flat_full = r.reshape(1, -1)
        for name, agg, M in sources:
            for tau in TAUS:
                with torch.no_grad():
                    W = agg.compute_W(P, r, tau)
                    mass = W.sum(dim=1)
                    rho = gram_offdiag(W)
                    rec = dict(inst=i, lam=float(lam), price=float(price),
                               source=name, M=M, tau=tau, rho=rho)
                    try:
                        y, rp, Hp, bp = solve_projected_nograd(P, r, gamma, W)
                        lp = lift_projected(y, W, mass)
                        rec["cond_proj"] = basis_cond(Hp, bp)
                    except RuntimeError as e:
                        rec["fail_proj"] = str(e); records.append(rec); continue
                    try:
                        w, rt, Am, be = solve_exact_nograd(P, r, gamma, W)
                        le = lift_exact(w, W)
                        rec["cond_exact"] = basis_cond(Am, be)
                    except RuntimeError as e:
                        rec["fail_exact"] = str(e); records.append(rec); continue
                    rec["occ_gap_rel"] = float((lp - le).abs().sum() / lp.abs().sum())
                    v_p = float((r_flat_full * lp).sum())
                    v_e = float((r_flat_full * le).sum())
                    rec["val_proj"], rec["val_exact"] = v_p, v_e
                    rec["val_gap_rel"] = abs(v_p - v_e) / max(abs(v_p), 1e-8)
                    rec["pol_disagree"] = policy_disagreement(lp, le)
                    rec["res_proj"] = flow_residual(lp, H_full, gamma)
                    rec["res_exact"] = flow_residual(le, H_full, gamma)
                    records.append(rec)
        print(f"[L1] instance {i+1}/{len(lams)} done")
    return records


# --- Level 2 --------------------------------------------------------------
def loss_and_grads(lam_pred_val, price, r_true_flat, agg, tau, layer,
                   train_lift=False):
    """Build the DFL decision loss L = -r_true^T y_lift(lambda_pred) through the
    given layer and return (L, dL/dlambda, flat dL/d(agg params))."""
    lam = torch.tensor([lam_pred_val], requires_grad=True)
    P, r, gamma, _ = build_mdp(lam, price)
    W = agg.compute_W(P, r, tau)
    mass = W.sum(dim=1)
    if layer == "proj":
        P_agg, r_agg, gamma_agg = aggregate_with_W(P, r, gamma, W)
        from mdp_lp.mdp import MDP
        H = MDP(BETA).build_H(P_agg)
        y = DifferentiableLPSolver.apply(H, r_agg.reshape(1, -1), gamma_agg)
        if train_lift:  # what train_elasticity.py actually does
            y_lift = SoftAggDense.disaggregate_y(y, W, A)
            y_lift = y_lift * ((1.0 / (1.0 - BETA)) / y_lift.sum(dim=1, keepdim=True))
        else:
            y_lift = lift_projected(y, W, mass)
    else:
        A_mat, r_t, b = galerkin_lp_coefficients(P, r, gamma, W, BETA)
        w = DifferentiableLPSolver.apply(A_mat, r_t, b)
        y_lift = lift_exact(w, W)
    L = -(r_true_flat * y_lift).sum()
    params = [p for p in agg.parameters() if p.requires_grad]
    grads = torch.autograd.grad(L, [lam] + params, allow_unused=True)
    g_lam = float(grads[0])
    g_par = torch.cat([g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1)
                       for g, p in zip(grads[1:], params)])
    return float(L.detach()), g_lam, g_par


def run_level2_stats(sources, lams, prices, tau=0.5):
    use = [s for s in sources if s[0] in
           ("random_s0_M10",) or s[0].startswith("trained_")]
    records = []
    for i in range(len(lams)):
        lam_t, price = lams[i:i + 1], prices[i:i + 1]
        with torch.no_grad():
            _, r_true, _, _ = build_mdp(lam_t, price)
            r_true_flat = r_true.reshape(1, -1)
        for name, agg, M in use:
            for scale in SCALES:
                lam_pred = float(lam_t) * scale
                rec = dict(inst=i, source=name, M=M, tau=tau, scale=scale,
                           lam_true=float(lam_t), lam_pred=lam_pred)
                try:
                    _, glp, gpp = loss_and_grads(lam_pred, price, r_true_flat,
                                                 agg, tau, "proj")
                    _, glpt, _ = loss_and_grads(lam_pred, price, r_true_flat,
                                                agg, tau, "proj", train_lift=True)
                    _, gle, gpe = loss_and_grads(lam_pred, price, r_true_flat,
                                                 agg, tau, "exact")
                except RuntimeError as e:
                    rec["fail"] = str(e); records.append(rec); continue
                rec["g_lam_proj"], rec["g_lam_exact"] = glp, gle
                rec["g_lam_proj_trainlift"] = glpt
                rec["sign_agree"] = bool(np.sign(glp) == np.sign(gle))
                rec["sign_agree_trainlift"] = bool(np.sign(glpt) == np.sign(gle))
                denom = (gpp.norm() * gpe.norm()).clamp_min(1e-12)
                rec["cos_aggparams"] = float((gpp @ gpe) / denom)
                rec["gnorm_ratio"] = float(gpp.norm() / gpe.norm().clamp_min(1e-12))
                records.append(rec)
        print(f"[L2] instance {i+1}/{len(lams)} done")
    return records


def run_landscape(sources, taus=(0.5, 0.1)):
    use = [s for s in sources if s[0] == "random_s0_M10"] + \
          [s for s in sources if s[0].startswith("trained_")][:1]
    price = torch.tensor([PRICE_LANDSCAPE])
    lam_t = torch.tensor([LAM_TRUE_LANDSCAPE])
    with torch.no_grad():
        _, r_true, _, _ = build_mdp(lam_t, price)
        r_true_flat = r_true.reshape(1, -1)
    grid = np.linspace(8.0, 12.0, 41)
    out = {}
    for (name, agg, M), tau in [(s, t) for t in taus for s in use]:
        name = f"{name}_tau{tau}"
        curves = {k: [] for k in ("L_proj", "g_proj", "L_exact", "g_exact")}
        n_fail = 0
        for lam in grid:
            try:
                Lp, gp, _ = loss_and_grads(float(lam), price, r_true_flat, agg, tau, "proj")
            except RuntimeError:
                Lp, gp = np.nan, np.nan
            try:
                Le, ge, _ = loss_and_grads(float(lam), price, r_true_flat, agg, tau, "exact")
            except RuntimeError:
                Le, ge = np.nan, np.nan
                n_fail += 1
            curves["L_proj"].append(Lp); curves["g_proj"].append(gp)
            curves["L_exact"].append(Le); curves["g_exact"].append(ge)
        out[name] = {k: np.array(v) for k, v in curves.items()}
        out[name]["n_fail_exact"] = np.array([n_fail])
        print(f"[Landscape] {name} done (exact-LP failures: {n_fail}/{len(grid)})")
    return grid, out


def probe_failure_statuses(l1, sources, lams, prices, n=60, seed=0):
    """Re-solve a sample of failed exact LPs, capturing the Gurobi status
    (INFEASIBLE vs UNBOUNDED vs NUMERIC), and check the boundedness identity
    column_sums(A) == (1-beta)*mu."""
    import gurobipy as gp
    from gurobipy import GRB
    from collections import Counter
    fails = [r for r in l1 if "fail_exact" in r]
    if not fails:
        return "no exact-LP failures to probe"
    src = {name: agg for name, agg, _ in sources}
    env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.start()
    names = {2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD",
             5: "UNBOUNDED", 12: "NUMERIC"}
    statuses, viol = Counter(), []
    rng = np.random.default_rng(seed)
    for k in rng.choice(len(fails), size=min(n, len(fails)), replace=False):
        rec = fails[k]
        with torch.no_grad():
            i = rec["inst"]
            P, r, gamma, _ = build_mdp(lams[i:i + 1], prices[i:i + 1])
            W = src[rec["source"]].compute_W(P, r, rec["tau"])
            A_mat, r_t, b = galerkin_lp_coefficients(P, r, gamma, W, BETA)
            mu = W.sum(dim=1)[0]
            viol.append(float((A_mat[0].sum(dim=0)
                               - ((1 - BETA) * mu).repeat_interleave(A)).abs().max()))
        m = gp.Model(env=env); m.Params.Threads = 1
        y = m.addMVar(shape=A_mat.shape[2], lb=0.0)
        m.setObjective(r_t[0].numpy() @ y, GRB.MAXIMIZE)
        m.addMConstr(A_mat[0].numpy(), y, "=", b[0].numpy())
        m.optimize()
        st = names.get(m.status, str(m.status))
        if st == "INF_OR_UNBD":
            m.Params.DualReductions = 0; m.reset(); m.optimize()
            st = "resolved:" + names.get(m.status, str(m.status))
        statuses[st] += 1
        m.dispose()
    return (f"exact-LP failure statuses ({sum(statuses.values())} sampled): "
            f"{dict(statuses)}; max boundedness-identity violation "
            f"{max(viol):.2e} (float32 noise => failures are true infeasibility)")


# --- figures --------------------------------------------------------------
C_PROJ, C_EXACT = "#2563EB", "#EA580C"          # validated pair (CVD dE>140)
TAU_RAMP = {0.1: "#BFDBFE", 0.25: "#93C5FD", 0.5: "#60A5FA",
            1.0: "#2563EB", 2.0: "#1E3A8A"}      # single-hue sequential


def fig_dose_response(recs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = [r for r in recs if "occ_gap_rel" in r]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    metrics = [("occ_gap_rel", "Relative occupancy gap"),
               ("val_gap_rel", "Relative decision-value gap"),
               ("pol_disagree", "Policy disagreement")]
    for ax, (key, label) in zip(axes, metrics):
        for tau in TAUS:
            for src_pref, marker in (("random", "o"), ("trained", "^")):
                pts = [(r["rho"], r[key]) for r in ok
                       if r["tau"] == tau and r["source"].startswith(src_pref)]
                if not pts:
                    continue
                x, y = zip(*pts)
                ax.scatter(x, np.maximum(y, 1e-6), s=9, marker=marker,
                           color=TAU_RAMP[tau], alpha=0.55, linewidths=0)
        ax.set_yscale("log")
        ax.set_xlabel(r"Gram off-diagonal mass $\rho$")
        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="", marker="s", color=TAU_RAMP[t],
                      label=fr"$\tau={t}$") for t in TAUS]
    handles += [Line2D([], [], ls="", marker="o", color="gray", label="random-init W"),
                Line2D([], [], ls="", marker="^", color="gray", label="trained W")]
    axes[2].legend(handles=handles, fontsize=7, frameon=False,
                   loc="lower right", ncol=2)
    fig.suptitle("Diagonal projection vs exact Galerkin LP — inventory MDP", y=1.02)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_landscape(grid, curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(curves)
    fig, axes = plt.subplots(2, n, figsize=(5.2 * n, 5.4), sharex=True)
    if n == 1:
        axes = axes.reshape(2, 1)
    for j, (name, c) in enumerate(curves.items()):
        ax = axes[0, j]
        ax.plot(grid, c["L_proj"], color=C_PROJ, lw=2, label="projected (ours)")
        ax.plot(grid, c["L_exact"], color=C_EXACT, lw=2, ls="--", label="exact Galerkin")
        for key, col in (("L_proj", C_PROJ), ("L_exact", C_EXACT)):
            if np.isfinite(c[key]).any():
                i = int(np.nanargmin(c[key]))
                ax.plot(grid[i], c[key][i], "o", color=col, ms=6)
        nf = int(c.get("n_fail_exact", np.array([0]))[0])
        if nf:
            ax.text(0.03, 0.05, f"exact LP infeasible {nf}/{len(grid)}",
                    transform=ax.transAxes, fontsize=8, color=C_EXACT)
        ax.axvline(LAM_TRUE_LANDSCAPE, color="gray", ls=":", lw=1)
        ax.set_title(name, fontsize=10)
        ax.set_ylabel("loss  $-r_{true}^\\top y$" if j == 0 else "")
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        if j == 0:
            ax.legend(fontsize=8, frameon=False)
        ax = axes[1, j]
        ax.plot(grid, c["g_proj"], color=C_PROJ, lw=2)
        ax.plot(grid, c["g_exact"], color=C_EXACT, lw=2, ls="--")
        ax.axvline(LAM_TRUE_LANDSCAPE, color="gray", ls=":", lw=1)
        ax.axhline(0, color="gray", lw=0.7)
        ax.set_xlabel(r"$\lambda_{pred}$")
        ax.set_ylabel(r"$dL/d\lambda$" if j == 0 else "")
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(r"Loss and gradient landscapes ($\lambda_{true}=10$, $\tau=0.5$)", y=1.0)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


# --- summary --------------------------------------------------------------
def q(vals, p):
    return float(np.percentile(vals, p)) if len(vals) else float("nan")


def summarize(l1, l2, grid, curves):
    lines = []
    ok = [r for r in l1 if "occ_gap_rel" in r]
    nfail = sum(1 for r in l1 if "fail_exact" in r)
    lines.append(f"Level 1: {len(l1)} points, exact-LP failures: {nfail} "
                 f"({100*nfail/max(len(l1),1):.1f}%)")
    header = f"{'tau':>5} {'rho med':>8} {'occ gap med[IQR]':>22} " \
             f"{'val gap med':>12} {'pol dis med':>12} {'cond_p med':>10} {'cond_e med':>10}"
    for pref in ("random", "trained"):
        sub_all = [r for r in ok if r["source"].startswith(pref)]
        if not sub_all:
            continue
        lines.append(f"-- source: {pref} --")
        lines.append(header)
        for tau in TAUS:
            sub = [r for r in sub_all if r["tau"] == tau]
            if not sub:
                continue
            og = [r["occ_gap_rel"] for r in sub]
            lines.append(
                f"{tau:>5} {q([r['rho'] for r in sub],50):>8.3f} "
                f"{q(og,50):>8.3f} [{q(og,25):.3f},{q(og,75):.3f}] "
                f"{q([r['val_gap_rel'] for r in sub],50):>12.4f} "
                f"{q([r['pol_disagree'] for r in sub],50):>12.3f} "
                f"{q([r['cond_proj'] for r in sub],50):>10.1f} "
                f"{q([r.get('cond_exact',np.nan) for r in sub],50):>10.1f}")
    ok2 = [r for r in l2 if "fail" not in r]
    lines.append(f"\nLevel 2: {len(l2)} points, failures: {len(l2)-len(ok2)}")
    for pref in ("random", "trained"):
        sub = [r for r in ok2 if r["source"].startswith(pref)]
        if not sub:
            continue
        cos = [r["cos_aggparams"] for r in sub]
        sa = np.mean([r["sign_agree"] for r in sub])
        sat = np.mean([r["sign_agree_trainlift"] for r in sub])
        ratio = [r["gnorm_ratio"] for r in sub]
        lines.append(
            f"  {pref}: dL/dlam sign agree {100*sa:.0f}% "
            f"(train-lift {100*sat:.0f}%), cos(agg-param grads) "
            f"med {q(cos,50):.3f} [IQR {q(cos,25):.3f},{q(cos,75):.3f}], "
            f"|g_proj|/|g_exact| med {q(ratio,50):.2f}")
    lines.append("\nLandscape argmin lambda* (true=10):")
    for name, c in curves.items():
        def amin(v):
            return (f"{grid[int(np.nanargmin(v))]:.2f}"
                    if np.isfinite(v).any() else "n/a (all infeasible)")
        mask = np.isfinite(c["g_proj"]) & np.isfinite(c["g_exact"])
        corr = (np.corrcoef(c["g_proj"][mask], c["g_exact"][mask])[0, 1]
                if mask.sum() > 2 else float("nan"))
        nf = int(c.get("n_fail_exact", np.array([0]))[0])
        lines.append(f"  {name}: proj {amin(c['L_proj'])}, exact {amin(c['L_exact'])}, "
                     f"grad corr {corr:.3f}, exact failures {nf}/{len(grid)}")
    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    sources = load_w_sources()
    print(f"[setup] W sources: {[s[0] for s in sources]}")
    lams, prices = sample_instances(N_INSTANCES, seed=0)
    print(f"[setup] lambda range [{lams.min():.2f}, {lams.max():.2f}], "
          f"price range [{prices.min():.2f}, {prices.max():.2f}]")

    l1 = run_level1(sources, lams, prices)
    l2 = run_level2_stats(sources, lams, prices)
    grid, curves = run_landscape(sources)
    probe = probe_failure_statuses(l1, sources, lams, prices)
    print(f"[probe] {probe}")

    (OUT_DIR / "records.json").write_text(json.dumps(
        {"level1": l1, "level2": l2,
         "landscape": {k: {kk: vv.tolist() for kk, vv in v.items()}
                       for k, v in curves.items()},
         "grid": grid.tolist()}, indent=1))
    fig_dose_response(l1, OUT_DIR / "fig_dose_response")
    fig_landscape(grid, curves, OUT_DIR / "fig_landscape")

    summary = summarize(l1, l2, grid, curves) + "\n\n" + probe
    (OUT_DIR / "summary.txt").write_text(summary)
    print("\n" + summary)
    print(f"\n[done] outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
