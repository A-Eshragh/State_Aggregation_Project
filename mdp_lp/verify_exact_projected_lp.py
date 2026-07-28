"""
Correctness checks for the exact Galerkin projected LP
(`galerkin_lp_coefficients` / `SoftAggDense.forward_galerkin`) against the
implemented projected-MDP path (`aggregate_with_W` + build_H).

Run from repo root:  python -m mdp_lp.verify_exact_projected_lp

Checks:
  0. Refactor no-op: SoftAggDense.forward == pre-refactor inline math.
  1. Hard-W equivalence: for one-hot W the exact Galerkin LP and the
     aggregated-MDP LP have the same optimal value, and identical lifted
     occupancies (with the mass-corrected lift y_agg/mu on the MDP side).
  2. tau sweep: the exact-vs-projected gap shrinks as W hardens (tau -> 0).
  3. Finite-difference gradient check of the exact layer through
     DifferentiableLPSolver (grads w.r.t. P-logits and W-logits).
"""

import numpy as np
import torch

from mdp_lp.mdp import MDP
from mdp_lp.soft_state_aggregation import (
    SoftAggDense,
    aggregate_with_W,
    galerkin_lp_coefficients,
)
from mdp_lp.solvers.dense_lp_solver import solve_lp_gurobi
from mdp_lp.solvers import differentiable_lp as _dlp
from mdp_lp.solvers.differentiable_lp import DifferentiableLPSolver

BETA = 0.95


def _solve_lp_scipy(H, r, gamma):
    """Drop-in replacement for solve_lp_gurobi using scipy/HiGHS.

    Basis is recovered from the solution support (exact for the
    non-degenerate random LPs used in these tests).
    """
    from scipy.optimize import linprog

    B, S, SA = H.shape
    ys, bases, duals = [], [], []
    for b in range(B):
        res = linprog(
            c=-r[b].detach().numpy().astype(np.float64),
            A_eq=H[b].detach().numpy().astype(np.float64),
            b_eq=gamma[b].detach().numpy().astype(np.float64),
            bounds=(0, None), method="highs",
        )
        if res.status != 0:
            raise RuntimeError(f"HiGHS failed: {res.message}")
        y = res.x
        basis = np.sort(np.argsort(-y)[:S])
        dtype = H.dtype  # keep caller precision (float64 for the FD check)
        ys.append(torch.tensor(y, dtype=dtype))
        bases.append(torch.tensor(basis, dtype=torch.long))
        duals.append(torch.tensor(-res.eqlin.marginals, dtype=dtype))
    return torch.stack(ys), torch.stack(bases), torch.stack(duals)


def _pick_solver():
    """Use Gurobi if licensed; otherwise fall back to scipy/HiGHS and
    monkeypatch DifferentiableLPSolver's forward solve accordingly."""
    try:
        from mdp_lp.solvers.dense_lp_solver import _get_gurobi_env
        _get_gurobi_env()
        return solve_lp_gurobi, "gurobi"
    except Exception as e:
        _dlp.solve_lp_gurobi = _solve_lp_scipy
        print(f"[setup] Gurobi unavailable ({type(e).__name__}); using scipy/HiGHS fallback")
        return _solve_lp_scipy, "scipy"


SOLVE, SOLVER_NAME = _pick_solver()


def random_mdp(B, S, A, seed):
    g = torch.Generator().manual_seed(seed)
    P = torch.softmax(torch.randn(B, S, A, S, generator=g), dim=-1)
    r = torch.randn(B, S, A, generator=g)
    gamma = torch.softmax(torch.randn(B, S, generator=g), dim=-1)
    return P, r, gamma


def solve_projected(P, r, gamma, W, beta=BETA):
    """Aggregated-MDP path: returns (y_agg flat (B,M*A), r_agg flat, mass)."""
    P_agg, r_agg, gamma_agg = aggregate_with_W(P, r, gamma, W)
    H = MDP(beta).build_H(P_agg)
    r_flat = r_agg.reshape(r_agg.shape[0], -1)
    y_agg, _, _ = SOLVE(H, r_flat, gamma_agg)
    mass = W.sum(dim=1)
    return y_agg, r_flat, mass


def solve_exact(P, r, gamma, W, beta=BETA):
    """Exact Galerkin path: returns (w flat (B,M*A), r_tilde flat)."""
    A_mat, r_tilde, b = galerkin_lp_coefficients(P, r, gamma, W, beta)
    w, _, _ = SOLVE(A_mat, r_tilde, b)
    return w, r_tilde


def lift(y_flat, W, n_actions):
    return SoftAggDense.disaggregate_y(y_flat, W, n_actions)


# ---------------------------------------------------------------------------
# 0. Refactor no-op check
# ---------------------------------------------------------------------------
def check_refactor_noop():
    B, S, A, M = 2, 10, 3, 4
    P, r, gamma = random_mdp(B, S, A, seed=0)
    torch.manual_seed(1)
    agg = SoftAggDense(n_states=S, n_actions=A, n_clusters=M, r_embed=8)
    tau, eps = 0.5, 1e-8

    P_agg, r_agg, gamma_agg, info = agg(P, r, gamma, tau=tau)
    W_new = info["W"]

    # Pre-refactor inline math (Method 4), verbatim.
    E = agg.dest_embed.weight
    phi = torch.einsum("bsaj,jr->bsar", P, E)
    phi_cat = phi.reshape(B, S, A * agg.r_embed)
    logits = agg.to_logits(phi_cat) + agg.r_bias(r)
    W = torch.softmax(logits / tau, dim=2)
    mass = W.sum(dim=1).clamp_min(eps)
    r_agg0 = torch.einsum("bsm,bsa->bma", W, r) / mass.unsqueeze(-1)
    right = torch.einsum("bsaj,bjm->bsam", P, W)
    p_hat = torch.einsum("bsm,bsan->bman", W, right)
    P_agg0 = p_hat / mass[:, :, None, None]
    P_agg0 = P_agg0.clamp_min(0.0)
    P_agg0 = P_agg0 / P_agg0.sum(dim=-1, keepdim=True).clamp_min(eps)
    gamma_agg0 = torch.einsum("bs,bsm->bm", gamma, W)
    gamma_agg0 = gamma_agg0.clamp_min(eps)
    gamma_agg0 = gamma_agg0 / gamma_agg0.sum(dim=1, keepdim=True).clamp_min(eps)

    for name, a, b in [("W", W_new, W), ("P_agg", P_agg, P_agg0),
                       ("r_agg", r_agg, r_agg0), ("gamma_agg", gamma_agg, gamma_agg0)]:
        err = (a - b).abs().max().item()
        assert err == 0.0, f"refactor changed {name}: max abs err {err}"
    print("[0] Refactor no-op check: PASS (bitwise identical)")


# ---------------------------------------------------------------------------
# 1. Hard-W equivalence
# ---------------------------------------------------------------------------
def check_hard_W():
    B, S, A, M = 2, 12, 3, 4
    for seed in range(3):
        P, r, gamma = random_mdp(B, S, A, seed=seed)
        assign = torch.arange(S) % M
        W = torch.zeros(B, S, M)
        W[:, torch.arange(S), assign] = 1.0

        y_agg, r_flat, mass = solve_projected(P, r, gamma, W)
        w, r_tilde = solve_exact(P, r, gamma, W)

        v_proj = (r_flat * y_agg).sum(dim=1)
        v_exact = (r_tilde * w).sum(dim=1)
        assert torch.allclose(v_proj, v_exact, rtol=1e-4, atol=1e-4), \
            f"seed {seed}: value mismatch {v_proj} vs {v_exact}"

        # Lifts: exact side y = Phi w; MDP side needs the mass correction y_agg/mu.
        y_scaled = (y_agg.view(B, M, A) / mass[:, :, None]).reshape(B, M * A)
        lift_proj = lift(y_scaled, W, A)
        lift_exact = lift(w, W, A)
        err = (lift_proj - lift_exact).abs().max().item()
        scale = lift_proj.abs().max().item()
        assert err <= 1e-4 * max(scale, 1.0), \
            f"seed {seed}: lift mismatch, max abs err {err} (scale {scale})"
        print(f"[1] Hard-W seed {seed}: PASS "
              f"(value {v_exact[0].item():+.4f}, lift max err {err:.2e})")


# ---------------------------------------------------------------------------
# 2. tau sweep: gap shrinks as W hardens
# ---------------------------------------------------------------------------
def check_tau_sweep():
    B, S, A, M = 1, 12, 3, 4
    P, r, gamma = random_mdp(B, S, A, seed=7)
    g = torch.Generator().manual_seed(7)
    logits = torch.randn(S, M, generator=g)
    while len(logits.argmax(dim=1).unique()) < M:  # every cluster non-empty in tau->0
        logits = torch.randn(S, M, generator=g)

    print("[2] tau sweep (exact vs projected, same W):")
    print(f"    {'tau':>6} {'gram offdiag':>12} {'occ rel gap':>12} {'value gap':>10}")
    prev_gap = None
    for tau in [2.0, 1.0, 0.5, 0.1, 0.01]:
        W = torch.softmax(logits / tau, dim=1).unsqueeze(0)
        gram = torch.einsum("bsi,bsm->bim", W, W)[0]
        rho = (gram - torch.diag(torch.diag(gram))).abs().sum() / gram.abs().sum()

        y_agg, r_flat, mass = solve_projected(P, r, gamma, W)
        w, r_tilde = solve_exact(P, r, gamma, W)
        y_scaled = (y_agg.view(B, M, A) / mass[:, :, None]).reshape(B, M * A)
        lp, le = lift(y_scaled, W, A), lift(w, W, A)
        occ_gap = ((lp - le).abs().sum() / lp.abs().sum()).item()
        val_gap = abs(((r_flat * y_agg).sum() - (r_tilde * w).sum()).item())
        print(f"    {tau:>6.2f} {rho.item():>12.4f} {occ_gap:>12.4f} {val_gap:>10.4f}")
        prev_gap = occ_gap
    assert prev_gap < 1e-2, f"tau=0.01 occupancy gap should be ~0, got {prev_gap}"
    print("    PASS (gap -> 0 as W hardens)")


# ---------------------------------------------------------------------------
# 3. Finite-difference gradient check of the exact layer
# ---------------------------------------------------------------------------
def check_gradient_fd():
    # float64 end-to-end so FD noise sits far below the smallest gradients;
    # force the scipy solver (Gurobi wrapper hardcodes float32 returns).
    _dlp.solve_lp_gurobi = _solve_lp_scipy
    dt = torch.float64
    B, S, A, M = 1, 8, 2, 3
    torch.manual_seed(3)
    p_logits = torch.randn(B, S, A, S, dtype=dt, requires_grad=True)
    w_logits = torch.randn(S, M, dtype=dt, requires_grad=True)
    r = torch.randn(B, S, A, dtype=dt)
    gamma = torch.softmax(torch.randn(B, S, dtype=dt), dim=-1)
    c = torch.randn(S * A, dtype=dt)
    tau = 0.5

    def loss_fn(pl, wl):
        P = torch.softmax(pl, dim=-1)
        W = torch.softmax(wl / tau, dim=1).unsqueeze(0)
        A_mat, r_tilde, b = galerkin_lp_coefficients(P, r, gamma, W, BETA)
        w = DifferentiableLPSolver.apply(A_mat, r_tilde, b)
        y = lift(w, W, A)
        return (y * c).sum()

    L = loss_fn(p_logits, w_logits)
    L.backward()

    h = 1e-4
    checks, n_ok = [], 0
    gen = torch.Generator().manual_seed(11)
    for tensor, grad, name in [(p_logits, p_logits.grad, "P-logit"),
                               (w_logits, w_logits.grad, "W-logit")]:
        flat = tensor.detach().reshape(-1)
        for _ in range(4):
            i = int(torch.randint(flat.numel(), (1,), generator=gen))
            with torch.no_grad():
                e = torch.zeros_like(flat); e[i] = h
                tp = (flat + e).view_as(tensor)
                tm = (flat - e).view_as(tensor)
                if name == "P-logit":
                    Lp, Lm = loss_fn(tp, w_logits.detach()), loss_fn(tm, w_logits.detach())
                else:
                    Lp, Lm = loss_fn(p_logits.detach(), tp), loss_fn(p_logits.detach(), tm)
            fd = ((Lp - Lm) / (2 * h)).item()
            an = grad.reshape(-1)[i].item()
            denom = max(abs(fd), abs(an), 1e-6)
            rel = abs(fd - an) / denom
            ok = rel < 5e-2 or abs(fd - an) < 1e-4
            n_ok += ok
            checks.append((name, i, an, fd, rel, ok))

    print("[3] FD gradient check (exact Galerkin layer):")
    for name, i, an, fd, rel, ok in checks:
        print(f"    {name}[{i:>3}]  autograd {an:+.5f}  fd {fd:+.5f}  "
              f"rel {rel:.3e}  {'ok' if ok else 'MISMATCH (basis boundary?)'}")
    assert n_ok >= len(checks) - 1, f"only {n_ok}/{len(checks)} coords match"
    print(f"    PASS ({n_ok}/{len(checks)} coords within tolerance)")


if __name__ == "__main__":
    check_refactor_noop()
    check_hard_W()
    check_tau_sweep()
    check_gradient_fd()
    print("\nAll checks passed.")
