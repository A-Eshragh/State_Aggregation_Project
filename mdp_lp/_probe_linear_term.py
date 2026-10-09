"""Quick probe: does the linear dual term (v*)^T(H_true y - gamma) cancel the
DFL slope at lambda_true? v* = dual of the TRUE LP (value function for H y = gamma).
Uses scipy/HiGHS (Gurobi license expired)."""
import torch
import numpy as np
from scipy.optimize import linprog
from mdp_lp.inventory_elasticity import create_inventory_from_lambda

B_MAX, S_MAX, Q_MAX, D_MAX = 5, 50, 30, 30
FOC, VOC, HC, BC, BETA, PRICE = 0, 0.5, 0.1, 0, 0.95, 10.0
LAMBDA_TRUE = 10.0


def build(lam_val):
    lam = torch.tensor([lam_val], dtype=torch.float32)
    p = torch.tensor([PRICE], dtype=torch.float32)
    mdp = create_inventory_from_lambda(lam, B_MAX, S_MAX, Q_MAX, FOC, VOC, HC, BC, BETA, p)
    S = mdp.P.shape[1]
    gamma = mdp.build_gamma(1, S, torch.device("cpu"))
    gamma = torch.clamp(gamma, min=1e-8)
    gamma = gamma / gamma.sum(dim=1, keepdim=True)
    H = mdp.build_H(mdp.P)
    r = mdp.r.reshape(1, -1)
    return (H[0].detach().numpy().astype(np.float64),
            r[0].detach().numpy().astype(np.float64),
            gamma[0].detach().numpy().astype(np.float64))


def solve_lp(H, r, gamma):
    """max r^T y s.t. H y = gamma, y>=0.  Returns y*, v* (dual for H y = gamma)."""
    res = linprog(c=-r, A_eq=H, b_eq=gamma, bounds=(0, None), method="highs")
    assert res.success, res.message
    y = res.x
    # eqlin.marginals: dual for min (-r)^T y.  Flip sign so v* is the value
    # function of the MAX problem: gamma^T v* = r^T y*, and H^T v* - r >= 0.
    v = -np.asarray(res.eqlin.marginals)
    return y, v


H_true, r_true, gamma_true = build(LAMBDA_TRUE)
y_true, v_star = solve_lp(H_true, r_true, gamma_true)

prim = r_true @ y_true
dual_obj = gamma_true @ v_star
rc = H_true.T @ v_star - r_true          # reduced cost H^T v* - r  (>= 0 for max)
print(f"strong duality:  r^T y* = {prim:.6f}   gamma^T v* = {dual_obj:.6f}   diff = {prim-dual_obj:.2e}")
print(f"reduced cost H^T v* - r:  min = {rc.min():.3e}  (want >= ~0)   max = {rc.max():.3e}")

h = 1e-2
dfl, lin, pen = {}, {}, {}
for lam in [LAMBDA_TRUE - h, LAMBDA_TRUE, LAMBDA_TRUE + h]:
    H, r, gamma = build(lam)
    y, _ = solve_lp(H, r, gamma)
    c = H_true @ y - gamma_true          # H_true y*(lam) - gamma
    dfl[lam] = -(r_true @ y)
    lin[lam] = v_star @ c
    pen[lam] = c @ c

a_dfl = (dfl[LAMBDA_TRUE + h] - dfl[LAMBDA_TRUE - h]) / (2 * h)
a_lin = (lin[LAMBDA_TRUE + h] - lin[LAMBDA_TRUE - h]) / (2 * h)
a_pen = (pen[LAMBDA_TRUE + h] - pen[LAMBDA_TRUE - h]) / (2 * h)
print(f"\nslope at lambda_true:")
print(f"  d/dl [-r^T y]     a_dfl = {a_dfl:+.4f}")
print(f"  d/dl [v*^T c]     a_lin = {a_lin:+.4f}")
print(f"  d/dl [||c||^2]    a_pen = {a_pen:+.4f}  (~0)")
print(f"  a_dfl + a_lin           = {a_dfl + a_lin:+.4f}   <-- ~0 if linear term removes the shift")
print(f"linear-term value at lambda_true: {lin[LAMBDA_TRUE]:.3e}  (~0)")
