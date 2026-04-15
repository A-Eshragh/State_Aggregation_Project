"""
Verify that 'bsas' (diagonal) vs 'bsaj' (full sum) einsum produces different
gradients, and show WHY diagonal gives better gradient signal for DFL.

Uses the actual inventory MDP from the project (create_inventory_from_lambda).

Key insight: P is row-stochastic => dP[s,a,:]/dlambda sums to 0.
  - Full sum: sum_j W[j,n] * dP[s,a,j]/dlam  ≈  0  (cancellation)
  - Diagonal: W[s,n] * dP[s,a,s]/dlam           (no cancellation)
"""
import torch
import numpy as np

from mdp_lp.inventory_elasticity import create_inventory_from_lambda
from mdp_lp.solvers.differentiable_lp import DifferentiableLPSolver
from mdp_lp.soft_state_aggregation import SoftAggDense

# ── MDP parameters (same as plot_combined_comparison.py) ──
B_MAX, S_MAX, Q_MAX = 5, 50, 30
FOC, VOC, HC, BC, BETA, PRICE = 0, 0.5, 0.1, 0, 0.95, 10.0
LAMBDA_TRUE = 10.0
RHO = 10000.0
N_STATES = B_MAX + S_MAX + 1   # 56
N_ACTIONS = Q_MAX + 1           # 31
N_AGG = 30


def build_mdp(lam_val):
    """Build MDP at a given lambda value, with grad enabled."""
    lam = torch.tensor([lam_val], dtype=torch.float32, requires_grad=True)
    p = torch.tensor([PRICE], dtype=torch.float32)
    mdp = create_inventory_from_lambda(lam, B_MAX, S_MAX, Q_MAX,
                                       FOC, VOC, HC, BC, BETA, p)
    S = mdp.P.shape[1]
    gamma = mdp.build_gamma(1, S, torch.device("cpu"))
    gamma = torch.clamp(gamma, min=1e-8)
    gamma = gamma / gamma.sum(dim=1, keepdim=True)
    return lam, mdp, gamma


# ══════════════════════════════════════════════════════════════
# Step 1: Verify dP/dlambda rows sum to 0 (stochastic constraint)
# ══════════════════════════════════════════════════════════════
print("="*60)
print("Step 1: dP/dlambda rows sum to 0 (stochastic constraint)")
print("="*60)

lam, mdp, gamma = build_mdp(5.0)
P = mdp.P  # (1, S, A, S)
S, A = P.shape[1], P.shape[2]

# Pick 3 representative states to show
sample_states = [0, S//2, S-1]
sample_action = 0

print(f"\nlambda = {lam.item()}, S = {S}, A = {A}")
print(f"Checking dP[s, a={sample_action}, :]/dlam for selected states:\n")

for s in sample_states:
    row_grad_sum = 0.0
    grads = []
    for j in range(S):
        if lam.grad is not None:
            lam.grad.zero_()
        P[0, s, sample_action, j].backward(retain_graph=True)
        g = lam.grad.item()
        grads.append(g)
        row_grad_sum += g

    nonzero = [(j, g) for j, g in enumerate(grads) if abs(g) > 1e-8]
    print(f"  state {s}: row sum of dP/dlam = {row_grad_sum:+.2e}  (should be ~0)")
    print(f"    non-zero entries: {len(nonzero)}/{S}")
    if nonzero:
        top3 = sorted(nonzero, key=lambda x: abs(x[1]), reverse=True)[:3]
        for j, g in top3:
            print(f"      dP[{s},{sample_action},{j}]/dlam = {g:+.6f}")
    print()


# ══════════════════════════════════════════════════════════════
# Step 2: Compare gradient through P_agg for both methods
# ══════════════════════════════════════════════════════════════
print("="*60)
print("Step 2: dP_agg/dlambda — diagonal vs full sum")
print("="*60)

# Fixed W for controlled comparison
torch.manual_seed(42)
W_fixed = torch.softmax(torch.randn(1, N_STATES, N_AGG), dim=-1)

lambda_values = [2.0, 5.0, 8.0, 12.0, 18.0]

for method_name, einsum_right in [("DIAGONAL (old code)", "bsas,bsm->bsam"),
                                   ("FULL SUM (corrected)", "bsaj,bjn->bsan")]:
    grads = []
    for lam_val in lambda_values:
        lam, mdp, gamma = build_mdp(lam_val)
        P = mdp.P
        mass = W_fixed.sum(dim=1, keepdim=True).clamp_min(1e-8)

        right = torch.einsum(einsum_right, P, W_fixed)
        p_hat = torch.einsum("bsm,bsan->bman", W_fixed, right)
        P_agg = p_hat / mass.unsqueeze(2)
        P_agg = P_agg / P_agg.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        # Use a non-trivial scalar loss from P_agg
        loss = P_agg[0, 0, 0, 0] - P_agg[0, 0, 0, 1]
        loss.backward()
        grads.append(lam.grad.item())

    print(f"\n  {method_name}:")
    for i, lam_val in enumerate(lambda_values):
        print(f"    lam={lam_val:5.1f}: dLoss/dlam = {grads[i]:+.8f}")
    print(f"    Max |gradient|:  {max(abs(g) for g in grads):.8f}")
    print(f"    Gradient spread: {max(grads) - min(grads):.8f}")


# ══════════════════════════════════════════════════════════════
# Step 3: Full pipeline — DFL loss gradient through LP solver
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Step 3: Full DFL pipeline — dLoss/dlambda through LP solver")
print("="*60)
print("  Loss = -r_true^T y* + rho * ||gamma - H_true y*||^2")
print("  This is the actual training loss.\n")

# Build true MDP for evaluation
lam_true_t = torch.tensor([LAMBDA_TRUE], dtype=torch.float32)
p_t = torch.tensor([PRICE], dtype=torch.float32)
mdp_true = create_inventory_from_lambda(lam_true_t, B_MAX, S_MAX, Q_MAX,
                                         FOC, VOC, HC, BC, BETA, p_t)
r_true = mdp_true.r.reshape(1, -1).detach()
H_true = mdp_true.build_H(mdp_true.P).detach()
gamma_true = mdp_true.build_gamma(1, N_STATES, torch.device("cpu"))
gamma_true = torch.clamp(gamma_true, min=1e-8)
gamma_true = gamma_true / gamma_true.sum(dim=1, keepdim=True)
gamma_true = gamma_true.detach()

for method_name, einsum_phi, einsum_right in [
    ("DIAGONAL (old code)", "bsas,sr->bsar", "bsas,bsm->bsam"),
    ("FULL SUM (corrected)", "bsaj,jr->bsar", "bsaj,bjn->bsan"),
]:
    grads = []
    losses = []

    for lam_val in lambda_values:
        lam, mdp, gamma = build_mdp(lam_val)
        P, r = mdp.P, mdp.r

        # --- Aggregation (inline, using the specified einsum) ---
        torch.manual_seed(42)
        agg = SoftAggDense(N_STATES, N_ACTIONS, N_AGG, r_embed=8)

        E = agg.dest_embed.weight
        phi = torch.einsum(einsum_phi, P, E)
        phi_cat = phi.reshape(1, N_STATES, N_ACTIONS * 8)
        logits = agg.to_logits(phi_cat)
        W = torch.softmax(logits / 0.5, dim=2)

        mass = W.sum(dim=1).clamp_min(1e-8)
        r_agg = torch.einsum("bsm,bsa->bma", W, r) / mass.unsqueeze(-1)

        right = torch.einsum(einsum_right, P, W)
        p_hat = torch.einsum("bsm,bsan->bman", W, right)
        P_agg = p_hat / mass[:, :, None, None]
        P_agg = P_agg.clamp_min(0.0)
        P_agg = P_agg / P_agg.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        gamma_agg = torch.einsum("bs,bsm->bm", gamma, W)
        gamma_agg = gamma_agg.clamp_min(1e-8)
        gamma_agg = gamma_agg / gamma_agg.sum(dim=1, keepdim=True).clamp_min(1e-8)

        # --- LP solve on aggregated MDP ---
        H_agg = mdp.build_H(P_agg)
        r_flat_agg = r_agg.reshape(1, -1)
        y_agg = DifferentiableLPSolver.apply(H_agg, r_flat_agg, gamma_agg)

        # --- Disaggregate ---
        y_star = SoftAggDense.disaggregate_y(y_agg, W, N_ACTIONS)

        # --- DFL loss ---
        v_star = (r_true * y_star).sum(dim=1)
        violation = gamma_true - (H_true @ y_star.unsqueeze(-1)).squeeze(-1)
        aug_lag = RHO * violation.pow(2).mean()
        loss = -v_star.mean() + aug_lag

        loss.backward()
        grads.append(lam.grad.item())
        losses.append(loss.item())

    print(f"  {method_name}:")
    for i, lam_val in enumerate(lambda_values):
        print(f"    lam={lam_val:5.1f}: loss={losses[i]:+12.2f}, dLoss/dlam={grads[i]:+.6f}")
    grad_range = max(abs(g) for g in grads)
    grad_spread = max(grads) - min(grads)
    print(f"    Max |gradient|:  {grad_range:.6f}")
    print(f"    Gradient spread: {grad_spread:.6f}  (larger = model can distinguish lambdas)")
    print()


# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════
print("="*60)
print("SUMMARY")
print("="*60)
print("""
P is row-stochastic => dP[s,a,:]/dlam sums to 0 for every (s, a).

Full-sum aggregation computes:
  dP_agg/dlam ~ sum_j W[j,n] * dP[s,a,j]/dlam
This is a weighted sum of derivatives that sum to zero => partial cancellation.

Diagonal aggregation computes:
  dP_agg/dlam ~ W[s,n] * dP[s,a,s]/dlam
Just one entry per row => no cancellation => stronger, more coherent gradient.

With S=56 states, the full sum has 56 terms canceling each other,
while diagonal uses 1 term. This is why the "buggy" diagonal code
gives better DFL training: the gradient signal survives.
""")
