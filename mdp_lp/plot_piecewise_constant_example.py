"""
Piecewise-CONSTANT decision map example (standard DFL).

When the prediction ĉ enters only the LP *cost vector*, the optimal
decision w*(ĉ) is piecewise-constant — it sits at the same LP vertex
until ĉ crosses a basis boundary, then jumps discontinuously.

Contrast: in our MDP LP, λ enters the *constraint matrix* H(λ),
so y*(λ) is piecewise-LINEAR (varies within each basis region).

Usage:
    python -m mdp_lp.plot_piecewise_constant_example
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# ── Simple 2-variable LP ──────────────────────────────────
#   max   ĉ · x₁  +  2 · x₂
#   s.t.  x₁ + x₂  ≤ 10      (labor)
#         2x₁ + x₂ ≤ 16      (material A)
#         x₁ + 2x₂ ≤ 14      (material B)
#         0 ≤ x₁ ≤ 7,   0 ≤ x₂ ≤ 6
#
# Feasible vertices: (0,0), (7,0), (7,2), (6,4), (2,6), (0,6)
#
# As ĉ increases, optimal vertex shifts:
#   ĉ < 1   →  x* = (2, 6)   value = 2ĉ + 12
#   1 < ĉ < 4  →  x* = (6, 4)   value = 6ĉ + 8
#   ĉ > 4   →  x* = (7, 2)   value = 7ĉ + 4
# ───────────────────────────────────────────────────────────

A_ub = np.array([[1, 1], [2, 1], [1, 2], [1, 0], [0, 1]])
b_ub = np.array([10, 16, 14, 7, 6])

CHAT_TRUE = 3.0
c_true = np.array([CHAT_TRUE, 2.0])

# Sweep ĉ
chat_vals = np.linspace(0.1, 6.0, 2000)
x1s, x2s, Ls = [], [], []

for c_hat in chat_vals:
    res = linprog([-c_hat, -2.0], A_ub=A_ub, b_ub=b_ub,
                  bounds=[(0, None), (0, None)], method='highs')
    x1s.append(round(res.x[0], 4))
    x2s.append(round(res.x[1], 4))
    Ls.append(-c_true @ res.x)   # DFL loss = -c_true^T x*(ĉ)

x1s, x2s, Ls = np.array(x1s), np.array(x2s), np.array(Ls)
grad = np.gradient(Ls, chat_vals)

# Boundary locations
bd1, bd2 = 1.0, 4.0

# ── Figure ─────────────────────────────────────────────────
fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, figsize=(11, 11),
    gridspec_kw={'height_ratios': [3, 2, 1.2]})

# ── Panel 1: decision map (piecewise-constant) ────────────
# Color each constant region differently
colors_x1 = ['#2196F3', '#1565C0', '#0D47A1']
colors_x2 = ['#FF9800', '#F57C00', '#E65100']

regions = [(0.1, bd1), (bd1, bd2), (bd2, 6.0)]
for i, (lo, hi) in enumerate(regions):
    mask = (chat_vals >= lo) & (chat_vals <= hi)
    ax1.plot(chat_vals[mask], x1s[mask], lw=3, color=colors_x1[i],
             label='$x_1^*(\\hat c)$' if i == 0 else None)
    ax1.plot(chat_vals[mask], x2s[mask], lw=3, color=colors_x2[i],
             label='$x_2^*(\\hat c)$' if i == 0 else None)

# Mark boundaries and jumps
for bd in [bd1, bd2]:
    ax1.axvline(bd, color='gray', ls=':', lw=1.5, alpha=0.6)

ax1.axvline(CHAT_TRUE, color='red', ls='--', lw=2,
            label=f'$\\hat c_{{true}} = {CHAT_TRUE}$')

# Annotate the jumps
ax1.annotate('', xy=(bd1 + 0.05, 6.0), xytext=(bd1 - 0.05, 6.0),
             arrowprops=dict(arrowstyle='->', color='#E53935', lw=2))
ax1.annotate('', xy=(bd1 + 0.05, 4.0), xytext=(bd1 + 0.05, 6.0),
             arrowprops=dict(arrowstyle='->', color='#E53935', lw=2))
ax1.text(bd1 + 0.15, 5.0, 'jump!', color='#E53935', fontsize=11,
         fontweight='bold', va='center')

ax1.annotate('', xy=(bd2 + 0.05, 4.0), xytext=(bd2 - 0.05, 4.0),
             arrowprops=dict(arrowstyle='->', color='#E53935', lw=2))
ax1.annotate('', xy=(bd2 + 0.05, 2.0), xytext=(bd2 + 0.05, 4.0),
             arrowprops=dict(arrowstyle='->', lw=2, color='#E53935'))
ax1.text(bd2 + 0.15, 3.0, 'jump!', color='#E53935', fontsize=11,
         fontweight='bold', va='center')

# Region labels
ax1.text(0.55, 8.0, 'Basis 1\n$x^*=(2,6)$', ha='center', fontsize=10,
         color='gray', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
ax1.text(2.5, 8.0, 'Basis 2\n$x^*=(6,4)$', ha='center', fontsize=10,
         color='gray', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
ax1.text(5.0, 8.0, 'Basis 3\n$x^*=(7,2)$', ha='center', fontsize=10,
         color='gray', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

ax1.set_ylabel('Decision  $w^*(\\hat c) = [x_1^*, x_2^*]$', fontsize=13)
ax1.set_title(
    'Piecewise-Constant Decision Map  (parameter in LP cost vector only)\n'
    'Small changes in $\\hat c$ near boundaries $\\Rightarrow$ '
    'large jumps in $w^*(\\hat c)$',
    fontsize=13)
ax1.legend(fontsize=11, loc='upper left')
ax1.set_ylim(-0.5, 9)
ax1.grid(True, alpha=0.3)

# ── Panel 2: DFL loss (piecewise-constant) ─────────────────
for i, (lo, hi) in enumerate(regions):
    mask = (chat_vals >= lo) & (chat_vals <= hi)
    color = ['#4CAF50', '#2E7D32', '#1B5E20'][i]
    ax2.plot(chat_vals[mask], Ls[mask], lw=3, color=color)

for bd in [bd1, bd2]:
    ax2.axvline(bd, color='gray', ls=':', lw=1.5, alpha=0.6)

ax2.axvline(CHAT_TRUE, color='red', ls='--', lw=2)

# Mark the true optimum
idx_true = np.argmin(np.abs(chat_vals - CHAT_TRUE))
ax2.plot(CHAT_TRUE, Ls[idx_true], 'r*', markersize=15, zorder=5)

# Annotate loss values
ax2.text(0.55, -18.5, '$L = -18$', ha='center', fontsize=11, color='#4CAF50')
ax2.text(2.5, -26.5, '$L = -26$ (optimal)', ha='center', fontsize=11, color='#2E7D32',
         fontweight='bold')
ax2.text(5.0, -25.5, '$L = -25$', ha='center', fontsize=11, color='#1B5E20')

ax2.set_ylabel('DFL Loss  $L(\\hat c) = -c_{true}^T \\, w^*(\\hat c)$', fontsize=13)
ax2.set_title(
    'DFL Loss is also piecewise-constant\n'
    'Gradient = 0 within each region $\\Rightarrow$ '
    'no learning signal except at boundaries',
    fontsize=13)
ax2.grid(True, alpha=0.3)

# ── Panel 3: gradient (zero everywhere) ────────────────────
for i, (lo, hi) in enumerate(regions):
    mask = (chat_vals >= lo) & (chat_vals <= hi)
    # Don't plot near boundaries (finite-diff artifacts)
    interior = mask & (np.abs(chat_vals - bd1) > 0.05) & (np.abs(chat_vals - bd2) > 0.05)
    ax3.plot(chat_vals[interior], grad[interior], lw=2, color='gray', alpha=0.8)

# Show the boundary spikes
for bd in [bd1, bd2]:
    near_bd = np.abs(chat_vals - bd) < 0.1
    ax3.plot(chat_vals[near_bd], grad[near_bd], lw=1.5, color='#E53935', alpha=0.8)

ax3.axvline(CHAT_TRUE, color='red', ls='--', lw=2)
ax3.axhline(0, color='black', lw=0.5)
for bd in [bd1, bd2]:
    ax3.axvline(bd, color='gray', ls=':', lw=1.5, alpha=0.6)

ax3.set_xlabel('Prediction  $\\hat c$', fontsize=13)
ax3.set_ylabel('$\\partial L / \\partial \\hat c$', fontsize=12)
ax3.set_title('Gradient: exactly zero within each piece, spikes only at boundaries', fontsize=12)
ax3.grid(True, alpha=0.3)

for ax in [ax1, ax2, ax3]:
    ax.set_xlim(0.1, 6.0)

plt.tight_layout()
plt.savefig('mdp_lp/plot_results/piecewise_constant_example.png', dpi=150, bbox_inches='tight')
plt.savefig('mdp_lp/plot_results/piecewise_constant_example.pdf', bbox_inches='tight')
print("Saved to mdp_lp/plot_results/piecewise_constant_example.{png,pdf}")
