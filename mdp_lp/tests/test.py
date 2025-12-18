import torch
from mdp_lp.mdp import MDP
from mdp_lp.solvers import solve_lp_dense

# Define a simple MDP: 2 states, 2 actions
beta = 0.9  # discount factor
mdp = MDP(beta)

# Transition probabilities: P[batch, state, action, next_state]
P = torch.tensor([[
    [[1.0, 0.0], [0.0, 1.0]],  # state 0: a0→s0, a1→s1
    [[0.0, 1.0], [1.0, 0.0]],  # state 1: a0→s1, a1→s0
]])

# Rewards: r[batch, state, action]
r = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])

# Convert to LP formulation
H, r_flat, gamma = mdp.lp_formulation(P, r)

# Solve LP
y_star, basis = solve_lp_dense(H[0], r_flat[0], gamma[0])

# Compute optimal value
v_star = (y_star * r_flat[0]).sum()
print(f"Optimal value: {v_star.item():.4f}")