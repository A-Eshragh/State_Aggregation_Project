# -------------------------------------------
"""
Value-Iteration Algorithm
Written by: Ali Eshragh & Ali Fattahi
Date: 21 May 2024
Revised by: Yuexing Li
Date: 21 Jan 2025
"""
# -------------------------------------------

import time
import torch
import numpy as np


def value_iteration(mdp, epsilon=1e-6, v_old=None):
    """
    Perform value iteration for a general MDP.

    Args:
        mdp: An instance of an MDP class.
        epsilon: Convergence threshold.
        v_old: Initial value function. If None, initializes to zero.

    Returns:
        policy: Optimal policy for each state.
        v_star: Optimal value for each state.
        run_time: Time taken to compute the result.
    """

    start_time = time.time()

    # Retrieve information from the MDP
    actions = mdp.actions
    trans_probs = mdp.trans_probs
    obj_fun = mdp.obj_fun
    gamma = mdp.gamma
    beta = mdp.beta

    n_samples, n, _, _ = trans_probs.shape
    if v_old is None:
        v_old = torch.zeros(n)  # Initialize value function if not provided

    policy = np.zeros((n_samples, n))
    v_star = np.zeros(n_samples)
    for s in range(n_samples):
        while True:
            v_new, v_new_index = torch.min(obj_fun + beta * torch.sum(trans_probs[s] * v_old.reshape(1, -1, 1), dim=1), dim=1)
            delta = torch.max(torch.abs(v_new - v_old)).item()
            if delta < epsilon:
                break
            else:
                v_old = v_new.clone()

        policy[s] = actions[v_new_index]

        # optimal value
        v_star[s] = torch.sum(gamma * v_new).item()

    run_time = time.time() - start_time

    return policy, v_star, run_time
