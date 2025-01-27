# -------------------------------------------
"""
Policy-Iteration Algorithm
Written by: Ali Eshragh & Ali Fattahi
Date: 29 May 2024
Revised: 24 July 2024
Revised by: Yuexing Li
Date: 22 Jan 2025
"""
# -------------------------------------------

import time
import torch
import numpy as np


def policy_iteration(mdp, pi_old=None):
    """
    Perform policy iteration for a general MDP.

    Args:
        mdp: An instance of an MDP class.
        pi_old: Initial policy.

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

    n_samples, n, _, k = trans_probs.shape
    if pi_old is None:
        pi_old = actions[torch.randint(0, k, (n,))]  # Initialize the policy randomly if not provided

    policy = np.zeros((n_samples, n))
    v_star = np.zeros(n_samples)
    for s in range(n_samples):
        while True:
            indices = torch.tensor([mdp.action_indices[action] for action in pi_old])
            p_pi = torch.gather(trans_probs[s], dim=2, index=indices[:, None, None].expand(n, n, 1)).squeeze(2)
            c_pi = torch.gather(obj_fun, dim=1, index=indices[:, None]).squeeze(1)
            coeff_matrix = torch.eye(n) - beta * p_pi
            v_pi = torch.linalg.solve(coeff_matrix, c_pi)

            # policy improvement
            _, v_pi_index = torch.min(obj_fun + beta * torch.sum(trans_probs[s] * v_pi.reshape(1, -1, 1), dim=1), dim=1)
            pi_new = actions[v_pi_index]

            if np.all(pi_old == pi_new):
                break
            else:
                pi_old = pi_new.copy()

        policy[s] = pi_new

        # optimal value
        v_star[s] = torch.sum(gamma * v_pi).item()

    run_time = time.time() - start_time

    return policy, v_star, run_time
