#--------------------------------------------
"""
Policy-Iteration Algorithm
Written by: Ali Eshragh & Ali Fattahi
Date: 29 May 2024
Revised: 24 July 2024
Revised by: Yuexing Li
Date: 10 Nov 2024
"""
#--------------------------------------------

import numpy as np
import time
import random

def policy_iteration(states, actions, trans_probs, cost_fun, beta, gamma, pi_old=None):
    """
    This function takes the state space S, action set A, the cost function C,\
    the discount factor beta, the initial distribution gamma, and the initial \
    policy pi_old (optional) as inputs to implement the Policy Iteration \
    algorithm for a given discounted MDP.
    """

    start_time = time.time()

    n = len(states)
    if pi_old is None:
        # Initialize the value function randomly
        pi_old = np.random.default_rng(42).choice(actions, size=n, replace=True)

    while True:
        # The first element of np.where is the row indices. We need the column indices
        action_indices = np.where(pi_old[:, np.newaxis] == actions)[1]
        action_indices_broadcast = np.tile(action_indices[:, np.newaxis], (1, n))
        p_pi = np.take_along_axis(trans_probs, action_indices_broadcast[:, :, np.newaxis], axis=2).squeeze(axis=2)
        c_pi = np.take_along_axis(cost_fun, action_indices[:, np.newaxis], axis=1).squeeze(axis=1)
        coeff_matrix = np.eye(n) - beta * p_pi
        v_pi = np.linalg.solve(coeff_matrix, c_pi)

        # policy improvement
        pi_new = np.array(actions)[np.argmin(cost_fun + beta * np.sum(trans_probs * v_pi.reshape(1, -1, 1), axis=1), axis=1)]

        if np.all(pi_old == pi_new):
            break
        else:
            pi_old = pi_new.copy()

    # optimal value
    v_star = np.sum(v_pi * gamma)

    run_time = time.time() - start_time

    return pi_new, v_pi, v_star, run_time
