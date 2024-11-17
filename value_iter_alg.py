#--------------------------------------------
"""
Value-Iteration Algorithm
Written by: Ali Eshragh & Ali Fattahi
Date: 21 May 2024
Revised by: Yuexing Li
Date: 21 Oct 2024
"""
#--------------------------------------------

import numpy as np
import time

def value_iteration(states, actions, trans_probs, cost_fun, beta, gamma, epsilon, v_old=None):
    """
    This function takes the state space S, action set A,
    the reward function R, the discount factor beta, and
    the error bound for the stopping criterion epsilon as
    inputs to implement the Value Iteration algorithm for
    a given discounted MDP.
    """

    start_time = time.time()

    if v_old is None:
        v_old = np.zeros(len(states))  # Initialize value function

    while True:
        v_new = np.min(cost_fun + beta * np.sum(trans_probs * v_old.reshape(1, -1, 1), axis=1), axis=1)
        delta = np.max(np.abs(v_new - v_old))
        if delta < epsilon:
            break
        else:
            v_old = v_new.copy()

    policy = np.array(actions)[np.argmin(cost_fun + beta * np.sum(trans_probs * v_new.reshape(1, -1, 1), axis=1), axis=1)]

    # optimal value
    v_star = np.sum(gamma * v_new)

    run_time = time.time() - start_time

    return policy, v_new, v_star, run_time
