# -------------------------------------------
"""
Main Body for creating the MDP for Birth-and-Death Problem
Written by: Ali Eshragh & Ali Fattahi
Date: 29 May 2024
Revised by: Yuexing Li
Date: 18 Jan 2025
"""
# -------------------------------------------

import numpy as np
import time


def create_mdp_bnd(n, k, c0, c1, pow1, pow2):
    """
    This function creates an MDP instance for the birth and death process.

    Args:
        n: the number of states
        k: the number of actions
        c0: feature 1 of the transition probabilities
        c1: feature 2 of the transition probabilities
        pow1: power 1 of the cost function
        pow2: power 2 of the cost function

    Returns:
        states: states of the MDP
        actions: actions of the MDP
        trans_probs: transition probabilities of the MDP
        cost_fun: cost function of the MDP
        gamma: initial distribution over the states
        run_time: running time
    """

    start_time = time.time()

    # construct the state space
    states = np.arange(1, n + 1)

    # construct the action set
    actions = np.arange(1, k + 1)

    # construct the transition probability matrix
    theta = 1 / (1 + np.exp((c0 + c1 * np.arange(2, n + 1).reshape(-1, 1)) / np.arange(1, k + 1).reshape(1, -1)))
    trans_probs = np.zeros((n, n, k))
    trans_probs[0, 1, :] = 1
    trans_probs[np.arange(1, n - 1), np.arange(2, n), :] = theta[:-1, :]
    trans_probs[-1, -1, :] = theta[-1, :]
    trans_probs[:, 0, :] = 1 - np.sum(trans_probs[:, 1:n, :], axis=1)

    # # construct the sparse transition probability matrix
    # import sparse
    # nz_indices = np.hstack((
    #     np.vstack((
    #         np.arange(k),
    #         np.zeros(k),
    #         np.ones(k)
    #     )),
    #     np.vstack((
    #         np.repeat(np.arange(k), n-1),
    #         np.tile(np.arange(1, n), k),
    #         np.zeros(k * (n-1))
    #     )),
    #     np.vstack((
    #         np.repeat(np.arange(k), n-1),
    #         np.tile(np.arange(1, n), k),
    #         np.tile(np.append(np.arange(2, n), n-1), k)
    #     ))
    # )).astype(int)
    # nz_prob = 1 / (1 + np.exp((c0 + c1 * np.arange(2, n+1).reshape(1, -1)) / np.arange(1, k+1).reshape(-1, 1)))
    # nz_values = np.concatenate((
    #     np.ones(k),
    #     1 - nz_prob.reshape(-1),
    #     nz_prob.reshape(-1)
    # ))
    # trans_probs = sparse.COO(nz_indices, nz_values, shape=(k, n, n))

    # construct the cost function
    cost_fun = np.exp((np.arange(1, k + 1).reshape(1, -1)) ** pow1 / (np.arange(1, n + 1).reshape(-1, 1)) ** pow2)

    # construct initial distribution
    gamma = 1 / n * np.ones(n)

    run_time = time.time() - start_time

    return states, actions, trans_probs, cost_fun, gamma, run_time


if __name__ == '__main__':
    create_mdp_bnd(100, 10, 0.1, 0.2, 0.8, 1)
