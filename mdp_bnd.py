#--------------------------------------------
"""
Main Body for creating the MDP for Birth-and-Death Problem
Written by: Ali Eshragh & Ali Fattahi
Date: 29 May 2024
Revised by: Yuexing Li
Date: 11 Oct 2024
"""
#--------------------------------------------

import numpy as np
import time

def create_mdp_bnd(n, k, c0, c1, pow1, pow2):
    """
    This function takes the number of states n,
    the number of actions k, the two constants c0
    and c1 used in the construction of transition
    probabilities, and the two powers pow1 and pow2
    used in the construction of the cost function to
    generate the MDP of the corresponding birth and
    death process
    """

    start_time = time.time()

    # construct the state space
    states = range(1, n+1)

    # construct the action set
    actions = range(1, k+1)
    
    # construct the transition probability matrix
    trans_probs = np.zeros((n, n, k))
    trans_probs[0, 1, :] = 1
    trans_probs[-1, -1, :] = 1 / (1 + np.exp((c0 + c1 * n) / np.arange(1, k+1)))
    diags = 1 / (1 + np.exp((c0 + c1 * np.arange(2, n).reshape(-1, 1)) / np.arange(1, k+1).reshape(1, -1)))
    trans_probs[np.arange(1, n-1), np.arange(2, n), :] = diags
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
    cost_fun = np.exp((np.arange(1, k+1).reshape(1, -1)) ** pow1 / (np.arange(1, n+1).reshape(-1, 1)) ** pow2)

    # construct initial distribution
    gamma = 1/n * np.ones(n)

    run_time = time.time() - start_time

    return states, actions, trans_probs, cost_fun, gamma, run_time

if __name__ == '__main__':
    create_mdp_bnd(100, 10, 0.1, 0.2, 0.8, 1)
