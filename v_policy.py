#--------------------------------------------
'''
Finding the value function for a given policy
Written by: Ali Eshragh & Ali Fattahi
Date: 12 June, 2024
'''
#--------------------------------------------

import numpy as np
import time

def val_policy(S, policy, P, C, beta, gamma):
    '''
    This function finds the value function for a given policy.
    '''

    start_time = time.time()

    n = len(S)
    P_pi = np.zeros((n, n))
    c_pi = np.zeros((n, 1))
    for s in S:
        for s_prime in S:
            # Since states range from min(S) to max(S), we need to shift them \
            # by min(S) unit as arrays in Python range from 0
            P_pi[s-min(S), s_prime-min(S)] = P[s][policy[s]][s_prime]
            c_pi[s-min(S),0] = C(s, policy[s])
    coeff_matrix = np.eye(n) - beta*P_pi
    v_pi = np.linalg.solve(coeff_matrix, c_pi)

    # optimal value
    v_star = np.dot(gamma.T, v_pi).item()

    run_time = time.time() - start_time

    return v_pi, v_star, run_time
