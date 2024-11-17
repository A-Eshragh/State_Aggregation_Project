# --------------------------------------------
'''
Main Body for state-aggregation algorithm
Written by: Ali Eshragh & Ali Fattahi
Date: 10 July, 2024
Updated: 17 July, 2024
Updated: 24 July, 2024
'''
# --------------------------------------------

import random
import time
import numpy as np

def find_max_pair(P):
    max_value = float('-inf')  # Start with the smallest possible value
    max_pair = (None, None)    # Initialize max_pair as a tuple of Nones

    # Iterate over each key-value pair in the dictionary
    for s in P:
        for s_prime in P[s]:
            # If the current value is greater than the max_value, update max_value and max_pair
            if P[s][s_prime] > max_value:
                max_value = P[s][s_prime]
                max_pair = (s, s_prime)

    return max_pair, max_value

def choose_d_deterministic(S, A, P, gamma, d, initial_s=None, policy=None):
    if policy is None:
        policy = {s: random.choices(A, k=1)[0] for s in S}

    S_d = []
    if initial_s is None:
        s = random.choices(S, weights = list(gamma.values()), k=1)[0]
    else:
        s = initial_s

    S_d.append(s)
    P_temp = {s: {s_prime: P[s][policy[s]][s_prime] for s_prime in S}}
    P_temp[s][s] = 0
    for _ in range(d-1):
        max_pair, max_value = find_max_pair(P_temp)
        s = max_pair[1]
        S_d.append(s)
        P_temp[s] = {}
        for s_prime in S:
            P_temp[s][s_prime] = P[s][policy[s]][s_prime]
        for s_prime in S_d:
            P_temp[s_prime][s] = 0
            P_temp[s][s_prime] = 0

    return  S_d


def state_agg(S, A, P, C, gamma, d, rnd=True, policy=None):

    start_time = time.time()

    n = len(S)
    if rnd is True:
        S_d = choose_d_deterministic(S, A, P, gamma, d, initial_s=None, policy=None)
    else:
        S_d = choose_d_deterministic(S, A, P, gamma, d, initial_s=None, policy=None)

    P_d = {s: {a: {s_prime: P[s][a][s_prime] for s_prime in S_d} for a in A} for s in S_d}
    S_d.append(n+1)
    P_d[n+1] = {}
    for a in A:
        P_d[n+1][a] = {}
        for s_prime in S_d:
            if s_prime != n+1:
                P_d[n+1][a][s_prime] = np.mean([P[s][a][s_prime] for s in S if s not in S_d])

    for s in S_d:
        for a in A:
            P_d[s][a][n+1] = 1 - np.sum([P_d[s][a][s_prime] for s_prime in S_d if s_prime != n+1])


    C_d = lambda s, a: C(s,a) if s != n+1 else np.mean([C(s, a) for s in S if s not in S_d for a in A])

    gamma_d = {}
    for s in S_d:
        if s != n+1:
            gamma_d[s] =gamma[s]
        else:
            gamma_d[s] = np.sum([gamma[s_prime] for s_prime in S if s_prime not in S_d])

    run_time = time.time() - start_time

    return S_d, P_d, C_d, gamma_d, run_time

def dis_agg_pi(S, S_d, pi_d):
    pi = {}
    n = len(S)
    for s in S:
        if s in S_d:
            pi[s] = pi_d[s]
        else:
            pi[s] = pi_d[n+1]

    return pi




