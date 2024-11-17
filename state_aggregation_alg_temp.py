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
        policy = {s: random.choice(A) for s in S}

    S_d = []
    if initial_s is None:
        s = random.choices(S, weights = [gamma[s] for s in S], k=1)[0]
    else:
        s = initial_s

    S_d.append(s)
    P_temp = {s: {s_prime: P[s][policy[s]][s_prime]
                 for s_prime in S if P[s][policy[s]][s_prime] > 0 and s_prime not in S_d}}
    for _ in range(d-1):
        max_pair, max_value = find_max_pair(P_temp)
        s = max_pair[1]
        S_d.append(s)
        P_temp[s] = {}
        P_temp[s] = {s_prime: P[s][policy[s]][s_prime]
                     for s_prime in S if P[s][policy[s]][s_prime] > 0 and s_prime not in S_d}
        if not P_temp[s]:  # Checks if P_temp[s] is empty
            del P_temp[s]  #
            # y

        for s_prime in S_d:
            if P_temp.get(s_prime, {}).get(s, 0) > 0:
                del P_temp[s_prime][s]  # Remove the key s from the inner dictionary
                if not P_temp[s_prime]:  # If the inner dictionary is empty
                    del P_temp[s_prime]  # Remove s_prime from P_temp

    return  S_d


def select_s_s_prime(P):
    # Flatten the dictionary into a list of tuples (s, s', P[s][s'])
    flat_list = [(s, s_prime, P[s][s_prime]) for s in P for s_prime in P[s]]

    # Extract s', weights, and create a list of tuples (s, s')
    pairs = [(s, s_prime) for s, s_prime, weight in flat_list]
    weights = [weight for s, s_prime, weight in flat_list]

    # Randomly select a pair (s, s') based on weights
    selected_pair = random.choices(pairs, weights=weights, k=1)[0]

    return selected_pair



def choose_d_stochastic(S, A, P, gamma, d, initial_s=None, policy=None):
    if policy is None:
        policy = {s: random.choice(A) for s in S}

    S_d = []
    if initial_s is None:
        s = random.choices(S, weights = [gamma[s] for s in S], k=1)[0]
    else:
        s = initial_s

    S_d.append(s)
    P_temp = {s: {s_prime: P[s][policy[s]][s_prime]
                 for s_prime in S if P[s][policy[s]][s_prime] > 0 and s_prime not in S_d}}
    for _ in range(d-1):
        selected_pair = select_s_s_prime(P_temp)
        s = selected_pair[1]
        S_d.append(s)
        P_temp[s] = {}
        P_temp[s] = {s_prime: P[s][policy[s]][s_prime]
                     for s_prime in S if P[s][policy[s]][s_prime] > 0 and s_prime not in S_d}
        if not P_temp[s]:  # Checks if P_temp[s] is empty
            del P_temp[s]  #
            # y

        for s_prime in S_d:
            if P_temp.get(s_prime, {}).get(s, 0) > 0:
                del P_temp[s_prime][s]  # Remove the key s from the inner dictionary
                if not P_temp[s_prime]:  # If the inner dictionary is empty
                    del P_temp[s_prime]  # Remove s_prime from P_temp

    return S_d


def state_agg(S, A, P, C, gamma, d, rnd=True, initial_s=None, policy=None):

    start_time = time.time()

    if rnd is True:
        S_d = choose_d_stochastic(S, A, P, gamma, d, initial_s, policy)
    else:
        S_d = choose_d_deterministic(S, A, P, gamma, d, initial_s, policy)

    P_d = {s: {a: {s_prime: P[s][a][s_prime] for s_prime in S_d} for a in A} for s in S_d}
    S_d.append(1000)
    P_d[1000] = {}
    for a in A:
        P_d[1000][a] = {}
        for s_prime in S_d:
            if s_prime != 1000:
                P_d[1000][a][s_prime] = np.mean([P[s][a][s_prime] for s in S if s not in S_d])

    for s in S_d:
        for a in A:
            P_d[s][a][1000] = 1 - np.sum([P_d[s][a][s_prime] for s_prime in S_d if s_prime != 1000])


    C_d = lambda s, a: C(s,a) if s != 1000 else np.mean([C(s, a) for s in S if s not in S_d for a in A])

    gamma_d = {}
    for s in S_d:
        if s != 1000:
            gamma_d[s] =gamma[s]
        else:
            gamma_d[s] = np.sum([gamma[s_prime] for s_prime in S if s_prime not in S_d])

    run_time = time.time() - start_time

    return S_d, P_d, C_d, gamma_d, run_time

def dis_agg_pi(S, S_d, pi_d):
    pi = {}
    for s in S:
        if s in S_d:
            pi[s] = pi_d[s]
        else:
            pi[s] = pi_d[1000]

    return pi




