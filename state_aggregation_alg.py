# --------------------------------------------
"""
Main Body for state-aggregation algorithm
Written by: actionsli Eshragh & actionsli Fattahi & Yuexing Li
Date: 10 July 2024
Updated: 17 July 2024
Updated: 24 July 2024
"""
# --------------------------------------------

import random
import time
import numpy as np
import math

# --------------------------------------------
def find_max_pair(trans_probs):
    max_value = float('-inf')  # statestart with the smallest possible value
    max_pair = (None, None)    # Initialize max_pair as a tuple of Nones

    # Iterate over each key-value pair in the dictionary
    for s in trans_probs:
        for s_prime in trans_probs[s]:
            # If the current value is greater than the max_value, update max_value and max_pair
            if trans_probs[s][s_prime] > max_value:
                max_value = trans_probs[s][s_prime]
                max_pair = (s, s_prime)

    return max_pair, max_value


# --------------------------------------------
def choose_d_deterministic(states, actions, trans_probs, gamma, d, initial_s=None, policy=None):
    if policy is None:
        policy = {s: random.choice(actions) for s in states}

    states_d = []
    if initial_s is None:
        s = random.choices(states, weights = [gamma[s] for s in states], k=1)[0]
    else:
        s = initial_s

    states_d.append(s)
    trans_probs_temp = {s: {s_prime: trans_probs[s][policy[s]][s_prime]
                 for s_prime in states if trans_probs[s][policy[s]][s_prime] > 0 and s_prime not in states_d}}
    for _ in range(d-1):
        max_pair, max_value = find_max_pair(trans_probs_temp)
        s = max_pair[1]
        states_d.append(s)
        trans_probs_temp[s] = {}
        trans_probs_temp[s] = {s_prime: trans_probs[s][policy[s]][s_prime]
                     for s_prime in states if trans_probs[s][policy[s]][s_prime] > 0 and s_prime not in states_d}
        if not trans_probs_temp[s]:  # Checks if trans_probs_temp[s] is empty
            del trans_probs_temp[s]  #
            # y

        for s_prime in states_d:
            if trans_probs_temp.get(s_prime, {}).get(s, 0) > 0:
                del trans_probs_temp[s_prime][s]  # Remove the key s from the inner dictionary
                if not trans_probs_temp[s_prime]:  # If the inner dictionary is empty
                    del trans_probs_temp[s_prime]  # Remove s_prime from trans_probs_temp

    return  states_d


# --------------------------------------------
def select_s_s_prime(trans_probs):
    # Flatten the dictionary into a list of tuples (s, s', trans_probs[s][s'])
    flat_list = [(s, s_prime, trans_probs[s][s_prime]) for s in trans_probs for s_prime in trans_probs[s]]

    # Extract s_prime, weights, and create a list of tuples (s, s_prime)
    pairs = [(s, s_prime) for s, s_prime, weight in flat_list]
    weights = [weight for s, s_prime, weight in flat_list]

    # Randomly select a pair (s, s_prime) based on weights
    selected_pair = random.choices(pairs, weights=weights, k=1)[0]

    return selected_pair


# --------------------------------------------
def choose_d_stochastic(states, actions, trans_probs, gamma, d, initial_s=None, policy=None):
    if policy is None:
        policy = np.random.default_rng(42).choice(actions, size=len(states), replace=True)

    states_d = []
    if initial_s is None:
        len_initial_s = 0
        s = np.random.default_rng(1).choice(states, p=gamma)
        states_d.append(s)
        trans_probs_temp = {s: {s_prime: trans_probs[s][policy[s]][s_prime]
                      for s_prime in states if trans_probs[s][policy[s]][s_prime] > 0 and s_prime not in states_d}}
    else:
        len_initial_s = len(initial_s)
        count = 0
        for s in initial_s:
            count += 1
            states_d.append(s)
            if (count == 1):
                trans_probs_temp = {s: {s_prime: trans_probs[s][policy[s]][s_prime]
                              for s_prime in states if trans_probs[s][policy[s]][s_prime] > 0 and s_prime not in states_d}}
            else:
                trans_probs_temp[s] = {}
                trans_probs_temp[s] = {s_prime: trans_probs[s][policy[s]][s_prime]
                             for s_prime in states if trans_probs[s][policy[s]][s_prime] > 0 and s_prime not in states_d}
                if not trans_probs_temp[s]:  # Checks if trans_probs_temp[s] is empty
                    del trans_probs_temp[s]  #

                for s_prime in states_d:
                    if trans_probs_temp.get(s_prime, {}).get(s, 0) > 0:
                        del trans_probs_temp[s_prime][s]  # Remove the key s from the inner dictionary
                        if not trans_probs_temp[s_prime]:  # If the inner dictionary is empty
                            del trans_probs_temp[s_prime]  # Remove s_prime from trans_probs_temp

    for _ in range(d-len_initial_s):
        selected_pair = select_s_s_prime(trans_probs_temp)
        s = selected_pair[1]
        states_d.append(s)
        trans_probs_temp[s] = {}
        trans_probs_temp[s] = {s_prime: trans_probs[s][policy[s]][s_prime]
                     for s_prime in states if trans_probs[s][policy[s]][s_prime] > 0 and s_prime not in states_d}
        if not trans_probs_temp[s]:  # Checks if trans_probs_temp[s] is empty
            del trans_probs_temp[s]  #
            # y

        for s_prime in states_d:
            if trans_probs_temp.get(s_prime, {}).get(s, 0) > 0:
                del trans_probs_temp[s_prime][s]  # Remove the key s from the inner dictionary
                if not trans_probs_temp[s_prime]:  # If the inner dictionary is empty
                    del trans_probs_temp[s_prime]  # Remove s_prime from trans_probs_temp

    return states_d


# --------------------------------------------
def state_agg(states, actions, trans_probs, cost_fun, gamma, d, rnd=True, initial_s=None, policy=None):

    start_time = time.time()

    if rnd is True:
        states_d = choose_d_stochastic(states, actions, trans_probs, gamma, d, initial_s, policy)
    else:
        states_d = choose_d_deterministic(states, actions, trans_probs, gamma, d, initial_s, policy)

    trans_probs_d = {s: {a: {s_prime: trans_probs[s][a][s_prime] for s_prime in states_d} for a in actions} for s in states_d}
    states_d.append('super_s')
    trans_probs_d['super_s'] = {}
    for a in actions:
        trans_probs_d['super_s'][a] = {}
        for s_prime in states_d:
            if s_prime != 'super_s':
                trans_probs_d['super_s'][a][s_prime] = np.mean([trans_probs[s][a][s_prime] for s in states if s not in states_d])

    for s in states_d:
        for a in actions:
            trans_probs_d[s][a]['super_s'] = 1 - np.sum([trans_probs_d[s][a][s_prime] for s_prime in states_d if s_prime != 'super_s'])


    #C_d = lambda s, a: cost_fun(s,a) if s != 'super_s' else np.mean([cost_fun(s_prime, a) for s_prime in states if s_prime not in states_d])
    C_d = {s: {a: 0 for a in actions} for s in states_d}
    for s in states_d:
        for a in actions:
            C_d[s][a] = cost_fun[s][a] if s != 'super_s' else np.mean([cost_fun[s_prime][a] for s_prime in states if s_prime not in states_d])

    gamma_d = {}
    for s in states_d:
        if s != 'super_s':
            gamma_d[s] =gamma[s]
        else:
            gamma_d[s] = np.sum([gamma[s_prime] for s_prime in states if s_prime not in states_d])

    run_time = time.time() - start_time

    return states_d, trans_probs_d, C_d, gamma_d, run_time


# --------------------------------------------
def state_agg_det(states, actions, trans_probs, cost_fun, gamma, d, states_d):

    start_time = time.time()

    trans_probs_d = {s: {a: {s_prime: trans_probs[s][a][s_prime] for s_prime in states_d} for a in actions} for s in states_d}
    states_d.append('super_s')
    trans_probs_d['super_s'] = {}
    for a in actions:
        trans_probs_d['super_s'][a] = {}
        for s_prime in states_d:
            if s_prime != 'super_s':
                trans_probs_d['super_s'][a][s_prime] = np.mean([trans_probs[s][a][s_prime] for s in states if s not in states_d])

    for s in states_d:
        for a in actions:
            trans_probs_d[s][a]['super_s'] = 1 - np.sum([trans_probs_d[s][a][s_prime] for s_prime in states_d if s_prime != 'super_s'])


    #C_d = lambda s, a: cost_fun(s,a) if s != 'super_s' else np.mean([cost_fun(s_prime, a) for s_prime in states if s_prime not in states_d])
    C_d = {s: {a: 0 for a in actions} for s in states_d}
    for s in states_d:
        for a in actions:
            C_d[s][a] = cost_fun[s][a] if s != 'super_s' else np.mean([cost_fun[s_prime][a] for s_prime in states if s_prime not in states_d])

    gamma_d = {}
    for s in states_d:
        if s != 'super_s':
            gamma_d[s] =gamma[s]
        else:
            gamma_d[s] = np.sum([gamma[s_prime] for s_prime in states if s_prime not in states_d])

    run_time = time.time() - start_time

    return trans_probs_d, C_d, gamma_d, run_time


def dis_agg_pi(states, states_d, pi_d):
    pi = {}
    for s in states:
        if s in states_d:
            pi[s] = pi_d[s]
        else:
            pi[s] = pi_d['super_s']

    return pi


def dis_agg_pi_inventory(states, states_d, pi_d):
    # Initialize pi as an empty dictionary
    pi = {}

    # Create a sorted list of states_d without "super_state"
    states_d_sorted = sorted([int(x) for x in states_d if x != 'super_s'])

    # Fill in pi for elements in states that are also in states_d
    for s in states:
        if s in states_d:
            pi[s] = pi_d[s]
        else:
            # Find the two closest elements a and b in states_d_sorted such that a < s < b
            s_min = max([x for x in states_d_sorted if x < s], default=None)
            s_max = min([x for x in states_d_sorted if x > s], default=None)

            # Handle cases where both a and b exist
            if s_min is not None and s_max is not None:
                pi[s] = math.ceil((pi_d[s_min] + pi_d[s_max]) / 2)
            # Handle edge cases where there is no smaller element (a is None)
            elif s_min is None:
                pi[s] = pi_d[s_max]  # If no a exists, use pi_d[b]
            # Handle edge cases where there is no larger element (b is None)
            elif s_max is None:
                pi[s] = pi_d[s_min]  # If no b exists, use pi_d[a]

    pi = dict(sorted(pi.items()))

    return pi
