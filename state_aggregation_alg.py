# --------------------------------------------
"""
Main Body for state-aggregation algorithm
Written by: actionsli Eshragh & actionsli Fattahi & Yuexing Li
Date: 10 July 2024
Updated: 17 July 2024
Updated: 24 July 2024
Updated: 18 January 2025
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
    # initial policy
    if policy is None:
        policy = np.random.default_rng(42).choice(actions, size=len(states), replace=True)

    initial_s = set() if initial_s is None else set(initial_s)
    assert len(initial_s) <= d
    initial_s_iterable = iter(initial_s)
    states_d = []  # cluster set
    candidate_pairs = {}

    for i in range(d):
        next_state = next(initial_s_iterable, -1)  # default -1 indicating no next elements in initial_s
        if next_state == -1:
            if len(candidate_pairs) > 0:
                # select a state pair from the candidate pairs dictionary according to the weights
                # ([0] extracts the element from the one-element list returned by random.choices)
                state_pair = random.Random(i).choices(list(candidate_pairs.keys()), list(candidate_pairs.values()))[0]
                # pick the second element of the state pair as the next state and add it to the cluster set
                next_state = state_pair[1]
            else:
                next_state = np.random.default_rng(i).choice(states, p=gamma)

        states_d.append(next_state)
        # update the candidate pairs dictionary
        state_index = states.tolist().index(next_state)
        action_index = actions.tolist().index(policy[state_index])
        new_pairs = {(next_state, states[j]): trans_probs[state_index, j, action_index]
                     for j in range(len(states)) if states[j] not in states_d and trans_probs[state_index, j, action_index] > 0}
        candidate_pairs.update(new_pairs)
        # delete all pairs whose second componenet is the next state just added to the cluster set
        candidate_pairs = {k: v for k, v in candidate_pairs.items() if k[1] != next_state}

    return states_d


# --------------------------------------------
def state_agg(states, actions, trans_probs, cost_fun, gamma, d, rnd=True, initial_s=None, policy=None):
    start_time = time.time()

    if rnd is True:
        states_d = choose_d_stochastic(states, actions, trans_probs, gamma, d, initial_s, policy)
    else:
        states_d = choose_d_deterministic(states, actions, trans_probs, gamma, d, initial_s, policy)

    n, k = cost_fun.shape
    # cluster_indices = np.where(np.isin(states, states_d))[0].tolist()
    cluster_indices = np.searchsorted(states, states_d).tolist()
    agg_indices = np.setdiff1d(np.arange(n), cluster_indices).tolist()
    states_d.append(-1)  # -1 represents the super state

    # transition probability
    trans_probs_d = np.zeros((d + 1, d + 1, k))
    cluster_probs = trans_probs[np.ix_(cluster_indices, cluster_indices, range(k))]
    trans_probs_d[:d, :d, :] = cluster_probs
    trans_probs_d[:d, -1, :] = 1 - np.sum(cluster_probs, axis=1)
    trans_probs_d[-1, :d, :] = np.mean(trans_probs[np.ix_(agg_indices, cluster_indices, range(k))], axis=0)
    trans_probs_d[-1, -1, :] = 1 - np.sum(trans_probs_d[-1, :d, :], axis=0)

    # cost function
    cost_fun_d = np.vstack((
        cost_fun[cluster_indices, :],
        np.mean(cost_fun[agg_indices, :], axis=0)
    ))

    # initial distribution
    gamma_d = np.append(gamma[cluster_indices], 1- np.sum(gamma[cluster_indices]))

    run_time = time.time() - start_time

    return np.array(states_d), trans_probs_d, cost_fun_d, gamma_d, run_time


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
    # Initialize pi with the action corresponding to the super state, i.e., the last one in pi_d
    pi = np.full(len(states), pi_d[-1])

    # Find the indices where states_d are in states
    mask = np.isin(states_d, states)
    original_states = states_d[mask]
    original_indices = np.searchsorted(states, original_states)

    # Assign the corresponding pi_d values to these positions in `pi`
    pi[original_indices] = pi_d[mask]

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
