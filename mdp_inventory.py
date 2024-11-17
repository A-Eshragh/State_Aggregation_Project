#--------------------------------------------
"""
Main Body for creating the MDP for Birth-and-Death Problem
Written by: Ali Eshragh & Ali Fattahi
Date: 12 June 2024
"""
#--------------------------------------------

import math
import time

import numpy as np
from scipy.stats import poisson

#--------------------------------------------
def indicator_fn(a):
    if a > 0:
        return 1
    else:
        return 0


#--------------------------------------------
def create_mdp_inventory(lambda_poisson, q, c1, c2, oc_f, oc_v, hc, bc, beta):
    '''
    This function takes the parameters of the demand Poisson distribution \
     lambda_poisson, the maximum ordering capacity q, maximum backlog c1, \
     maximum storage capacity c2, fixed ordering cost oc_f, variable ordering cost oc_v, holding cost hc, backlog \
     cost bc, and the discount factor to construct the MDP for an inventory \
     model.
    '''

    start_time = time.time()

    # construct the state space
    S = [i for i in range(-c1, c2 + 1)]

    # construct the action set
    A = [i for i in range(q + 1)]
    
    # construct the transition probability matrix
    P = {s: {a: {s_prime: 0 for s_prime in S} for a in A} for s in S}
    for s in S:
        for a in A:
            if s+a > c2:
                for s_prime in range(-c1+1, c2+1):
                    P[s][a][s_prime] = poisson.pmf(c2-s_prime, lambda_poisson)
                P[s][a][-c1] = 1 - poisson.cdf(c2+c1-1, lambda_poisson)
            else:
                for s_prime in range(-c1+1, s+a+1):
                    P[s][a][s_prime] = poisson.pmf(s+a-s_prime, lambda_poisson)
                P[s][a][-c1] = 1 - poisson.cdf(s+a+c1-1, lambda_poisson)

    # construct the cost function
    #C = lambda s, a: (oc_f*indicator_fn(a) + oc_v*a + sum((hc*indicator_fn(s+a-d) \
    #                               + bc*indicator_fn(d-s-a))*abs(s+a-d)\
    #                              *poisson.pmf(d, lambda_poisson) \
    #                              for d in range(s+a+c1+1)) \
    #                   + bc*(c1)*(1 - poisson.cdf(s+a+c1, lambda_poisson)) )\
    #                 *indicator_fn(c2+1-s-a) + np.max([oc_f, oc_v, hc, bc])*max([c1, c2])/(1-beta)*indicator_fn(s+a-c2)
    C = {s: {a: 0 for a in A} for s in S}
    for s in S:
        for a in A:
            C[s][a] = (oc_f * indicator_fn(a) + oc_v * a + sum((hc * indicator_fn(s + a - d) \
                                                               + bc * indicator_fn(d - s - a)) * abs(s + a - d) \
                                                              * poisson.pmf(d, lambda_poisson) \
                                                              for d in range(s + a + c1 + 1)) \
                      + bc * (c1) * (1 - poisson.cdf(s + a + c1, lambda_poisson))) \
                     * indicator_fn(c2 + 1 - s - a) + np.max([oc_f, oc_v, hc, bc]) * max([c1, c2]) / (
                                 1 - beta) * indicator_fn(s + a - c2)

    n = c1 + c2 + 1  # number of states
    gamma = {s: 1/n for s in S}

    run_time = time.time() - start_time

    return S, A, P, C, gamma, run_time


#--------------------------------------------
def update_and_normalize(p, s, a, c1, c2, d_max):
    # Set range of s_prime from min(c2, s + a) down to c1
    upper_bound = min(c2, s + a)

    # Set values from (min(c2, s + a) - d_max) to c1 to zero
    for s_prime in range(upper_bound - d_max - 1, -c1 - 1, -1):
        p[s][a][s_prime] = 0

    # For the remaining positive values, normalize them
    remaining_values = [p[s][a][s_prime] for s_prime in range(max(-c1, upper_bound - d_max), upper_bound + 1)]
    total_sum = sum(remaining_values)

    for s_prime in range(max(-c1, upper_bound - d_max), upper_bound + 1):
        p[s][a][s_prime] /= total_sum

    return p[s][a]


#--------------------------------------------
def create_mdp_inventory_bounded_demand(lambda_poisson, d_max, q, c1, c2, oc_f, oc_v, hc, bc, beta):
    '''
    This function takes the parameters of the demand Poisson distribution \
     lambda_poisson, the maximum ordering capacity q, maximum backlog c1, \
     maximum storage capacity c2, fixed ordering cost oc_f, variable ordering cost oc_v, holding cost hc, backlog \
     cost bc, and the discount factor to construct the MDP for an inventory \
     model.
    '''

    start_time = time.time()

    # construct the state space
    S = [i for i in range(-c1, c2 + 1)]

    # construct the action set
    A = [i for i in range(q + 1)]

    # construct the transition probability matrix
    P = {s: {a: {s_prime: 0 for s_prime in S} for a in A} for s in S}
    for s in S:
        for a in A:
            if s + a > c2:
                for s_prime in range(-c1 + 1, c2 + 1):
                    P[s][a][s_prime] = poisson.pmf(c2 - s_prime, lambda_poisson)
                P[s][a][-c1] = 1 - poisson.cdf(c2 + c1 - 1, lambda_poisson)
            else:
                for s_prime in range(-c1 + 1, s + a + 1):
                    P[s][a][s_prime] = poisson.pmf(s + a - s_prime, lambda_poisson)
                P[s][a][-c1] = 1 - poisson.cdf(s + a + c1 - 1, lambda_poisson)

            P[s][a] = update_and_normalize(P, s, a, c1, c2, d_max)
            #for s_prime in S:
            #    P[s][a][s_prime] *= 0.90
            #P[s][a][c2] += 0.10

    # construct the cost function
    C = {s: {a: 0 for a in A} for s in S}
    for s in S:
        for a in A:
            C[s][a] = (oc_f * indicator_fn(a) + oc_v * a + sum((hc * indicator_fn(s + a - d) \
                                                               + bc * indicator_fn(d - s - a)) * abs(s + a - d) \
                                                              * poisson.pmf(d, lambda_poisson) \
                                                              for d in range(s + a + c1 + 1)) \
                      + bc * (c1) * (1 - poisson.cdf(s + a + c1, lambda_poisson))) \
                     * indicator_fn(c2 + 1 - s - a) + np.max([oc_f, oc_v, hc, bc]) * max([c1, c2]) / (
                                 1 - beta) * indicator_fn(s + a - c2)


    n = c1 + c2 + 1  # number of states
    gamma = {s: 1 / n for s in S}

    run_time = time.time() - start_time

    return S, A, P, C, gamma, run_time


#--------------------------------------------
