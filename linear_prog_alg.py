#--------------------------------------------
"""
Linear Programming Model for Solving MDP
Written by: Ali Eshragh & Ali Fattahi
Date: 4 June 2024
Revised: 24 July 2024
Revised by: Yuexing Li
Date: 10 Nov 2024
"""
#--------------------------------------------


from gurobipy import Model, GRB, quicksum
import numpy as np
import time


def lin_prog(states, actions, trans_probs, cost_fun, beta, gamma):
    """
    This function takes the state space S, action set A, the cost function C,\
    and the discount factor beta as inputs to construct the space of \
    discounted occupational measures and solve the linear program corresponding to the \
    given discounted MDP.
    """

    start_time = time.time()

    n = len(states)
    k = len(actions)

    # s_num = {}
    # s_num_inv = {}
    # count = 0
    # for s in states:
    #     s_num[s] = count
    #     s_num_inv[count] = s
    #     count += 1
    #
    # a_num = {}
    # a_num_inv = {}
    # count = 0
    # for a in actions:
    #     a_num[a] = count
    #     a_num_inv[count] = a
    #     count += 1

    # construct the model
    m = Model('linear program for solving discounted MDP')

    # suppress Gurobi output
    m.Params.OutputFlag = 0

    # decision variables
    y = m.addVars(n, k, name='y')

    # objective function
    m.setObjective(
        quicksum(
            cost_fun[s][a] * y[s, a]
            for a in range(k)
            for s in range(n)
        ),
        GRB.MINIMIZE)

    # constraints
    m.addConstrs(
        (quicksum(
            (int(s == s_prime) - beta * trans_probs[s][s_prime][a]) * y[s, a]
            for a in range(k)
            for s in range(n)
        ) == gamma[s_prime]
        for s_prime in range(n)), name='space of discounted occupational measures'
    )

    # update the model (after adding variables and constraints) and solve
    m.update()
    # m.display()
    m.optimize()

    v_opt = m.ObjVal
    y_opt = np.array([[y[s, a].X for a in range(k)] for s in range(n)])

    policy = np.array(actions)[np.argmax(y_opt, axis=1)]

    run_time = time.time() - start_time

    return policy, v_opt, run_time
