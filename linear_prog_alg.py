# -------------------------------------------
"""
Linear Programming Model for Solving MDP
Written by: Ali Eshragh & Ali Fattahi
Date: 4 June 2024
Revised: 24 July 2024
Revised by: Yuexing Li
Date: 22 Jan 2025
"""
# -------------------------------------------

import time
import numpy as np
from gurobipy import Model, GRB, quicksum


def lin_prog(mdp):
    """
    Solves a general MDP using linear programming.

    Args:
        mdp: An instance of an MDP class.

    Returns:
        policy: Optimal policy for each state.
        v_star: Optimal value for each state.
        run_time: Time taken to compute the result.
    """

    start_time = time.time()

    # Retrieve information from the MDP
    actions = mdp.actions
    trans_probs = mdp.trans_probs.numpy()
    obj_fun = mdp.obj_fun.numpy()
    gamma = mdp.gamma
    beta = mdp.beta

    n_samples, n, _, k = trans_probs.shape
    policy = np.zeros((n_samples, n))
    v_star = np.zeros(n_samples)
    for s in range(n_samples):
        # construct the model
        m = Model('linear program for solving discounted MDP')
        # suppress Gurobi output
        m.Params.OutputFlag = 0
        # decision variables
        y = m.addVars(n, k, name='y')
        # objective function
        m.setObjective(
            quicksum(
                obj_fun[i, a] * y[i, a]
                for a in range(k)
                for i in range(n)
            ),
            GRB.MINIMIZE)
        # constraints
        m.addConstrs(
            (quicksum(
                (int(i == j) - beta * trans_probs[s, i, j, a]) * y[i, a]
                for a in range(k)
                for i in range(n)
            ) == gamma[j]
             for j in range(n)), name='space of discounted occupational measures'
        )
        # update the model (after adding variables and constraints) and solve
        m.update()
        m.optimize()

        v_star[s] = m.ObjVal
        y_opt = np.array([[y[i, a].X for a in range(k)] for i in range(n)])
        policy[s] = actions[np.argmax(y_opt, axis=1)]

    run_time = time.time() - start_time

    return policy, v_star, run_time
