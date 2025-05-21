# -------------------------------------------
"""
Main body for solving the input MDP
Written by: Ali Eshragh & Ali Fattahi
Date: 29 May 2024
Revised by: Yuexing Li
Date: 26 Jan 2025
"""
# -------------------------------------------

from mdp import *
from utils import *
from functools import partial
from state_aggregation_alg import state_agg, dis_agg_pi
from qpth.qp import QPFunction


if __name__ == '__main__':
    set_seed(42)
    # generate the MDP of the input birth and death process
    n = 5000  # number of states: |S|
    k = 10  # number of actions: |A|
    pow1 = 0.8  # constant for calculating cost function
    pow2 = 1  # constant for calculating cost function
    beta = 0.9  # discount factor
    c0 = 0.1  # constant for calculating transition probabilities
    c1 = 0.2  # constant for calculating transition probabilities
    n_feats = 2

    gt_mode = 'nn'
    feats = [c0, c1]
    # feats = torch.rand(n_feats)
    gt_mdl = partial(expon_mdl, n=n, k=k) if gt_mode == 'exp' else GTNN(n_feats, (n - 1) * k)

    print('\n--------------------')
    print('State Aggregation:')
    print('--------------------')

    d = 10  # the number of states in the cluster set
    mdp = BirthDeath(n, k, [c0, c1], pow1, pow2, beta, gt_mdl)
    mdp_d, time_sa = state_agg(mdp, d)
    print(f'Aggregated States: {mdp_d.states}')
    print(f'Run Time: {time_sa:.4f}s')

    # use QP to solve the MDP approximately
    regularize_coef = 0.5
    qp_solver = QPFunction()
    # opt_qp_full, obj_qp_full, qp_run_time_full = solve_qp(mdp, regularize_coef, qp_solver)
    opt_qp, obj_qp, qp_run_time = solve_qp(mdp_d, regularize_coef, qp_solver)
    print(f'QP Solution: {opt_qp}')
    print(f'QP Objective: {obj_qp}')
    print(f'QP Run Time: {qp_run_time:.4f}')

    # solve by various algorithms
    algos = {
        'Value Iteration': 'value_iteration',
        'Policy Iteration': 'policy_iteration',
        'Linear Programming': 'lin_prog'
    }
    for algo_name in list(algos.keys()):
        print('\n--------------------')
        print(f'{algo_name}:')
        print('--------------------')
        print('\nOriginal MDP:')
        opt_pi, v_star, run_time = getattr(mdp, algos[algo_name])()
        print(f'Optimal Policy: {opt_pi}')
        print(f'Optimal Value: {v_star}')
        print(f'Run Time: {run_time:.4f}s')

        print(f'\nAggregated MDP:')
        opt_pi_agg, v_star_agg, run_time_agg = getattr(mdp_d, algos[algo_name])()
        print(f'Optimal Policy: {opt_pi_agg}')
        print(f'Optimal Value: {v_star_agg}')
        print(f'Run Time: {run_time_agg:.4f}')
        pi_dis_agg = dis_agg_pi(mdp, mdp_d, opt_pi_agg)
        vbar_policy = mdp.eval_policy(pi_dis_agg)[1]
        print(f'Disaggregated Policy: {pi_dis_agg}')
        print(f'vbar_dis_policy: {vbar_policy}')
        rel_err = (vbar_policy - v_star) / v_star
        print(f'Relative error: {rel_err}')
