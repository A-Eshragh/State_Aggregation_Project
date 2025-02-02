# -------------------------------------------
"""
Main body for solving the input MDP
Written by: Ali Eshragh & Ali Fattahi
Date: 29 May 2024
Revised by: Yuexing Li
Date: 26 Jan 2025
"""
# -------------------------------------------

import random
from mdp import *
from state_aggregation_alg import state_agg, dis_agg_pi


def set_seed(seed):
    # Set Python built-in random seed
    random.seed(seed)

    # Set NumPy seed
    np.random.seed(seed)

    # Set PyTorch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior (PyTorch only)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    set_seed(42)
    # generate the MDP of the input birth and death process
    n = 50  # number of states: |S|
    k = 10  # number of actions: |A|
    c0 = 0.1  # constant for calculating transition probabilities
    c1 = 0.2  # constant for calculating transition probabilities
    pow1 = 0.8  # constant for calculating cost function
    pow2 = 1  # constant for calculating cost function
    beta = 0.9  # discount factor
    mdp = BirthDeath(n, k, [c0, c1], pow1, pow2, beta)

    print('\n--------------------')
    print('State Aggregation:')
    print('--------------------')

    d = 5
    mdp_d, time_sa = state_agg(mdp, d)
    print(f'Aggregated States: {mdp_d.states}')
    print(f'Run Time: {time_sa: .4f}s')

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

