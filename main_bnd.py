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
from value_iter_alg import *
from policy_iter_alg import *
from linear_prog_alg import *
from state_aggregation_alg import state_agg, dis_agg_pi
from v_policy_calc import val_policy


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    # generate the MDP of the input birth and death process
    n = 50  # number of states: |S|
    k = 10  # number of actions: |A|
    c0 = 0.1  # constant for calculating transition probabilities
    c1 = 0.2  # constant for calculating transition probabilities
    pow1 = 0.8  # constant for calculating cost function
    pow2 = 1  # constant for calculating cost function
    beta = 0.9  # discount factor
    bnd_instance = BirthDeath(n, k, [c0, c1], pow1, pow2, beta)

    # solve by value iteration algorithm
    print('\n--------------------')
    print('Value Iteration:')
    print('--------------------')
    opt_pi_vi, v_star_vi, time_vi = value_iteration(bnd_instance)
    print(f'Optimal Policy: {opt_pi_vi}')
    print(f'Optimal Value: {v_star_vi}')
    print(f'Run Time: {time_vi:.4f}s')

    # solve by policy iteration algorithm
    print('\n--------------------')
    print('Policy Iteration:')
    print('--------------------')
    opt_pi_pi, v_star_pi, time_pi = policy_iteration(bnd_instance)
    print(f'Optimal Policy: {opt_pi_pi}')
    print(f'Optimal Value: {v_star_pi}')
    print(f'Run Time: {time_pi:.4f}')

    # solve by linear programming
    print('\n--------------------')
    print('Linear Programming:')
    print('--------------------')
    opt_pi_lp, v_star_lp, time_lp = lin_prog(bnd_instance)
    print(f'Optimal Policy: {opt_pi_lp}')
    print(f'Optimal Value: {v_star_lp}')
    print(f'Run Time: {time_lp:.4f}')

    # -------------------
    # State aggregation
    # -------------------
    print('\n--------------------')
    print('State Aggregation:')
    print('--------------------')

    d = 5
    rnd = True
    S_d, P_d, C_d, gamma_d, run_time = state_agg(S, A, P, C, gamma, d, rnd)
    print(f'S_d: {S_d}')

    # solve by value iteration algorithm
    print(f'\n--------------------')
    print(f"Value Iteration:")
    print('--------------------')
    opt_pi_vi_agg, opt_v_vi_agg, v_star_vi_agg, time_vi_agg = value_iteration(S_d, A, P_d, C_d, beta, gamma_d, epsilon)
    # print(f"Optimal Policy: {opt_pi_vi_agg}")
    # print(f"Value Function: " , [f"{s}: {opt_v_vi_agg[s]:.4f}" for s in S])
    print(f"Optimal Value: {v_star_vi_agg:.4f}")
    pi_dis_agg = dis_agg_pi(S, S_d, opt_pi_vi_agg)
    _, vbar_policy, _ = val_policy(S, A, pi_dis_agg, P, C, beta, gamma)
    print(f"\nDisaggregated Policy: {pi_dis_agg}")
    print(f'vbar_dis_policy = {vbar_policy:.4f}')
    print(f"Run Time: {time_vi_agg:.4f}")

    # solve by policy iteration algorithm
    print(f'\n--------------------')
    print(f"Policy Iteration:")
    print('--------------------')
    opt_pi_pi_agg, opt_v_pi_agg, v_star_pi_agg, time_pi_agg = policy_iteration(S_d, A, P_d, C_d, beta, gamma_d)
    # print(f"Optimal Policy: {opt_pi_pi_agg}")
    # print(f"Value Function: " , [f"{s}: {opt_v_pi_agg[s-1,0]:.4f}" for s in S])
    # print(f"Optimal Value: {v_star_pi_agg:.4f}")
    pi_dis_agg = dis_agg_pi(S, S_d, opt_pi_pi_agg)
    _, vbar_policy, _ = val_policy(S, A, pi_dis_agg, P, C, beta, gamma)
    # print(f"Disaggregated Policy: {pi_dis_agg}")
    print(f'vbar_dis_policy = {vbar_policy:.4f}')
    print(f"Run Time: {time_pi_agg:.4f}")

    # solve by linear programming
    print(f'\n--------------------')
    print(f"Linear Programming:")
    print('--------------------')
    opt_pi_lp_agg, v_star_lp_agg, time_lp_agg = lin_prog(S_d, A, P_d, C_d, beta, gamma_d)
    # print(f"Optimal Policy: {opt_pi_lp_agg}")
    # print(f"Optimal Value: {v_star_lp_agg:.4f}")
    pi_dis_agg = dis_agg_pi(S, S_d, opt_pi_lp_agg)
    _, vbar_policy, _ = val_policy(S, A, pi_dis_agg, P, C, beta, gamma)
    # print(f"Disaggregated Policy: {pi_dis_agg}")
    print(f'vbar_dis_policy = {vbar_policy:.4f}')
    print(f"Run Time: {time_lp_agg:.4f}")

    rel_err = (vbar_policy - v_star_lp)/v_star_lp*100
    print(f'\nRelative error = {rel_err:.2f}%')

    for i in range(3):
        print(f'\nIteration: {i}')
        print('--------------------')
        S_d, P_d, C_d, gamma_d, run_time = state_agg(S, A, P, C, gamma, d, rnd, None, pi_dis_agg)
        print(f'S_d: {S_d}')
        opt_pi_lp_agg, v_star_lp_agg, time_lp_agg = lin_prog(S_d, A, P_d, C_d, beta, gamma_d)
        pi_dis_agg = dis_agg_pi(S, S_d, opt_pi_lp_agg)
        v_policy, vbar_policy, v_policy_time = val_policy(S, A, pi_dis_agg, P, C, beta, gamma)
        # print(f"Disaggregated Policy: {pi_dis_agg}")
        print(f'vbar_dis_policy = {vbar_policy:.4f}')
        print(f"Run Time: {time_lp:.4f}")

        rel_err = (vbar_policy - v_star_lp) / v_star_lp * 100
        print(f'\nRelative error = {rel_err:.2f}%')