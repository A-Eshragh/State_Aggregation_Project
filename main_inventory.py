#--------------------------------------------
"""
Main body for solving the input MDP
Written by: Ali Eshragh & Ali Fattahi
Date: 12 June 2024
Revised: 12 Oct 2024
"""
#--------------------------------------------

# from mdp_inventory import create_mdp_inventory
from mdp_inventory import create_mdp_inventory_bounded_demand
# from value_iter_alg import value_iteration
from policy_iter_alg import policy_iteration
# from linear_prog_alg import lin_prog
from v_policy_calc import val_policy, val_policy_2
from state_aggregation_alg import state_agg, state_agg_det, dis_agg_pi, dis_agg_pi_inventory
# from state_aggregation_alg2 import state_agg, dis_agg_pi
import numpy as np


if __name__ == '__main__':
    # generate the MDP of the input inventory model
    lambda_poisson = 30 # the parameters of the demand Poisson distribution
    q = 20 # maximum ordering capacity
    c1 = 100 # maximum backlog
    c2 = 50 # maximum storage capacity
    oc_f = 0 # fixed ordering cost
    oc_v = 1 # variable ordering cost
    hc = 2 # holding cost
    bc = 3 # backlog cost
    beta = 0.95 # discount factor
    d_max = 3 # maximum possible number of orders
    # S, A, P, C, gamma, time_create_mdp = create_mdp_inventory(lambda_poisson, q, c1, c2, oc_f, oc_v, hc, bc, beta)
    S, A, P, C, gamma, time_create_mdp = create_mdp_inventory_bounded_demand(lambda_poisson, d_max, q, c1, c2, oc_f, oc_v, hc, bc, beta)

    # solve by value iteration algorithm
    '''
    print(f'\n--------------------')
    print(f"Value Iteration:")
    print('--------------------')
    epsilon = 0.000001
    opt_pi_vi, opt_v_vi, v_star_vi, time_vi = value_iteration(S, A, P, C, beta, gamma, epsilon)
    print(f"Optimal Policy: {opt_pi_vi}")
    #print(f"Value Function: " , [f"{s}: {opt_v_vi[s]:.4f}" for s in S])
    print(f"Optimal Value: {v_star_vi:.4f}")
    print(f"Run Time: {time_vi:.4f}")'''

    # solve by policy iteration algorithm
    print(f'\n--------------------')
    print(f"Policy Iteration:")
    print('--------------------')
    opt_pi_pi, opt_v_pi, v_star_pi, time_pi = policy_iteration(S, A, P, C, beta, gamma)
    print(f"Optimal Policy: {opt_pi_pi}")
    #print(f"Value Function: " , [f"{s}: {opt_v_pi[s-1,0]:.4f}" for s in S])
    print(f"Optimal Value: {v_star_pi:.4f}")
    print(f"Run Time: {time_pi:.4f}")

    # solve by linear programming
    '''print(f'\n--------------------')
    print(f"Linear Programming:")
    print('--------------------')
    opt_pi_lp, v_star_lp, time_lp = lin_prog(S, A, P, C, beta, gamma)
    print(f"Optimal Policy: {opt_pi_lp}")
    print(f"Optimal Value: {v_star_lp:.4f}")
    print(f"Run Time: {time_lp:.4f}")'''


    #-------------------
    # State aggregation
    #-------------------
    print(f'\n--------------------')
    print(f'State Aggregation:')
    print('--------------------')

    d = 20
    rnd = True
    initial_s = list(np.linspace(-c1, c2, 10, dtype=int))
    S_d, P_d, C_d, gamma_d, run_time = state_agg(S, A, P, C, gamma, d, rnd, initial_s)
    #S_d = list(np.linspace(-c1, c2, d, dtype=int))
    #P_d, C_d, gamma_d, run_time = state_agg_det(S, A, P, C, gamma, d, S_d)

    print(f'S_d: {S_d}')

    # solve by value iteration algorithm
    '''print(f'\n--------------------')
    print(f"Value Iteration:")
    print('--------------------')
    epsilon = 0.000001
    opt_pi_vi_agg, opt_v_vi_agg, v_star_vi_agg, time_vi_agg = value_iteration(S_d, A, P_d, C_d, beta, gamma_d, epsilon)
    #print(f"Optimal Policy: {opt_pi_vi_agg}")
    #print(f"Value Function: " , [f"{s}: {opt_v_vi_agg[s]:.4f}" for s in S])
    print(f"Optimal Value: {v_star_vi_agg:.4f}")
    pi_dis_agg = dis_agg_pi(S, S_d, opt_pi_vi_agg)
    v_policy, vbar_policy, v_policy_time = val_policy(S, pi_dis_agg, P, C, beta, gamma)
    print(f"\nDisaggregated Policy: {pi_dis_agg}")
    print(f'vbar_dis_policy = {vbar_policy:.4f}')
    print(f"Run Time: {time_vi_agg:.4f}")'''

    # solve by policy iteration algorithm
    print(f'\n--------------------')
    print(f"Policy Iteration:")
    print('--------------------')
    opt_pi_pi_agg, opt_v_pi_agg, v_star_pi_agg, time_pi_agg = policy_iteration(S_d, A, P_d, C_d, beta, gamma_d)
    opt_pi_pi_agg = dict(sorted(opt_pi_pi_agg.items(), key=lambda x: (isinstance(x[0], str), x[0])))
    print(f"Optimal Policy: {opt_pi_pi_agg}")
    #print(f"Value Function: " , [f"{s}: {opt_v_pi_agg[s-1,0]:.4f}" for s in S])
    #print(f"Optimal Value: {v_star_pi_agg:.4f}")
    #pi_dis_agg = dis_agg_pi(S, S_d, opt_pi_pi_agg)
    pi_dis_agg = dis_agg_pi_inventory(S, S_d, opt_pi_pi_agg)
    print(f"Disaggregated Policy: {pi_dis_agg}")
    v_policy_agg, vbar_policy, v_policy_time = val_policy(S, pi_dis_agg, P, C, beta, gamma)
    print(f'vbar_dis_policy = {vbar_policy:.4f}')
    print(f"Run Time: {time_pi_agg:.4f}")

    # solve by linear programming
    '''print(f'\n--------------------')
    print(f"Linear Programming:")
    print('--------------------')
    opt_pi_lp_agg, v_star_lp_agg, time_lp_agg = lin_prog(S_d, A, P_d, C_d, beta, gamma_d)
    #print(f"Optimal Policy: {opt_pi_lp_agg}")
    #print(f"Optimal Value: {v_star_lp_agg:.4f}")
    #pi_dis_agg = dis_agg_pi(S, S_d, opt_pi_lp_agg)
    pi_dis_agg = dis_agg_pi_inventory(S, S_d, opt_pi_lp_agg)
    v_policy, vbar_policy, v_policy_time = val_policy(S, pi_dis_agg, P, C, beta, gamma)
    print(f"Disaggregated Policy: {pi_dis_agg}")
    print(f'vbar_dis_policy = {vbar_policy:.4f}')
    print(f"Run Time: {time_lp_agg:.4f}")'''

    rel_err = (vbar_policy - v_star_pi)/v_star_pi*100
    print(f'\nRelative error = {rel_err:.2f}%')

    for i in range(10):
        print(f'\nIteration: {i}')
        print('--------------------')
        S_d, P_d, C_d, gamma_d, run_time = state_agg(S, A, P, C, gamma, d, rnd, initial_s, pi_dis_agg)
        #S_d = list(np.linspace(-c1, c2, d, dtype=int))
        #P_d, C_d, gamma_d, run_time = state_agg_det(S, A, P, C, gamma, d, S_d)
        print(f'S_d: {S_d}')
        opt_pi_pi_agg, opt_v_pi_agg, v_star_pi_agg, time_pi_agg = policy_iteration(S_d, A, P_d, C_d, beta, gamma_d)
        #opt_pi_lp_agg, v_star_lp_agg, time_lp_agg = lin_prog(S_d, A, P_d, C_d, beta, gamma_d)
        opt_pi_pi_agg = dict(sorted(opt_pi_pi_agg.items(), key=lambda x: (isinstance(x[0], str), x[0])))
        print(f"Optimal Policy: {opt_pi_pi_agg}")
        #pi_dis_agg = dis_agg_pi(S, S_d, opt_pi_pi_agg)
        pi_dis_agg = dis_agg_pi_inventory(S, S_d, opt_pi_pi_agg)
        print(f"Disaggregated Policy: {pi_dis_agg}")
        print(f"Run Time: {time_pi_agg:.4f}")

        print(f'\nval_policy_1...')
        v_policy, vbar_policy, v_policy_time = val_policy(S, pi_dis_agg, P, C, beta, gamma)
        print(f'vbar_dis_policy = {vbar_policy:.4f}')
        print(f"Run Time: {v_policy_time:.4f}")

        print(f'\nval_policy_2...')
        v_policy2, vbar_policy2, v_policy_time2 = val_policy_2(S, pi_dis_agg, P, C, beta, gamma)
        print(f'vbar_dis_policy_2 = {vbar_policy2:.4f}')
        print(f"Run Time: {v_policy_time2:.4f}")

        rel_err = (vbar_policy - v_star_pi) / v_star_pi * 100
        print(f'\nRelative error = {rel_err:.2f}%')

    '''------------
    The following graph displays the value function for a certain policy versus /
    actions for a fixed state. This policy is formed by using the optimal policy /
    at all states except for the fixed state.
    ------------'''

    '''policy = opt_pi_pi
    policy[-10] = q
    v_policy, vbar_policy, v_policy_time = val_policy(S, policy, P, C, beta, gamma)
    print(f'\nvbar_policy = {vbar_policy:.4f}')
    
    import matplotlib.pyplot as plt
    
    policy_temp = np.zeros((q + 1, 1))
    no_of_graphs = 6
    step_size = np.floor((c2+c1)/no_of_graphs)
    for i in range(no_of_graphs):
        s = np.floor(-c1+i*step_size)
        policy = opt_pi_lp.copy()
        vbar_policy = np.zeros((q + 1, 1))
        for a in range(q+1):
            policy[s] = a
            policy_temp[a,0] = a
            if s+a <= c2:
                v_policy, vbar_policy[a, 0], v_policy_time = val_policy(S, policy, P, C, beta, gamma)
            else:
                a_fixed = int(c2-s)
                vbar_policy[a, 0] = vbar_policy[a_fixed, 0]
    
        plt.plot(policy_temp, vbar_policy, label=f's = {s:.0f}', marker='o', linestyle='-')
    
    # Adjust layout and show plot
    plt.xlabel('a')
    plt.ylabel('vbar(s,a)')
    plt.legend()
    plt.show()
    '''