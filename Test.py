import random
import numpy as np
from mdp_bnd import create_mdp_bnd

# Example dictionary P
P = {
    's1': {'s1': 0.2, 's2': 0.5, 's3': 0.3},
    's2': {'s1': 0.4, 's2': 0.1, 's3': 0.6},
    's3': {'s1': 10.997, 's2': 0.3, 's3': 0.2}
}


def select_s_s_prime(P):
    # Flatten the dictionary into a list of tuples (s, s', P[s][s'])
    flat_list = [(s, s_prime, P[s][s_prime]) for s in P for s_prime in P[s]]

    # Extract s', weights, and create a list of tuples (s, s')
    pairs = [(s, s_prime) for s, s_prime, weight in flat_list]
    weights = [weight for s, s_prime, weight in flat_list]

    # Randomly select a pair (s, s') based on weights
    selected_pair = random.choices(pairs, weights=weights, k=1)[0]

    return selected_pair


# Example usage
selected_pair = select_s_s_prime(P)
print(f"Randomly selected pair (s, s') is {selected_pair}.")
print(selected_pair[1])
input()






def choose_random_number(S, P):
    return random.choices(S, weights=P, k=1)[0]

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
        policy = {s: random.choices(A, k=1)[0] for s in S}

    S_d = []
    if initial_s is None:
        s = random.choices(S, weights=gamma, k=1)[0]
    else:
        s = initial_s

    S_d.append(s)
    P_temp = {s: {s_prime: P[s][policy[s]][s_prime] for s_prime in S}}
#    P_temp = {s: {s_prime: P[s][s_prime] for s_prime in S}}
    P_temp[s][s] = 0
    for _ in range(d-1):
        max_pair, max_value = find_max_pair(P_temp)
        s = max_pair[1]
        S_d.append(s)
        P_temp[s] = {}
        for s_prime in S:
            P_temp[s][s_prime] = P[s][policy[s]][s_prime]
#            P_temp[s][s_prime] = P[s][s_prime]
        for s_prime in S_d:
            P_temp[s_prime][s] = 0
            P_temp[s][s_prime] = 0

        keys = sorted(P.keys())
        n = len(keys)

        # Create an empty matrix of size n x n
        matrix = np.zeros((n, n))

        # Fill the matrix with values from the dictionary
        for i, key in enumerate(keys):
            for j, subkey in enumerate(keys):
                matrix[i, j] = P[key][policy[key]][subkey]
#                matrix[i, j] = P[key][subkey]

        # Format the matrix to display values with two decimal places
        formatted_matrix = np.around(matrix, decimals=4)

        print(formatted_matrix)

        matrix = np.zeros((len(S_d), n))

        for i in range(len(S_d)):
            for j in range(len(S)):
                matrix[i, j] = P_temp[S_d[i]][S[j]]

        # Format the matrix to display values with two decimal places
        formatted_matrix = np.around(matrix, decimals=4)

        print(f'\n{formatted_matrix}')
        print(f'\n{S_d}')
        input()

    return  S_d

'''np.set_printoptions(formatter={'float': '{:0.2f}'.format})
n = 10
S = [i for i in range(n)]
A = [i for i in range(5)]
matrix = np.random.exponential(1, (n, n))
matrix[0,3] = 100
matrix[3,0] = 100
matrix[3,6] = 99
matrix[6,3] = 100
matrix_dict = {}
for i in range(matrix.shape[0]):
    matrix_dict[i] = {}
    for j in range(matrix.shape[1]):
        matrix_dict[i][j] = matrix[i, j]
P = matrix_dict

n = 10

d = 10
gamma = np.zeros((n,1))
gamma[0,0] = 1'''

'''n = 10
k = 5
c0 = 1
c1 = 2
pow1 = 3
pow2 = 1
S, A, P, C, run_time = create_mdp_bnd(n, k, c0, c1, pow1, pow2)

for a in A:
    P[6][a][10] = 100'''

import numpy as np

from mdp_inventory import create_mdp_inventory
from value_iter_alg import value_iteration
from policy_iter_alg import policy_iteration
from linear_prog_alg import lin_prog
from v_policy_calc import val_policy

# generate the MDP of the input inventory model
lambda_poisson = 1 # the parameters of the demand Poisson distribution
q = 10 # maximum ordering capacity
c1 = 5 # maximum backlog
c2 = 5 # maximum storage capacity
oc_f = 0 # fixed ordering cost
oc_v = 1 # variable ordering cost
hc = 2 # holding cost
bc = 3 # backlog cost
beta = 0.9 # discount factor
S, A, P, C, time_create_mdp = create_mdp_inventory(lambda_poisson, q, c1, c2, oc_f, oc_v, hc, bc, beta)
n = c1+c2+1 # number of states
gamma = 1/n*np.ones((n,1)) # initial distribution

d = 5

#S_d = choose_d_deterministic(S, A, P, gamma, d)

def state_agg(S, A, P, C, gamma, d, rnd, policy=None):
    n = len(S)
    if rnd is True:
        S_d = choose_d_deterministic(S, A, P, gamma, d, initial_s=None, policy=None)
    else:
        S_d = choose_d_deterministic(S, A, P, gamma, d, initial_s=None, policy=None)

    P_d = {s: {a: {s_prime: P[s][a][s_prime] for s_prime in S_d} for a in A} for s in S_d}
    S_d.append('super_s')
    print(S_d)
    P_d['super_s'] = {}
    for a in A:
        P_d['super_s'][a] = {}
        for s_prime in S_d:
            if s_prime != 'super_s':
                P_d['super_s'][a][s_prime] = np.mean([P[s][a][s_prime] for s in S if s not in S_d])

    for s in S_d:
        for a in A:
            P_d[s][a]['super_s'] = 1 - np.sum([P_d[s][a][s_prime] for s_prime in S_d if s_prime != 'super_s'])

    C_d = lambda s, a: C(s, a) if s != 'super_s' else np.mean([C(s, a) for s in S if s not in S_d for a in A])

    gamma_d = np.zeros((d+1, 1))
    for s in S_d:
        if s != 'super_s':
            gamma_d[s, 0] = gamma[s, 0]
    gamma_d[d, 0] = 1 - np.sum([gamma_d[s,0] for s in S_d if s != 'super_s'])

    return S_d, P_d, C_d, gamma_d


S_d, P_d, C_d, gamma_d = state_agg(S, A, P, C, gamma, d, False, policy=None)

np.set_printoptions(formatter={'float': '{:0.2f}'.format})

# Print the matrix
print("Formatted matrix:")
#print(matrix)
print(S_d)
print(P_d)

print(gamma)
print(gamma_d)
print(np.sum(gamma_d))