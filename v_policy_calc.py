# -------------------------------------------
"""
Finding the value function for a given policy
Written by: Ali Eshragh & Ali Fattahi
Date: 12 June, 2024
Updated: 24 July, 2024
Revsied by: Yuexing Li
Updated: 21 Jan, 2025
"""
# -------------------------------------------

import numpy as np
# from scipy.sparse import dok_matrix, identity
# from scipy.sparse.linalg import spsolve
import time


# def val_policy(S, policy, P, C, beta, gamma):
#     """
#     This function finds the value function for a given policy.
#
#     Args:
#         S:
#         policy:
#         P:
#         C:
#         beta:
#         gamma:
#
#     Returns:
#
#     """
#
#     start_time = time.time()
#
#     # Dictionary comprehension to map states to indices
#     s_num = {s: idx for idx, s in enumerate(S)}
#
#     n = len(S)
#
#     # Initialize sparse matrix for P_pi using dictionary of keys (DOK) format
#     P_pi = dok_matrix((n, n), dtype=np.float32)
#
#     # Populate P_pi using the policy and transition probabilities in P
#     for s in S:
#         for s_prime in P[s][policy[s]]:
#             P_pi[s_num[s], s_num[s_prime]] = P[s][policy[s]][s_prime]
#
#     # Vectorized calculation for cost vector c_pi
#     c_pi = np.array([C[s][policy[s]] for s in S]).reshape(n, 1)
#
#     # Convert P_pi to Compressed Sparse Row format for faster operations
#     P_pi = P_pi.tocsr()
#
#     # Create a sparse identity matrix
#     I = identity(n, format='csr')
#
#     # Calculate the coefficient matrix (I - beta * P_pi) in CSR format
#     coeff_matrix = I - beta * P_pi
#
#     # Solve for v_pi using sparse solver
#     v_pi = spsolve(coeff_matrix, c_pi)
#
#     # Convert gamma to a NumPy array (assuming it's a dictionary)
#     gamma_vec = np.array([gamma[s] for s in S])
#
#     # Calculate the optimal value v_star
#     v_star = np.dot(gamma_vec, v_pi)
#
#     run_time = time.time() - start_time
#
#     return v_pi, v_star, run_time
