import time
import numpy as np
from scipy.sparse import csr_matrix, identity, linalg


class MDP:
    def __init__(self, states, actions, trans_probs, obj_fun, init_dist, disc_factor):
        self.states = states
        self.actions = actions
        self.trans_probs = trans_probs
        self.obj_fun = obj_fun
        self.init_dist = init_dist
        self.disc_factor = disc_factor

    def eval_policy(self, policy, instance_indices=None):
        """
        Evaluates a given policy.

        Args:
            policy: Policy to be evaluated.
            instance_indices: Which MDPs are used to evaluate the policy.

        Returns:
            v_pi: Value of the policy
            v_star: Value of the MDP.
        """

        if instance_indices is None:
            assert policy.ndim == 2 and policy.shape[0] == len(self.trans_probs)
            instance_indices = list(range(policy.shape[0]))
        else:
            instance_indices = list(instance_indices)

        n_instances = len(instance_indices)
        n_states = len(self.states)
        policy = policy.reshape(n_instances, -1)  # ensure a single policy has the shape (1, n_states)
        v_pi = np.zeros((n_instances, n_states))
        if isinstance(self.trans_probs, tuple):
            # sparse representation of the transition probability
            row_indices, col_indices, probs = self.trans_probs
            row_indices = row_indices.numpy()
            col_indices = col_indices.numpy()
            probs = probs.detach().numpy()
            for idx, s in enumerate(instance_indices):
                # get the actions of the policy corresponding to the row indices of the nnz
                action_indices = policy[idx][row_indices]  # Shape: (nnz, )
                # advanced indexing: for each i in 0,...,nnz-1, get the item at position [i, action_indices[i]]
                p_pi_nnz = probs[s][np.arange(len(action_indices)), action_indices]
                p_pi = csr_matrix((p_pi_nnz, (row_indices, col_indices)), shape=(n_states, n_states))
                c_pi = self.obj_fun[np.arange(n_states), policy[idx]]
                coeff_matrix = identity(n_states, format='csr') - self.disc_factor * p_pi
                v_pi[idx] = linalg.spsolve(coeff_matrix, c_pi)
        else:
            for idx, s in enumerate(instance_indices):
                # dense representation of the transition probability
                # advanced indexing: for each i, j in 0,...,n_states-1, get the item at position [i, j, pi[i]]
                p_pi = self.trans_probs[s][np.arange(n_states)[:, None], np.arange(n_states)[None, :], policy[idx][:, None]]
                c_pi = self.obj_fun[np.arange(n_states), policy[idx]]
                coeff_matrix = np.eye(n_states) - self.disc_factor * p_pi
                v_pi[idx] = np.linalg.solve(coeff_matrix, c_pi)

        v_star = np.sum(self.init_dist.numpy() * v_pi, axis=1)

        return np.squeeze(v_pi), v_star

    def policy_iteration(self, pi_old=None, max_iter=1000):
        """
        Perform policy iteration for a batch of MDPs.

        Args:
            pi_old: Initial policy.
            max_iter: Maximum number of iterations

        Returns:
            policy: Optimal policy for each state.
            v_star: Optimal value for each state.
            run_time: Time taken to compute the result.
        """

        start_time = time.time()

        # Retrieve information from the MDP
        trans_probs = self.trans_probs
        obj_fun = self.obj_fun.numpy()
        gamma = self.init_dist.numpy()
        beta = self.disc_factor
        n_states = len(self.states)
        n_actions = len(self.actions)

        if pi_old is None:
            pi_old = np.random.randint(0, n_actions, size=n_states)  # Initialize the policy randomly if not provided

        if isinstance(trans_probs, tuple):
            row_indices, col_indices, probs = self.trans_probs
            n_samples = len(probs)
            row_indices = row_indices.numpy()
            col_indices = col_indices.numpy()
            probs = probs.detach().numpy()
        else:
            row_indices = None
            col_indices = None
            probs = None
            n_samples = trans_probs.shape[0]
            trans_probs = trans_probs.numpy()

        policy = np.zeros((n_samples, n_states))
        v_star = np.zeros(n_samples)
        for s in range(n_samples):
            counter = 1
            while True:
                # policy evaluation
                v_pi = self.eval_policy(pi_old, [s])[0]
                # policy improvement
                if isinstance(trans_probs, tuple):
                    # compute contribution of each nonzero: P[i, j, a] * v[j]
                    contrib = probs[s] * v_pi[col_indices][:, None]  # shape: (nnz, n_actions)
                    # accumulate contributions into q[s, a]
                    q = np.zeros((n_states, n_actions))
                    # np.add.at(a, indices, b) adds the values from array "b" to elements of array "a" at the indices
                    np.add.at(q, (row_indices[:, None], np.arange(n_actions)[None, :]), contrib)
                    q = obj_fun + beta * q
                else:
                    q = obj_fun + beta * np.sum(trans_probs[s] * v_pi.reshape(1, -1, 1), axis=1)
                pi_new = np.argmin(q, axis=1)

                if np.all(pi_old == pi_new) or counter >= max_iter:
                    break
                else:
                    pi_old = pi_new.copy()
                    counter += 1

            policy[s] = pi_new
            # optimal value
            v_star[s] = np.sum(gamma * v_pi)

        run_time = time.time() - start_time

        return policy, v_star, run_time
