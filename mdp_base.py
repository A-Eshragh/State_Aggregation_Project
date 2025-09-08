import time
import torch
import numpy as np
from abc import ABC, abstractmethod
from scipy.sparse import coo_matrix, eye, linalg, issparse


class MDP:
    def __init__(self, states, actions, trans_probs, obj_fun, init_dist, disc_factor):
        self.states = states
        self.actions = actions
        self.trans_probs = trans_probs
        self.obj_fun = obj_fun
        self.init_dist = init_dist
        self.disc_factor = disc_factor

        # ===== Representation backends (Strategy pattern) =========================

    class _Backend(ABC):
        """Abstracts away sparse vs. dense operations for one sample."""

        @abstractmethod
        def eval_policy(self, mdp: "MDP", policy, obj_fun_s):
            """Return v_pi for this sample using the given policy."""
            ...

        @abstractmethod
        def vsa(self, mdp: "MDP", v_pi):
            """Return gamma * P @ v_pi for this sample (shape (S*A,) -> (S*A,) then reshape later)."""
            ...

    class _SparseBackend(_Backend):
        def __init__(self, ps_csr):
            # ps_csr: (S*A, S) CSR matrix
            self.ps_csr = ps_csr

        def eval_policy(self, mdp: "MDP", policy, obj_fun_s):
            return mdp.eval_policy(policy, self.ps_csr, obj_fun_s)[0]

        def vsa(self, mdp: "MDP", v_pi):
            return mdp.disc_factor * self.ps_csr.dot(v_pi)

    class _DenseBackend(_Backend):
        def __init__(self, trans_prob):
            # P: (S*A, S) ndarray
            self.trans_prob = trans_prob

        def eval_policy(self, mdp: "MDP", policy, obj_fun_s):
            return mdp.eval_policy(policy, self.trans_prob, obj_fun_s)[0]

        def vsa(self, mdp: "MDP", v_pi):
            # Apply discount consistently here too
            return mdp.disc_factor * (self.trans_prob @ v_pi)

    def eval_policy(self, policy, trans_prob, obj_fun):
        """
        Evaluates a given policy.

        Args:
            policy: Policy to be evaluated.
            trans_prob: Transition probability
            obj_fun: Objective function

        Returns:
            v_pi: Value of the policy
            v_star: Value of the MDP
        """

        n_states = len(self.states)
        n_actions = len(self.actions)
        sa_idx = np.arange(n_states) * n_actions + policy  # linearized index for the (state, action) pair
        p_pi = trans_prob[sa_idx, :]
        c_pi = obj_fun[np.arange(n_states), policy]
        if issparse(trans_prob):
            # sparse representation of the transition probability: trans_prob is in CSR format
            coeff_matrix = eye(n_states, format='csr') - self.disc_factor * p_pi
            # from scipy.sparse.linalg import svds
            # sigma_max = svds(coeff_matrix, k=1, return_singular_vectors=False, which='LM')[0] # LM = Largest Magnitude
            # sigma_min = svds(coeff_matrix, k=1, return_singular_vectors=False, which='SM')[0] # SM = Smallest Magnitude
            # cond_number = sigma_max / sigma_min
            v_pi = linalg.spsolve(coeff_matrix, c_pi)
        else:
            # dense representation of the transition probability
            coeff_matrix = np.eye(n_states) - self.disc_factor * p_pi
            v_pi = np.linalg.solve(coeff_matrix, c_pi)

        v_star = np.sum(self.init_dist.numpy() * v_pi)

        return v_pi, v_star

    def solve_optimal_policy(self, backend: "_Backend", obj_fun_s, pi_init, max_iter):
        n_states = len(self.states)
        n_actions = len(self.actions)
        pi_old = pi_init.copy()
        counter = 1
        while True:
            # policy evaluation
            v_pi = backend.eval_policy(self, pi_old, obj_fun_s)

            # policy improvement
            v_sa = backend.vsa(self, v_pi)                     # shape: (S*A,)
            q = obj_fun_s + v_sa.reshape(n_states, n_actions)  # (S, A)
            pi_new = np.argmin(q, axis=1)

            if np.all(pi_old == pi_new) or counter >= max_iter:
                break
            pi_old = pi_new
            counter += 1

        v_star_s = float(np.sum(self.init_dist.numpy() * v_pi))

        return pi_new, v_star_s

    def policy_iteration(self, pi_init=None, max_iter=1000):
        """
        Perform policy iteration for a batch of MDPs.

        Args:
            pi_init: Initial policy (optional)
            max_iter: Maximum policy improvement iterations

        Returns:
            policy: (N, S) optimal action per state for each sample
            v_star: (N, ) optimal value under init_dist for each sample
            run_time: float seconds
        """

        start_time = time.time()

        trans_probs = self.trans_probs
        n_samples = trans_probs.shape[0]
        obj_fun = self.obj_fun.detach().numpy()

        n_states = len(self.states)
        n_actions = len(self.actions)
        policy = np.zeros((n_samples, n_states))
        v_star = np.zeros(n_samples)

        if pi_init is None:
            pi_init = np.random.randint(0, n_actions, size=n_states)

        if trans_probs.layout != torch.strided:
            # Sparse batch in torch.sparse_coo
            idx = trans_probs.indices()  # (3, nnz)
            val = trans_probs.values()   # (nnz,)
            for s in range(n_samples):
                mask = (idx[0] == s)
                row_id = idx[1, mask].numpy()
                col_id = idx[2, mask].numpy()
                prob = val[mask].detach().numpy()

                ps_csr = coo_matrix(
                    (prob, (row_id, col_id)),
                    shape=(n_states * n_actions, n_states)
                ).tocsr()

                backend = MDP._SparseBackend(ps_csr)
                pi_new, v_star_s = self.solve_optimal_policy(backend, obj_fun[s], pi_init, max_iter)
                policy[s] = pi_new
                v_star[s] = v_star_s
        else:
            # Dense batch
            for s in range(n_samples):
                p = trans_probs[s]  # (S*A, S) ndarray-like
                backend = MDP._DenseBackend(p)
                pi_new, v_star_s = self.solve_optimal_policy(backend, obj_fun[s], pi_init, max_iter)
                policy[s] = pi_new
                v_star[s] = v_star_s

        run_time = time.time() - start_time

        return policy, v_star, run_time
