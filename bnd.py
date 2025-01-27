import numpy as np
import torch
import time
from nn_mdl import *
from gurobipy import Model, GRB, quicksum


class BirthDeath:
    def __init__(self, n, k, feats, pow1, pow2, beta, gt_mdl='exp'):
        self.states = np.arange(1, n + 1)
        self.actions = np.arange(1, k + 1)
        # for faster indexing
        self.state_indices = {state: pos for pos, state in enumerate(self.states)}
        self.action_indices = {action: pos for pos, action in enumerate(self.actions)}
        feats = torch.tensor(feats)
        feats = feats.view(1, -1) if feats.ndim == 1 else feats
        self.feats = feats
        self.n_samples, self.n_feats = self.feats.shape
        self.pow1 = pow1
        self.pow2 = pow2
        self.beta = beta
        self.gt_mdl = gt_mdl
        self.trans_probs = self.gen_trans_probs()
        self.cost_fun = self.gen_cost_fun()
        self.gamma = 1 / n * torch.ones(n)


    def gen_trans_probs(self):
        n = len(self.states)
        k = len(self.actions)
        if self.gt_mdl == 'exp':
            assert self.n_feats == 2
            transform_matrix = torch.stack([torch.ones(n - 1), torch.arange(2, n + 1)])
            numerator = torch.repeat_interleave(torch.matmul(self.feats, transform_matrix), repeats=k, dim=1)
            denominator = torch.arange(1, k + 1).repeat(n - 1)
            expon = torch.exp(- numerator / denominator)  # avoid numerical overflow when n is large
            thetas = expon / (expon + 1)
        else:
            mdl = GTNN(self.n_feats, (n - 1) * k)
            thetas = mdl(self.feats)

        theta_array = thetas.reshape(self.n_samples, n - 1, k)
        trans_probs = torch.zeros(self.n_samples, n, n, k)
        trans_probs[:, 0, 1, :] = 1
        trans_probs[:, torch.arange(1, n - 1), torch.arange(2, n), :] = theta_array[:, :-1, :]
        trans_probs[:, -1, -1, :] = theta_array[:, -1, :]
        trans_probs[:, :, 0, :] = 1 - torch.sum(trans_probs[:, :, 1:n, :], dim=2)

        return trans_probs


    def gen_cost_fun(self):
        n = len(self.states)
        k = len(self.actions)
        cost_fun = torch.exp((torch.arange(1, k + 1).reshape(1, -1)) ** self.pow1 /
                             (torch.arange(1, n + 1).reshape(-1, 1)) ** self.pow2)

        return cost_fun


    def value_iteration(self, epsilon=1e-6, v_old=None):
        start_time = time.time()

        n_samples, n, _, _ = self.trans_probs.shape
        if v_old is None:
            v_old = torch.zeros(n)  # Initialize value function if not provided

        policy = np.zeros((n_samples, n))
        v_star = np.zeros(n_samples)
        for s in range(n_samples):
            while True:
                v_new, v_new_index = torch.min(self.cost_fun + self.beta * torch.sum(self.trans_probs[s] * v_old.reshape(1, -1, 1), dim=1), dim=1)
                delta = torch.max(torch.abs(v_new - v_old)).item()
                if delta < epsilon:
                    break
                else:
                    v_old = v_new.clone()

            policy[s] = self.actions[v_new_index]

            # optimal value
            v_star[s] = torch.sum(self.gamma * v_new).item()

        run_time = time.time() - start_time

        return policy, v_star, run_time


    def policy_iteration(self, pi_old=None):
        start_time = time.time()

        n_samples, n, _, k = self.trans_probs.shape
        if pi_old is None:
            pi_old = self.actions[torch.randint(0, k, (n, ))]  # Initialize the policy randomly if not provided

        policy = np.zeros((n_samples, n))
        v_star = np.zeros(n_samples)
        for s in range(n_samples):
            while True:
                indices = torch.tensor([self.action_indices[action] for action in pi_old])
                p_pi = torch.gather(self.trans_probs[s], dim=2, index=indices[:, None, None].expand(n, n, 1)).squeeze(2)
                c_pi = torch.gather(self.cost_fun, dim=1, index=indices[:, None]).squeeze(1)
                coeff_matrix = torch.eye(n) - self.beta * p_pi
                v_pi = torch.linalg.solve(coeff_matrix, c_pi)

                # policy improvement
                _, v_pi_index = torch.min(self.cost_fun + self.beta * torch.sum(self.trans_probs[s] * v_pi.reshape(1, -1, 1), dim=1), dim=1)
                pi_new = self.actions[v_pi_index]

                if np.all(pi_old == pi_new):
                    break
                else:
                    pi_old = pi_new.copy()

            policy[s] = pi_new

            # optimal value
            v_star[s] = torch.sum(self.gamma * v_pi).item()

        run_time = time.time() - start_time

        return policy, v_star, run_time


    def lin_prog(self):
        start_time = time.time()

        n_samples, n, _, k = self.trans_probs.shape
        policy = np.zeros((n_samples, n))
        v_star = np.zeros(n_samples)
        trans_probs = self.trans_probs.numpy()
        cost_fun = self.cost_fun.numpy()
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
                    cost_fun[i, a] * y[i, a]
                    for a in range(k)
                    for i in range(n)
                ),
                GRB.MINIMIZE)
            # constraints
            m.addConstrs(
                (quicksum(
                    (int(i == j) - self.beta * trans_probs[s, i, j, a]) * y[i, a]
                    for a in range(k)
                    for i in range(n)
                ) == self.gamma[j]
                 for j in range(n)), name='space of discounted occupational measures'
            )
            # update the model (after adding variables and constraints) and solve
            m.update()
            m.optimize()

            v_star[s] = m.ObjVal
            y_opt = np.array([[y[i, a].X for a in range(k)] for i in range(n)])
            policy[s] = self.actions[np.argmax(y_opt, axis=1)]

        run_time = time.time() - start_time

        return policy, v_star, run_time


if __name__ == '__main__':
    torch.manual_seed(42)
    bnd_instance = BirthDeath(50, 10, [0.1, 0.2], 0.8, 1, 0.9)
    opt_pi_vi, v_star_vi, time_vi = bnd_instance.value_iteration()
    print('Value Iteration:')
    print(f'Optimal Policy: {opt_pi_vi}')
    print(f'Optimal Value: {v_star_vi}')
    print(f'Run Time: {time_vi:.4f}s')
    opt_pi_pi, v_star_pi, time_pi = bnd_instance.policy_iteration()
    print('Policy Iteration:')
    print(f'Optimal Policy: {opt_pi_pi}')
    print(f'Optimal Value: {v_star_pi}')
    print(f'Run Time: {time_pi:.4f}s')
    opt_pi_lp, v_star_lp, time_lp = bnd_instance.lin_prog()
    print('Linear Program:')
    print(f'Optimal Policy: {opt_pi_lp}')
    print(f'Optimal Value: {v_star_lp}')
    print(f'Run Time: {time_lp:.4f}s')
