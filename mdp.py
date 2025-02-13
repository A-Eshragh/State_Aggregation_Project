from utils import *
from gurobipy import Model, GRB, quicksum


class MDP:
    def __init__(self, states, actions, trans_probs, obj_fun, init_dist, disc_factor):
        self.states = states
        self.actions = actions
        self.trans_probs = trans_probs
        self.obj_fun = obj_fun
        self.init_dist = init_dist
        self.disc_factor = disc_factor
        # for faster indexing
        self.state_indices = self._gen_state_index_dict()
        self.action_indices = {action: pos for pos, action in enumerate(actions)}

    def _gen_state_index_dict(self):
        if self.states.ndim == 1:
            state_indices = {state: pos for pos, state in enumerate(self.states)}
        elif self.states.ndim == 2:
            state_indices = [{state: pos for pos, state in enumerate(row)} for row in self.states]
        else:
            raise ValueError('States array must be 1D or 2D.')

        return state_indices

    def value_iteration(self, epsilon=1e-6, v_old=None):
        """
        Perform value iteration for a batch of MDPs.

        Args:
            epsilon: Convergence threshold.
            v_old: Initial value function. If None, initializes to zero.

        Returns:
            policy: Optimal policy for each state.
            v_star: Optimal value for each state.
            run_time: Time taken to compute the result.
        """

        start_time = time.time()

        # Retrieve information from the MDP
        actions = self.actions
        trans_probs = self.trans_probs
        obj_fun = self.obj_fun
        gamma = self.init_dist
        beta = self.disc_factor

        n_samples, n, _, _ = trans_probs.shape
        if v_old is None:
            v_old = torch.zeros(n)  # Initialize value function if not provided

        policy = torch.zeros(n_samples, n)
        v_star = torch.zeros(n_samples)
        for s in range(n_samples):
            while True:
                v_new, v_new_index = torch.min(obj_fun[s] + beta * torch.sum(trans_probs[s] * v_old.reshape(1, -1, 1), dim=1), dim=1)
                delta = torch.max(torch.abs(v_new - v_old)).item()
                if delta < epsilon:
                    break
                else:
                    v_old = v_new.clone()

            policy[s] = torch.tensor(actions[v_new_index])

            # optimal value
            v_star[s] = torch.sum(gamma[s] * v_new).item()

        run_time = time.time() - start_time

        return policy, v_star, run_time

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
            assert policy.ndim == 2 and policy.shape[0] == self.trans_probs.shape[0]
            instance_indices = list(range(policy.shape[0]))
        else:
            policy = policy.reshape(1, -1)
            instance_indices = list(instance_indices)

        n_instances = len(instance_indices)
        n = self.trans_probs.shape[1]
        v_pi = torch.zeros(n_instances, n)
        v_star = torch.zeros(n_instances)
        for idx, s in enumerate(instance_indices):
            indices = torch.tensor([self.action_indices[action] for action in policy[idx]])
            p_pi = torch.gather(self.trans_probs[s], dim=2, index=indices[:, None, None].expand(n, n, 1)).squeeze(2)
            c_pi = torch.gather(self.obj_fun[s], dim=1, index=indices[:, None]).squeeze(1)
            coeff_matrix = torch.eye(n) - self.disc_factor * p_pi
            v_pi[idx] = torch.linalg.solve(coeff_matrix, c_pi)
            v_star[idx] = torch.sum(self.init_dist[s] * v_pi).item()

        return torch.squeeze(v_pi), torch.squeeze(v_star)

    def policy_iteration(self, pi_old=None):
        """
        Perform policy iteration for a batch of MDPs.

        Args:
            pi_old: Initial policy.

        Returns:
            policy: Optimal policy for each state.
            v_star: Optimal value for each state.
            run_time: Time taken to compute the result.
        """

        start_time = time.time()

        # Retrieve information from the MDP
        actions = self.actions
        trans_probs = self.trans_probs
        obj_fun = self.obj_fun
        gamma = self.init_dist
        beta = self.disc_factor

        n_samples, n, _, k = trans_probs.shape
        if pi_old is None:
            pi_old = actions[torch.randint(0, k, (n,))]  # Initialize the policy randomly if not provided

        policy = torch.zeros((n_samples, n))
        v_star = torch.zeros(n_samples)
        for s in range(n_samples):
            while True:
                v_pi = self.eval_policy(pi_old, [s])[0]

                # policy improvement
                _, v_pi_index = torch.min(obj_fun[s] + beta * torch.sum(trans_probs[s] * v_pi.reshape(1, -1, 1), dim=1), dim=1)
                pi_new = actions[v_pi_index]

                if np.all(pi_old == pi_new):
                    break
                else:
                    pi_old = pi_new.copy()

            policy[s] = torch.tensor(pi_new)

            # optimal value
            v_star[s] = torch.sum(gamma[s] * v_pi).item()

        run_time = time.time() - start_time

        return policy, v_star, run_time

    def lin_prog(self):
        """
        Solve a batch of MDPs using linear programming.

        Args: None

        Returns:
            policy: Optimal policy for each state.
            v_star: Optimal value for each state.
            run_time: Time taken to compute the result.
        """

        start_time = time.time()

        # Retrieve information from the MDP
        actions = self.actions
        trans_probs = self.trans_probs
        obj_fun = self.obj_fun
        gamma = self.init_dist
        beta = self.disc_factor

        n_samples, n, _, k = trans_probs.shape
        policy = torch.zeros(n_samples, n)
        v_star = torch.zeros(n_samples)
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
                    obj_fun[s, i, a] * y[i, a]
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
                ) == gamma[s, j]
                 for j in range(n)), name='space of discounted occupational measures'
            )
            # update the model (after adding variables and constraints) and solve
            m.update()
            m.optimize()

            v_star[s] = m.ObjVal
            y_opt = torch.tensor([[y[i, a].X for a in range(k)] for i in range(n)])
            policy[s] = torch.tensor(actions[torch.argmax(y_opt, dim=1)])

        run_time = time.time() - start_time

        return policy, v_star, run_time


class BirthDeath(MDP):
    def __init__(self, n, k, feats, pow1, pow2, beta, mdl):
        feats = torch.tensor(feats) if not isinstance(feats, torch.Tensor) else feats
        self.feats = feats.unsqueeze(0) if feats.ndim == 1 else feats
        self.n_samples, self.n_feats = self.feats.shape
        self.pow1 = pow1
        self.pow2 = pow2
        self.mdl = mdl
        states = np.arange(1, n + 1)
        actions = np.arange(1, k + 1)
        trans_probs = self.gen_trans_probs(n, k, mdl)
        obj_fun = self.gen_obj_fun(n, k)
        gamma = torch.ones(self.n_samples, n) / n
        super().__init__(states, actions, trans_probs, obj_fun, gamma, beta)

    def gen_trans_probs(self, n, k, mdl):
        thetas = mdl(self.feats)
        theta_array = thetas.reshape(self.n_samples, n - 1, k)
        trans_probs = torch.zeros(self.n_samples, n, n, k)
        trans_probs[:, 0, 1, :] = 1
        trans_probs[:, torch.arange(1, n - 1), torch.arange(2, n), :] = theta_array[:, :-1, :]
        trans_probs[:, -1, -1, :] = theta_array[:, -1, :]
        trans_probs[:, :, 0, :] = 1 - torch.sum(trans_probs[:, :, 1:n, :], dim=2)

        return trans_probs

    def gen_obj_fun(self, n, k):
        obj_fun = torch.exp((torch.arange(1, k + 1).reshape(1, -1)) ** self.pow1 /
                            (torch.arange(1, n + 1).reshape(-1, 1)) ** self.pow2)

        return obj_fun.unsqueeze(0).expand(self.n_samples, -1, -1)
