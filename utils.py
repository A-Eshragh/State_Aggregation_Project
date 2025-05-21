import random
import torch
import time
import numpy as np
import torch.nn as nn
# import cvxpy as cp
# from cvxpylayers.torch import CvxpyLayer


class GTNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()  # Ensure outputs are between 0 and 1
        )

    def forward(self, x):
        return self.model(x)


class ThetaPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


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


def expon_mdl(feats, n, k):
    assert feats.ndim == 2 and feats.shape[1] == 2
    transform_matrix = torch.stack([torch.ones(n - 1), torch.arange(2, n + 1)])
    numerator = torch.repeat_interleave(torch.matmul(feats, transform_matrix), repeats=k, dim=1)
    denominator = torch.arange(1, k + 1).repeat(n - 1)
    expon = torch.exp(- numerator / denominator)  # avoid numerical overflow when n is large
    thetas = expon / (expon + 1)

    return thetas


def solve_qp(mdp, quad_coef, qp_solver):
    start_time = time.time()

    trans_probs = mdp.trans_probs
    obj_fun = mdp.obj_fun
    gamma = mdp.init_dist
    beta = mdp.disc_factor
    n_samples, n, k = obj_fun.shape

    # Objective: 0.5 z^T P z + q^T z
    # P = 2 * quad_coef * I
    # q = flattened obj_fun
    quad_mat = 2 * quad_coef * torch.eye(n * k).unsqueeze(0).repeat(n_samples, 1, 1)  # Shape: (n_samples, n*k, n*k)
    lin_coef = obj_fun.reshape(n_samples, -1)

    # Gz <= h (inequality constraints): y >= 0 => -I y <= 0
    ineq_coef = - torch.eye(n * k).unsqueeze(0).repeat(n_samples, 1, 1)  # Shape: (n_samples, n*k, n*k)
    ineq_rhs = torch.zeros(n_samples, n * k)  # Shape: (n_samples, n*k)

    # Construct the equality coefficient array
    # A z = b, where A has shape (n_samples, n, n*k)
    eq_coef = torch.eye(n).unsqueeze(0).unsqueeze(3).repeat(n_samples, 1, 1, k) - beta * trans_probs  # Shape: (n_samples, n, n, k)
    # permute dimensions 1 and 2 because we need the constraint for each j (dim 2)
    eq_coef = eq_coef.permute(0, 2, 1, 3).reshape(n_samples, n, n * k)
    eq_rhs = gamma  # Shape: (n_samples, n)

    # Solve QP: min 0.5 z^T P z + q^T z
    # Subject to Az = b and Gz <= h
    y = qp_solver(quad_mat, lin_coef, ineq_coef, ineq_rhs, eq_coef, eq_rhs)
    obj = torch.sum(lin_coef * y, dim=1)

    run_time = time.time() - start_time

    return y, obj, run_time


# def solve_intpoint(mdp, regularize_coef):
#     start_time = time.time()
#
#     trans_probs = mdp.trans_probs
#     obj_fun = mdp.obj_fun
#     gamma = mdp.init_dist
#     beta = mdp.disc_factor
#     n_samples, n, k = obj_fun.shape
#
#     # Define LP variables in CVXPY
#     y = cp.Variable(n * k)  # Decision variable
#     a_param = cp.Parameter((n, n * k))  # Constraints matrix
#     b_param = cp.Parameter(n)  # RHS vector
#     c_param = cp.Parameter(n * k)  # Objective coefficients
#     # Define LP problem
#     objective = cp.Minimize(c_param.T @ y - regularize_coef * cp.sum(cp.log(y)))
#     constraints = [a_param @ y == b_param, y >= 0]
#     problem = cp.Problem(objective, constraints)
#     # Convert to a differentiable layer
#     cvxpylayer = CvxpyLayer(problem, parameters=[a_param, b_param, c_param], variables=[y])
#
#     y_opt = torch.zeros(n_samples, n * k)
#     obj = torch.zeros(n_samples)
#     for s in range(n_samples):
#         a_input = torch.eye(n).unsqueeze(2).repeat(1, 1, k) - beta * trans_probs[s]
#         a_input = a_input.permute(1, 0, 2).reshape(n, n * k)
#         b_input = gamma[s]
#         c_input = obj_fun[s].reshape(-1)
#
#         y_opt[s], = cvxpylayer(a_input, b_input, c_input)
#         obj[s] = torch.sum(c_input * y_opt[s])
#
#     run_time = time.time() - start_time
#
#     return y_opt, obj, run_time
