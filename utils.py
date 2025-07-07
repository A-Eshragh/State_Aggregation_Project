import random
import torch
import time
import json
import numpy as np
import torch.nn as nn


class GroundTruth(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)


class ThetaPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)


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


def collect_tune_results(path, algo):
    files = list(path.glob('*.json'))
    if not files:
        raise FileNotFoundError(f'No *.json tune files found in {path}. Train first or supply --hyperparam.')
    all_results = [json.loads(f.read_text()) for f in files]
    key = 'eval_loss' if algo == 'dfl' else 'pred_loss'

    return min(all_results, key=lambda x: x[key])['hyperparam']


# def expon_mdl(feats, n, k):
#     assert feats.ndim == 2 and feats.shape[1] == 2
#     transform_matrix = torch.stack([torch.ones(n - 1), torch.arange(2, n + 1)])
#     numerator = torch.repeat_interleave(torch.matmul(feats, transform_matrix), repeats=k, dim=1)
#     denominator = torch.arange(1, k + 1).repeat(n - 1)
#     expon = torch.exp(- numerator / denominator)  # avoid numerical overflow when n is large
#     thetas = expon / (expon + 1)
#
#     return thetas


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
