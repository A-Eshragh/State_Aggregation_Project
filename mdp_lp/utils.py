import random
import torch
import torch.nn as nn
import time
import json
import ast
import numpy as np
from pathlib import Path


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


def set_hp_output(args, task_id, save_dir):
    # default hyperparameter configurations
    hyperparams = [{'lr': lr, 'hidden_dim': hd}
                   for lr in [5e-3, 1e-3, 5e-4, 1e-4]
                   for hd in [32, 64]]

    if args.hyperparam is None:
        if args.mode == 'train':
            if task_id == -1:  # default task_id "-1" indicates no job arrays for hyperparameter tuning are provided
                raise ValueError('--hyperparam must be given when not performing hyperparameter tuning.')
            if task_id >= len(hyperparams):
                raise ValueError(f'task_id {task_id} out of range 0–{len(hyperparams) - 1}.')
            hp = hyperparams[task_id]
            result_file = save_dir / Path(f'train_result_{task_id}.json')
        else:  # test mode, after hyperparam tuning
            if task_id != -1:
                raise ValueError('hyperparameter tuning should not be done in test mode.')
            hp = collect_tune_results(save_dir, args.algo)
            result_file = save_dir / Path(f'test_result.json')
    else:
        if task_id != -1:
            raise ValueError('hyperparameter tuning is not needed when a specific hyperparameter is provided.')
        try:
            hp = ast.literal_eval(args.hyperparam)
        except (SyntaxError, ValueError):
            raise ValueError('--hyperparam must be a valid Python literal (e.g., \'{"lr": 0.001}\').')
        if not isinstance(hp, dict):
            raise ValueError('--hyperparam must be a dict, e.g., \'{"lr": 0.001, "hidden_dim": 64}\'')
        result_file = save_dir / Path(f'{args.mode}_result_{hp}.json')

    return hp, result_file


def solve_lp(mdp_lp_obj, lp_solver):
    """
    Solve LP for MDP using mdp_lp's LP formulation.
    
    Args:
        mdp_lp_obj: mdp_lp.mdp.MDP instance
        lp_solver: Solver function (e.g., solve_lp_dense, solve_lp_sparse)
    
    Returns:
        y: (batch, S*A) occupancy measure
        obj: (batch,) optimal values
        run_time: float seconds
    """
    start_time = time.time()
    
    P = mdp_lp_obj.P  # (batch, S, A, S)
    r = mdp_lp_obj.r  # (batch, S, A)
    
    H, r_flat, gamma = mdp_lp_obj.lp_formulation(P, r)
    
    # Solve LP for each batch element
    batch = P.shape[0]
    y_list = []
    obj_list = []
    
    for b in range(batch):
        y_b, _ = lp_solver(H[b:b+1], r_flat[b:b+1], gamma[b:b+1])
        y_list.append(y_b[0])  # Remove batch dimension
        obj_b = torch.sum(r_flat[b] * y_b[0])
        obj_list.append(obj_b)
    
    y = torch.stack(y_list, dim=0)  # (batch, S*A)
    obj = torch.stack(obj_list, dim=0)  # (batch,)
    
    run_time = time.time() - start_time
    
    return y, obj, run_time


