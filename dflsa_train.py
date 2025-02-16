import json
import os
import argparse
import torch.optim as optim
from functools import partial
from mdp import *
from utils import *
from qpth.qp import QPFunction
from state_aggregation_alg import state_agg
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path


def train_eval(config, params, train_loader, eval_loader):
    learning_rate = config['lr']
    hidden_dim = config['hidden_dim']

    n = params['n_states']
    k = params['n_actions']
    pow1 = params['pow1']
    pow2 = params['pow2']
    beta = params['beta']
    d = params['cluster_size']
    n_feats = params['n_feats']
    num_epochs = params['num_epochs']
    quad_coef = params['quad_coef']
    gt_mode = params['gt_mode']
    algo_mode = params['algo_mode']

    gt_mdl = partial(expon_mdl, n=n, k=k) if gt_mode == 'exp' else GTNN(n_feats, (n - 1) * k)
    loss_fn = nn.L1Loss()
    qp_solver = QPFunction()

    pred_model = ThetaPredictor(n_feats, (n - 1) * k, hidden_dim)
    optimizer = optim.Adam(pred_model.parameters(), lr=learning_rate)

    # train_loss_list = []
    obj_loss_list = []
    for epoch in range(num_epochs):
        train_loss = 0.0
        obj_loss = 0.0
        pred_model.train()
        for batch in train_loader:
            feats = batch[0]
            mdp_hat = BirthDeath(n, k, feats, pow1, pow2, beta, pred_model)
            mdp_hat_agg = state_agg(mdp_hat, d)[0]
            obj_hat = solve_qp(mdp_hat_agg, quad_coef, qp_solver)[1]

            mdp_true = BirthDeath(n, k, feats, pow1, pow2, beta, gt_mdl)
            obj_true = mdp_true.policy_iteration()[1]
            obj_diff = loss_fn(obj_hat, obj_true)
            if algo_mode == 'dflsa':
                loss = obj_diff
            else:
                theta_hat = pred_model(feats)
                theta_true = gt_mdl(feats)
                loss = loss_fn(theta_hat, theta_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            obj_loss += obj_diff.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_obj_loss = obj_loss / len(train_loader)
        # train_loss_list.append(avg_train_loss)
        obj_loss_list.append(avg_obj_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Objective Loss: {avg_obj_loss:.4f}')

    pred_model.eval()
    eval_time = 0.0
    estimation_loss = 0.0
    eval_loss = 0.0
    with torch.no_grad():
        for batch in eval_loader:
            feats = batch[0]
            start_time = time.time()
            mdp_hat = BirthDeath(n, k, feats, pow1, pow2, beta, pred_model)
            mdp_hat_agg = state_agg(mdp_hat, d)[0]
            obj_hat = solve_qp(mdp_hat_agg, quad_coef, qp_solver)[1]
            eval_time += time.time() - start_time

            mdp_true = BirthDeath(n, k, feats, pow1, pow2, beta, gt_mdl)
            obj_true = mdp_true.policy_iteration()[1]
            eval_loss += loss_fn(obj_hat, obj_true).item()

            theta_hat = pred_model(feats)
            theta_true = gt_mdl(feats)
            estimation_loss += loss_fn(theta_hat, theta_true).item()

    eval_time /= len(eval_loader.dataset)
    estimation_loss /= len(eval_loader)
    eval_loss /= len(eval_loader)
    print(f'Average Run Time: {eval_time:.4f}')
    print(f'Evaluation Loss: {eval_loss:.4f}')

    return {'config': config, 'params': params, 'eval_time': eval_time, 'estimation_loss': estimation_loss,
            'eval_loss': eval_loss, 'obj_loss': obj_loss_list}


def set_params(args):
    # generate the MDP of the input birth and death process
    n = args.n_states  # number of states: |S|
    k = args.n_actions  # number of actions: |A|
    pow1 = 0.8  # constant for calculating cost function
    pow2 = 1  # constant for calculating cost function
    beta = 0.9  # discount factor
    d = 5  # the number of states in the cluster set
    n_feats = 2
    num_epochs = 100
    quad_coef = 5
    gt_mode = args.gt_mode
    algo_mode = args.algo_mode
    n_samples = 100
    batch_size = 16

    params = {'n_states': n, 'n_actions': k, 'pow1': pow1, 'pow2': pow2, 'beta': beta, 'cluster_size': d,
              'n_feats': n_feats, 'num_epochs': num_epochs, 'quad_coef': quad_coef,
              'gt_mode': gt_mode, 'algo_mode': algo_mode, 'n_samples': n_samples, 'batch_size': batch_size}

    # Define hyperparameter configurations
    hyperparams = [{'lr': lr, 'hidden_dim': hd}
                   for lr in [1e-3, 5e-4, 1e-4, 5e-5]
                   for hd in [32, 64, 128]]

    return params, hyperparams


def split_data(params, mode):
    n_samples = params['n_samples']
    n_feats = params['n_feats']
    batch_size = params['batch_size']

    feats = torch.rand(n_samples, n_feats)
    train_size = int(0.8 * n_samples)
    test_size = n_samples - train_size
    train_dataset, test_dataset = random_split(TensorDataset(feats), [train_size, test_size])
    if mode == 'train':
        train_tune_size = int(0.8 * train_size)
        val_size = train_size - train_tune_size
        train_dataset, test_dataset = random_split(train_dataset, [train_tune_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def main(args):
    set_seed(42)
    params, hyperparams = set_params(args)
    train_loader, test_loader = split_data(params, 'train')
    config = hyperparams[task_id]

    tune_result = train_eval(config, params, train_loader, test_loader)

    # Save result to a file
    gt_mode = params['gt_mode']
    algo_mode = params['algo_mode']
    save_dir = Path(f'results/{gt_mode}/{algo_mode}')
    save_dir.mkdir(parents=True, exist_ok=True)
    result_file = save_dir / Path(f'train_result_{task_id}.json')
    result_file.write_text(json.dumps(tune_result, indent=4))


if __name__ == '__main__':
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID'))

    # Parse input arguments
    parser = argparse.ArgumentParser(description='Process input parameters.')
    parser.add_argument('n_states', type=int, help='the number of states')
    parser.add_argument('n_actions', type=int, help='the number of actions')
    parser.add_argument('gt_mode', type=str, help='exp or nn')
    parser.add_argument('algo_mode', type=str, help='dflsa or twostage')
    main(parser.parse_args())
