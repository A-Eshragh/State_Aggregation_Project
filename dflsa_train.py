import json
import os
import argparse
import torch.optim as optim
from functools import partial
from mdp_sparse import *
from sa_sparse import state_agg
from utils import *
from qpth.qp import QPFunction
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path


def train_eval(config, params, train_loader, eval_loader):
    learning_rate = config['lr']
    hidden_dim = config['hidden_dim']

    n = params['n_states']
    k = params['n_actions']
    d = params['cluster_size']
    gt_mode = params['gt_mode']
    algo_mode = params['algo_mode']
    pow1 = params['pow1']
    pow2 = params['pow2']
    beta = params['beta']
    n_feats = params['n_feats']
    num_epochs = params['num_epochs']
    reg_coef = params['reg_coef']
    loss_weight = params['loss_weight']

    gt_mdl = partial(expon_mdl, n=n, k=k) if gt_mode == 'exp' else GTNN(n_feats, (n - 1) * k)
    pred_model = ThetaPredictor(n_feats, (n - 1) * k, hidden_dim)
    loss_fn = nn.L1Loss()
    qp_solver = QPFunction()
    optimizer = optim.Adam(pred_model.parameters(), lr=learning_rate)

    train_time_list = []
    obj_loss_list = []  # list of optimality gaps
    start_time = time.time()
    for epoch in range(num_epochs):
        train_loss = 0.0
        obj_loss = 0.0
        pred_model.train()
        for feats, obj_true in train_loader:
            theta_hat = pred_model(feats)
            theta_true = gt_mdl(feats)
            pred_loss = loss_fn(theta_hat, theta_true)
            mdp_hat = BirthDeath(n, k, theta_hat, pow1, pow2, beta)
            if d == -1:
                # in this case, twostage method with policy iteration will be used
                obj_hat = torch.from_numpy(mdp_hat.policy_iteration()[1]).float()  # gradient not required
                obj_diff = loss_fn(obj_hat, obj_true)
                loss = pred_loss
            else:
                mdp_hat_agg = state_agg(mdp_hat, d)[0]
                obj_hat = solve_qp(mdp_hat_agg, reg_coef, qp_solver)[1]
                obj_diff = loss_fn(obj_hat, obj_true)
                loss = pred_loss if algo_mode == 'twostage' else obj_diff + loss_weight * pred_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            obj_loss += obj_diff.item()

        train_time_list.append(time.time() - start_time)
        avg_train_loss = train_loss / len(train_loader)
        avg_obj_loss = obj_loss / len(train_loader)
        obj_loss_list.append(avg_obj_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Objective Loss: {avg_obj_loss:.4f}')

    pred_model.eval()
    pred_loss = 0.0
    eval_loss = 0.0
    start_time = time.time()
    with torch.no_grad():
        for feats, obj_true in eval_loader:
            theta_hat = pred_model(feats)
            theta_true = gt_mdl(feats)
            pred_loss += loss_fn(theta_hat, theta_true).item()
            mdp_hat = BirthDeath(n, k, theta_hat, pow1, pow2, beta)
            if d == -1:
                _, obj_hat, pi_time = mdp_hat.policy_iteration()
                print(f'Policy Iteration Runtime: {pi_time / len(obj_true):.4f}s')
                obj_hat = torch.from_numpy(obj_hat).float()
            else:
                mdp_hat_agg, sa_time = state_agg(mdp_hat, d)
                print(f'State Aggregation Runtime: {sa_time / len(obj_true):.4f}s')
                _, obj_hat, qp_time = solve_qp(mdp_hat_agg, reg_coef, qp_solver)
                print(f'Quadratic Programming Runtime: {qp_time / len(obj_true):.4f}s')

            eval_loss += torch.mean(torch.abs(obj_hat - obj_true) / obj_true).item()

    eval_time = time.time() - start_time
    eval_time /= len(eval_loader.dataset)
    pred_loss /= len(eval_loader)
    eval_loss /= len(eval_loader)
    print(f'Average Runtime: {eval_time:.4f}s')
    print(f'Evaluation Loss: {eval_loss * 100:.4f}%')

    return {'config': config, 'params': params, 'eval_time': eval_time, 'pred_loss': pred_loss,
            'eval_loss': eval_loss, 'obj_loss': obj_loss_list, 'train_time': train_time_list}


def set_params(args):
    # generate the MDP of the input birth and death process
    n = args.n_states  # number of states: |S|
    k = args.n_actions  # number of actions: |A|
    d = args.cluster_size  # the number of states in the cluster set
    gt_mode = args.gt_mode
    algo_mode = args.algo_mode if d != -1 else 'twostage'  # d=-1 indicates full MDP, which is only solved by twostage

    pow1 = 0.8  # constant for calculating cost function
    pow2 = 1  # constant for calculating cost function
    beta = 0.9  # discount factor
    n_feats = 2
    num_epochs = 100
    n_samples = 100
    batch_size = int(0.8 * 0.2 * n_samples)  # ensure the validation set contains exactly one batch
    reg_coef = 0.5  # regularizing coefficient of the quadratic term of the decision variable
    loss_weight = 0  # weight coefficient for the prediction loss in the DFL objective

    params = {'n_states': n, 'n_actions': k, 'cluster_size': d, 'pow1': pow1, 'pow2': pow2, 'beta': beta,
              'n_feats': n_feats, 'num_epochs': num_epochs, 'gt_mode': gt_mode, 'algo_mode': algo_mode,
              'n_samples': n_samples, 'batch_size': batch_size, 'reg_coef': reg_coef, 'loss_weight': loss_weight}

    # Define hyperparameter configurations
    hyperparams = [{'lr': lr, 'hidden_dim': hd}
                   for lr in [5e-3, 1e-3, 5e-4, 1e-4]
                   for hd in [32, 64]]

    return params, hyperparams


def prepare_data(params, mode):
    n = params['n_states']
    k = params['n_actions']
    gt_mode = params['gt_mode']
    pow1 = params['pow1']
    pow2 = params['pow2']
    beta = params['beta']
    n_samples = params['n_samples']
    n_feats = params['n_feats']
    batch_size = params['batch_size']

    # feats = torch.rand(n_samples, n_feats)
    feats = torch.distributions.Cauchy(loc=0.0, scale=1.0).sample(torch.Size([n_samples, n_feats]))
    feats = torch.clamp(feats, min=-50, max=50)  # clip the feature values for numerical stability of neural networks
    gt_mdl = partial(expon_mdl, n=n, k=k) if gt_mode == 'exp' else GTNN(n_feats, (n - 1) * k)
    thetas = gt_mdl(feats)
    mdp_true = BirthDeath(n, k, thetas, pow1, pow2, beta)
    obj_true = torch.from_numpy(mdp_true.policy_iteration()[1]).float()

    train_size = int(0.8 * n_samples)
    test_size = n_samples - train_size
    train_dataset, test_dataset = random_split(TensorDataset(feats, obj_true), [train_size, test_size])
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
    train_loader, test_loader = prepare_data(params, 'train')
    config = hyperparams[task_id]

    tune_result = train_eval(config, params, train_loader, test_loader)

    # Save result to a file
    n = params['n_states']
    k = params['n_actions']
    d = params['cluster_size']
    gt_mode = params['gt_mode']
    algo_mode = params['algo_mode']
    save_dir = Path(f'results/n={n};k={k};d={d}/{gt_mode}/{algo_mode}')
    save_dir.mkdir(parents=True, exist_ok=True)
    result_file = save_dir / Path(f'train_result_{task_id}.json')
    result_file.write_text(json.dumps(tune_result, indent=4))


if __name__ == '__main__':
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID'))

    # Parse input arguments
    parser = argparse.ArgumentParser(description='Process input parameters.')
    parser.add_argument('n_states', type=int, help='the number of states')
    parser.add_argument('n_actions', type=int, help='the number of actions')
    parser.add_argument('cluster_size', type=int, help='the cluster size of the aggregated MDP; -1 means no aggregation')
    parser.add_argument('gt_mode', type=str, help='exp or nn')
    parser.add_argument('algo_mode', type=str, help='dfl or twostage')

    main(parser.parse_args())
