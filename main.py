import os
import argparse
import ast
import json
from pathlib import Path
from train_eval import train_eval, prepare_data
from utils import set_seed, collect_tune_results


def config(args):
    # MDP configuration
    mdp_type = args.mdp_type
    n = args.n_states  # number of states: |S|
    k = args.n_actions  # number of actions: |A|
    m = args.n_neighbors  # number of neighboring states per row in the transition probability
    two_way = args.two_way  # allow states to transition in both ways?
    assert m < (n // 2 if two_way else n)
    pow1 = 0.8  # constant for calculating cost function
    pow2 = 1  # constant for calculating cost function
    beta = 0.9  # discount factor

    d = args.cluster_size  # the number of states in the cluster set
    algo = args.algo if d != -1 else 'twostage'  # d=-1 indicates full MDP, which is only solved by twostage
    mode = args.mode  # train or test
    n_feats = 2
    num_epochs = 100
    n_samples = 100
    batch_size = int(0.8 * 0.2 * n_samples) if mode == 'train' else int(0.2 * n_samples)
    reg_coef = 0.5  # regularizing coefficient for the quadratic term of the decision variable
    loss_weight = 0.5  # weight coefficient for the prediction loss in the DFL objective function

    params = {'n_states': n, 'n_actions': k, 'n_neighbors': m, 'two_way': two_way, 'pow1': pow1, 'pow2': pow2, 'beta': beta,
              'cluster_size': d, 'algo': algo, 'mode': mode, 'n_feats': n_feats, 'num_epochs': num_epochs,
              'n_samples': n_samples, 'batch_size': batch_size, 'reg_coef': reg_coef, 'loss_weight': loss_weight}

    save_dir = Path(f'results/{mdp_type}/n={n};k={k}/m={m};twoway={two_way}/d={d}/{algo}')
    save_dir.mkdir(parents=True, exist_ok=True)

    # Define hyperparameter configurations
    hyperparams = [{'lr': lr, 'hidden_dim': hd}
                   for lr in [5e-3, 1e-3, 5e-4, 1e-4]
                   for hd in [32, 64]]

    if args.hyperparam is None:
        if mode == 'train':
            if task_id == -1:  # default task_id "-1" indicates no job arrays for hyperparameter tuning are provided
                parser.error('--hyperparam must be given when not performing hyperparameter tuning.')
            if task_id >= len(hyperparams):
                parser.error(f'task_id {task_id} out of range 0–{len(hyperparams) - 1}.')
            hp = hyperparams[task_id]
            result_file = save_dir / Path(f'train_result_{task_id}.json')
        else:  # test mode, after hyperparam tuning
            if task_id != -1:
                parser.error('hyperparameter tuning should not be done in test mode.')
            hp = collect_tune_results(save_dir, algo)
            result_file = save_dir / Path(f'test_result.json')
    else:
        if task_id != -1:
            parser.error('hyperparameter tuning is not needed when a specific hyperparameter is provided.')
        try:
            hp = ast.literal_eval(args.hyperparam)
        except (SyntaxError, ValueError):
            parser.error('--hyperparam must be a valid Python literal (e.g., \'{"lr": 0.001}\').')
        if not isinstance(hp, dict):
            parser.error('--hyperparam must be a dict, e.g., \'{"lr": 0.001, "hidden_dim": 64}\'')

        result_file = save_dir / Path(f'{mode}_result_{hp}.json')

    return params, hp, result_file


def main(args):
    set_seed(42)
    params, hp, result_file = config(args)
    train_loader, test_loader = prepare_data(params, args.mode)

    tune_result = train_eval(params, hp, train_loader, test_loader)

    # Save result to a file
    result_file.write_text(json.dumps(tune_result, indent=4))


if __name__ == '__main__':
    # if the job array is not used, task_id is -1 by default
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', -1))

    # Parse input arguments
    parser = argparse.ArgumentParser(description='Process input parameters.')
    parser.add_argument('--mdp_type', type=str, default='bnd', help='the type of the MDPs: bnd or inventory')
    parser.add_argument('--n_states', type=int, default=10000, help='the number of states')
    parser.add_argument('--n_actions', type=int, default=10, help='the number of actions')
    parser.add_argument('--n_neighbors', type=int, default=5,
                        help='the number of neighboring states per row in the transition probs')
    parser.add_argument('--two_way', type=lambda s: s.lower() == 'y', default=True,
                        help='whether two-way transitions are allowed')
    parser.add_argument('--cluster_size', type=int, default=20,
                        help='the cluster size of the aggregated MDP; -1 means no aggregation')
    parser.add_argument('--algo', type=str, default='dfl', help='algorithm: dfl or twostage')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--hyperparam', type=str, default=None, help='hyperparameter of user choice')

    main(parser.parse_args())
