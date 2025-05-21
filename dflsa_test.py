import ast
from dflsa_train import *

def collect_tune_results(path, algo_mode):
    files = list(path.glob('*.json'))
    all_results = [json.loads(file.read_text()) for file in files]
    if algo_mode == 'dfl':
        best_config = min(all_results, key=lambda x: x['eval_loss'])['config']
    else:
        best_config = min(all_results, key=lambda x: x['pred_loss'])['config']

    return best_config


def main(args):
    set_seed(42)
    params, hyperparams = set_params(args)
    params['batch_size'] = int(0.2 * params['n_samples'])
    train_loader, test_loader = prepare_data(params, 'test')

    n = params['n_states']
    k = params['n_actions']
    d = params['cluster_size']
    gt_mode = params['gt_mode']
    algo_mode = params['algo_mode']
    tune_results_path = Path(f'results/n={n};k={k};d={d}/{gt_mode}/{algo_mode}')
    config = collect_tune_results(tune_results_path, algo_mode) if args.hyperparam is None else ast.literal_eval(args.hyperparam)

    test_result = train_eval(config, params, train_loader, test_loader)

    # Save result to a file
    result_file = tune_results_path / Path(f'test_result.json')
    result_file.write_text(json.dumps(test_result, indent=4))


if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser(description='Process input parameters.')
    parser.add_argument('n_states', type=int, help='the number of states')
    parser.add_argument('n_actions', type=int, help='the number of actions')
    parser.add_argument('cluster_size', type=int, help='the cluster size of the aggregated MDP; -1 means no aggregation')
    parser.add_argument('gt_mode', type=str, help='exp or nn')
    parser.add_argument('algo_mode', type=str, help='dfl or twostage')
    parser.add_argument('--hyperparam', type=str, default=None, help='hyperparameter of user choice')
    main(parser.parse_args())
