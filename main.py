import os
import argparse
import json
from pathlib import Path
from train_eval import train_eval, prepare_data
from utils import set_seed, set_hp_output


def config(args):
    # preprocess input values
    # args.algo = args.algo if args.n_agg != -1 else 'twostage'  # n_agg=-1 indicates full MDP, only solved by twostage
    save_dir = Path()
    match args.mdp_type:
        case 'bnd':
            assert args.n_neighbors < (args.n_states // 2 if args.two_way else args.n_state)
            save_dir = Path(f'results/bnd/n={args.n_states};k={args.n_actions}/m={args.n_neighbors};twoway={args.two_way}')
        case 'inventory':
            assert args.q_max + args.d_max < args.b_max + args.s_max
            save_dir = Path(f'results/inventory/b_max={args.b_max};s_max={args.s_max}/q_max={args.q_max};d_max={args.d_max}')
        case _:
            raise ValueError('No such MDP type!')
    save_dir = save_dir / Path(f'n_agg={args.n_agg}/{args.algo}')
    save_dir.mkdir(parents=True, exist_ok=True)
    hp, result_file = set_hp_output(args, task_id, save_dir)
    args.hyperparam = hp

    return args, result_file


def main(args):
    set_seed(42)
    args, result_file = config(args)
    train_loader, test_loader = prepare_data(args)

    tune_result = train_eval(args, train_loader, test_loader)
    # Save results to a file
    result_file.write_text(json.dumps(tune_result, indent=4))


if __name__ == '__main__':
    # if the job array is not used, task_id is -1 by default
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', -1))

    # Define common arguments in a shared parser
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--beta', type=float, default=0.95, help='discount factor')
    common_parser.add_argument('--n_agg', type=int, default=10,
                               help='the number of standalone states after aggregation; -1 for no aggregation')
    common_parser.add_argument('--algo', type=str, default='dfl', help='algorithm: dfl or twostage')
    common_parser.add_argument('--mode', type=str, default='train', help='train or test')
    common_parser.add_argument('--n_feats', type=int, default=2, help='dimension of the input features')
    common_parser.add_argument('--n_epochs', type=int, default=100, help='the number of training epochs')
    common_parser.add_argument('--n_samples', type=int, default=100, help='the number of MDP instances')
    common_parser.add_argument('--reg_coef', type=float, default=0.5,
                               help='regularizing coefficient for the quadratic term')
    common_parser.add_argument('--loss_weight', type=float, default=0.5,
                               help='weight coefficient for the prediction loss')
    common_parser.add_argument('--hyperparam', type=str, default=None,
                               help='hyperparameter of user choice')
    common_parser.add_argument('--train_log', type=lambda s: s.lower() == 'y', default=False, help='train log')

    # Define the main parser
    parser = argparse.ArgumentParser(description='Process MDP input parameters.')
    subparsers = parser.add_subparsers(dest='mdp_type', required=True, help='Type of MDP: bnd or inventory')

    # Subparser for BND
    parser_bnd = subparsers.add_parser('bnd', parents=[common_parser], help='Birth-and-death MDP')
    parser_bnd.add_argument('--n_states', type=int, default=10000, help='the number of states')
    parser_bnd.add_argument('--n_actions', type=int, default=10, help='the number of actions')
    parser_bnd.add_argument('--n_neighbors', type=int, default=5,
                            help='the number of neighboring states per row')
    parser_bnd.add_argument('--two_way', type=lambda s: s.lower() == 'y', default=True,
                            help='two-way transitions allowed')
    parser_bnd.add_argument('--pow1', type=float, default=0.5, help='exponent of action in the objective')
    parser_bnd.add_argument('--pow2', type=float, default=1.5, help='exponent of state in the objective')

    # Subparser for Inventory
    parser_inv = subparsers.add_parser('inventory', parents=[common_parser], help='Inventory MDP')
    parser_inv.add_argument('--b_max', type=int, default=300, help='maximum backlog')
    parser_inv.add_argument('--s_max', type=int, default=500, help='maximum storage')
    parser_inv.add_argument('--q_max', type=int, default=10, help='maximum order quantity')
    parser_inv.add_argument('--d_max', type=int, default=8, help='maximum demand')
    parser_inv.add_argument('--foc', type=float, default=2, help='fixed ordering cost')
    parser_inv.add_argument('--voc', type=float, default=0.4, help='variable ordering cost')
    parser_inv.add_argument('--hc', type=float, default=0.2, help='holding cost')
    parser_inv.add_argument('--bc', type=float, default=0.5, help='backorder cost')

    main(parser.parse_args())
