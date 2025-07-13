import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def plot_epoch(mdp_type, n, k, m, two_way, d):
    sa_dir = Path(f'results/{mdp_type}/n={n};k={k}/m={m};twoway={two_way}/d={d}')
    full_dir = Path(f'results/{mdp_type}/n={n};k={k}/m={m};twoway={two_way}/d={-1}')
    dflsa_path = sa_dir / Path(f'dfl/test_result.json')
    twostagesa_path = sa_dir / Path(f'twostage/test_result.json')
    twostage_path = full_dir / Path(f'twostage/test_result.json')

    paths = [dflsa_path, twostagesa_path, twostage_path]
    losses = []
    times = []
    for path in paths:
        losses.append(json.loads(path.read_text())['obj_loss'])
        times.append(json.loads(path.read_text())['train_time'])

    epochs = list(range(1, 101))
    # Plot the optimality gap
    fig, ax = plt.subplots(figsize=(6.5, 1.5))
    ax.plot(epochs, losses[0], label='DFL+SA', linewidth=2, color='blue')
    ax.plot(epochs, losses[1], label='Two-stage+SA', linewidth=2, linestyle='dashed', color='red')
    ax.plot(epochs, losses[2], label='Two-stage', linewidth=2, linestyle='dotted', color='green')
    # Labels and title
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Optimality Gap')
    ax.legend(frameon=False, fontsize='small')
    # Remove only the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Save the figure
    plt.savefig(sa_dir / Path('training_optimality_gap.png'), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

    # Plot the training time
    fig, ax = plt.subplots(figsize=(6.5, 1.5))
    ax.plot(epochs, times[0], label='DFL+SA', linewidth=2, color='blue')
    ax.plot(epochs, times[1], label='Two-stage+SA', linewidth=2, linestyle='dashed', color='red')
    ax.plot(epochs, times[2], label='Two-stage', linewidth=2, linestyle='dotted', color='green')
    # Labels and title
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Computation Time')
    ax.legend(frameon=False, fontsize='small')
    # Remove only the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Save the figure
    plt.savefig(sa_dir / Path('training_computation_time.png'), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_d(mdp_type, n, k, m, two_way, d_list):
    master_dir = Path(f'results/{mdp_type}/n={n};k={k}/m={m};twoway={two_way}')
    paths = []
    for d in d_list:
        sa_path = master_dir / Path(f'd={d}/dfl/test_result.json')
        paths.append(sa_path)

    full_path = master_dir / Path(f'd={-1}/twostage/test_result.json')
    paths.append(full_path)

    losses = []
    times = []
    for path in paths:
        losses.append(json.loads(path.read_text())['eval_loss'])
        times.append(json.loads(path.read_text())['eval_time'])

    losses = [loss * 100 for loss in losses]  # convert to percentage scale
    # Plot the optimality gap
    fig, ax = plt.subplots(figsize=(6.5, 1.5))
    ax.plot(d_list, losses[:-1], label='DFL+SA', linewidth=2, color='blue')
    ax.plot([min(d_list), max(d_list)], [losses[-1], losses[-1]], label='Two-stage', linewidth=2, linestyle='dotted', color='green')
    ax.set_xticks(d_list)
    # Labels and title
    ax.set_xlabel('d')
    ax.set_ylabel('Optimality Gap (%)')
    ax.legend(frameon=False, fontsize='small')
    # Remove only the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Save the figure
    plt.savefig(master_dir / Path(f'test_optimality_gap.png'), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

    # Plot the computation time
    fig, ax = plt.subplots(figsize=(6.5, 1.5))
    ax.plot(d_list, times[:-1], label='DFL+SA', linewidth=2, color='blue')
    ax.plot([min(d_list), max(d_list)], [times[-1], times[-1]], label='Two-stage', linewidth=2, linestyle='dotted', color='green')
    ax.set_xticks(d_list)
    ax.set_yscale('log')
    # Labels and title
    ax.set_xlabel('d')
    ax.set_ylabel('Log-Scale Time')
    ax.legend(frameon=False, fontsize='small')
    # Remove only the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Save the figure
    plt.savefig(master_dir / Path(f'test_computation_time.png'), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()


def main(args):
    mdp_type = args.mdp_type
    n = args.n_states
    k = args.n_actions
    m = args.n_neighbors
    two_way = args.two_way
    d = args.base_cluster_size
    d_list = args.cluster_size_list
    plot_epoch(mdp_type, n, k, m, two_way, d)
    plot_d(mdp_type, n, k, m, two_way, d_list)


if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser(description='Process input parameters.')
    parser.add_argument('--mdp_type', type=str, default='bnd', help='the type of the MDPs: bnd or inventory')
    parser.add_argument('--n_states', type=int, default=10000, help='the number of states')
    parser.add_argument('--n_actions', type=int, default=10, help='the number of actions')
    parser.add_argument('--n_neighbors', type=int, default=5,
                        help='the number of neighboring states per row in the transition probs')
    parser.add_argument('--two_way', type=lambda s: s.lower() == 'y', default=True,
                        help='whether two-way transitions are allowed')
    parser.add_argument('--base_cluster_size', type=int, default=20,
                        help='the base cluster size of the aggregated MDP for training comparison')
    parser.add_argument('--cluster_size_list', type=int, nargs='+', default=[10, 15, 20, 25, 30, 35, 40, 45, 50],
                        help='list of cluster sizes of the aggregated MDP for test comparison')

    main(parser.parse_args())
