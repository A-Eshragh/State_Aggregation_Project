import json
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def plot_epoch(mdp_type, n, k, m, d):
    sa_dir = Path(f'results/{mdp_type}/m={m}/n={n};k={k};d={d}')
    full_dir = Path(f'results/{mdp_type}/m={m}/n={n};k={k};d={-1}')
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
    plt.savefig(sa_dir / Path('optimality_gap.png'), dpi=300, bbox_inches='tight')
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
    plt.savefig(sa_dir / Path('computation_time.png'), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_d(mdp_type, n, k, m, d_list):
    paths = []
    for d in d_list:
        sa_path = Path(f'results/{mdp_type}/m={m}/n={n};k={k};d={d}/dfl/test_result.json')
        paths.append(sa_path)

    full_path = Path(f'results/{mdp_type}/m={m}/n={n};k={k};d={-1}/twostage/test_result.json')
    paths.append(full_path)

    losses = []
    times = []
    for path in paths:
        losses.append(json.loads(path.read_text())['eval_loss'])
        times.append(json.loads(path.read_text())['eval_time'])
    # losses = [loss * 100 for loss in losses]  # convert to percentage scale
    # Plot the optimality gap
    fig, ax = plt.subplots(figsize=(6.5, 1.5))
    ax.plot(d_list, losses[:-1], label='DFL+SA', linewidth=2, color='blue')
    ax.plot([min(d_list), max(d_list)], [losses[-1], losses[-1]], label='Two-stage', linewidth=2, linestyle='dotted', color='green')
    ax.set_xticks(d_list)
    # Labels and title
    ax.set_xlabel('d')
    ax.set_ylabel('Optimality Gap')
    ax.legend(frameon=False, fontsize='small')
    # Remove only the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Save the figure
    plt.savefig(Path(f'results/{mdp_type}/m={m}/n={n}_k={k}_optimality_gap.png'), dpi=300, bbox_inches='tight')
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
    plt.savefig(Path(f'results/{mdp_type}/m={m}/n={n}_k={k}_computation_time.png'), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()


def main(args):
    mdp_type = args.mdp_type
    n = args.n_states
    k = args.n_actions
    m = args.n_neighbors
    d = args.cluster_size
    plot_epoch(mdp_type, n, k, m, d)
    plot_d(mdp_type, n, k, m, [10, 20, 30, 40, 50])


if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser(description='Process input parameters.')
    parser.add_argument('--mdp_type', type=str, default='bnd', choices=['bnd', 'inventory'], help='the type of the MDPs')
    parser.add_argument('--n_states', type=int, default=200, help='the number of states')
    parser.add_argument('--n_actions', type=int, default=10, help='the number of actions')
    parser.add_argument('--n_neighbors', type=int, default=1, help='the number of neighboring states per row in the transition probs')
    parser.add_argument('--cluster_size', type=int, default=10, help='the cluster size of the aggregated MDP; -1 means no aggregation')

    main(parser.parse_args())
