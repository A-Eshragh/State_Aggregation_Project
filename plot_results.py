import json
import matplotlib.pyplot as plt
from pathlib import Path


def plot_epoch(n, k, d):
    sa_dir = Path(f'results/n={n};k={k};d={d}/{gt_mode}')
    full_dir = Path(f'results/n={n};k={k};d={-1}/{gt_mode}')
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
    plt.show()
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
    plt.show()
    plt.close()


def plot_d(n, k, d_list):
    paths = []
    for d in d_list:
        sa_path = Path(f'results/n={n};k={k};d={d}/{gt_mode}/dfl/test_result.json')
        paths.append(sa_path)

    full_path = Path(f'results/n={n};k={k};d={-1}/{gt_mode}/twostage/test_result.json')
    paths.append(full_path)

    losses = []
    times = []
    for path in paths:
        losses.append(json.loads(path.read_text())['eval_loss'])
        times.append(json.loads(path.read_text())['eval_time'])

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
    plt.savefig(Path(f'results/n={n}_k={k}_optimality_gap.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Plot the computation time
    fig, ax = plt.subplots(figsize=(6.5, 1.5))
    ax.plot(d_list, times[:-1], label='DFL+SA', linewidth=2, color='blue')
    ax.plot([min(d_list), max(d_list)], [times[-1], times[-1]], label='Two-stage', linewidth=2, linestyle='dotted', color='green')
    ax.set_xticks(d_list)
    # Labels and title
    ax.set_xlabel('d')
    ax.set_ylabel('Computation Time')
    ax.legend(frameon=False, fontsize='small')
    # Remove only the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Save the figure
    plt.savefig(Path(f'results/n={n}_k={k}_computation_time.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == '__main__':
    n_states = 2000
    n_actions = 10
    cluster_size = 10
    gt_mode = 'nn'
    plot_epoch(n_states, n_actions, cluster_size)
    plot_d(n_states, n_actions, [10, 20, 30, 40, 50])
