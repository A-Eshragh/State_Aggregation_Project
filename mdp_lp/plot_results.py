import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def plot_epoch(mdp_type, n, k, m, two_way):
    """
    Plot training curves: optimality gap and computation time vs epochs.
    Compares DFL vs TwoStage algorithms.
    """
    master_dir = Path(__file__).parent / Path(f'results_lp/{mdp_type}/n={n};k={k}/m={m};twoway={two_way}/solver=dense')
    
    # Try test_result.json first, then fall back to train_result.json
    dfl_test = master_dir / Path(f'dfl/test_result.json')
    twostage_test = master_dir / Path(f'twostage/test_result.json')
    
    # Find train_result.json files (they have hyperparam in filename)
    dfl_train_files = list((master_dir / Path('dfl')).glob('train_result_*.json'))
    twostage_train_files = list((master_dir / Path('twostage')).glob('train_result_*.json'))
    
    dfl_path = dfl_test if dfl_test.exists() else (dfl_train_files[0] if dfl_train_files else None)
    twostage_path = twostage_test if twostage_test.exists() else (twostage_train_files[0] if twostage_train_files else None)

    paths = [dfl_path, twostage_path]
    losses = []
    times = []
    
    for path in paths:
        if path and path.exists():
            data = json.loads(path.read_text())
            losses.append(data.get('obj_loss', []))
            times.append(data.get('train_time', []))
        else:
            print(f"Warning: {path} not found, skipping...")
            losses.append([])
            times.append([])

    if not losses[0] and not losses[1]:
        print("No data found to plot!")
        return

    epochs = list(range(1, len(losses[0]) + 1)) if losses[0] else list(range(1, len(losses[1]) + 1))
    
    # Plot the optimality gap
    fig, ax = plt.subplots(figsize=(6.5, 1.5))
    if losses[0]:
        ax.plot(epochs, losses[0], label='DFL', linewidth=2, color='blue')
    if losses[1]:
        ax.plot(epochs, losses[1], label='Two-stage', linewidth=2, linestyle='dashed', color='red')
    # Labels and title
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Optimality Gap')
    ax.legend(frameon=False, fontsize='small')
    # Remove only the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Save the figure
    plt.savefig(master_dir / Path('training_optimality_gap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot the training time
    fig, ax = plt.subplots(figsize=(6.5, 1.5))
    if times[0]:
        ax.plot(epochs, times[0], label='DFL', linewidth=2, color='blue')
    if times[1]:
        ax.plot(epochs, times[1], label='Two-stage', linewidth=2, linestyle='dashed', color='red')
    # Labels and title
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Computation Time')
    ax.legend(frameon=False, fontsize='small')
    # Remove only the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Save the figure
    plt.savefig(master_dir / Path('training_computation_time.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_eval_comparison(mdp_type, n, k, m, two_way):
    """
    Plot evaluation metrics: optimality gap and computation time.
    Compares DFL vs TwoStage algorithms.
    """
    master_dir = Path(__file__).parent / Path(f'results_lp/{mdp_type}/n={n};k={k}/m={m};twoway={two_way}/solver=dense')
    
    # Try test_result.json first, then fall back to train_result.json
    dfl_test = master_dir / Path(f'dfl/test_result.json')
    twostage_test = master_dir / Path(f'twostage/test_result.json')
    
    dfl_train_files = list((master_dir / Path('dfl')).glob('train_result_*.json'))
    twostage_train_files = list((master_dir / Path('twostage')).glob('train_result_*.json'))
    
    dfl_path = dfl_test if dfl_test.exists() else (dfl_train_files[0] if dfl_train_files else None)
    twostage_path = twostage_test if twostage_test.exists() else (twostage_train_files[0] if twostage_train_files else None)

    losses = []
    times = []
    labels = []
    
    for path, label in [(dfl_path, 'DFL'), (twostage_path, 'Two-stage')]:
        if path and path.exists():
            data = json.loads(path.read_text())
            losses.append(data.get('eval_loss', 0) * 100)  # Convert to percentage
            times.append(data.get('eval_time', 0))
            labels.append(label)
        else:
            print(f"Warning: {path} not found, skipping {label}...")

    if not losses:
        print("No data found to plot!")
        return

    # Plot the optimality gap
    fig, ax = plt.subplots(figsize=(6.5, 1.5))
    ax.bar(labels, losses, color=['blue', 'red'][:len(labels)], alpha=0.7)
    ax.set_ylabel('Optimality Gap (%)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(master_dir / Path(f'test_optimality_gap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot the computation time
    fig, ax = plt.subplots(figsize=(6.5, 1.5))
    ax.bar(labels, times, color=['blue', 'red'][:len(labels)], alpha=0.7)
    ax.set_ylabel('Computation Time (s)')
    ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(master_dir / Path(f'test_computation_time.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main(args):
    mdp_type = args.mdp_type
    n = args.n_states
    k = args.n_actions
    m = args.n_neighbors
    two_way = args.two_way
    
    plot_epoch(mdp_type, n, k, m, two_way)
    plot_eval_comparison(mdp_type, n, k, m, two_way)
    print(f"Plots saved to mdp_lp/results_lp/{mdp_type}/n={n};k={k}/m={m};twoway={two_way}/solver=dense/")


if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser(description='Plot LP results.')
    parser.add_argument('--mdp_type', type=str, default='bnd', help='the type of the MDPs: bnd or inventory')
    parser.add_argument('--n_states', type=int, default=2000, help='the number of states')
    parser.add_argument('--n_actions', type=int, default=10, help='the number of actions')
    parser.add_argument('--n_neighbors', type=int, default=2,
                        help='the number of neighboring states per row in the transition probs')
    parser.add_argument('--two_way', type=lambda s: s.lower() == 'y', default=True,
                        help='whether two-way transitions are allowed')

    main(parser.parse_args())
