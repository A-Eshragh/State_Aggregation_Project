import json
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    gt_mode = 'nn'
    algos = ['dflsa', 'twostage']
    data_dir = Path(f'results/{gt_mode}')
    results = {}
    for algo_mode in algos:
        file_path = data_dir / Path(f'{algo_mode}/test_result.json')
        results[algo_mode] = json.loads(file_path.read_text())['obj_loss']

    # Plot the losses
    epochs = list(range(1, 101))
    fig, ax = plt.subplots(figsize=(6.5, 1.5))
    ax.plot(epochs, results['dflsa'], label='SAILO', linewidth=2, color='blue')
    ax.plot(epochs, results['twostage'], label='Two-stage', linewidth=2, linestyle='dashed', color='red')

    # Labels and title
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Optimality Gap')
    ax.legend(frameon=False)

    # Remove only the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save the figure
    plt.savefig(data_dir / Path('loss_plot.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
