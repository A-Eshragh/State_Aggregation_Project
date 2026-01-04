"""
Plotting functions for elasticity-based inventory experiments.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_elasticity_results(results, save_dir, algo):
    """Generate all required plots for elasticity experiments."""
    save_dir = Path(save_dir)
    
    # 1. Average total discounted reward per epoch
    plot_reward_per_epoch(results['train_reward_per_epoch'], save_dir, algo)
    
    # 2. Predicted vs true lambda histograms (final epoch)
    final_lambda_pred = results['train_lambda_pred'][-1]
    final_lambda_true = results['train_lambda_true'][-1]
    test_lambda_pred = results['test_lambda_pred']
    test_lambda_true = results['test_lambda_true']
    plot_lambda_histograms(final_lambda_pred, final_lambda_true, 
                          test_lambda_pred, test_lambda_true, save_dir, algo)
    
    # 3. Side-by-side box plot of final optimal value on test set
    plot_test_obj_boxplot(results['test_obj_values'], save_dir, algo)


def plot_reward_per_epoch(train_reward_per_epoch, save_dir, algo):
    """Plot average total discounted reward per epoch."""
    fig, ax = plt.subplots(figsize=(6.5, 1.5))
    epochs = range(1, len(train_reward_per_epoch) + 1)
    ax.plot(epochs, train_reward_per_epoch, linewidth=2, color='blue' if algo == 'dfl' else 'red')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Avg Discounted Reward')
    ax.set_title(f'{algo.upper()} - Training Reward per Epoch')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(save_dir / 'train_reward_per_epoch.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_lambda_histograms(train_lambda_pred, train_lambda_true, 
                          test_lambda_pred, test_lambda_true, save_dir, algo):
    """Plot predicted vs true lambda histograms."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Training set
    axes[0].hist(train_lambda_true, bins=30, alpha=0.5, label='True', color='blue')
    axes[0].hist(train_lambda_pred, bins=30, alpha=0.5, label='Predicted', color='red')
    axes[0].set_xlabel('λ (Poisson Rate)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{algo.upper()} - Training Set')
    axes[0].legend()
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # Test set
    axes[1].hist(test_lambda_true, bins=30, alpha=0.5, label='True', color='blue')
    axes[1].hist(test_lambda_pred, bins=30, alpha=0.5, label='Predicted', color='red')
    axes[1].set_xlabel('λ (Poisson Rate)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'{algo.upper()} - Test Set')
    axes[1].legend()
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'lambda_histograms.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_test_obj_boxplot(test_obj_values, save_dir, algo):
    """Plot box plot of final optimal value on test set."""
    fig, ax = plt.subplots(figsize=(6.5, 1.5))
    ax.boxplot([test_obj_values], labels=[algo.upper()])
    ax.set_ylabel('Optimal Value')
    ax.set_title(f'{algo.upper()} - Test Set Optimal Values')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(save_dir / 'test_obj_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_comparison_boxplot(dfl_values, twostage_values, save_dir):
    """Plot side-by-side box plot comparing DFL and Two-stage."""
    fig, ax = plt.subplots(figsize=(6.5, 1.5))
    ax.boxplot([dfl_values, twostage_values], labels=['DFL', 'Two-stage'])
    ax.set_ylabel('Optimal Value')
    ax.set_title('Test Set Optimal Values Comparison')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(save_dir / 'comparison_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()







